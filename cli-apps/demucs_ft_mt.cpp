#include "dsp.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include "threaded_inference.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <sstream>
#include <string>
#include <thread>
#include <unsupported/Eigen/FFT>
#include <vector>

using namespace demucscpp;
using namespace nqr;

static Eigen::MatrixXf load_audio_file(std::string filename)
{
    // load a wav file with libnyquist
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    NyquistIO loader;

    loader.Load(fileData.get(), filename);

    if (fileData->sampleRate != demucscpp::SUPPORTED_SAMPLE_RATE)
    {
        std::cerr << "[ERROR] demucs.cpp only supports the following sample "
                     "rate (Hz): "
                  << SUPPORTED_SAMPLE_RATE << std::endl;
        exit(1);
    }

    std::cout << "Input samples: "
              << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;

    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr << "[ERROR] demucs.cpp only supports mono and stereo audio"
                  << std::endl;
        exit(1);
    }

    // number of samples per channel
    size_t N = fileData->samples.size() / fileData->channelCount;

    // create a struct to hold two float vectors for left and right channels
    Eigen::MatrixXf ret(2, N);

    if (fileData->channelCount == 1)
    {
        // Mono case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[i]; // left channel
            ret(1, i) = fileData->samples[i]; // right channel
        }
    }
    else
    {
        // Stereo case
        for (size_t i = 0; i < N; ++i)
        {
            ret(0, i) = fileData->samples[2 * i];     // left channel
            ret(1, i) = fileData->samples[2 * i + 1]; // right channel
        }
    }

    return ret;
}

// write a function to write a StereoWaveform to a wav file
static void write_audio_file(const Eigen::MatrixXf &waveform,
                             std::string filename)
{
    // create a struct to hold the audio data
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();

    // set the sample rate
    fileData->sampleRate = SUPPORTED_SAMPLE_RATE;

    // set the number of channels
    fileData->channelCount = 2;

    // set the number of samples
    fileData->samples.resize(waveform.cols() * 2);

    // write the left channel
    for (long int i = 0; i < waveform.cols(); ++i)
    {
        fileData->samples[2 * i] = waveform(0, i);
        fileData->samples[2 * i + 1] = waveform(1, i);
    }

    int encoderStatus =
        encode_wav_to_disk({fileData->channelCount, PCM_FLT, DITHER_TRIANGLE},
                           fileData.get(), filename);
    std::cout << "Encoder Status: " << encoderStatus << std::endl;
}

int main(int argc, const char **argv)
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <model dir> <wav file> <out dir> <num threads>"
                  << std::endl;
        exit(1);
    }

    std::cout << "demucs_ft_mt.cpp (Multi-threaded Fine-tuned) driver program"
              << std::endl;

    // load model passed as argument
    std::string model_dir = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // output dir passed as argument
    std::string out_dir = argv[3];

    // get num threads from user parameter argv[4]
    // cast it to int
    int num_threads = std::stoi(argv[4]);

    Eigen::MatrixXf audio = load_audio_file(wav_file);
    Eigen::Tensor3dXf out_targets;

    // initialize nested 4 fine-tuned struct demucs_model
    std::array<struct demucs_model, 4> models = {
        demucs_model(), demucs_model(), demucs_model(), demucs_model()};

    // iterate over all files in model_dir
    // and load the model
    for (const auto &entry : std::filesystem::directory_iterator(model_dir))
    {
        bool ret = false;

        // check if entry contains the name "htdemucs_ft_drums"
        if (entry.path().string().find("htdemucs_ft_drums") !=
            std::string::npos)
        {
            ret = load_demucs_model(entry.path().string(), &models[0]);
            std::cout << "Loading ft model " << entry.path().string()
                      << " for drums" << std::endl;
        }
        else if (entry.path().string().find("htdemucs_ft_bass") !=
                 std::string::npos)
        {
            ret = load_demucs_model(entry.path().string(), &models[1]);
            std::cout << "Loading ft model " << entry.path().string()
                      << " for bass" << std::endl;
        }
        else if (entry.path().string().find("htdemucs_ft_other") !=
                 std::string::npos)
        {
            ret = load_demucs_model(entry.path().string(), &models[2]);
            std::cout << "Loading ft model " << entry.path().string()
                      << " for other" << std::endl;
        }
        else if (entry.path().string().find("htdemucs_ft_vocals") !=
                 std::string::npos)
        {
            ret = load_demucs_model(entry.path().string(), &models[3]);
            std::cout << "Loading ft model " << entry.path().string()
                      << " for vocals" << std::endl;
        }
        else
        {
            continue;
        }

        // debug some members of model
        std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
                  << std::endl;
        if (!ret)
        {
            std::cerr << "Error loading model" << std::endl;
            exit(1);
        }
    }

    const int nb_sources = 4;

    std::cout << "Starting Demucs fine-tuned (" << std::to_string(nb_sources)
              << "-source) inference" << std::endl;

    // create 4 audio matrix same size, to hold output
    Eigen::Tensor3dXf drums_targets = demucscppthreaded::threaded_inference(
        models[0], audio, num_threads, "DRUMS\t ");

    Eigen::Tensor3dXf bass_targets = demucscppthreaded::threaded_inference(
        models[1], audio, num_threads, "BASS\t ");

    Eigen::Tensor3dXf other_targets = demucscppthreaded::threaded_inference(
        models[2], audio, num_threads, "OTHER\t ");

    Eigen::Tensor3dXf vocals_targets = demucscppthreaded::threaded_inference(
        models[3], audio, num_threads, "VOCALS\t ");

    out_targets = Eigen::Tensor3dXf(drums_targets.dimension(0),
                                    drums_targets.dimension(1),
                                    drums_targets.dimension(2));

    // simply use the respective stem from each independent fine-tuned model
    out_targets.chip<0>(0) = drums_targets.chip<0>(0);
    out_targets.chip<0>(1) = bass_targets.chip<0>(1);
    out_targets.chip<0>(2) = other_targets.chip<0>(2);
    out_targets.chip<0>(3) = vocals_targets.chip<0>(3);

    const int nb_out_sources = 4;

    for (int target = 0; target < nb_out_sources; ++target)
    {
        // now write the 4 audio waveforms to files in the output dir
        // using libnyquist
        // join out_dir with "/target_0.wav"
        // using std::filesystem::path;

        std::filesystem::path p = out_dir;
        // make sure the directory exists
        std::filesystem::create_directories(p);

        auto p_target = p / "target_0.wav";

        // target 0,1,2,3 map to drums,bass,other,vocals

        std::string target_name;

        switch (target)
        {
        case 0:
            target_name = "drums";
            break;
        case 1:
            target_name = "bass";
            break;
        case 2:
            target_name = "other";
            break;
        case 3:
            target_name = "vocals";
            break;
        case 4:
            target_name = "guitar";
            break;
        case 5:
            target_name = "piano";
            break;
        default:
            std::cerr << "Error: target " << target << " not supported"
                      << std::endl;
            exit(1);
        }

        // insert target_name into the path after the digit
        // e.g. target_name_0_drums.wav
        p_target.replace_filename("target_" + std::to_string(target) + "_" +
                                  target_name + ".wav");

        std::cout << "Writing wav file " << p_target << std::endl;

        Eigen::MatrixXf target_waveform(2, audio.cols());

        // copy the input stereo wav file into all 4 targets
        for (int channel = 0; channel < 2; ++channel)
        {
            for (int sample = 0; sample < audio.cols(); ++sample)
            {
                target_waveform(channel, sample) =
                    out_targets(target, channel, sample);
            }
        }

        write_audio_file(target_waveform, p_target);
    }
}
