#include "dsp.hpp"
#include "model.hpp"
#include "tensor.hpp"
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
#include <unsupported/Eigen/FFT>
#include <vector>

using namespace demucscpp_v3;
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
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <model file> <wav file> <out dir>" << std::endl;
        exit(1);
    }

    std::cout << "demucs.cpp Main driver program" << std::endl;

    // load model passed as argument
    std::string model_file = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // output dir passed as argument
    std::string out_dir = argv[3];

    Eigen::MatrixXf audio = load_audio_file(wav_file);
    Eigen::Tensor3dXf out_targets;

    // initialize a struct demucs_model
    struct demucs_v3_model model
    {
    };

    // debug some members of model
    auto ret = load_demucs_v3_model(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    const int nb_sources = 4;

    std::cout << "Starting Demucs v3 MMI inference" << std::endl;

    // set output precision to 3 decimal places
    std::cout << std::fixed << std::setprecision(3);

    demucscpp::ProgressCallback progressCallback =
        [](float progress, const std::string &log_message)
    {
        std::cout << "(" << std::setw(3) << std::setfill(' ')
                  << progress * 100.0f << "%) " << log_message << std::endl;
    };

    // create 4 audio matrix same size, to hold output
    Eigen::Tensor3dXf audio_targets =
        demucscpp_v3::demucs_v3_inference(model, audio, progressCallback);

    out_targets = audio_targets;

    const int nb_out_sources = nb_sources;

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
