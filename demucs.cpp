#include "dsp.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unsupported/Eigen/FFT>
#include <vector>

using namespace demucscpp;

int main(int argc, const char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <model file> <wav file> <out dir>" << std::endl;
        exit(1);
    }

    // set eigen nb threads to physical cores minus 1
    // discover number of physical cores through C++ stdlib
    // https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
    int nb_cores = std::thread::hardware_concurrency();
    std::cout << "Number of physical cores: " << nb_cores << std::endl;
    Eigen::setNbThreads(nb_cores - 1);

    std::cout << "demucs.cpp Main driver program" << std::endl;

    // load model passed as argument
    std::string model_file = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // output dir passed as argument
    std::string out_dir = argv[3];

    Eigen::MatrixXf audio = load_audio(wav_file);
    Eigen::Tensor3dXf out_targets;

    // initialize a struct demucs_model
    struct demucs_model model
    {
    };

    // debug some members of model
    auto ret = load_demucs_model(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    int nb_sources = model.is_4sources ? 4 : 6;

    std::cout << "Starting Demucs (" << std::to_string(nb_sources) <<"-source) inference" << std::endl;

    // create 4 audio matrix same size, to hold output
    Eigen::Tensor3dXf audio_targets =
        demucscpp::demucs_inference(model, audio);
    std::cout << "returned!" << std::endl;

    out_targets = audio_targets;

    int nb_out_sources = model.is_4sources ? 4 : 6;

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
        // generate p_target = p / "target_{target}.wav"
        p_target.replace_filename("target_" + std::to_string(target) + ".wav");

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

        demucscppdebug::debug_matrix_xf(target_waveform,
                                        "target_waveform for target " +
                                            std::to_string(target));

        demucscpp::write_audio_file(target_waveform, p_target);
    }
}
