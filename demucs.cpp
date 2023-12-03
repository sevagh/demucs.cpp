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

    // enable openmp parallelization for Eigen
    // init parallelism for eigen
    // Eigen::initParallel();

    //// set eigen nb threads to physical cores minus 1
    //// discover number of physical cores through C++ stdlib
    ////
    ///https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
    // int nb_cores = std::thread::hardware_concurrency();
    // std::cout << "Number of physical cores: " << nb_cores << std::endl;
    // Eigen::setNbThreads(nb_cores - 1);

    std::cout << "demucs.cpp Main driver program" << std::endl;

    // load model passed as argument
    std::string model_file = argv[1];

    // load audio passed as argument
    std::string wav_file = argv[2];

    // output dir passed as argument
    std::string out_dir = argv[3];

    Eigen::MatrixXf audio = load_audio(wav_file);
    Eigen::Tensor3dXf out_targets;

    std::cout << "Using 4s model" << std::endl;

    // initialize a struct demucs_model
    struct demucs_model_4s model
    {
    };

    auto ret = load_demucs_model_4s(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    std::cout << "Starting demucs inference" << std::endl;

    // create 4 audio matrix same size, to hold output
    Eigen::Tensor3dXf audio_targets =
        demucscpp::demucs_inference_4s(model, audio);
    std::cout << "returned!" << std::endl;

    out_targets = audio_targets;

    for (int target = 0; target < 4; ++target)
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
