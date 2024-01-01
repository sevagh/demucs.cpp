// use gtest to test the load_audio_for_kissfft function

#include "dsp.hpp"
#include "tensor.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <random>

#define NEAR_TOLERANCE 1e-4

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
                  << demucscpp::SUPPORTED_SAMPLE_RATE << std::endl;
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

// write a basic test case for a mono file
TEST(LoadAudioForKissfft, LoadMonoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_mono.wav";
    Eigen::MatrixXf ret = load_audio_file(filename);

    // check the number of samples
    EXPECT_EQ(ret.cols(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret(0, 0), ret(1, 0));
    EXPECT_EQ(ret(0, 262143), ret(1, 262143));
}

// write a basic test case for a stereo file
TEST(LoadAudioForKissfft, LoadStereoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_stereo.wav";

    Eigen::MatrixXf ret = load_audio_file(filename);

    // check the number of samples
    EXPECT_EQ(ret.cols(), 262144);

    // check the first and last samples
    EXPECT_EQ(ret(0, 0), ret(1, 0));
    EXPECT_EQ(ret(0, 262143), ret(1, 262143));
}

// write a basic test case for the stft function
TEST(DSP_STFT, STFTRoundtripRandWaveform)
{
    Eigen::MatrixXf audio_in(2, 4096);

    // populate the audio_in with some random data
    // between -1 and 1
    for (size_t i = 0; i < 4096; ++i)
    {
        audio_in(0, i) = (float)rand() / (float)RAND_MAX;
        audio_in(1, i) = (float)rand() / (float)RAND_MAX;
    }

    // initialize stft_buffers struct
    demucscpp::stft_buffers stft_buf(audio_in.cols());

    // copy the audio_in into stft_buf.waveform

    // print rows and cols dimensions of stft_buf.waveform and audio_in
    std::cout << "stft_buf.waveform.rows(): " << stft_buf.waveform.rows()
              << std::endl;
    std::cout << "stft_buf.waveform.cols(): " << stft_buf.waveform.cols()
              << std::endl;
    std::cout << "audio_in.rows(): " << audio_in.rows() << std::endl;
    std::cout << "audio_in.cols(): " << audio_in.cols() << std::endl;

    stft_buf.waveform = audio_in;

    // compute the stft
    demucscpp::stft(stft_buf);

    Eigen::Tensor3dXcf spec = stft_buf.spec;

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.dimension(2);

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.dimension(1), 2049);

    demucscpp::istft(stft_buf);

    Eigen::MatrixXf audio_out = stft_buf.waveform;

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (long int i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}

// write a basic test case for the stft function
// with real gspi.wav
TEST(DSP_STFT, STFTRoundtripGlockenspiel)
{
    Eigen::MatrixXf audio_in = load_audio_file("../test/data/gspi_mono.wav");

    // create buffers
    demucscpp::stft_buffers stft_buf(audio_in.cols());

    // copy the audio_in into stft_buf.waveform
    stft_buf.waveform = audio_in;

    // compute the stft
    demucscpp::stft(stft_buf);
    Eigen::Tensor3dXcf spec = stft_buf.spec;

    // check the number of frequency bins per frames, first and last
    auto n_frames = spec.dimension(2);

    std::cout << "n_frames: " << n_frames << std::endl;

    EXPECT_EQ(spec.dimension(1), 2049);

    demucscpp::istft(stft_buf);

    Eigen::MatrixXf audio_out = stft_buf.waveform;

    EXPECT_EQ(audio_in.rows(), audio_out.rows());
    EXPECT_EQ(audio_in.cols(), audio_out.cols());

    for (long int i = 0; i < audio_in.cols(); ++i)
    {
        // expect similar samples with a small floating point error
        EXPECT_NEAR(audio_in(0, i), audio_out(0, i), NEAR_TOLERANCE);
        EXPECT_NEAR(audio_in(1, i), audio_out(1, i), NEAR_TOLERANCE);
    }
}
