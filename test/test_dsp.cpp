// use gtest to test the load_audio_for_kissfft function

#include "dsp.hpp"
#include <gtest/gtest.h>
#include <random>

#define NEAR_TOLERANCE 1e-4

// write a basic test case for a mono file
TEST(LoadAudioForKissfft, LoadMonoAudio)
{
    // load a wav file with libnyquist
    std::string filename = "../test/data/gspi_mono.wav";
    Eigen::MatrixXf ret = demucscpp::load_audio(filename);

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

    Eigen::MatrixXf ret = demucscpp::load_audio(filename);

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
    Eigen::MatrixXf audio_in =
        demucscpp::load_audio("../test/data/gspi_mono.wav");

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
