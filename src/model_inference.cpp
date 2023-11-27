#include "crosstransformer.hpp"
#include "dsp.hpp"
#include "encdec.hpp"
#include "layers.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>

// Function to do reflection padding
static void reflect_padding(Eigen::MatrixXf &padded_mix,
                            const Eigen::MatrixXf &mix, int left_padding,
                            int right_padding)
{
    // Assumes padded_mix has size (2, N + left_padding + right_padding)
    // Assumes mix has size (2, N)

    int N = mix.cols(); // The original number of columns

    // Copy the original mix into the middle of padded_mix
    padded_mix.block(0, left_padding, 2, N) = mix;

    // Reflect padding on the left
    for (int i = 0; i < left_padding; ++i)
    {
        padded_mix.block(0, left_padding - 1 - i, 2, 1) = mix.block(0, i, 2, 1);
    }

    // Reflect padding on the right
    for (int i = 0; i < right_padding; ++i)
    {
        padded_mix.block(0, N + left_padding + i, 2, 1) =
            mix.block(0, N - 1 - i, 2, 1);
    }
}

void demucscpp::model_inference_4s(
    struct demucscpp::demucs_model_4s &model,
    struct demucscpp::demucs_segment_buffers_4s &buffers,
    struct demucscpp::stft_buffers &stft_buf)
{
    // apply demucs inference
    std::cout << "3., apply_model mix shape: (" << buffers.mix.rows() << ", "
              << buffers.mix.cols() << ")" << std::endl;

    // pad buffers.pad on the left, reflect
    // pad buffers.pad_end on the right, reflect
    // copy buffers.mix into buffers.padded_mix with reflect padding as above
    reflect_padding(buffers.padded_mix, buffers.mix, buffers.pad,
                    buffers.pad_end);

    // copy buffers.padded_mix into stft_buf.waveform
    stft_buf.waveform = buffers.padded_mix;

    // let's get a stereo complex spectrogram first
    demucscpp::stft(stft_buf);

    // remove 2: 2 + le of stft
    // same behavior as _spec in the python apply.py code
    buffers.z = stft_buf.spec.slice(
        Eigen::array<int, 3>{0, 0, 2},
        Eigen::array<int, 3>{2, (int)stft_buf.spec.dimension(1),
                             (int)stft_buf.spec.dimension(2) - 4});

    // print z shape
    std::cout << "buffers.z: " << buffers.z.dimension(0) << ", "
              << buffers.z.dimension(1) << ", " << buffers.z.dimension(2)
              << std::endl;

    // demucscppdebug::debug_tensor_3dxcf(buffers.z, "z!");

    // x = mag = z.abs(), but for CaC we're simply stacking the complex
    // spectrogram along the channel dimension
    for (int i = 0; i < buffers.z.dimension(0); ++i)
    {
        // limiting to j-1 because we're dropping 2049 to 2048 bins
        for (int j = 0; j < buffers.z.dimension(1) - 1; ++j)
        {
            for (int k = 0; k < buffers.z.dimension(2); ++k)
            {
                buffers.x(2 * i, j, k) = buffers.z(i, j, k).real();
                buffers.x(2 * i + 1, j, k) = buffers.z(i, j, k).imag();
            }
        }
    }

    // x shape is complex*chan, nb_frames, nb_bins (2048)
    // using CaC (complex-as-channels)
    // print x shape
    std::cout << "buffers.x: " << buffers.x.dimension(0) << ", "
              << buffers.x.dimension(1) << ", " << buffers.x.dimension(2)
              << std::endl;

    demucscppdebug::debug_tensor_3dxf(buffers.x, "x pre-std/mean");

    // apply following pytorch operations to buffers.x in Eigen C++ code:
    //  mean = x.mean(dim=(1, 2, 3), keepdim=True)
    //  std = x.std(dim=(1, 2, 3), keepdim=True)
    //  x = (x - mean) / (1e-5 + std)

    // Compute mean and standard deviation using Eigen
    Eigen::Tensor<float, 0> mean_tensor = buffers.x.mean();
    float mean = mean_tensor(0);
    float variance = demucscpp::calculate_variance(buffers.x, mean);
    float std_ = std::sqrt(variance);

    // Normalize x
    const float epsilon = 1e-5;

    buffers.x = (buffers.x - mean) / (std_ + epsilon);

    // buffers.x will be the freq branch input
    demucscppdebug::debug_tensor_3dxf(buffers.x, "x post-std/mean");

    // prepare time branch input by copying buffers.mix into buffers.xt(0, ...)
    for (int i = 0; i < buffers.mix.rows(); ++i)
    {
        for (int j = 0; j < buffers.mix.cols(); ++j)
        {
            buffers.xt(0, i, j) = buffers.mix(i, j);
        }
    }

    demucscppdebug::debug_tensor_3dxf(buffers.xt, "xt pre-std/mean");

    // apply similar mean, std normalization as above using 2d mean, std
    Eigen::Tensor<float, 0> meant_tensor = buffers.xt.mean();
    float meant = meant_tensor(0);
    float variancet = demucscpp::calculate_variance(buffers.xt, meant);
    float stdt = std::sqrt(variancet);

    // Normalize x
    buffers.xt = (buffers.xt - meant) / (stdt + epsilon);

    demucscppdebug::debug_tensor_3dxf(buffers.xt, "xt post-std/mean");

    // buffers.xt will be the time branch input

    /* HEART OF INFERENCE CODE HERE !! */
    // ITERATION 0

    // apply tenc, enc

    demucscppdebug::debug_tensor_3dxf(buffers.xt, "xt pre-encoder");
    demucscppdebug::debug_tensor_3dxf(buffers.x, "x pre-encoder");

    demucscpp::apply_time_encoder(model, 0, buffers.xt, buffers.xt_0);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_0, "xt_0 post-tencoder-0");

    demucscpp::apply_freq_encoder(model, 0, buffers.x, buffers.x_0);
    demucscppdebug::debug_tensor_3dxf(buffers.x_0, "x_0 post-encoder-0");

    // absorb both scaling factors in one expression
    //   i.e. eliminate const float freq_emb_scale = 0.2f;
    const float emb_scale = 10.0f * 0.2f;

    Eigen::MatrixXf emb =
        model.freq_emb_embedding_weight.transpose() * emb_scale;
    demucscppdebug::assert_(emb.rows() == 48);
    demucscppdebug::assert_(emb.cols() == 512);

    // apply embedding to buffers.x_0
    for (int i = 0; i < 48; ++i)
    {
        for (int j = 0; j < 512; ++j)
        {
            for (int k = 0; k < buffers.x_0.dimension(2); ++k)
            {
                // implicit broadcasting
                buffers.x_0(i, j, k) += emb(i, j);
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(buffers.x_0,
                                      "x_0 post-encoder-freq-emb-0");

    buffers.saved_0 = buffers.x_0;
    buffers.savedt_0 = buffers.xt_0;

    apply_time_encoder(model, 1, buffers.xt_0, buffers.xt_1);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_1, "xt_1 post-tencoder-1");

    apply_freq_encoder(model, 1, buffers.x_0, buffers.x_1);
    demucscppdebug::debug_tensor_3dxf(buffers.x_1, "x_1 post-encoder-1");

    buffers.saved_1 = buffers.x_1;
    buffers.savedt_1 = buffers.xt_1;

    apply_time_encoder(model, 2, buffers.xt_1, buffers.xt_2);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_2, "xt_2 post-tencoder-2");

    apply_freq_encoder(model, 2, buffers.x_1, buffers.x_2);
    demucscppdebug::debug_tensor_3dxf(buffers.x_2, "x_2 post-encoder-2");

    buffers.saved_2 = buffers.x_2;
    buffers.savedt_2 = buffers.xt_2;

    apply_time_encoder(model, 3, buffers.xt_2, buffers.xt_3);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_3, "xt_3 post-tencoder-3");

    apply_freq_encoder(model, 3, buffers.x_2, buffers.x_3);
    demucscppdebug::debug_tensor_3dxf(buffers.x_3, "x_3 post-encoder-3");

    buffers.saved_3 = buffers.x_3;
    buffers.savedt_3 = buffers.xt_3;

    // bottom channels = 512

    /*****************************/
    /*  FREQ CHANNEL UPSAMPLING  */
    /*****************************/

    // Reshape buffers.x_3 into x_3_reshaped
    // Apply the conv1d function
    // Reshape back to 512x8x336 and store in buffers.x_3_channel_upsampled
    Eigen::Tensor3dXf x_3_reshaped =
        buffers.x_3.reshape(Eigen::array<int, 3>({1, 384, 8 * 336}));
    Eigen::Tensor3dXf x_3_reshaped_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(x_3_reshaped, model.channel_upsampler_weight,
                          model.channel_upsampler_bias);
    buffers.x_3_channel_upsampled =
        x_3_reshaped_upsampled.reshape(Eigen::array<int, 3>({512, 8, 336}));

    /*****************************/
    /*  TIME CHANNEL UPSAMPLING  */
    /*****************************/

    // for time channel upsampling
    // apply upsampler directly to xt_3 no reshaping drama needed
    buffers.xt_3_channel_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(buffers.xt_3, model.channel_upsampler_t_weight,
                          model.channel_upsampler_t_bias);

    demucscppdebug::debug_tensor_3dxf(buffers.x_3_channel_upsampled,
                                      "x pre-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(buffers.xt_3_channel_upsampled,
                                      "xt pre-crosstransformer");

    /*************************/
    /*  CROSS-TRANSFORMER!  */
    /************************/
    demucscpp::apply_crosstransformer(model, buffers.x_3_channel_upsampled,
                                      buffers.xt_3_channel_upsampled);

    // reshape buffers.x_3_channel_upsampled into 1, 512, 2688
    // when skipping the crosstransformer
    // Eigen::Tensor3dXf x_3_reshaped_upsampled_2 =
    // buffers.x_3_channel_upsampled.reshape(Eigen::array<int, 3>({1, 512,
    // 8*336})); buffers.x_3_channel_upsampled = x_3_reshaped_upsampled_2;

    demucscppdebug::debug_tensor_3dxf(buffers.x_3_channel_upsampled,
                                      "x post-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(buffers.xt_3_channel_upsampled,
                                      "xt post-crosstransformer");
    // then apply the conv1d_2d function

    Eigen::Tensor3dXf x_3_reshaped_downsampled = demucscpp::conv1d<512, 384, 1, 1, 0, 0>(
        buffers.x_3_channel_upsampled, model.channel_downsampler_weight,
        model.channel_downsampler_bias);
    buffers.x_3 =
        x_3_reshaped_downsampled.reshape(Eigen::array<int, 3>({384, 8, 336}));

    // apply upsampler directly to xt_3
    buffers.xt_3 = demucscpp::conv1d<512, 384, 1, 1, 0, 0>(
        buffers.xt_3_channel_upsampled, model.channel_downsampler_t_weight,
        model.channel_downsampler_t_bias);

    // now decoder time!

    // skip == saved_3
    demucscppdebug::debug_tensor_3dxf(buffers.x_3, "x_3 pre-decoder");
    demucscppdebug::debug_tensor_3dxf(buffers.saved_3,
                                      "saved_3/skip pre-decoder");
    demucscpp::apply_freq_decoder(model, 0, buffers.x_3, buffers.x_2,
                                  buffers.saved_3);
    demucscppdebug::debug_tensor_3dxf(buffers.x_2, "buffers.x_2 post-decoder");

    demucscppdebug::debug_tensor_3dxf(buffers.xt_3, "xt_3 pre-tdecoder");
    demucscppdebug::debug_tensor_3dxf(buffers.savedt_3,
                                      "savedt_3/skip pre-tdecoder");
    demucscpp::apply_time_decoder(model, 0, buffers.xt_3, buffers.xt_2,
                                  buffers.savedt_3);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_2,
                                      "buffers.xt_2 post-tdecoder");

    demucscppdebug::debug_tensor_3dxf(buffers.x_2, "x_2 pre-decoder");
    demucscppdebug::debug_tensor_3dxf(buffers.saved_2,
                                      "saved_2/skip pre-decoder");
    demucscpp::apply_freq_decoder(model, 1, buffers.x_2, buffers.x_1,
                                  buffers.saved_2);
    demucscppdebug::debug_tensor_3dxf(buffers.x_1, "buffers.x_1 post-decoder");

    demucscppdebug::debug_tensor_3dxf(buffers.xt_2, "xt_2 pre-tdecoder");
    demucscppdebug::debug_tensor_3dxf(buffers.savedt_2,
                                      "savedt_2/skip pre-tdecoder");
    demucscpp::apply_time_decoder(model, 1, buffers.xt_2, buffers.xt_1,
                                  buffers.savedt_2);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_1,
                                      "buffers.xt_1 post-tdecoder");

    demucscppdebug::debug_tensor_3dxf(buffers.x_1, "x_1 pre-decoder");
    demucscppdebug::debug_tensor_3dxf(buffers.saved_1,
                                      "saved_1/skip pre-decoder");
    demucscpp::apply_freq_decoder(model, 2, buffers.x_1, buffers.x_0,
                                  buffers.saved_1);
    demucscppdebug::debug_tensor_3dxf(buffers.x_0, "buffers.x_0 post-decoder");

    demucscppdebug::debug_tensor_3dxf(buffers.xt_1, "xt_1 pre-tdecoder");
    demucscppdebug::debug_tensor_3dxf(buffers.savedt_1,
                                      "savedt_1/skip pre-tdecoder");
    demucscpp::apply_time_decoder(model, 2, buffers.xt_1, buffers.xt_0,
                                  buffers.savedt_1);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_0,
                                      "buffers.xt_0 post-tdecoder");

    demucscppdebug::debug_tensor_3dxf(buffers.x_0, "x_0 pre-decoder");
    demucscppdebug::debug_tensor_3dxf(buffers.saved_0,
                                      "saved_0/skip pre-decoder");
    demucscpp::apply_freq_decoder(model, 3, buffers.x_0, buffers.x_out,
                                  buffers.saved_0);
    demucscppdebug::debug_tensor_3dxf(buffers.x_out,
                                      "buffers.x_out post-decoder");

    demucscppdebug::debug_tensor_3dxf(buffers.xt_0, "xt_0 pre-tdecoder");
    demucscppdebug::debug_tensor_3dxf(buffers.savedt_0,
                                      "savedt_0/skip pre-tdecoder");
    demucscpp::apply_time_decoder(model, 3, buffers.xt_0, buffers.xt_out,
                                  buffers.savedt_0);
    demucscppdebug::debug_tensor_3dxf(buffers.xt_out,
                                      "buffers.xt_out post-tdecoder");

    std::cout << "Now mask time!" << std::endl;
    demucscppdebug::assert_(4 * 4 == buffers.x_out.dimension(0));

    // xt dim 1 is a fake dim of 1
    // so we could have symmetry between the tensor3dxf of the freq and time
    // branches
    demucscppdebug::assert_(4 * 2 == buffers.xt_out.dimension(1));

    // 4 sources, 2 channels * 2 complex channels (real+imag), F bins, T frames
    Eigen::Tensor4dXf x_4d =
        Eigen::Tensor4dXf(4, 4, buffers.x.dimension(1), buffers.x.dimension(2));

    // 4 sources, 2 channels, N samples
    std::array<Eigen::MatrixXf, 4> xt_3d = {
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2)),
        Eigen::MatrixXf(2, buffers.xt.dimension(2))};

    demucscppdebug::debug_tensor_3dxf(buffers.x_out, "x (freq out) pre-norm");
    demucscppdebug::debug_tensor_3dxf(buffers.xt_out, "xt (time out) pre-norm");

    // distribute the channels of buffers.x into x_4d
    // in pytorch it's (16, 2048, 336) i.e. (chan, freq, time)
    // then apply `.view(4, -1, freq, time)

    // implement above logic in Eigen C++
    // copy buffers.x into x_4d
    // apply opposite of
    // buffers.x(i, j, k) = (buffers.x(i, j, k) - mean) / (epsilon + std_);
    for (int s = 0; s < 4; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < buffers.x.dimension(1); ++j)
            {
                for (int k = 0; k < buffers.x.dimension(2); ++k)
                {
                    x_4d(s, i, j, k) =
                        std_ * buffers.x_out(s * 4 + i, j, k) + mean;
                }
            }
        }
    }

    demucscppdebug::debug_tensor_4dxf(x_4d, "x (freq out) post-norm");

    // let's also copy buffers.xt into xt_4d
    for (int s = 0; s < 4; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < buffers.xt.dimension(2); ++j)
            {
                xt_3d[s](i, j) = stdt * buffers.xt_out(0, s * 2 + i, j) + meant;
            }
        }
    }
    demucscppdebug::debug_matrix_xf(xt_3d[0], "xt (0) (time out) post-norm");
    demucscppdebug::debug_matrix_xf(xt_3d[1], "xt (1) (time out) post-norm");
    demucscppdebug::debug_matrix_xf(xt_3d[2], "xt (2) (time out) post-norm");
    demucscppdebug::debug_matrix_xf(xt_3d[3], "xt (3) (time out) post-norm");

    // If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
    // undo complex-as-channels by splitting the 2nd dim of x_4d into (2, 2)
    for (int source = 0; source < 4; ++source)
    {
        Eigen::Tensor3dXcf x_target = Eigen::Tensor3dXcf(
            2, buffers.x.dimension(1), buffers.x.dimension(2));

        // in the CaC case, we're simply unstacking the complex
        // spectrogram from the channel dimension
        for (int i = 0; i < buffers.z.dimension(0); ++i)
        {
            // limiting to j-1 because we're dropping 2049 to 2048 bins
            for (int j = 0; j < buffers.z.dimension(1) - 1; ++j)
            {
                for (int k = 0; k < buffers.z.dimension(2); ++k)
                {
                    // buffers.x(2*i, j, k) = buffers.z(i, j, k).real();
                    // buffers.x(2*i + 1, j, k) = buffers.z(i, j, k).imag();
                    x_target(i, j, k) =
                        std::complex<float>(x_4d(source, 2 * i, j, k),
                                            x_4d(source, 2 * i + 1, j, k));
                }
            }
        }

        // need to re-pad 2: 2 + le on spectrogram
        // opposite of this
        // buffers.z = stft_buf.spec.slice(Eigen::array<int, 3>{0, 0, 2},
        //         Eigen::array<int, 3>{2, (int)stft_buf.spec.dimension(1),
        //         (int)stft_buf.spec.dimension(2) - 4});
        // Add padding to spectrogram

        Eigen::array<std::pair<int, int>, 3> paddings = {
            std::make_pair(0, 0), std::make_pair(0, 1), std::make_pair(2, 2)};
        Eigen::Tensor3dXcf x_target_padded =
            x_target.pad(paddings, std::complex<float>(0.0f, 0.0f));

        stft_buf.spec = x_target_padded;

        demucscpp::istft(stft_buf);

        demucscppdebug::debug_matrix_xf(stft_buf.waveform,
                                        "x (freq out) post-mask-istft");

        // now we have waveform from istft(x), the frequency branch
        // that we need to sum with xt, the time branch
        Eigen::MatrixXf padded_waveform = stft_buf.waveform;

        // undo the reflect pad 1d by copying padded_mix into mix
        // from range buffers.pad:buffers.pad + buffers.segment_samples
        Eigen::MatrixXf unpadded_waveform =
            padded_waveform.block(0, buffers.pad, 2, buffers.segment_samples);

        // sum with xt
        // choose a different source to sum with in case
        // they're in different orders...
        unpadded_waveform += xt_3d[source];

        demucscppdebug::debug_matrix_xf(unpadded_waveform,
                                        "unpadded waveform post-sum target: " +
                                            std::to_string(source));

        std::cout << "mix: " << buffers.mix.rows() << ", " << buffers.mix.cols()
                  << std::endl;

        // copy target waveform into all 4 dims of targets_out
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < buffers.mix.cols(); ++k)
            {
                buffers.targets_out(source, j, k) = unpadded_waveform(j, k);
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(buffers.targets_out,
                                      "x total out from model inference");
}
