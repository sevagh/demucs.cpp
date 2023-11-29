// use gtest to test the load_audio_for_kissfft function

#include "crosstransformer.hpp"
#include "dsp.hpp"
#include "conv.hpp"
#include "encdec.hpp"
#include "layers.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>

#define NEAR_TOLERANCE 1e-4

// initialize a struct demucs_model
static struct demucscpp::demucs_model_4s model
{
};
static bool loaded = false;

// google test global setup for model before all tests
static void setUpTestSuite()
{
    if (loaded)
    {
        return;
    }

    // load model from "../ggml-demucs/ggml-model-htdemucs-f16.bin"
    std::string model_file = "../ggml-demucs/ggml-model-htdemucs-f16.bin";

    auto ret = load_demucs_model_4s(model_file, &model);
    std::cout << "demucs_model_load returned " << (ret ? "true" : "false")
              << std::endl;
    if (!ret)
    {
        std::cerr << "Error loading model" << std::endl;
        exit(1);
    }

    loaded = true;
}

//TEST(DemucsCPPLayers, Im2colCol2im)
//{
//    setUpTestSuite();
//
//    std::cout << std::fixed << std::setprecision(20) << std::endl;
//
//    Eigen::Tensor3dXf x_fake(2, 3, 3);
//
//    // fill x_fake with values 1-18
//    for (size_t i = 0; i < 2; ++i)
//    {
//        for (size_t j = 0; j < 3; ++j)
//        {
//            for (size_t k = 0; k < 3; ++k)
//            {
//                x_fake(i, j, k) = i * 9 + j * 3 + k + 1;
//            }
//        }
//    }
//
//    demucscppdebug::debug_tensor_3dxf(x_fake, "x_fake");
//
//    // use kernel=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1)
//    Eigen::MatrixXf x_im2col = demucscpp::im2col(x_fake, 1, 1, 1, 1, 0, 0, 1, 1);
//
//    demucscppdebug::debug_matrix_xf(x_im2col, "x_im2col");
//
//    Eigen::Tensor3dXf x_col2im = demucscpp::col2im(x_im2col, 2, 3, 3, 1, 1, 1, 1, 0, 0, 1, 1);
//
//    demucscppdebug::debug_tensor_3dxf(x_col2im, "x_col2im");
//
//    Eigen::Tensor3dXf x_fake_2(2, 4, 4);
//
//    // Fill x_fake_2 with values
//    int counter = 1;
//    for (int i = 0; i < x_fake_2.dimension(0); ++i) {
//        for (int j = 0; j < x_fake_2.dimension(1); ++j) {
//            for (int k = 0; k < x_fake_2.dimension(2); ++k) {
//                x_fake_2(i, j, k) = counter++;
//            }
//        }
//    }
//
//    demucscppdebug::debug_tensor_3dxf(x_fake_2, "x_fake_2");
//
//    // Parameters: kernel=(2,2), stride=(2,2), padding=(0,0)
//    Eigen::MatrixXf x_im2col_2 = demucscpp::im2col(x_fake_2, 2, 2, 2, 2, 0, 0, 1, 1);
//    demucscppdebug::debug_matrix_xf(x_im2col_2, "x_im2col_2");
//
//    // Reverse im2col
//    Eigen::Tensor3dXf x_col2im_2 = demucscpp::col2im(x_im2col_2, 2, 4, 4, 2, 2, 2, 2, 0, 0, 1, 1);
//    demucscppdebug::debug_tensor_3dxf(x_col2im_2, "x_col2im_2");
//
//    Eigen::Tensor3dXf x_fake_3(2, 4, 4);
//
//    // Fill x_fake_3 with values
//    int counter_2 = 1;
//    for (int i = 0; i < x_fake_3.dimension(0); ++i) {
//        for (int j = 0; j < x_fake_3.dimension(1); ++j) {
//            for (int k = 0; k < x_fake_3.dimension(2); ++k) {
//                x_fake_3(i, j, k) = counter_2++;
//            }
//        }
//    }
//
//    demucscppdebug::debug_tensor_3dxf(x_fake_3, "x_fake_3");
//
//    // Parameters: kernel=(2,2), stride=(2,2), padding=(0,0)
//    Eigen::MatrixXf x_im2col_3 = demucscpp::im2col(x_fake_3, 2, 2, 2, 2, 1, 1, 1, 1);
//    demucscppdebug::debug_matrix_xf(x_im2col_3, "x_im2col_3");
//
//    // Reverse im2col
//    Eigen::Tensor3dXf x_col2im_3 = demucscpp::col2im(x_im2col_3, 2, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1);
//    demucscppdebug::debug_tensor_3dxf(x_col2im_3, "x_col2im_3");
//
//    // dilation test
//    Eigen::Tensor3dXf x_dilation_test(2, 6, 6);  // Example tensor
//
//    // Fill x_dilation_test with values
//    int counter_3 = 1;
//    for (int i = 0; i < x_dilation_test.dimension(0); ++i) {
//        for (int j = 0; j < x_dilation_test.dimension(1); ++j) {
//            for (int k = 0; k < x_dilation_test.dimension(2); ++k) {
//                x_dilation_test(i, j, k) = counter_3++;
//            }
//        }
//    }
//
//    demucscppdebug::debug_tensor_3dxf(x_dilation_test, "x_dilation_test");
//
//    // Dilation test parameters
//    int kernel_height = 3;
//    int kernel_width = 3;
//    int stride_height = 1;
//    int stride_width = 1;
//    int pad_height = 0;
//    int pad_width = 0;
//    int dilation_height = 2;
//    int dilation_width = 2;
//
//    // Apply im2col with dilation
//    Eigen::MatrixXf x_im2col_dilated = demucscpp::im2col(x_dilation_test, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, dilation_height, dilation_width);
//    demucscppdebug::debug_matrix_xf(x_im2col_dilated, "x_im2col_dilated");
//
//    // Reverse with col2im
//    Eigen::Tensor3dXf x_col2im_dilated = demucscpp::col2im(x_im2col_dilated, 2, 6, 6, kernel_height, kernel_width, stride_height, stride_width, pad_height, pad_width, dilation_height, dilation_width);
//    demucscppdebug::debug_tensor_3dxf(x_col2im_dilated, "x_col2im_dilated");
//}

TEST(DemucsCPPLayers, GemmConv)
{
    constexpr int in_channels = 1;
    constexpr int in_height = 13;
    constexpr int in_width = 9;

    constexpr int out_channels = 5;
    constexpr int kernel_height = 2;
    constexpr int kernel_width = 3;

    // x: 2 in channels, 3x4 image per channel
    Eigen::Tensor3dXf x(in_channels, in_height, in_width);

    // fill x with values
    //int counter = 0;
    for (size_t i = 0; i < x.dimension(0); ++i)
    {
        for (size_t j = 0; j < x.dimension(1); ++j)
        {
            for (size_t k = 0; k < x.dimension(2); ++k)
            {
                if (k % 2 == 0) {
                    x(i, j, k) = 0.5;//counter++;
                }
                else {
                    x(i, j, k) = -0.75;
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_3dxf(x, "x");
    Eigen::Tensor4dXf w(out_channels, in_channels, kernel_height, kernel_width);

    int counter = 1;
    for (size_t i = 0; i < w.dimension(0); ++i)
    {
        for (size_t j = 0; j < w.dimension(1); ++j)
        {
            for (size_t k = 0; k < w.dimension(2); ++k)
            {
                for (size_t l = 0; l < w.dimension(3); ++l)
                {
                    w(i, j, k, l) = 0.1f*((float)counter);
                    if (j % 2 == 0) {
                        w(i, j, k, l) *= -1.0f;
                    }
                    counter++;
                    // set weights to be -0.5, 0.5 alternating over kernel_height
                    //if (k % 2 == 0) {
                    //    w(i, j, k, l) = 0.5;
                    //}
                    //else {
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //if (l % 2 == 0)
                    //{
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //else
                    //{
                    //    w(i, j, k, l) = 0.5;
                    //}
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_4dxf(w, "w");

    // bias: 4 out channels
    Eigen::Tensor1dXf b(out_channels);

    // set bias to be 0.75 or -0.25 alternating
    for (size_t i = 0; i < out_channels; ++i)
    {
        b(i) = 0.0f;
        //if (i % 2 == 0)
        //{
        //    b(i) = 0.75;
        //}
        //else
        //{
        //    b(i) = -0.25;
        //}
    }

    //demucscppdebug::debug_tensor_1dxf(b, "b");

    // apply conv2d_gemm with some params
    Eigen::Tensor3dXf y_gemm = demucscpp::conv2d_gemm<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // apply regular conv2d with some params
    Eigen::Tensor3dXf y_conv2d = demucscpp::conv2d<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // print every single value of x with its index; prepend a tab character
    // to each line

    std::cout << std::fixed << std::setprecision(4) << std::endl;

    std::cout << "x (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < x.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < x.dimension(1); ++j) {
            for (size_t k = 0; k < x.dimension(2); ++k) {
                std::cout << x(i, j, k) << "\t";
            }
            std::cout << std::endl;  // New line at the end of each row
        }
        std::cout << std::endl;  // Extra line to separate channels
    }
    std::cout << "w (Out Channel, In Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < w.dimension(0); ++i) {
        for (size_t j = 0; j < w.dimension(1); ++j) {
            std::cout << "Out Channel " << i << ", In Channel " << j << ":" << std::endl;
            for (size_t k = 0; k < w.dimension(2); ++k) {
                for (size_t l = 0; l < w.dimension(3); ++l) {
                    std::cout << w(i, j, k, l) << "\t";
                }
                std::cout << std::endl;  // New line for each row
            }
            std::cout << std::endl;  // Extra line to separate filters
        }
    }

    std::cout << "b (Bias):" << std::endl;
    for (size_t i = 0; i < b.dimension(0); ++i) {
        std::cout << b(i) << "\t";
    }
    std::cout << std::endl << std::endl;  // New line after printing all biases

    std::cout << "y_gemm (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_gemm.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_gemm.dimension(1); ++j) {
            for (size_t k = 0; k < y_gemm.dimension(2); ++k) {
                std::cout << y_gemm(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "y_conv2d (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_conv2d.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_conv2d.dimension(1); ++j) {
            for (size_t k = 0; k < y_conv2d.dimension(2); ++k) {
                std::cout << y_conv2d(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // compare y_gemm and y_conv2d
    demucscppdebug::debug_tensor_3dxf(y_gemm, "y_gemm");
    demucscppdebug::debug_tensor_3dxf(y_conv2d, "y_conv2d");
}

TEST(DemucsCPPLayers, GemmConv2)
{
    constexpr int in_channels = 3;
    constexpr int in_height = 13;
    constexpr int in_width = 9;

    constexpr int out_channels = 5;
    constexpr int kernel_height = 2;
    constexpr int kernel_width = 3;

    // x: 2 in channels, 3x4 image per channel
    Eigen::Tensor3dXf x(in_channels, in_height, in_width);

    // fill x with values
    //int counter = 0;
    for (size_t i = 0; i < x.dimension(0); ++i)
    {
        for (size_t j = 0; j < x.dimension(1); ++j)
        {
            for (size_t k = 0; k < x.dimension(2); ++k)
            {
                if (k % 2 == 0) {
                    x(i, j, k) = 0.5;//counter++;
                }
                else {
                    x(i, j, k) = -0.75;
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_3dxf(x, "x");
    Eigen::Tensor4dXf w(out_channels, in_channels, kernel_height, kernel_width);

    int counter = 1;
    for (size_t i = 0; i < w.dimension(0); ++i)
    {
        for (size_t j = 0; j < w.dimension(1); ++j)
        {
            for (size_t k = 0; k < w.dimension(2); ++k)
            {
                for (size_t l = 0; l < w.dimension(3); ++l)
                {
                    w(i, j, k, l) = 0.1f*((float)counter);
                    if (j % 2 == 0) {
                        w(i, j, k, l) *= -1.0f;
                    }
                    counter++;
                    // set weights to be -0.5, 0.5 alternating over kernel_height
                    //if (k % 2 == 0) {
                    //    w(i, j, k, l) = 0.5;
                    //}
                    //else {
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //if (l % 2 == 0)
                    //{
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //else
                    //{
                    //    w(i, j, k, l) = 0.5;
                    //}
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_4dxf(w, "w");

    // bias: 4 out channels
    Eigen::Tensor1dXf b(out_channels);

    // set bias to be 0.75 or -0.25 alternating
    for (size_t i = 0; i < out_channels; ++i)
    {
        b(i) = 0.0f;
        //if (i % 2 == 0)
        //{
        //    b(i) = 0.75;
        //}
        //else
        //{
        //    b(i) = -0.25;
        //}
    }

    //demucscppdebug::debug_tensor_1dxf(b, "b");

    // apply conv2d_gemm with some params
    Eigen::Tensor3dXf y_gemm = demucscpp::conv2d_gemm<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // apply regular conv2d with some params
    Eigen::Tensor3dXf y_conv2d = demucscpp::conv2d<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // print every single value of x with its index; prepend a tab character
    // to each line

    std::cout << std::fixed << std::setprecision(4) << std::endl;

    std::cout << "x (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < x.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < x.dimension(1); ++j) {
            for (size_t k = 0; k < x.dimension(2); ++k) {
                std::cout << x(i, j, k) << "\t";
            }
            std::cout << std::endl;  // New line at the end of each row
        }
        std::cout << std::endl;  // Extra line to separate channels
    }
    std::cout << "w (Out Channel, In Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < w.dimension(0); ++i) {
        for (size_t j = 0; j < w.dimension(1); ++j) {
            std::cout << "Out Channel " << i << ", In Channel " << j << ":" << std::endl;
            for (size_t k = 0; k < w.dimension(2); ++k) {
                for (size_t l = 0; l < w.dimension(3); ++l) {
                    std::cout << w(i, j, k, l) << "\t";
                }
                std::cout << std::endl;  // New line for each row
            }
            std::cout << std::endl;  // Extra line to separate filters
        }
    }

    std::cout << "b (Bias):" << std::endl;
    for (size_t i = 0; i < b.dimension(0); ++i) {
        std::cout << b(i) << "\t";
    }
    std::cout << std::endl << std::endl;  // New line after printing all biases

    std::cout << "y_gemm (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_gemm.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_gemm.dimension(1); ++j) {
            for (size_t k = 0; k < y_gemm.dimension(2); ++k) {
                std::cout << y_gemm(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "y_conv2d (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_conv2d.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_conv2d.dimension(1); ++j) {
            for (size_t k = 0; k < y_conv2d.dimension(2); ++k) {
                std::cout << y_conv2d(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // compare y_gemm and y_conv2d
    demucscppdebug::debug_tensor_3dxf(y_gemm, "y_gemm");
    demucscppdebug::debug_tensor_3dxf(y_conv2d, "y_conv2d");
}

TEST(DemucsCPPLayers, GemmTrConv)
{
    constexpr int in_channels = 1;
    constexpr int in_height = 13;
    constexpr int in_width = 9;

    constexpr int out_channels = 5;
    constexpr int kernel_height = 2;
    constexpr int kernel_width = 3;

    // x: 2 in channels, 3x4 image per channel
    Eigen::Tensor3dXf x(in_channels, in_height, in_width);

    // fill x with values
    //int counter = 0;
    for (size_t i = 0; i < x.dimension(0); ++i)
    {
        for (size_t j = 0; j < x.dimension(1); ++j)
        {
            for (size_t k = 0; k < x.dimension(2); ++k)
            {
                if (k % 2 == 0) {
                    x(i, j, k) = 0.5;//counter++;
                }
                else {
                    x(i, j, k) = -0.75;
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_3dxf(x, "x");
    Eigen::Tensor4dXf w(in_channels, out_channels, kernel_height, kernel_width);

    int counter = 1;
    for (size_t i = 0; i < w.dimension(0); ++i)
    {
        for (size_t j = 0; j < w.dimension(1); ++j)
        {
            for (size_t k = 0; k < w.dimension(2); ++k)
            {
                for (size_t l = 0; l < w.dimension(3); ++l)
                {
                    w(i, j, k, l) = 0.1f*((float)counter);
                    if (i % 2 == 0) {
                        w(i, j, k, l) *= -1.0f;
                    }
                    counter++;
                    // set weights to be -0.5, 0.5 alternating over kernel_height
                    //if (k % 2 == 0) {
                    //    w(i, j, k, l) = 0.5;
                    //}
                    //else {
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //if (l % 2 == 0)
                    //{
                    //    w(i, j, k, l) = -0.5;
                    //}
                    //else
                    //{
                    //    w(i, j, k, l) = 0.5;
                    //}
                }
            }
        }
    }

    //demucscppdebug::debug_tensor_4dxf(w, "w");

    // bias: 4 out channels
    Eigen::Tensor1dXf b(out_channels);

    // set bias to be 0.75 or -0.25 alternating
    for (size_t i = 0; i < out_channels; ++i)
    {
        b(i) = 0.0f;
        //if (i % 2 == 0)
        //{
        //    b(i) = 0.75;
        //}
        //else
        //{
        //    b(i) = -0.25;
        //}
    }

    //demucscppdebug::debug_tensor_1dxf(b, "b");

    // apply conv2d_gemm with some params
    Eigen::Tensor3dXf y_gemm = demucscpp::conv2d_tr_gemm<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // apply regular conv2d with some params
    Eigen::Tensor3dXf y_conv2d = demucscpp::conv2d_tr_old<in_channels, out_channels, kernel_height, kernel_width, 1, 1, 0, 0, 1, 1>(x, w, b);

    // print every single value of x with its index; prepend a tab character
    // to each line

    std::cout << std::fixed << std::setprecision(4) << std::endl;

    std::cout << "x (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < x.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < x.dimension(1); ++j) {
            for (size_t k = 0; k < x.dimension(2); ++k) {
                std::cout << x(i, j, k) << "\t";
            }
            std::cout << std::endl;  // New line at the end of each row
        }
        std::cout << std::endl;  // Extra line to separate channels
    }
    std::cout << "w (Out Channel, In Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < w.dimension(0); ++i) {
        for (size_t j = 0; j < w.dimension(1); ++j) {
            std::cout << "Out Channel " << i << ", In Channel " << j << ":" << std::endl;
            for (size_t k = 0; k < w.dimension(2); ++k) {
                for (size_t l = 0; l < w.dimension(3); ++l) {
                    std::cout << w(i, j, k, l) << "\t";
                }
                std::cout << std::endl;  // New line for each row
            }
            std::cout << std::endl;  // Extra line to separate filters
        }
    }

    std::cout << "b (Bias):" << std::endl;
    for (size_t i = 0; i < b.dimension(0); ++i) {
        std::cout << b(i) << "\t";
    }
    std::cout << std::endl << std::endl;  // New line after printing all biases

    std::cout << "y_gemm (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_gemm.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_gemm.dimension(1); ++j) {
            for (size_t k = 0; k < y_gemm.dimension(2); ++k) {
                std::cout << y_gemm(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "y_conv2d (Channel, Height, Width):" << std::endl;
    for (size_t i = 0; i < y_conv2d.dimension(0); ++i) {
        std::cout << "Channel " << i << ":" << std::endl;
        for (size_t j = 0; j < y_conv2d.dimension(1); ++j) {
            for (size_t k = 0; k < y_conv2d.dimension(2); ++k) {
                std::cout << y_conv2d(i, j, k) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // compare y_gemm and y_conv2d
    demucscppdebug::debug_tensor_3dxf(y_gemm, "y_gemm");
    demucscppdebug::debug_tensor_3dxf(y_conv2d, "y_conv2d");
}

// write a basic test case for a stereo file
TEST(DemucsCPPLayers, FreqEncoders)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf x_fake(4, 2048, 336);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 2048; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    Eigen::Tensor3dXf x_fake_enc_0(48, 512, 336);
    demucscpp::apply_freq_encoder(model, 0, x_fake, x_fake_enc_0);

    demucscppdebug::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug::debug_tensor_3dxf(x_fake_enc_0, "x_fake_enc_0");

    Eigen::Tensor3dXf x_fake_enc_1(96, 128, 336);
    demucscpp::apply_freq_encoder(model, 1, x_fake_enc_0, x_fake_enc_1);
    demucscppdebug::debug_tensor_3dxf(x_fake_enc_1, "x_fake_enc_1");

    Eigen::Tensor3dXf x_fake_enc_2(192, 32, 336);
    demucscpp::apply_freq_encoder(model, 2, x_fake_enc_1, x_fake_enc_2);
    demucscppdebug::debug_tensor_3dxf(x_fake_enc_2, "x_fake_enc_2");

    Eigen::Tensor3dXf x_fake_enc_3(384, 8, 336);
    demucscpp::apply_freq_encoder(model, 3, x_fake_enc_2, x_fake_enc_3);
    demucscppdebug::debug_tensor_3dxf(x_fake_enc_3, "x_fake_enc_3");
}

TEST(DemucsCPPLayers, FreqDecoders)
{
    setUpTestSuite();

    Eigen::Tensor3dXf x_fake_dec_0(384, 8, 336);
    Eigen::Tensor3dXf skip_fake_dec_0(384, 8, 336);

    Eigen::Tensor3dXf x_fake_dec_1(192, 32, 336);
    Eigen::Tensor3dXf skip_fake_dec_1(192, 32, 336);

    Eigen::Tensor3dXf x_fake_dec_2(96, 128, 336);
    Eigen::Tensor3dXf skip_fake_dec_2(96, 128, 336);

    Eigen::Tensor3dXf x_fake_dec_3(48, 512, 336);
    Eigen::Tensor3dXf skip_fake_dec_3(48, 512, 336);

    Eigen::Tensor3dXf x_fake_dec_4(4, 2048, 336);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 384; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake_dec_0(i, j, k) = -1.0;
                    skip_fake_dec_0(i, j, k) = 0.5;
                }
                else
                {
                    x_fake_dec_0(i, j, k) = 1.0;
                    skip_fake_dec_0(i, j, k) = -0.5;
                }
            }
        }
    }
    for (size_t i = 0; i < 192; ++i)
    {
        for (size_t j = 0; j < 32; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_1(i, j, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_1(i, j, k) = -0.5;
                }
            }
        }
    }
    for (size_t i = 0; i < 96; ++i)
    {
        for (size_t j = 0; j < 128; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_2(i, j, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_2(i, j, k) = -0.5;
                }
            }
        }
    }
    for (size_t i = 0; i < 48; ++i)
    {
        for (size_t j = 0; j < 512; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    skip_fake_dec_3(i, j, k) = 0.5;
                }
                else
                {
                    skip_fake_dec_3(i, j, k) = -0.5;
                }
            }
        }
    }

    demucscpp::apply_freq_decoder(model, 0, x_fake_dec_0, x_fake_dec_1,
                                  skip_fake_dec_0);

    demucscppdebug::debug_tensor_3dxf(x_fake_dec_0, "x_fake_dec_0");
    demucscppdebug::debug_tensor_3dxf(x_fake_dec_1, "x_fake_dec_1");

    demucscpp::apply_freq_decoder(model, 1, x_fake_dec_1, x_fake_dec_2,
                                  skip_fake_dec_1);

    demucscppdebug::debug_tensor_3dxf(x_fake_dec_2, "x_fake_dec_2");

    demucscpp::apply_freq_decoder(model, 2, x_fake_dec_2, x_fake_dec_3,
                                  skip_fake_dec_2);

    demucscppdebug::debug_tensor_3dxf(x_fake_dec_3, "x_fake_dec_3");

    demucscpp::apply_freq_decoder(model, 3, x_fake_dec_3, x_fake_dec_4,
                                  skip_fake_dec_3);

    demucscppdebug::debug_tensor_3dxf(x_fake_dec_4, "x_fake_dec_4");
}

// write a basic test case for a stereo file
TEST(DemucsCPPLayers, TimeEncoders)
{
    setUpTestSuite();

    std::cout << std::fixed << std::setprecision(20) << std::endl;

    Eigen::Tensor3dXf xt_fake(1, 2, 343980);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 1; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            for (size_t k = 0; k < 343980; ++k)
            {
                if (k % 2 == 0)
                {
                    xt_fake(i, j, k) = -1.0;
                }
                else
                {
                    xt_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt_fake");

    Eigen::Tensor3dXf xt_fake_enc_0(1, 48, 85995);
    demucscpp::apply_time_encoder(model, 0, xt_fake, xt_fake_enc_0);

    demucscppdebug::debug_tensor_3dxf(xt_fake_enc_0, "xt_fake_enc_0");

    Eigen::Tensor3dXf xt_fake_enc_1(1, 96, 21499);
    demucscpp::apply_time_encoder(model, 1, xt_fake_enc_0, xt_fake_enc_1);

    demucscppdebug::debug_tensor_3dxf(xt_fake_enc_1, "xt_fake_enc_1");

    Eigen::Tensor3dXf xt_fake_enc_2(1, 192, 5375);

    demucscpp::apply_time_encoder(model, 2, xt_fake_enc_1, xt_fake_enc_2);
    demucscppdebug::debug_tensor_3dxf(xt_fake_enc_2, "xt_fake_enc_2");

    Eigen::Tensor3dXf xt_fake_enc_3(1, 384, 1344);

    demucscpp::apply_time_encoder(model, 3, xt_fake_enc_2, xt_fake_enc_3);
    demucscppdebug::debug_tensor_3dxf(xt_fake_enc_3, "xt_fake_enc_3");
}

TEST(DemucsCPPLayers, TimeDecoders)
{
    setUpTestSuite();

    Eigen::Tensor3dXf xt_fake_dec_0(1, 384, 1344);
    Eigen::Tensor3dXf skipt_fake_dec_0(1, 384, 1344);

    Eigen::Tensor3dXf xt_fake_dec_1(1, 192, 5375);
    Eigen::Tensor3dXf skipt_fake_dec_1(1, 192, 5375);

    Eigen::Tensor3dXf xt_fake_dec_2(1, 96, 21499);
    Eigen::Tensor3dXf skipt_fake_dec_2(1, 96, 21499);

    Eigen::Tensor3dXf xt_fake_dec_3(1, 48, 85995);
    Eigen::Tensor3dXf skipt_fake_dec_3(1, 48, 85995);

    Eigen::Tensor3dXf xt_fake_dec_4(1, 8, 343980);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 384; ++i)
    {
        for (size_t j = 0; j < 1344; ++j)
        {
            if (j % 2 == 0)
            {
                xt_fake_dec_0(0, i, j) = -1.0;
                skipt_fake_dec_0(0, i, j) = 0.5;
            }
            else
            {
                xt_fake_dec_0(0, i, j) = 1.0;
                skipt_fake_dec_0(0, i, j) = -0.5;
            }
        }
    }
    for (size_t i = 0; i < 192; ++i)
    {
        for (size_t j = 0; j < 5375; ++j)
        {
            if (j % 2 == 0)
            {
                skipt_fake_dec_1(0, i, j) = 0.5;
            }
            else
            {
                skipt_fake_dec_1(0, i, j) = -0.5;
            }
        }
    }
    for (size_t i = 0; i < 96; ++i)
    {
        for (size_t j = 0; j < 21499; ++j)
        {
            if (j % 2 == 0)
            {
                skipt_fake_dec_2(0, i, j) = 0.5;
            }
            else
            {
                skipt_fake_dec_2(0, i, j) = -0.5;
            }
        }
    }
    for (size_t i = 0; i < 48; ++i)
    {
        for (size_t j = 0; j < 85995; ++j)
        {
            if (j % 2 == 0)
            {
                skipt_fake_dec_3(0, i, j) = 0.5;
            }
            else
            {
                skipt_fake_dec_3(0, i, j) = -0.5;
            }
        }
    }

    demucscpp::apply_time_decoder(model, 0, xt_fake_dec_0, xt_fake_dec_1,
                                  skipt_fake_dec_0);

    demucscppdebug::debug_tensor_3dxf(xt_fake_dec_0, "xt_fake_dec_0");
    demucscppdebug::debug_tensor_3dxf(xt_fake_dec_1, "xt_fake_dec_1");

    demucscpp::apply_time_decoder(model, 1, xt_fake_dec_1, xt_fake_dec_2,
                                  skipt_fake_dec_1);

    demucscppdebug::debug_tensor_3dxf(xt_fake_dec_2, "xt_fake_dec_2");

    demucscpp::apply_time_decoder(model, 2, xt_fake_dec_2, xt_fake_dec_3,
                                  skipt_fake_dec_2);

    demucscppdebug::debug_tensor_3dxf(xt_fake_dec_3, "xt_fake_dec_3");

    demucscpp::apply_time_decoder(model, 3, xt_fake_dec_3, xt_fake_dec_4,
                                  skipt_fake_dec_3);

    demucscppdebug::debug_tensor_3dxf(xt_fake_dec_4, "xt_fake_dec_4");

    // compare first and last element of waveform_outputs and normalized_audio
    // EXPECT_NEAR(waveform_outputs(0, 0, 0), normalized_audio(0, 0),
    // NEAR_TOLERANCE); EXPECT_NEAR(waveform_outputs(0, 0, 44099),
    // normalized_audio(0, 44099), NEAR_TOLERANCE);
}

TEST(DemucsCPPLayers, CrossTransformer)
{
    setUpTestSuite();

    /*****************************/
    /*  FREQ CHANNEL UPSAMPLING  */
    /*****************************/

    Eigen::Tensor3dXf x_fake(384, 8, 336);
    Eigen::Tensor3dXf xt_fake(1, 384, 1344);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 384; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    for (size_t j = 0; j < 384; ++j)
    {
        for (size_t k = 0; k < 1344; ++k)
        {
            if (k % 2 == 0)
            {
                xt_fake(0, j, k) = -1.0;
            }
            else
            {
                xt_fake(0, j, k) = 1.0;
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt_fake");

    // Reshape buffers.x_3 into x_3_reshaped
    // Apply the conv1d function
    // Reshape back to 512x8x336 and store in buffers.x_3_channel_upsampled
    Eigen::Tensor3dXf x_fake_reshaped =
        x_fake.reshape(Eigen::array<int, 3>({1, 384, 8 * 336}));
    Eigen::Tensor3dXf x_fake_reshaped_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(x_fake_reshaped, model.channel_upsampler_weight,
                          model.channel_upsampler_bias);
    Eigen::Tensor3dXf x_fake_upsampled =
        x_fake_reshaped_upsampled.reshape(Eigen::array<int, 3>({512, 8, 336}));

    /*****************************/
    /*  TIME CHANNEL UPSAMPLING  */
    /*****************************/

    // for time channel upsampling
    // apply upsampler directly to xt_3 no reshaping drama needed
    Eigen::Tensor3dXf xt_fake_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(xt_fake, model.channel_upsampler_t_weight,
                          model.channel_upsampler_t_bias);

    demucscppdebug::debug_tensor_3dxf(x_fake_upsampled,
                                      "x pre-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(xt_fake_upsampled,
                                      "xt pre-crosstransformer");

    /*************************/
    /*  CROSS-TRANSFORMER!  */
    /************************/
    demucscpp::apply_crosstransformer(model, x_fake_upsampled,
                                      xt_fake_upsampled);

    // reshape buffers.x_3_channel_upsampled into 1, 512, 2688
    // when skipping the crosstransformer
    // Eigen::Tensor3dXf x_3_reshaped_upsampled_2 =
    // buffers.x_3_channel_upsampled.reshape(Eigen::array<int, 3>({1, 512,
    // 8*336})); buffers.x_3_channel_upsampled = x_3_reshaped_upsampled_2;

    demucscppdebug::debug_tensor_3dxf(x_fake_upsampled,
                                      "x post-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(xt_fake_upsampled,
                                      "xt post-crosstransformer");
    // then apply the conv1d_2d function

    Eigen::Tensor3dXf x_fake_reshaped_downsampled =
        demucscpp::conv1d<512, 384, 1, 1, 0, 1>(x_fake_upsampled, model.channel_downsampler_weight,
                          model.channel_downsampler_bias);
    Eigen::Tensor3dXf x_fake_downsampled = x_fake_reshaped_downsampled.reshape(
        Eigen::array<int, 3>({384, 8, 336}));

    // apply upsampler directly to xt_3
    Eigen::Tensor3dXf xt_fake_downsampled =
        demucscpp::conv1d<512, 384, 1, 1, 0, 1>(xt_fake_upsampled, model.channel_downsampler_t_weight,
                          model.channel_downsampler_t_bias);

    demucscppdebug::debug_tensor_3dxf(x_fake_downsampled, "x post-downsampler");
    demucscppdebug::debug_tensor_3dxf(xt_fake_downsampled,
                                      "xt post-downsampler");
}

TEST(DemucsCPPLayers, CrossTransformerNoUpsamp)
{
    setUpTestSuite();

    Eigen::Tensor3dXf x_fake(512, 8, 336);
    Eigen::Tensor3dXf xt_fake(1, 512, 1344);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 512; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    for (size_t j = 0; j < 512; ++j)
    {
        for (size_t k = 0; k < 1344; ++k)
        {
            if (k % 2 == 0)
            {
                xt_fake(0, j, k) = -1.0;
            }
            else
            {
                xt_fake(0, j, k) = 1.0;
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(x_fake, "x pre-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt pre-crosstransformer");

    /*************************/
    /*  CROSS-TRANSFORMER!  */
    /************************/
    demucscpp::apply_crosstransformer(model, x_fake, xt_fake);

    demucscppdebug::debug_tensor_3dxf(x_fake, "x post-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt post-crosstransformer");
}

TEST(DemucsCPPLayers, Upsamplers)
{
    setUpTestSuite();

    /*****************************/
    /*  FREQ CHANNEL UPSAMPLING  */
    /*****************************/

    Eigen::Tensor3dXf x_fake(384, 8, 336);
    Eigen::Tensor3dXf xt_fake(1, 384, 1344);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 384; ++i)
    {
        for (size_t j = 0; j < 8; ++j)
        {
            for (size_t k = 0; k < 336; ++k)
            {
                if (k % 2 == 0)
                {
                    x_fake(i, j, k) = -1.0;
                }
                else
                {
                    x_fake(i, j, k) = 1.0;
                }
            }
        }
    }

    for (size_t j = 0; j < 384; ++j)
    {
        for (size_t k = 0; k < 1344; ++k)
        {
            if (k % 2 == 0)
            {
                xt_fake(0, j, k) = -1.0;
            }
            else
            {
                xt_fake(0, j, k) = 1.0;
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(x_fake, "x_fake");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt_fake");

    // Reshape buffers.x_3 into x_3_reshaped
    // Apply the conv1d function
    // Reshape back to 512x8x336 and store in buffers.x_3_channel_upsampled
    Eigen::Tensor3dXf x_fake_reshaped =
        x_fake.reshape(Eigen::array<int, 3>({1, 384, 8 * 336}));
    Eigen::Tensor3dXf x_fake_reshaped_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(x_fake_reshaped, model.channel_upsampler_weight,
                          model.channel_upsampler_bias);
    Eigen::Tensor3dXf x_fake_upsampled =
        x_fake_reshaped_upsampled.reshape(Eigen::array<int, 3>({512, 8, 336}));

    /*****************************/
    /*  TIME CHANNEL UPSAMPLING  */
    /*****************************/

    // for time channel upsampling
    // apply upsampler directly to xt_3 no reshaping drama needed
    Eigen::Tensor3dXf xt_fake_upsampled =
        demucscpp::conv1d<384, 512, 1, 1, 0, 1>(xt_fake, model.channel_upsampler_t_weight,
                          model.channel_upsampler_t_bias);

    demucscppdebug::debug_tensor_3dxf(x_fake_upsampled, "x upsampled");
    demucscppdebug::debug_tensor_3dxf(xt_fake_upsampled, "xt upsampled");

    // reshape x_fake_upsampled to 1, 512, 2688

    Eigen::Tensor3dXf x_fake_upsampled_reshaped =
        x_fake_upsampled.reshape(Eigen::array<int, 3>({1, 512, 8 * 336}));
    Eigen::Tensor3dXf x_fake_downsampled_reshaped = demucscpp::conv1d<512, 384, 1, 1, 0, 0>(
        x_fake_upsampled_reshaped, model.channel_downsampler_weight,
        model.channel_downsampler_bias);
    Eigen::Tensor3dXf x_fake_downsampled = x_fake_downsampled_reshaped.reshape(
        Eigen::array<int, 3>({384, 8, 336}));

    // apply upsampler directly to xt_3
    Eigen::Tensor3dXf xt_fake_downsampled =
        demucscpp::conv1d<512, 384, 1, 1, 0, 0>(xt_fake_upsampled, model.channel_downsampler_t_weight,
                          model.channel_downsampler_t_bias);

    demucscppdebug::debug_tensor_3dxf(x_fake_downsampled, "x post-downsampler");
    demucscppdebug::debug_tensor_3dxf(xt_fake_downsampled,
                                      "xt post-downsampler");
}

TEST(DemucsCPPLayers, CTLayers)
{
    setUpTestSuite();

    Eigen::Tensor3dXf x_fake(1, 2688, 512);
    Eigen::Tensor3dXf xt_fake(1, 1344, 512);

    // fill with -1, 1 alternating
    for (size_t i = 0; i < 2688; ++i)
    {
        for (size_t j = 0; j < 512; ++j)
        {
            if (j % 2 == 0)
            {
                x_fake(0, i, j) = -1.0;
            }
            else
            {
                x_fake(0, i, j) = 1.0;
            }
        }
    }

    for (size_t i = 0; i < 1344; ++i)
    {
        for (size_t j = 0; j < 512; ++j)
        {
            if (j % 2 == 0)
            {
                xt_fake(0, i, j) = -1.0;
            }
            else
            {
                xt_fake(0, i, j) = 1.0;
            }
        }
    }

    demucscppdebug::debug_tensor_3dxf(x_fake, "x pre-crosstransformer");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt pre-crosstransformer");

    // make copies of each
    Eigen::Tensor3dXf x_fake_copy = x_fake;
    Eigen::Tensor3dXf xt_fake_copy = xt_fake;

    int freq_or_time = 0;
    int weight_idx = 0;
    float eps = 1e-5;

    demucscpp::common_encoder_layer(
        x_fake, // pass x as q
        x_fake, // pass x as k
        model.crosstransformer_my_layers_norm1_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_self_attn_in_proj_weight[freq_or_time]
                                                                 [weight_idx],
        model.crosstransformer_my_layers_self_attn_in_proj_bias[freq_or_time]
                                                               [weight_idx],
        model.crosstransformer_my_layers_self_attn_out_proj_weight[freq_or_time]
                                                                  [weight_idx],
        model.crosstransformer_my_layers_self_attn_out_proj_bias[freq_or_time]
                                                                [weight_idx],
        model
            .crosstransformer_my_layers_gamma_1_scale[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm2_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm2_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_linear1_weight[freq_or_time]
                                                       [weight_idx],
        model.crosstransformer_my_layers_linear1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_linear2_weight[freq_or_time]
                                                       [weight_idx],
        model.crosstransformer_my_layers_linear2_bias[freq_or_time][weight_idx],
        model
            .crosstransformer_my_layers_gamma_2_scale[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm_out_weight[freq_or_time]
                                                        [weight_idx],
        model
            .crosstransformer_my_layers_norm_out_bias[freq_or_time][weight_idx],
        eps);

    freq_or_time = 1;

    demucscpp::common_encoder_layer(
        xt_fake, // pass x as q
        xt_fake, // pass x as k
        model.crosstransformer_my_layers_norm1_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_self_attn_in_proj_weight[freq_or_time]
                                                                 [weight_idx],
        model.crosstransformer_my_layers_self_attn_in_proj_bias[freq_or_time]
                                                               [weight_idx],
        model.crosstransformer_my_layers_self_attn_out_proj_weight[freq_or_time]
                                                                  [weight_idx],
        model.crosstransformer_my_layers_self_attn_out_proj_bias[freq_or_time]
                                                                [weight_idx],
        model
            .crosstransformer_my_layers_gamma_1_scale[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm2_weight[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm2_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_linear1_weight[freq_or_time]
                                                       [weight_idx],
        model.crosstransformer_my_layers_linear1_bias[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_linear2_weight[freq_or_time]
                                                       [weight_idx],
        model.crosstransformer_my_layers_linear2_bias[freq_or_time][weight_idx],
        model
            .crosstransformer_my_layers_gamma_2_scale[freq_or_time][weight_idx],
        model.crosstransformer_my_layers_norm_out_weight[freq_or_time]
                                                        [weight_idx],
        model
            .crosstransformer_my_layers_norm_out_bias[freq_or_time][weight_idx],
        eps);

    demucscppdebug::debug_tensor_3dxf(x_fake, "x post-layer-0");
    demucscppdebug::debug_tensor_3dxf(xt_fake, "xt post-tlayer-0");

    //demucscppdebug::debug_tensor_1dxf(model.crosstransformer_norm_in_weight,
    //                                  "norm_in weight");
    //demucscppdebug::debug_tensor_1dxf(model.crosstransformer_norm_in_bias,
    //                                  "norm_in bias");

    //demucscppdebug::debug_tensor_1dxf(model.crosstransformer_norm_in_t_weight,
    //                                  "norm_in_t weight");
    //demucscppdebug::debug_tensor_1dxf(model.crosstransformer_norm_in_t_bias,
    //                                  "norm_in_t bias");

    //Eigen::Tensor3dXf x_norm_in = demucscpp::layer_norm(
    //    x_fake_copy, model.crosstransformer_norm_in_weight,
    //    model.crosstransformer_norm_in_bias, eps);
    //Eigen::Tensor3dXf xt_norm_in = demucscpp::layer_norm(
    //    xt_fake_copy, model.crosstransformer_norm_in_t_weight,
    //    model.crosstransformer_norm_in_t_bias, eps);
    //Eigen::Tensor3dXf x_norm_in_t = demucscpp::layer_norm(
    //    xt_fake_copy, model.crosstransformer_norm_in_weight,
    //    model.crosstransformer_norm_in_bias, eps);
    //Eigen::Tensor3dXf xt_norm_in_f = demucscpp::layer_norm(
    //    x_fake_copy, model.crosstransformer_norm_in_t_weight,
    //    model.crosstransformer_norm_in_t_bias, eps);

    //demucscppdebug::debug_tensor_3dxf(x_norm_in, "x norm-in");
    //demucscppdebug::debug_tensor_3dxf(xt_norm_in, "xt norm-in-t");

    //demucscppdebug::debug_tensor_3dxf(x_norm_in_t, "x norm-in_t");
    //demucscppdebug::debug_tensor_3dxf(xt_norm_in_f, "xt norm-in-t_f");
}

TEST(DemucsCPPLayers, LayerNormBasic)
{
    Eigen::Tensor3dXf x(1, 2, 3);
    Eigen::Tensor1dXf w(3);
    Eigen::Tensor1dXf b(3);

    x(0, 0, 0) = 1.0;
    x(0, 0, 1) = 2.0;
    x(0, 0, 2) = 3.0;
    x(0, 1, 0) = 4.0;
    x(0, 1, 1) = 5.0;
    x(0, 1, 2) = 6.0;

    w(0) = 0.75;
    w(1) = -0.5;
    w(2) = -1.35;

    b(0) = 0.5;
    b(1) = -0.25;
    b(2) = 0.75;

    demucscppdebug::debug_tensor_3dxf(x, "x");
    demucscppdebug::debug_tensor_1dxf(w, "w");
    demucscppdebug::debug_tensor_1dxf(b, "b");

    Eigen::Tensor3dXf x_out = demucscpp::layer_norm(x, w, b, 1e-5);

    demucscppdebug::debug_tensor_3dxf(x_out, "x_out");
}

TEST(DemucsCPPLayers, LayerNormBigger)
{
    Eigen::Tensor3dXf x(1, 2688, 512);
    Eigen::Tensor1dXf w(512);
    Eigen::Tensor1dXf b(512);

    // fill x with alternating -1, 1
    for (size_t i = 0; i < 2688; ++i)
    {
        for (size_t j = 0; j < 512; ++j)
        {
            if (j % 2 == 0)
            {
                x(0, i, j) = -1.0;
            }
            else
            {
                x(0, i, j) = 1.0;
            }
        }
    }

    // fill w with alternating -0.25, 0.25
    // fill b with alternating 0.5, -0.5
    for (size_t i = 0; i < 512; ++i)
    {
        if (i % 2 == 0)
        {
            w(i) = -0.25 + i * 0.03;
            b(i) = 0.5;
        }
        else
        {
            w(i) = 0.25;
            b(i) = -0.5 + i * 0.57;
        }
    }

    demucscppdebug::debug_tensor_3dxf(x, "x");
    demucscppdebug::debug_tensor_1dxf(w, "w");
    demucscppdebug::debug_tensor_1dxf(b, "b");

    Eigen::Tensor3dXf x_out = demucscpp::layer_norm(x, w, b, 1e-5);

    demucscppdebug::debug_tensor_3dxf(x_out, "x_out");
}
