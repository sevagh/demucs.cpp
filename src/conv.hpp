#ifndef CONV_HPP
#define CONV_HPP

#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

namespace demucscpp
{

template<int in_channels, int out_channels, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d_tr(const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w, const Eigen::Tensor1dXf &b)
{
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    // Transposed convolution output size calculation
    int out_height =
        (in_height - 1) * stride_height - 2 * pad_height + kernel_height;
    int out_width =
        (in_width - 1) * stride_width - 2 * pad_width + kernel_width;

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);

    // Initialize y_out to b
    for (int chout = 0; chout < out_channels; ++chout)
    {
        y_out.chip<0>(chout).setConstant(b(chout));
    }

    // Transposed convolution loop
    for (int n = 0; n < kernel_width; ++n)
    {
        for (int m = 0; m < kernel_height; ++m)
        {
            for (int chin = 0; chin < in_channels; ++chin)
            {
                for (int j = 0; j < in_width; ++j)
                {
                    for (int i = 0; i < in_height; ++i)
                    {
                        for (int chout = 0; chout < out_channels; ++chout)
                        {
                            int oh = i * stride_height + m * dilation_height -
                                     pad_height;
                            int ow = j * stride_width + n * dilation_width -
                                     pad_width;

                            // Check if the indices are within the bounds of the
                            // output tensor
                            if (oh >= 0 && oh < out_height && ow >= 0 &&
                                ow < out_width)
                            {
                                y_out(chout, oh, ow) += x(chin, i, j) * w(chin, chout, m, n);
                            }
                        }
                    }
                }
            }
        }
    }

    return y_out;
}

template<int in_channels, int out_channels, int kernel_size, int stride, int pad, int dilation>
Eigen::Tensor3dXf conv1d_tr(const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &w, const Eigen::Tensor1dXf &b)
{
    // Convert 1D convolution to 2D convolution by adding an extra dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // Move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = x.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // Call the 2D transposed convolution function
    Eigen::Tensor3dXf y_out = conv2d_tr<in_channels, out_channels, kernel_size, 1, stride, 1, pad, 0, dilation, 1>(x_shuff, w_4d, b);

    // Move end axis to the front
    Eigen::Tensor3dXf y_out_shuf = y_out.shuffle(Eigen::array<int, 3>({2, 0, 1}));

    return y_out_shuf;
}

template<int in_channels, int out_channels, int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d(const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w, const Eigen::Tensor1dXf &b)
{
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    int out_height = static_cast<int>(std::floor(in_height + 2 * pad_height -
                                                 kernel_height) /
                                      stride_height) +
                     1;
    int out_width =
        static_cast<int>(std::floor(in_width + 2 * pad_width - kernel_width) /
                         stride_width) +
        1;

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);

    // Initialize y_out to b
    for (int chout = 0; chout < out_channels; ++chout)
    {
        y_out.chip<0>(chout).setConstant(b(chout));
    }

    // 2d convolution loop
    for (int n = 0; n < kernel_width; ++n)
    {
        for (int m = 0; m < kernel_height; ++m)
        {
            for (int chin = 0; chin < in_channels; ++chin)
            {
                for (int j = 0; j < out_width; ++j)
                {
                    for (int i = 0; i < out_height; ++i)
                    {
                        for (int chout = 0; chout < out_channels; ++chout)
                        {
                            int ih = i * stride_height + m * dilation_height -
                                     pad_height;
                            int jw = j * stride_width + n * dilation_width -
                                     pad_width;
                            if (ih >= 0 && ih < in_height && jw >= 0 &&
                                jw < in_width)
                            {
                                y_out(chout, i, j) += x(chin, ih, jw) * w(chout, chin, m, n);
                            }
                        }
                    }
                }
            }
        }
    }

    return y_out;
}

template<int in_channels, int out_channels, int kernel_size, int stride, int pad, int dilation>
Eigen::Tensor3dXf conv1d(const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &w, const Eigen::Tensor1dXf &b)
{
    // copy w into a 4d tensor with trailing (,1) dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = x.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // do 2d convolution inference here
    // treating the in_freq dimension as a width dimension with a no-op kernel
    Eigen::Tensor3dXf y_out = demucscpp::conv2d<in_channels, out_channels, kernel_size, 1, stride, 1, pad, 0, dilation, 1>(
        x_shuff, w_4d, b);

    // move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        y_out.shuffle(Eigen::array<int, 3>({2, 0, 1}));
    return y_out_shuf;
}
} // namespace demucscpp

#endif // CONV_HPP
