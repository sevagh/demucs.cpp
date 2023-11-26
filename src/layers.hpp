#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

namespace demucscpp
{

void apply_dconv(struct demucscpp::demucs_model_4s &model, Eigen::Tensor3dXf &y,
                 int freq_idx, int encdec_idx, int layer_idx, int mid_crop);

// used for implementing both self-attention and cross-attention
// let's not modify the second argument
void common_encoder_layer(
    Eigen::Tensor3dXf &q,       // q = x = frequency|time
    const Eigen::Tensor3dXf &k, // k = xt = time|frequency, _or_ k == q
    const Eigen::Tensor1dXf &norm1_weight, const Eigen::Tensor1dXf &norm1_bias,
    const Eigen::Tensor1dXf &norm2_weight, const Eigen::Tensor1dXf &norm2_bias,
    const Eigen::MatrixXf &in_proj_weight, const Eigen::VectorXf &in_proj_bias,
    const Eigen::MatrixXf &out_proj_weight,
    const Eigen::VectorXf &out_proj_bias, const Eigen::VectorXf &gamma_1_scale,
    const Eigen::Tensor1dXf &norm3_weight, const Eigen::Tensor1dXf &norm3_bias,
    const Eigen::MatrixXf &linear1_weight, const Eigen::VectorXf &linear1_bias,
    const Eigen::MatrixXf &linear2_weight, const Eigen::VectorXf &linear2_bias,
    const Eigen::VectorXf &gamma_2_scale,
    const Eigen::Tensor1dXf &norm_out_weight,
    const Eigen::Tensor1dXf &norm_out_bias, float eps = 1e-5,
    const int num_heads = 8);

Eigen::Tensor3dXf conv2d_tr(const Eigen::Tensor3dXf &x,
                            const Eigen::Tensor4dXf &w,
                            const Eigen::Tensor1dXf &b, const int kernel_height,
                            const int kernel_width, const int stride_height,
                            const int stride_width, const int pad_height,
                            const int pad_width, int dilation_height = 1,
                            int dilation_width = 1);

Eigen::Tensor3dXf conv1d_tr(const Eigen::Tensor3dXf &x,
                            const Eigen::Tensor3dXf &w,
                            const Eigen::Tensor1dXf &b, int kernel_size,
                            int stride, int pad, int dilation);

Eigen::Tensor3dXf group_norm(const Eigen::Tensor3dXf &x,
                             const Eigen::Tensor1dXf &w,
                             const Eigen::Tensor1dXf &b, int num_groups,
                             float eps);

Eigen::Tensor3dXf layer_norm(const Eigen::Tensor3dXf &x,
                             const Eigen::Tensor1dXf &weight,
                             const Eigen::Tensor1dXf &b, float eps);

Eigen::Tensor3dXf glu(const Eigen::Tensor3dXf &x, const int dim);

inline Eigen::Tensor3dXf gelu(const Eigen::Tensor3dXf &x)
{
    return x.unaryExpr(
        [](float a)
        { return 0.5f * a * (1.0f + std::erf(a / std::sqrt(2.0f))); });
}

inline Eigen::MatrixXf gelu(const Eigen::MatrixXf &x)
{
    return x.unaryExpr(
        [](float a)
        { return 0.5f * a * (1.0f + std::erf(a / std::sqrt(2.0f))); });
}

inline Eigen::Tensor3dXf layer_scale(const Eigen::Tensor3dXf &x,
                                     const Eigen::Tensor1dXf &scale_weights)
{
    Eigen::Tensor3dXf y_out(x.dimensions());
    for (int i = 0; i < x.dimension(1); ++i)
    {
        y_out.chip<1>(i) = x.chip<1>(i) * scale_weights(i);
    }
    return y_out;
}

inline float calculate_variance(const Eigen::Tensor3dXf &tensor, float mean)
{
    Eigen::Tensor<float, 0> sum_squares = (tensor - mean).square().sum();
    float variance = sum_squares(0) / (tensor.size() - 1);
    return variance;
}

inline float calculate_variance(const Eigen::Tensor1dXf &tensor, float mean)
{
    Eigen::Tensor<float, 0> sum_squares = (tensor - mean).square().sum();
    float variance = sum_squares(0) / (tensor.size() - 1);
    return variance;
}

template<int kernel_height, int kernel_width, int stride_height, int stride_width, int pad_height, int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d(const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w, const Eigen::Tensor1dXf &b)
{
    int in_channels = x.dimension(0);
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    int out_channels = w.dimension(0);

    int out_height = static_cast<int>(std::floor(in_height + 2 * pad_height -
                                                 kernel_height) /
                                      stride_height) +
                     1;
    int out_width =
        static_cast<int>(std::floor(in_width + 2 * pad_width - kernel_width) /
                         stride_width) +
        1;

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    y_out.setZero();

    // 2d convolution loop
    for (int chout = 0; chout < out_channels; ++chout)
    {
        for (int i = 0; i < out_height; ++i)
        {
            for (int j = 0; j < out_width; ++j)
            {
                float sum = 0.0f;
                for (int chin = 0; chin < in_channels; ++chin)
                {
                    for (int m = 0; m < kernel_height; ++m)
                    {
                        for (int n = 0; n < kernel_width; ++n)
                        {
                            int ih = i * stride_height + m * dilation_height -
                                     pad_height;
                            int jw = j * stride_width + n * dilation_width -
                                     pad_width;
                            if (ih >= 0 && ih < in_height && jw >= 0 &&
                                jw < in_width)
                            {
                                sum += x(chin, ih, jw) * w(chout, chin, m, n);
                            }
                        }
                    }
                }
                y_out(chout, i, j) = sum + b(chout);
            }
        }
    }

    return y_out;
}

template<int kernel_size, int stride, int pad, int dilation>
Eigen::Tensor3dXf conv1d(const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &w, const Eigen::Tensor1dXf &b)
{
    // copy w into a 4d tensor with trailing (,1) dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = x.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // do 2d convolution inference here
    // treating the in_freq dimension as a width dimension with a no-op kernel
    Eigen::Tensor3dXf y_out = demucscpp::conv2d<kernel_size, 1, stride, 1, pad, 0, dilation, 1>(
        x_shuff, w_4d, b);

    // move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        y_out.shuffle(Eigen::array<int, 3>({2, 0, 1}));
    return y_out_shuf;
}

} // namespace demucscpp

#endif // LAYERS_HPP
