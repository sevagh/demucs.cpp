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

Eigen::Tensor3dXf conv2d(const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w,
                         const Eigen::Tensor1dXf &b, const int kernel_height,
                         const int kernel_width, const int stride_height,
                         const int stride_width, const int pad_height,
                         const int pad_width, int dilation_height = 1,
                         int dilation_width = 1);

Eigen::Tensor3dXf conv1d(const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &w,
                         const Eigen::Tensor1dXf &b, int kernel_size,
                         int stride, int pad, int dilation);

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
    float sum_squares = 0.0f;
    float c = 0.0f; // A running compensation for lost low-order bits.
    for (int i = 0; i < tensor.size(); ++i)
    {
        float y = (tensor.data()[i] - mean) * (tensor.data()[i] - mean);
        float t = sum_squares + y;
        if (abs(sum_squares) >= abs(y))
        {
            c += (sum_squares - t) + y;
        }
        else
        {
            c += (y - t) + sum_squares;
        }
        sum_squares = t;
    }
    float variance = (sum_squares + c) / (tensor.size() - 1);
    return variance;
}

inline float calculate_variance(const Eigen::Tensor1dXf &tensor, float mean)
{
    float sum_squares = 0.0f;
    float c = 0.0f; // A running compensation for lost low-order bits.
    for (int i = 0; i < tensor.size(); ++i)
    {
        float y = (tensor.data()[i] - mean) * (tensor.data()[i] - mean);
        float t = sum_squares + y;
        if (abs(sum_squares) >= abs(y))
        {
            c += (sum_squares - t) + y;
        }
        else
        {
            c += (y - t) + sum_squares;
        }
        sum_squares = t;
    }
    float variance = (sum_squares + c) / (tensor.size() - 1);
    return variance;
}

} // namespace demucscpp

#endif // LAYERS_HPP
