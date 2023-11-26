#include "layers.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor3dXf demucscpp::group_norm(const Eigen::Tensor3dXf &x,
                                        const Eigen::Tensor1dXf &weight,
                                        const Eigen::Tensor1dXf &b,
                                        int num_groups, float eps)
{
    int freq = x.dimension(0);
    int channels = x.dimension(1);
    int width = x.dimension(2);

    Eigen::Tensor3dXf y_out(freq, channels, width);
    y_out.setZero();

    int group_size = channels / num_groups;

    for (int i = 0; i < freq; ++i)
    {
        for (int g = 0; g < num_groups; ++g)
        {
            int start = g * group_size;
            int end = (g + 1) * group_size;

            Eigen::Tensor3dXf slice =
                x.slice(Eigen::array<int, 3>{i, start, 0},
                        Eigen::array<int, 3>{1, group_size, width});
            Eigen::Tensor<float, 0> mean_tensor = slice.mean();
            float mean = mean_tensor(0);
            float var = demucscpp::calculate_variance(slice, mean);

            for (int c = start; c < end; ++c)
            {
                for (int w = 0; w < width; ++w)
                {
                    float norm_val = (x(i, c, w) - mean) / std::sqrt(var + eps);
                    y_out(i, c, w) = norm_val * weight(c) + b(c);
                }
            }
        }
    }

    return y_out;
}

Eigen::Tensor3dXf demucscpp::glu(const Eigen::Tensor3dXf &x, const int dim)
{
    if (x.dimension(dim) % 2 != 0)
    {
        throw std::invalid_argument(
            "Dimension size must be evenly divisible by 2");
    }

    int split_size = x.dimension(dim) / 2;
    demucscppdebug::assert_(split_size > 0);

    Eigen::array<int, 3> start_indices = {0, 0, 0};
    Eigen::array<int, 3> sizes = {(int)x.dimension(0), (int)x.dimension(1),
                                  (int)x.dimension(2)};
    start_indices[dim] = split_size;
    sizes[dim] = split_size;

    auto first_half = x.slice(Eigen::array<int, 3>({0, 0, 0}), sizes);
    auto second_half = x.slice(start_indices, sizes);
    auto sigmoid_second_half = second_half.unaryExpr(
        [](float v) { return 1.0f / (1.0f + std::exp(-v)); });

    return first_half * sigmoid_second_half;
}

Eigen::Tensor3dXf demucscpp::conv2d_tr(
    const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w,
    const Eigen::Tensor1dXf &b, const int kernel_height, const int kernel_width,
    const int stride_height, const int stride_width, const int pad_height,
    const int pad_width, int dilation_height, int dilation_width)
{
    // Always ensure dilation is at least 1
    dilation_height = (dilation_height == 0) ? 1 : dilation_height;
    dilation_width = (dilation_width == 0) ? 1 : dilation_width;

    int in_channels = x.dimension(0);
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    int out_channels = w.dimension(1);
    int kernel_height_w = w.dimension(2);
    int kernel_width_w = w.dimension(3);

    demucscppdebug::assert_(kernel_height == kernel_height_w);
    demucscppdebug::assert_(kernel_width == kernel_width_w);

    // Transposed convolution output size calculation
    int out_height =
        (in_height - 1) * stride_height - 2 * pad_height + kernel_height;
    int out_width =
        (in_width - 1) * stride_width - 2 * pad_width + kernel_width;

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    y_out.setZero();

    for (int chout = 0; chout < out_channels; ++chout)
    {
        for (int i = 0; i < in_height; ++i)
        {
            for (int j = 0; j < in_width; ++j)
            {
                for (int chin = 0; chin < in_channels; ++chin)
                {
                    for (int m = 0; m < kernel_height; ++m)
                    {
                        for (int n = 0; n < kernel_width; ++n)
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
                                float y = x(chin, i, j) * w(chin, chout, m, n);
                                float sum = y_out(chout, oh, ow);
                                float t = sum + y;
                                float c = 0.0f;
                                // Kahan's sum correction
                                if (abs(sum) >= abs(y))
                                {
                                    c = (sum - t) + y;
                                }
                                else
                                {
                                    c = (y - t) + sum;
                                }
                                y_out(chout, oh, ow) = t + c;
                            }
                        }
                    }
                }
            }
        }
    }

    // Second pass to add the bias
    for (int chout = 0; chout < out_channels; ++chout)
    {
        for (int oh = 0; oh < out_height; ++oh)
        {
            for (int ow = 0; ow < out_width; ++ow)
            {
                y_out(chout, oh, ow) += b(chout);
            }
        }
    }

    return y_out;
}

Eigen::Tensor3dXf demucscpp::conv1d_tr(const Eigen::Tensor3dXf &x,
                                       const Eigen::Tensor3dXf &w,
                                       const Eigen::Tensor1dXf &b,
                                       int kernel_size, int stride, int pad,
                                       int dilation)
{
    // always ensure dilation is at least 1
    dilation = (dilation == 0) ? 1 : dilation;

    // copy w into a 4d tensor with trailing (,1) dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>(
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}));

    // move 0 axis to the end
    // we may not need this in the transposed case...
    Eigen::Tensor3dXf x_shuff = x.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // do 2d transposed convolution inference here
    // treating the in_freq dimension as a width dimension with a no-op kernel
    Eigen::Tensor3dXf y_out = demucscpp::conv2d_tr(
        x_shuff, w_4d, b, kernel_size, 1, stride, 1, pad, 0, dilation, 1);

    // move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        y_out.shuffle(Eigen::array<int, 3>({2, 0, 1}));
    return y_out_shuf;
}

Eigen::Tensor3dXf demucscpp::layer_norm(const Eigen::Tensor3dXf &x,
                                        const Eigen::Tensor1dXf &weight,
                                        const Eigen::Tensor1dXf &bias,
                                        float eps)
{
    int freq = x.dimension(0);
    int channels = x.dimension(1);
    int width = x.dimension(2);

    Eigen::Tensor3dXf y_out(freq, channels, width);

    for (int i = 0; i < freq; ++i)
    {
        for (int c = 0; c < channels; ++c)
        {
            Eigen::Tensor1dXf slice = x.chip(i, 0).chip(c, 0);
            Eigen::Tensor<float, 0> mean_tensor = slice.mean();
            float mean = mean_tensor(0);
            float var = demucscpp::calculate_variance(slice, mean);

            for (int w = 0; w < width; ++w)
            {
                float norm_val = (x(i, c, w) - mean) / std::sqrt(var + eps);
                y_out(i, c, w) = norm_val * weight(w) + bias(w);
            }
        }
    }

    return y_out;
}

void demucscpp::apply_dconv(struct demucscpp::demucs_model_4s &model,
                            Eigen::Tensor3dXf &y, int freq_idx, int encdec_idx,
                            int layer_idx, int mid_crop)
{
    // store another copy of y to sum back later
    Eigen::Tensor3dXf y_copy = y;

    // now dconv time

    // Conv1d(48, 6, kernel_size=(3,), stride=(1,), padding=(1,))
    y = demucscpp::conv1d<3, 1, 1, 1>(
        y,
        model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx][0],
        model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx][0]);

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_1_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [0],
        model.dconv_layers_1_groupnorm_bias[freq_idx][encdec_idx][layer_idx][0],
        1, 1e-05);

    y = demucscpp::gelu(y);

    // Conv1d(6, 96, kernel_size=(1,), stride=(1,))
    y = demucscpp::conv1d<1, 1, 0, 0>(
        y,
        model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx][0],
        model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx][0]);

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_4_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [0],
        model.dconv_layers_4_groupnorm_bias[freq_idx][encdec_idx][layer_idx][0],
        1, 1e-05);

    y = demucscpp::glu(y, 1);

    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][encdec_idx][layer_idx][0]);

    // now we add y to itself
    y = y + y_copy;

    // store another copy of y to sum back later
    y_copy = y;

    // NEXT ENTIRE SUBSEQUENCE OF DCONV WITH SLIGHTLY DIFFERENT PARAMS

    // Conv1d(48, 6, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    y = demucscpp::conv1d<3, 1, 2, 2>(
        y,
        model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx][1],
        model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx][1]);

    Eigen::Tensor3dXf y_cropped =
        y.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                Eigen::array<Eigen::Index, 3>(
                    {y.dimension(0), y.dimension(1), mid_crop}));

    y = y_cropped;

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_1_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [1],
        model.dconv_layers_1_groupnorm_bias[freq_idx][encdec_idx][layer_idx][1],
        1, 1e-05);

    y = demucscpp::gelu(y);

    // Conv1d(6, 96, kernel_size=(1,), stride=(1,))
    y = demucscpp::conv1d<1, 1, 0, 0>(
        y,
        model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx][1],
        model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx][1]);

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_4_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [1],
        model.dconv_layers_4_groupnorm_bias[freq_idx][encdec_idx][layer_idx][1],
        1, 1e-05);

    y = demucscpp::glu(y, 1);
    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][encdec_idx][layer_idx][1]);

    // if y_copy is shorter than y in the last dim
    // pad the last dim with zeros to match

    if (y_copy.dimension(2) < y.dimension(2))
    {
        // pad the last dim with zeros to match
        Eigen::Tensor3dXf padded_tensor_copy(
            y_copy.dimension(0), y_copy.dimension(1), y.dimension(2));
        padded_tensor_copy.setZero();
        padded_tensor_copy.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                                 y_copy.dimensions()) = y_copy;
        y_copy = padded_tensor_copy;
    }

    // now sum with itself
    y = y + y_copy;
}

void demucscpp::common_encoder_layer(
    Eigen::Tensor3dXf &q,       // q = x = frequency
    const Eigen::Tensor3dXf &k, // k = xt = time
    const Eigen::Tensor1dXf &norm1_weight, const Eigen::Tensor1dXf &norm1_bias,
    const Eigen::Tensor1dXf &norm2_weight, const Eigen::Tensor1dXf &norm2_bias,
    const Eigen::MatrixXf &in_proj_weight, const Eigen::VectorXf &in_proj_bias,
    const Eigen::MatrixXf &out_proj_weight,
    const Eigen::VectorXf &out_proj_bias, const Eigen::VectorXf &gamma1_scale,
    const Eigen::Tensor1dXf &norm3_weight, const Eigen::Tensor1dXf &norm3_bias,
    const Eigen::MatrixXf &linear1_weight, const Eigen::VectorXf &linear1_bias,
    const Eigen::MatrixXf &linear2_weight, const Eigen::VectorXf &linear2_bias,
    const Eigen::VectorXf &gamma2_scale,
    const Eigen::Tensor1dXf &norm_out_weight,
    const Eigen::Tensor1dXf &norm_out_bias, float eps, const int num_heads)
{
    // Normalize x using the norm1 weights and biases
    Eigen::Tensor3dXf q_norm =
        demucscpp::layer_norm(q, norm1_weight, norm1_bias, eps);
    Eigen::Tensor3dXf k_norm =
        demucscpp::layer_norm(k, norm2_weight, norm2_bias, eps);

    // Cross-attention block
    // Compute Q, K, V matrices

    int B = q.dimension(0);
    int T = q.dimension(1);
    int C = q.dimension(2);

    int B_k = k.dimension(0);
    int S = k.dimension(1);
    int C_k = k.dimension(2);

    demucscppdebug::assert_(B == B_k);
    demucscppdebug::assert_(B == 1);
    demucscppdebug::assert_(C == C_k);

    // Reshape q, k to 2D matrix of dimensions (T*B, C)

    // Use Eigen::Map to avoid manual loops for reshaping
    Eigen::MatrixXf q_norm_2d =
        Eigen::Map<const Eigen::MatrixXf>(q_norm.data(), T, C);
    Eigen::MatrixXf k_norm_2d =
        Eigen::Map<const Eigen::MatrixXf>(k_norm.data(), S, C);

    // Compute Q, K, V matrices
    Eigen::MatrixXf Q =
        q_norm_2d * in_proj_weight.block(0, 0, C, C).transpose();
    Eigen::MatrixXf K =
        k_norm_2d * in_proj_weight.block(C, 0, C, C).transpose();
    Eigen::MatrixXf V =
        k_norm_2d * in_proj_weight.block(2 * C, 0, C, C).transpose();

    Eigen::VectorXf q_bias = in_proj_bias.segment(0, C);
    Eigen::VectorXf k_bias = in_proj_bias.segment(C, C);
    Eigen::VectorXf v_bias = in_proj_bias.segment(2 * C, C);

    int head_split = C / num_heads;

    Eigen::Tensor3dXf Q_heads(T, num_heads, head_split);
    Eigen::Tensor3dXf K_heads(S, num_heads, head_split);
    Eigen::MatrixXf V_heads_2d(S * num_heads, head_split);

    // reverse loop order to combine with K and V
    for (int d = 0; d < head_split; ++d)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            for (int t = 0; t < T; ++t)
            {
                Q_heads(t, h, d) =
                    Q(t, h * head_split + d) + q_bias(h * head_split + d);
            }
            for (int s = 0; s < S; ++s)
            {
                K_heads(s, h, d) =
                    K(s, h * head_split + d) + k_bias(h * head_split + d);
                V_heads_2d(s * num_heads + h, d) =
                    V(s, h * head_split + d) + v_bias(h * head_split + d);
            }
        }
    }

    // Compute cross-attention scores
    Eigen::MatrixXf scores(num_heads * T, S); // Initialize to zeros

    for (int h = 0; h < num_heads; ++h)
    {
        // Extract the h-th head from Q_heads and K_heads
        Eigen::Tensor<float, 2> Q_head_tensor = Q_heads.chip(h, 1);
        Eigen::Tensor<float, 2> K_head_tensor = K_heads.chip(h, 1);

        // Reshape the tensors to matrices
        Eigen::Map<Eigen::MatrixXf> Q_head(Q_head_tensor.data(), T, head_split);
        Eigen::Map<Eigen::MatrixXf> K_head(K_head_tensor.data(), S, head_split);

        // Compute the dot product of Q_head and K_head
        Eigen::MatrixXf dot_product = Q_head * K_head.transpose();

        // Store the result in scores
        scores.block(h * T, 0, T, S) = dot_product / std::sqrt((float)head_split);
    }

    // Apply softmax to scores
    Eigen::ArrayXf max_vals = scores.rowwise().maxCoeff();
    Eigen::MatrixXf max_vals_expanded = max_vals.replicate(1, scores.cols());
    scores = (scores - max_vals_expanded).array().exp().matrix();
    Eigen::VectorXf row_sums = scores.rowwise().sum();
    Eigen::MatrixXf divisor = row_sums.replicate(1, scores.cols());
    scores = (scores.array() / divisor.array()).matrix();

    // Compute cross-attention output
    std::vector<Eigen::MatrixXf> cross_attn_out_3d;
    std::vector<Eigen::MatrixXf> V_heads_3d;
    std::vector<Eigen::MatrixXf> scores_3d;

    for (int h = 0; h < num_heads; ++h)
    {
        V_heads_3d.push_back(Eigen::MatrixXf(S, head_split));
        scores_3d.push_back(Eigen::MatrixXf(T, S));
        cross_attn_out_3d.push_back(Eigen::MatrixXf(T, head_split));
    }

    // first copy V_heads_2d, scores into 3d tensors
    for (int h = 0; h < num_heads; ++h)
    {
        for (int s = 0; s < S; ++s)
        {
            for (int t = 0; t < T; ++t)
            {
                scores_3d[h](t, s) = scores(h * T + t, s);
            }
            for (int d = 0; d < head_split; ++d)
            {
                V_heads_3d[h](s, d) = V_heads_2d(s * num_heads + h, d);
            }
        }
    }

    // now loop over 8 and do inner matmuls, assigning
    // results to cross_attn_out_3d
    for (int h = 0; h < num_heads; ++h)
    {
        cross_attn_out_3d[h] = scores_3d[h] * V_heads_3d[h];
    }

    Eigen::MatrixXf cross_attn_out(T, C);

    // now copy cross_attn_out_3d into cross_attn_out
    // from shape (8, T, 64) to (T, C)
    for (int t = 0; t < T; ++t)
    {
        for (int c = 0; c < C; ++c)
        {
            int h = c / head_split;
            int k_ = c % head_split;
            cross_attn_out(t, c) = cross_attn_out_3d[h](t, k_);
        }
    }

    // Apply output projection
    Eigen::MatrixXf out_proj = cross_attn_out * out_proj_weight.transpose();
    out_proj.array().rowwise() += out_proj_bias.transpose().array();

    // now we need x = q + out_proj, but let's store that in 3d q
    for (int t = 0; t < T; ++t)
    {
        for (int c = 0; c < C; ++c)
        {
            q(0, t, c) += out_proj(t, c) * gamma1_scale(c);
        }
    }

    // copy q into x_2d
    Eigen::MatrixXf q_2d(T, C);
    q_2d = Eigen::Map<const Eigen::MatrixXf>(q.data(), T, C);

    // before feedforward, apply norm3 to x i.e. q
    q_norm = demucscpp::layer_norm(q, norm3_weight, norm3_bias, eps);
    q_norm_2d = Eigen::Map<const Eigen::MatrixXf>(q_norm.data(), T, C);

    // Feedforward block
    // Linear layer 1
    Eigen::MatrixXf ff1 = q_norm_2d * linear1_weight.transpose();
    ff1.rowwise() += linear1_bias.transpose();

    ff1 = demucscpp::gelu(ff1);

    // Linear layer 2
    Eigen::MatrixXf ff2 = ff1 * linear2_weight.transpose();
    ff2.rowwise() += linear2_bias.transpose();

    // Apply gamma_2 scale directly on 2D matrix
    // ff2.array().colwise() *= gamma_2_scale.array();
    ff2 = ff2.array().rowwise() * gamma2_scale.transpose().array();

    // now x = x + self.gamma_2(self._ff_block(self.norm3(q))))
    q_2d += ff2;

    // Map the 2D data back into a 3D tensor with dimensions (T, B, C)
    q = Eigen::TensorMap<Eigen::Tensor3dXf>(q_2d.data(), T, B, C);

    // Swap the first and last dimensions to get a tensor with dimensions (B, C,
    // T)
    Eigen::array<int, 3> permute_dims = {1, 2, 0};
    Eigen::Tensor3dXf q_shuf = q.shuffle(permute_dims);

    // Normalize the output with norm_out/MyGroupNorm
    q = demucscpp::group_norm(q_shuf, norm_out_weight, norm_out_bias, 1, eps);

    Eigen::array<int, 3> permute_dims_2 = {0, 2, 1};
    q_shuf = q.shuffle(permute_dims_2);

    q = q_shuf;
}
