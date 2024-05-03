#include "encdec.hpp"
#include "layers.hpp"
#include "model.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

void demucscpp::apply_freq_encoder(const struct demucscpp::demucs_model &model,
                                   int encoder_idx,
                                   const Eigen::Tensor3dXf &x_in,
                                   Eigen::Tensor3dXf &x_out)
{
    Eigen::Tensor3dXf x_shuf = x_in.shuffle(Eigen::array<int, 3>({2, 0, 1}));

    // 2D Convolution operation
    Eigen::Tensor3dXf y;

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d_fused_gelu<4, 48, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 1:
        y = demucscpp::conv1d_fused_gelu<48, 96, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 2:
        y = demucscpp::conv1d_fused_gelu<96, 192, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 3:
        y = demucscpp::conv1d_fused_gelu<192, 384, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    };

    // reverse all dims
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({2, 1, 0}));
    demucscpp::apply_dconv(model, y_shuff, 0, 0, encoder_idx,
                           y_shuff.dimension(2));

    // swap back from H,C,W to C,H,W
    // then put W in front to use conv1d function for width=1 conv2d
    y = y_shuff.shuffle(Eigen::array<int, 3>({2, 1, 0}));

    // need rewrite, norm2, glu
    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 96, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 192, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 384, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 768, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    };

    y_shuff = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // copy into x_out
    x_out = demucscpp::glu(y_shuff, 0);
}

void demucscpp::apply_time_encoder(const struct demucscpp::demucs_model &model,
                                   int tencoder_idx,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out)
{
    int crop = demucscpp::TIME_BRANCH_LEN_0;
    // switch case for tencoder_idx
    switch (tencoder_idx)
    {
    case 0:
        break;
    case 1:
        crop = demucscpp::TIME_BRANCH_LEN_1;
        break;
    case 2:
        crop = demucscpp::TIME_BRANCH_LEN_2;
        break;
    case 3:
        crop = demucscpp::TIME_BRANCH_LEN_3;
        break;
    }

    // now implement the forward pass
    // first, apply the convolution
    // Conv1d(2, 48, kernel_size=(8,), stride=(4,), padding=(2,))
    Eigen::Tensor3dXf yt;

    switch (tencoder_idx)
    {
    case 0:
        yt = demucscpp::conv1d_fused_gelu<2, 48, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 1:
        yt = demucscpp::conv1d_fused_gelu<48, 96, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 2:
        yt = demucscpp::conv1d_fused_gelu<96, 192, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 3:
        yt = demucscpp::conv1d_fused_gelu<192, 384, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    };

    // now dconv time
    demucscpp::apply_dconv(model, yt, 1, 0, tencoder_idx, crop);

    // end of dconv?

    // need rewrite, norm2, glu
    switch (tencoder_idx)
    {
    case 0:
        yt = demucscpp::conv1d<48, 96, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 1:
        yt = demucscpp::conv1d<96, 192, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 2:
        yt = demucscpp::conv1d<192, 384, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 3:
        yt = demucscpp::conv1d<384, 768, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    };

    xt_out = demucscpp::glu(yt, 1);
}

void demucscpp::apply_freq_decoder(const struct demucscpp::demucs_model &model,
                                   int decoder_idx,
                                   const Eigen::Tensor3dXf &x_in,
                                   Eigen::Tensor3dXf &x_out,
                                   const Eigen::Tensor3dXf &skip)
{
    Eigen::Tensor3dXf y = x_in + skip;

    // need rewrite, norm2, glu
    switch (decoder_idx)
    {
    case 0:
        y = demucscpp::conv2d<384, 768, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.decoder_rewrite_weight[decoder_idx],
            model.decoder_rewrite_bias[decoder_idx]);
        break;
    case 1:
        y = demucscpp::conv2d<192, 384, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.decoder_rewrite_weight[decoder_idx],
            model.decoder_rewrite_bias[decoder_idx]);
        break;
    case 2:
        y = demucscpp::conv2d<96, 192, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.decoder_rewrite_weight[decoder_idx],
            model.decoder_rewrite_bias[decoder_idx]);
        break;
    case 3:
        y = demucscpp::conv2d<48, 96, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.decoder_rewrite_weight[decoder_idx],
            model.decoder_rewrite_bias[decoder_idx]);
        break;
    };

    y = demucscpp::glu(y, 0);

    // swap first and second dimensions
    // from C,H,W into H,C,W
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));
    y = y_shuff;

    // start the DConv
    demucscpp::apply_dconv(model, y, 0, 1, 4 - decoder_idx - 1, y.dimension(2));

    // dconv finished

    // swap back from H,C,W to C,H,W
    Eigen::Tensor3dXf y_shuff_2 = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));

    // now time for the transpose convolution

    // 2D Convolution operation
    switch (decoder_idx)
    {
    case 0:
        y = demucscpp::conv2d_tr_gemm_fused_gelu<384, 192, 8, 1, 4, 1, 0, 0, 1,
                                                 1>(
            y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
            model.decoder_conv_tr_bias[decoder_idx]);
        break;
    case 1:
        y = demucscpp::conv2d_tr_gemm_fused_gelu<192, 96, 8, 1, 4, 1, 0, 0, 1,
                                                 1>(
            y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
            model.decoder_conv_tr_bias[decoder_idx]);
        break;
    case 2:
        y = demucscpp::conv2d_tr_gemm_fused_gelu<96, 48, 8, 1, 4, 1, 0, 0, 1,
                                                 1>(
            y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
            model.decoder_conv_tr_bias[decoder_idx]);
        break;
    case 3:
        if (model.is_4sources)
        {
            y = demucscpp::conv2d_tr<48, 16, 8, 1, 4, 1, 0, 0, 1, 1>(
                y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
                model.decoder_conv_tr_bias[decoder_idx]);
        }
        else
        {
            y = demucscpp::conv2d_tr<48, 24, 8, 1, 4, 1, 0, 0, 1, 1>(
                y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
                model.decoder_conv_tr_bias[decoder_idx]);
        }
    };

    int y_dim1_begin = 2;
    int y_dim1_end = y.dimension(1) - 4;

    // remove 2 elements from begin and end of y along dimension 1 (0, 1, 2)
    x_out = y.slice(Eigen::array<Eigen::Index, 3>({0, y_dim1_begin, 0}),
                    Eigen::array<Eigen::Index, 3>(
                        {y.dimension(0), y_dim1_end, y.dimension(2)}));
}

void demucscpp::apply_time_decoder(const struct demucscpp::demucs_model &model,
                                   int tdecoder_idx,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out,
                                   const Eigen::Tensor3dXf &skip)
{
    int crop = demucscpp::TIME_BRANCH_LEN_3;
    int out_length = demucscpp::TIME_BRANCH_LEN_2;
    // switch case for tdecoder_idx
    switch (tdecoder_idx)
    {
    case 0:
        break;
    case 1:
        crop = demucscpp::TIME_BRANCH_LEN_2;
        out_length = demucscpp::TIME_BRANCH_LEN_1;
        break;
    case 2:
        crop = demucscpp::TIME_BRANCH_LEN_1;
        out_length = demucscpp::TIME_BRANCH_LEN_0;
        break;
    case 3:
        crop = demucscpp::TIME_BRANCH_LEN_0;
        out_length = demucscpp::TIME_BRANCH_LEN_IN;
        break;
    }

    // need rewrite, norm2, glu
    Eigen::Tensor3dXf yt;
    switch (tdecoder_idx)
    {
    case 0:
        yt = demucscpp::conv1d<384, 768, 3, 1, 1, 1>(
            xt_in + skip, model.tdecoder_rewrite_weight[tdecoder_idx],
            model.tdecoder_rewrite_bias[tdecoder_idx]);
        break;
    case 1:
        yt = demucscpp::conv1d<192, 384, 3, 1, 1, 1>(
            xt_in + skip, model.tdecoder_rewrite_weight[tdecoder_idx],
            model.tdecoder_rewrite_bias[tdecoder_idx]);
        break;
    case 2:
        yt = demucscpp::conv1d<96, 192, 3, 1, 1, 1>(
            xt_in + skip, model.tdecoder_rewrite_weight[tdecoder_idx],
            model.tdecoder_rewrite_bias[tdecoder_idx]);
        break;
    case 3:
        yt = demucscpp::conv1d<48, 96, 3, 1, 1, 1>(
            xt_in + skip, model.tdecoder_rewrite_weight[tdecoder_idx],
            model.tdecoder_rewrite_bias[tdecoder_idx]);
        break;
    };

    yt = demucscpp::glu(yt, 1);

    // start the DConv
    demucscpp::apply_dconv(model, yt, 1, 1, 4 - tdecoder_idx - 1, crop);

    // dconv finished

    // next, apply the final transpose convolution
    Eigen::Tensor3dXf yt_tmp;

    switch (tdecoder_idx)
    {
    case 0:
        yt_tmp = demucscpp::conv1d_tr_fused_gelu<384, 192, 8, 4, 0, 1>(
            yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
            model.tdecoder_conv_tr_bias[tdecoder_idx]);
        break;
    case 1:
        yt_tmp = demucscpp::conv1d_tr_fused_gelu<192, 96, 8, 4, 0, 1>(
            yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
            model.tdecoder_conv_tr_bias[tdecoder_idx]);
        break;
    case 2:
        yt_tmp = demucscpp::conv1d_tr_fused_gelu<96, 48, 8, 4, 0, 1>(
            yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
            model.tdecoder_conv_tr_bias[tdecoder_idx]);
        break;
    case 3:
        if (model.is_4sources)
        {
            yt_tmp = demucscpp::conv1d_tr<48, 8, 8, 4, 0, 1>(
                yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
                model.tdecoder_conv_tr_bias[tdecoder_idx]);
        }
        else
        {
            yt_tmp = demucscpp::conv1d_tr<48, 12, 8, 4, 0, 1>(
                yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
                model.tdecoder_conv_tr_bias[tdecoder_idx]);
        }
        break;
    };

    yt = yt_tmp;

    // remove padding
    // 2:2+length
    xt_out = yt.slice(Eigen::array<Eigen::Index, 3>({0, 0, 2}),
                      Eigen::array<Eigen::Index, 3>(
                          {yt.dimension(0), yt.dimension(1), out_length}));
}

void demucscpp_v3::apply_freq_encoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                                   int encoder_idx,
                                   const Eigen::Tensor3dXf &x_in,
                                   Eigen::Tensor3dXf &x_out)
{
    Eigen::Tensor3dXf x_shuf = x_in.shuffle(Eigen::array<int, 3>({2, 0, 1}));

    // 2D Convolution operation
    Eigen::Tensor3dXf y;

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d_fused_gelu<4, 48, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 1:
        y = demucscpp::conv1d_fused_gelu<48, 96, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 2:
        y = demucscpp::conv1d_fused_gelu<96, 192, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    case 3:
        y = demucscpp::conv1d_fused_gelu<192, 384, 8, 4, 2, 1>(
            x_shuf, model.encoder_conv_weight[encoder_idx],
            model.encoder_conv_bias[encoder_idx]);
        break;
    };

    // reverse all dims
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({2, 1, 0}));
    demucscpp_v3::apply_dconv_v3(model, y_shuff, 0, encoder_idx,
                           y_shuff.dimension(2));

    // swap back from H,C,W to C,H,W
    // then put W in front to use conv1d function for width=1 conv2d
    y = y_shuff.shuffle(Eigen::array<int, 3>({2, 1, 0}));

    // need rewrite, norm2, glu
    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 96, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 192, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 384, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 768, 1, 1, 0, 1>(
            y, model.encoder_rewrite_weight[encoder_idx],
            model.encoder_rewrite_bias[encoder_idx]);
        break;
    };

    y_shuff = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // copy into x_out
    x_out = demucscpp::glu(y_shuff, 0);
}

void demucscpp_v3::apply_time_encoder_v3(const struct demucscpp_v3::demucs_v3_model &model,
                                   int tencoder_idx,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out)
{
    int crop = demucscpp::TIME_BRANCH_LEN_0;
    // switch case for tencoder_idx
    switch (tencoder_idx)
    {
    case 0:
        break;
    case 1:
        crop = demucscpp::TIME_BRANCH_LEN_1;
        break;
    case 2:
        crop = demucscpp::TIME_BRANCH_LEN_2;
        break;
    case 3:
        crop = demucscpp::TIME_BRANCH_LEN_3;
        break;
    }

    // now implement the forward pass
    // first, apply the convolution
    // Conv1d(2, 48, kernel_size=(8,), stride=(4,), padding=(2,))
    Eigen::Tensor3dXf yt;

    switch (tencoder_idx)
    {
    case 0:
        yt = demucscpp::conv1d_fused_gelu<2, 48, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 1:
        yt = demucscpp::conv1d_fused_gelu<48, 96, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 2:
        yt = demucscpp::conv1d_fused_gelu<96, 192, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    case 3:
        yt = demucscpp::conv1d_fused_gelu<192, 384, 8, 4, 2, 1>(
            xt_in, model.tencoder_conv_weight[tencoder_idx],
            model.tencoder_conv_bias[tencoder_idx]);
        break;
    };

    // now dconv time
    demucscpp_v3::apply_dconv_v3(model, yt, 1, tencoder_idx, crop);

    // end of dconv?

    // need rewrite, norm2, glu
    switch (tencoder_idx)
    {
    case 0:
        yt = demucscpp::conv1d<48, 96, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 1:
        yt = demucscpp::conv1d<96, 192, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 2:
        yt = demucscpp::conv1d<192, 384, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    case 3:
        yt = demucscpp::conv1d<384, 768, 1, 1, 0, 1>(
            yt, model.tencoder_rewrite_weight[tencoder_idx],
            model.tencoder_rewrite_bias[tencoder_idx]);
        break;
    };

    xt_out = demucscpp::glu(yt, 1);
}

void demucscpp_v3::apply_time_encoder_4(const struct demucscpp_v3::demucs_v3_model &model,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out)
{
    // now implement the forward pass
    // first, apply the convolution
    // Conv1d(2, 48, kernel_size=(8,), stride=(4,), padding=(2,))
    Eigen::Tensor3dXf yt = demucscpp::conv1d<384, 768, 8, 4, 2, 1>(
            xt_in, model.tencoder_4_conv_weight,
            model.tencoder_4_conv_bias);

    xt_out = yt;
}

void demucscpp_v3::apply_freq_shared_encoder_4_5(const struct demucscpp_v3::demucs_v3_model &model,
                                   const Eigen::Tensor3dXf &x_in,
                                   const Eigen::Tensor3dXf &x_inject,
                                   const int encoder_idx,
                                   Eigen::Tensor3dXf &x_out,
                                   struct demucscpp_v3::demucs_v3_segment_buffers &buffers)
{
    // 2D Convolution operation
    Eigen::Tensor3dXf y;
    Eigen::Tensor3dXf y_shuff;

    //demucscppdebug::debug_tensor_3dxf(x_in, "x_in enc_4_5 before conv");

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv2d<384, 768, 8, 1, 4, 1, 0, 0, 1, 1>(
            x_in, model.encoder_4_conv_weight,
            model.encoder_4_5_conv_bias[encoder_idx]);
        // encoder 4 (i.e. encoder_idx 0) has the inject param
        // swap first two dims of x_inject
        y += x_inject.shuffle(Eigen::array<int, 3>({1, 0, 2}));
        break;
    case 1:
        // shuffle first two dims of x_in
        // first assign y to x_in with first two dims swapped
        y = x_in.shuffle(Eigen::array<int, 3>({1, 0, 2}));
        y = demucscpp::conv1d<768, 1536, 4, 2, 1, 1>(
            y, model.encoder_5_conv_weight,
            model.encoder_4_5_conv_bias[encoder_idx]);
        // shuffle dims of y for group norm + gelu
        y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));
        y = y_shuff;
        break;
    };

    //demucscppdebug::debug_tensor_3dxf(y, "y enc_4_5 before group norm");

    // apply groupnorm
    y = demucscpp_v3::group_norm_fused_gelu(y, model.encoder_4_5_norm1_weight[encoder_idx],
                             model.encoder_4_5_norm1_bias[encoder_idx], 4, 1e-05);

    //demucscppdebug::debug_tensor_3dxf(y, "y enc_4_5 before dconv");

    // swap first two dims
    y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));

    // special dconv with bilstm + local attn
    demucscpp_v3::apply_dconv_v3_encoder_4_5(model, y_shuff, encoder_idx,
                           y_shuff.dimension(2), buffers);

    //demucscppdebug::debug_tensor_3dxf(y_shuff, "y enc_4_5 after dconv");

    // swap back from H,C,W to C,H,W
    // then put W in front to use conv1d function for width=1 conv2d
    //y = y_shuff.shuffle(Eigen::array<int, 3>({1, 0, 2}));
    y = y_shuff;

    //demucscppdebug::debug_tensor_3dxf(y, "y enc_4_5 after shuff");
    //demucscppdebug::debug_tensor_3dxf(y, "y before rewrite");

    // need rewrite, norm2, glu
    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<768, 1536, 1, 1, 0, 1>(
            y, model.encoder_4_5_rewrite_weight[encoder_idx],
            model.encoder_4_5_rewrite_bias[encoder_idx]);
        break;
    case 1:
        y = demucscpp::conv1d<1536, 3072, 1, 1, 0, 1>(
            y, model.encoder_4_5_rewrite_weight[encoder_idx],
            model.encoder_4_5_rewrite_bias[encoder_idx]);
        break;
    };

    //demucscppdebug::debug_tensor_3dxf(y, "y after rewrite");

    // apply groupnorm
    y = demucscpp::group_norm(y, model.encoder_4_5_norm2_weight[encoder_idx],
                             model.encoder_4_5_norm2_bias[encoder_idx], 4, 1e-05);

    //demucscppdebug::debug_tensor_3dxf(y, "y after group norm");

    // copy into x_out
    if (encoder_idx == 0) {
        y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));
        x_out = demucscpp::glu(y_shuff, 0);
    } else {
        y_shuff = y;
        x_out = demucscpp::glu(y_shuff, 1);
    }

    //demucscppdebug::debug_tensor_3dxf(x_out, "y after glu");
}

Eigen::Tensor3dXf demucscpp_v3::apply_freq_shared_decoder_0_1(
    const struct demucscpp_v3::demucs_v3_model &model,
    const int decoder_idx,
    const Eigen::Tensor3dXf &x_in,
    Eigen::Tensor3dXf &x_out,
    const Eigen::Tensor3dXf &skip)
{
    //demucscppdebug::debug_tensor_3dxf(skip, "skip");
    //demucscppdebug::debug_tensor_3dxf(x_in, "x_in pre-skip");

    // TODO: after done, collapse all switch cases into one
    // wont need print statements

    Eigen::Tensor3dXf y = x_in.shuffle(Eigen::array<int, 3>({1, 0, 2})) + skip;

    //demucscppdebug::debug_tensor_3dxf(y, "x_in post-skip");

    // first glu(norm1(rewrite))
    switch (decoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<1536, 3072, 3, 1, 1, 1>(
            y, model.decoder_0_rewrite_weight,
            model.decoder_0_1_rewrite_bias[decoder_idx]);
        break;
    case 1:
        y = demucscpp::conv2d<768, 1536, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.decoder_1_rewrite_weight,
            model.decoder_0_1_rewrite_bias[decoder_idx]);

        // debug weight and bias
        //demucscppdebug::debug_tensor_4dxf(model.decoder_1_rewrite_weight, "decoder_1_rewrite_weight");
        //demucscppdebug::debug_tensor_1dxf(model.decoder_0_1_rewrite_bias[decoder_idx], "decoder_0_1_rewrite_bias");

        break;
    };

    //demucscppdebug::debug_tensor_3dxf(y, "y after rewrite");

    switch (decoder_idx)
    {
    case 0:
        // apply groupnorm1 with norm1 weights
        y = demucscpp::group_norm(y, model.decoder_0_1_norm1_weight[decoder_idx],
                                model.decoder_0_1_norm1_bias[decoder_idx], 4, 1e-05);

        //demucscppdebug::debug_tensor_3dxf(y, "y after group norm");

        y = demucscpp::glu(y, 1);
        break;
    case 1:
        // apply groupnorm1 with norm1 weights
        y = demucscpp_v3::group_norm_2(y, model.decoder_0_1_norm1_weight[decoder_idx],
                                model.decoder_0_1_norm1_bias[decoder_idx], 4, 1e-05);

        //demucscppdebug::debug_tensor_3dxf(y, "y after group norm");

        y = demucscpp::glu(y, 0);
        break;
    }

    //demucscppdebug::debug_tensor_3dxf(y, "y after glu");

    // return pre, to be used optionally (as first input to time decoder)
    Eigen::Tensor3dXf pre_ret = y;

    Eigen::Tensor3dXf y2;

    // no dconv for decoders
    // simply conv_tr -> norm2

    // 2D Convolution operation
    switch (decoder_idx)
    {
    case 0:
        y = demucscpp::conv1d_tr<1536, 768, 4, 2, 0, 1>(
            y, model.decoder_0_conv_tr_weight,
            model.decoder_0_1_conv_tr_bias[decoder_idx]);

        //demucscppdebug::debug_tensor_3dxf(y, "y after conv_tr");

        // make a y2 first which does group norm and gelu separately for debugging
        // group_norm_2 should be similar to group_norm but operating on the flipped first two axes
        // to avoid needing to shuffle
        y2 = demucscpp::group_norm(y, model.decoder_0_1_norm2_weight[decoder_idx],
                                model.decoder_0_1_norm2_bias[decoder_idx], 4, 1e-05);

        //demucscppdebug::debug_tensor_3dxf(y2, "y2 after group norm");

        y2 = demucscpp::gelu(y2);

        //demucscppdebug::debug_tensor_3dxf(y2, "y2 after group norm + gelu");

        // now apply groupnorm2 with norm2 weights
        y = demucscpp_v3::group_norm_fused_gelu_2(y, model.decoder_0_1_norm2_weight[decoder_idx],
                                model.decoder_0_1_norm2_bias[decoder_idx], 4, 1e-05);


        break;
    case 1:
        y = demucscpp::conv2d_tr<768, 384, 8, 1, 4, 1, 0, 0, 1, 1>(
            y, model.decoder_1_conv_tr_weight,
            model.decoder_0_1_conv_tr_bias[decoder_idx]);

        //demucscppdebug::debug_tensor_3dxf(y, "y after conv_tr");

        // now apply groupnorm2 with norm2 weights
        y = demucscpp_v3::group_norm_fused_gelu(y, model.decoder_0_1_norm2_weight[decoder_idx],
                                model.decoder_0_1_norm2_bias[decoder_idx], 4, 1e-05);

        break;
    };

    //demucscppdebug::debug_tensor_3dxf(y, "y after group norm + gelu");

    if (decoder_idx == 0) {
        int y_dim2_begin = 2;
        int y_dim2_end = y.dimension(2) - 2;

        // remove 2 elements from begin and end of y along dimension 2
        x_out = y.slice(Eigen::array<Eigen::Index, 3>({0, 0, y_dim2_begin}),
                        Eigen::array<Eigen::Index, 3>(
                            {y.dimension(0), y.dimension(1), y_dim2_end}));
    } else {
        x_out = y;
    }

    return pre_ret;
}

void demucscpp_v3::apply_time_decoder_0(
    const struct demucscpp_v3::demucs_v3_model &model,
    const Eigen::Tensor3dXf &x_in,
    Eigen::Tensor3dXf &x_out,
    const Eigen::Tensor3dXf &skip)
{
    // simple decoder
    // rewrite and conv_tr, no group norms
    //demucscppdebug::debug_tensor_3dxf(skip, "skip");
    //demucscppdebug::debug_tensor_3dxf(x_in, "x_in pre-skip");

    // TODO: after done, collapse all switch cases into one
    // wont need print statements

    Eigen::Tensor3dXf y = x_in + skip;

    //demucscppdebug::debug_tensor_3dxf(y, "y post-skip");

    // swap first two dims
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));

    // no norm1, rewrite, dconv for tdecoder0
    // simply conv_tr -> norm2

    // 2D Convolution operation
    y = demucscpp::conv1d_tr<768, 384, 8, 4, 0, 1>(
        y_shuff, model.tdecoder_0_conv_tr_weight,
        model.tdecoder_0_conv_tr_bias);

    //demucscppdebug::debug_tensor_3dxf(y, "y after conv1d_tr");

    // now apply groupnorm2 with norm2 weights
    y = demucscpp_v3::group_norm_fused_gelu_2(y, model.tdecoder_0_norm2_weight,
                             model.tdecoder_0_norm2_bias, 4, 1e-05);

    //demucscppdebug::debug_tensor_3dxf(y, "y after group norm + gelu");

    int y_dim2_begin = 4;
    int y_dim2_end = y.dimension(2) - 4;

    // remove 2 elements from begin and end of y along dimension 2
    x_out = y.slice(Eigen::array<Eigen::Index, 3>({0, 0, y_dim2_begin}),
                    Eigen::array<Eigen::Index, 3>(
                        {y.dimension(0), y.dimension(1), y_dim2_end}));
}

void demucscpp_v3::apply_common_decoder(
    const struct demucscpp_v3::demucs_v3_model &model,
    const int freq_or_time_idx,
    const int decoder_idx,
    const Eigen::Tensor3dXf &x_in,
    Eigen::Tensor3dXf &x_out,
    const Eigen::Tensor3dXf &skip)
{
    // simple decoder
    // rewrite and conv_tr, no group norms
    //demucscppdebug::debug_tensor_3dxf(skip, "skip");
    //demucscppdebug::debug_tensor_3dxf(x_in, "x_in pre-skip");

    // TODO: after done, collapse all switch cases into one
    // wont need print statements

    Eigen::Tensor3dXf y = x_in + skip;

    //demucscppdebug::debug_tensor_3dxf(y, "y post-skip");

    // first glu(norm1(rewrite))
    if ((freq_or_time_idx == 0) && (decoder_idx == 0)) {
        y = demucscpp::conv2d<384, 768, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.freq_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 0)) {
        y = demucscpp::conv1d<384, 768, 3, 1, 1, 1>(
            y, model.time_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 1)) {
        y = demucscpp::conv2d<192, 384, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.freq_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 1)) {
        y = demucscpp::conv1d<192, 384, 3, 1, 1, 1>(
            y, model.time_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 2)) {
        y = demucscpp::conv2d<96, 192, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.freq_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 2)) {
        y = demucscpp::conv1d<96, 192, 3, 1, 1, 1>(
            y, model.time_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 3)) {
        y = demucscpp::conv2d<48, 96, 3, 3, 1, 1, 1, 1, 1, 1>(
            y, model.freq_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 3)) {
        y = demucscpp::conv1d<48, 96, 3, 1, 1, 1>(
            y, model.time_decoders_rewrite_weight[decoder_idx],
            model.decoders_rewrite_bias[freq_or_time_idx][decoder_idx]);
    }

    //demucscppdebug::debug_tensor_3dxf(y, "y after rewrite conv2d");

    y = demucscpp::glu(y, freq_or_time_idx);

    //demucscppdebug::debug_tensor_3dxf(y, "y after glu");

    // no dconv for decoders, no norm2
    // simply conv_tr

    // 2D Convolution operation
    if ((freq_or_time_idx == 0) && (decoder_idx == 0)) {
        y = demucscpp::conv2d_tr<384, 192, 8, 1, 4, 1, 0, 0, 1, 1>(
            y, model.freq_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 0)) {
        y = demucscpp::conv1d_tr<384, 192, 8, 4, 0, 1>(
            y, model.time_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 1)) {
        y = demucscpp::conv2d_tr<192, 96, 8, 1, 4, 1, 0, 0, 1, 1>(
            y, model.freq_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 1)) {
        y = demucscpp::conv1d_tr<192, 96, 8, 4, 0, 1>(
            y, model.time_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 2)) {
        y = demucscpp::conv2d_tr<96, 48, 8, 1, 4, 1, 0, 0, 1, 1>(
            y, model.freq_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 2)) {
        y = demucscpp::conv1d_tr<96, 48, 8, 4, 0, 1>(
            y, model.time_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 0) && (decoder_idx == 3)) {
        y = demucscpp::conv2d_tr<48, 16, 8, 1, 4, 1, 0, 0, 1, 1>(
            y, model.freq_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    } else if ((freq_or_time_idx == 1) && (decoder_idx == 3)) {
        y = demucscpp::conv1d_tr<48, 8, 8, 4, 0, 1>(
            y, model.time_decoders_conv_tr_weight[decoder_idx],
            model.decoders_conv_tr_bias[freq_or_time_idx][decoder_idx]);
    }

    //demucscppdebug::debug_tensor_3dxf(y, "y after conv1d_tr");

    // apply gelu activation unless it's the last layer
    // self.last in pytorch
    if (decoder_idx < 3) {
        y = demucscpp::gelu(y);
    }

    //demucscppdebug::debug_tensor_3dxf(y, "y after gelu");

    if (freq_or_time_idx == 1) {
        // for time branch
        // remove padding
        // 2:2+length
        int out_length = x_out.dimension(2);
        x_out = y.slice(Eigen::array<Eigen::Index, 3>({0, 0, 2}),
                        Eigen::array<Eigen::Index, 3>(
                            {y.dimension(0), y.dimension(1), out_length}));

    } else {
        // for freq branch
        int y_dim1_begin = 2;
        int y_dim1_end = y.dimension(1) - 4;

        // remove 2 elements from begin and end of y along dimension 1 (0, 1, 2)
        x_out = y.slice(Eigen::array<Eigen::Index, 3>({0, y_dim1_begin, 0}),
                        Eigen::array<Eigen::Index, 3>(
                            {y.dimension(0), y_dim1_end, y.dimension(2)}));

    }
}
