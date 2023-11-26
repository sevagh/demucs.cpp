#include "encdec.hpp"
#include "layers.hpp"
#include "model.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

// forward declaration to apply a frequency encoder
void demucscpp::apply_freq_encoder(struct demucscpp::demucs_model_4s &model,
                                   int encoder_idx,
                                   const Eigen::Tensor3dXf &x_in,
                                   Eigen::Tensor3dXf &x_out)
{

    Eigen::Tensor3dXf x_shuf = x_in.shuffle(Eigen::array<int, 3>({2, 0, 1}));
    // 2D Convolution operation
    Eigen::Tensor3dXf y =
        demucscpp::conv1d<8, 4, 2, 1>(x_shuf, model.encoder_conv_weight[encoder_idx],
                          model.encoder_conv_bias[encoder_idx]);

    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    y = demucscpp::gelu(y_shuff);

    // swap first and second dimensions
    // from C,H,W into H,C,W
    y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));
    y = y_shuff;
    demucscpp::apply_dconv(model, y, 0, 0, encoder_idx, 336);

    // swap back from H,C,W to C,H,W
    // then put W in front to use conv1d function for width=1 conv2d
    Eigen::Tensor3dXf y_shuff_2 = y.shuffle(Eigen::array<int, 3>({2, 1, 0}));

    // need rewrite, norm2, glu
    y = demucscpp::conv1d<1, 1, 0, 1>(y_shuff_2, model.encoder_rewrite_weight[encoder_idx],
                          model.encoder_rewrite_bias[encoder_idx]);

    y_shuff_2 = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // copy into x_out
    x_out = demucscpp::glu(y_shuff_2, 0);
}

// forward declaration to apply a time encoder
void demucscpp::apply_time_encoder(struct demucscpp::demucs_model_4s &model,
                                   int tencoder_idx,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out)
{
    int crop = 85995;
    // switch case for tencoder_idx
    switch (tencoder_idx)
    {
    case 0:
        crop = 85995;
        break;
    case 1:
        crop = 21499;
        break;
    case 2:
        crop = 5375;
        break;
    case 3:
        crop = 1344;
        break;
    default:
        std::cout << "invalid tencoder_idx" << std::endl;
        break;
    }

    // now implement the forward pass
    // first, apply the convolution
    // Conv1d(2, 48, kernel_size=(8,), stride=(4,), padding=(2,))
    Eigen::Tensor3dXf yt =
        demucscpp::conv1d<8, 4, 2, 1>(xt_in, model.tencoder_conv_weight[tencoder_idx],
                          model.tencoder_conv_bias[tencoder_idx]);

    yt = demucscpp::gelu(yt);

    // now dconv time
    demucscpp::apply_dconv(model, yt, 1, 0, tencoder_idx, crop);

    // end of dconv?

    // need rewrite, norm2, glu
    yt = demucscpp::conv1d<1, 1, 0, 1>(yt, model.tencoder_rewrite_weight[tencoder_idx],
                           model.tencoder_rewrite_bias[tencoder_idx]);

    xt_out = demucscpp::glu(yt, 1);
}

// forward declaration to apply a frequency decoder
void demucscpp::apply_freq_decoder(struct demucscpp::demucs_model_4s &model,
                                   int decoder_idx,
                                   const Eigen::Tensor3dXf &x_in,
                                   Eigen::Tensor3dXf &x_out,
                                   const Eigen::Tensor3dXf &skip)
{
    // need rewrite, norm2, glu
    Eigen::Tensor3dXf y = demucscpp::conv2d<3, 3, 1, 1, 1, 1, 1, 1>(
        x_in + skip, model.decoder_rewrite_weight[decoder_idx],
        model.decoder_rewrite_bias[decoder_idx]);

    y = demucscpp::glu(y, 0);

    // pre-dconv freq reshape: y.shape: torch.Size([8, 384, 336])
    // post-dconv freq reshape: y.shape: torch.Size([1, 384, 8, 336])
    // CHW -> HWC

    // swap first and second dimensions
    // from C,H,W into H,C,W
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));
    y = y_shuff;

    // start the DConv

    demucscpp::apply_dconv(model, y, 0, 1, 4-decoder_idx-1, 336);

    // dconv finished

    // swap back from H,C,W to C,H,W
    Eigen::Tensor3dXf y_shuff_2 = y.shuffle(Eigen::array<int, 3>({1, 0, 2}));

    // now time for the transpose convolution

    // 2D Convolution operation
    y = demucscpp::conv2d_tr(
        y_shuff_2, model.decoder_conv_tr_weight[decoder_idx],
        model.decoder_conv_tr_bias[decoder_idx], 8, 1, 4, 1, 0, 0, 1, 1);

    int y_dim1_begin = 2;
    int y_dim1_end = y.dimension(1) - 4;

    // remove 2 elements from begin and end of y along dimension 1 (0, 1, 2)
    Eigen::Tensor3dXf y_cropped_2 =
        y.slice(Eigen::array<Eigen::Index, 3>({0, y_dim1_begin, 0}),
                Eigen::array<Eigen::Index, 3>(
                    {y.dimension(0), y_dim1_end, y.dimension(2)}));

    if (decoder_idx < 3)
    {
        x_out = demucscpp::gelu(y_cropped_2);
    }
    else
    {
        std::cout << "last decoder, no gelu" << std::endl;
        // last, no gelu
        x_out = y_cropped_2;
    }
}

// forward declaration to apply a time decoder
void demucscpp::apply_time_decoder(struct demucscpp::demucs_model_4s &model,
                                   int tdecoder_idx,
                                   const Eigen::Tensor3dXf &xt_in,
                                   Eigen::Tensor3dXf &xt_out,
                                   const Eigen::Tensor3dXf &skip)
{
    int crop = 1344;
    int out_length = 5375;
    // switch case for tdecoder_idx
    switch (tdecoder_idx)
    {
    case 0:
        break;
    case 1:
        crop = 5375;
        out_length = 21499;
        break;
    case 2:
        crop = 21499;
        out_length = 85995;
        break;
    case 3:
        crop = 85995;
        out_length = 343980;
        break;
    default:
        std::cout << "invalid tdecoder_idx" << std::endl;
        break;
    }

    // need rewrite, norm2, glu
    Eigen::Tensor3dXf yt = demucscpp::conv1d<3, 1, 1, 1>(
        xt_in + skip, model.tdecoder_rewrite_weight[tdecoder_idx],
        model.tdecoder_rewrite_bias[tdecoder_idx]);

    yt = demucscpp::glu(yt, 1);

    // start the DConv
    demucscpp::apply_dconv(model, yt, 1, 1, 4-tdecoder_idx-1, crop);

    // dconv finished

    // next, apply the final transpose convolution
    Eigen::Tensor3dXf yt_tmp = demucscpp::conv1d_tr(
        yt, model.tdecoder_conv_tr_weight[tdecoder_idx],
        model.tdecoder_conv_tr_bias[tdecoder_idx], 8, 4, 0, 1);

    yt = yt_tmp;

    // remove padding
    // 2:2+length
    Eigen::Tensor3dXf yt_crop =
        yt.slice(Eigen::array<Eigen::Index, 3>({0, 0, 2}),
                 Eigen::array<Eigen::Index, 3>(
                     {yt.dimension(0), yt.dimension(1), out_length}));

    // gelu activation if not last
    if (tdecoder_idx < 3)
    {
        xt_out = demucscpp::gelu(yt_crop);
    }
    else
    {
        xt_out = yt_crop;
    }
}
