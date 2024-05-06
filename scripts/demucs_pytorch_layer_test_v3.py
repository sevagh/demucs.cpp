#!/usr/bin/env python
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.pretrained import SOURCES
import torch
import torchaudio.backend.sox_io_backend
import torchaudio
import argparse
import numpy as np
import os
from  einops import rearrange
import sys


def debug_tensor_demucscpp(x, name):
    #check if x is of type TensorChunk
    if hasattr(x, 'tensor'):
        # split into subchunk from self.offset:self.offset+self.length
        x = x.tensor[..., x.offset:x.offset+x.length]

    print(f"Debugging tensor!: {name}")
    print(f"\tshape: {tuple(x.shape)}")
    x_min, x_min_idx = torch.min(x.reshape(-1), dim=0)
    x_max, x_max_idx = torch.max(x.reshape(-1), dim=0)
    x_mean = torch.mean(x)
    x_stddev = torch.std(x)
    x_sum = torch.sum(x)
    print(f"\tmin: {x_min.item()}")
    print(f"\tmax: {x_max.item()}")
    print(f"\tmean: {x_mean.item()}")
    print(f"\tstddev: {x_stddev.item()}")
    print(f"\tsum: {x_sum.item()}")
    print(f"\tmin idx: {tuple(np.unravel_index(x_min_idx.item(), x.shape))}")
    print(f"\tmax idx: {tuple(np.unravel_index(x_max_idx.item(), x.shape))}")
    print(f"FINISHED DEBUG FOR TENSOR {name}")


if __name__ == '__main__':
    #input_file = "./test/data/gspi_stereo_short.wav"

    ## load audio file and resample to 44100 Hz
    #metadata = torchaudio.info(input_file)
    #print(metadata)
    #audio, rate = torchaudio.load(input_file)
    #print(rate)

    # demucs v3 hybrid
    model = get_model('hdemucs_mmi')
    print(model)

    try:
        test_name = sys.argv[1]
    except IndexError:
        test_name = "all"

    if test_name == "all" or test_name == "freq-enc":
        # get the henclayer
        henclayer_0 = model.models[0].encoder[0]

        # create a fake tensor of shape (1, 4, 2048, 336)
        x = torch.ones((1, 4, 2048, 336))

        # set alternating odd index values to -1
        x[..., ::2] = -1

        debug_tensor_demucscpp(x, "x")

        x_enc_0 = henclayer_0(x)

        debug_tensor_demucscpp(x_enc_0, "x_enc_0")

        # continue for the rest of the encoder layers
        # generate tensors for each layer
        # shapes are:
        #    (96, 128, 336) -> (192, 32, 336) -> (384, 8, 336)
        # continue with x_enc_1,2,3

        henclayer_1 = model.models[0].encoder[1]
        x_enc_1 = henclayer_1(x_enc_0)

        debug_tensor_demucscpp(x_enc_1, "x_enc_1")

        henclayer_2 = model.models[0].encoder[2]
        x_enc_2 = henclayer_2(x_enc_1)

        debug_tensor_demucscpp(x_enc_2, "x_enc_2")

        henclayer_3 = model.models[0].encoder[3]
        x_enc_3 = henclayer_3(x_enc_2)

        debug_tensor_demucscpp(x_enc_3, "x_enc_3")

    if test_name == "all" or test_name == "time-enc":
        # create fake xt tensor of shape (1, 2, 343980)
        xt = torch.ones((1, 2, 343980))
        xt[..., ::2] = -1

        htenclayer_0 = model.models[0].tencoder[0]

        debug_tensor_demucscpp(xt, "xt")

        xt_enc_0 = htenclayer_0(xt)

        debug_tensor_demucscpp(xt_enc_0, "xt_enc_0")

        htenclayer_1 = model.models[0].tencoder[1]
        xt_enc_1 = htenclayer_1(xt_enc_0)

        debug_tensor_demucscpp(xt_enc_1, "xt_enc_1")

        htenclayer_2 = model.models[0].tencoder[2]
        xt_enc_2 = htenclayer_2(xt_enc_1)

        debug_tensor_demucscpp(xt_enc_2, "xt_enc_2")

        htenclayer_3 = model.models[0].tencoder[3]
        xt_enc_3 = htenclayer_3(xt_enc_2)

        debug_tensor_demucscpp(xt_enc_3, "xt_enc_3")

    if test_name == "all" or test_name == "encoder45":
        # get the henclayer
        henclayer_0 = model.models[0].encoder[0]

        # create a fake tensor of shape (1, 4, 2048, 336)
        x = torch.ones((1, 4, 2048, 336))

        # set alternating odd index values to -1
        x[..., ::2] = -1

        debug_tensor_demucscpp(x, "x")

        x_enc_0 = henclayer_0(x)

        debug_tensor_demucscpp(x_enc_0, "x_enc_0")

        # continue for the rest of the encoder layers
        # generate tensors for each layer
        # shapes are:
        #    (96, 128, 336) -> (192, 32, 336) -> (384, 8, 336)
        # continue with x_enc_1,2,3

        henclayer_1 = model.models[0].encoder[1]
        x_enc_1 = henclayer_1(x_enc_0)

        debug_tensor_demucscpp(x_enc_1, "x_enc_1")

        henclayer_2 = model.models[0].encoder[2]
        x_enc_2 = henclayer_2(x_enc_1)

        debug_tensor_demucscpp(x_enc_2, "x_enc_2")

        henclayer_3 = model.models[0].encoder[3]
        x_enc_3 = henclayer_3(x_enc_2)

        debug_tensor_demucscpp(x_enc_3, "x_enc_3")

        # create fake xt tensor of shape (1, 2, 343980)
        xt = torch.ones((1, 2, 343980))
        xt[..., ::2] = -1

        htenclayer_0 = model.models[0].tencoder[0]

        debug_tensor_demucscpp(xt, "xt")

        xt_enc_0 = htenclayer_0(xt)

        debug_tensor_demucscpp(xt_enc_0, "xt_enc_0")

        htenclayer_1 = model.models[0].tencoder[1]
        xt_enc_1 = htenclayer_1(xt_enc_0)

        debug_tensor_demucscpp(xt_enc_1, "xt_enc_1")

        htenclayer_2 = model.models[0].tencoder[2]
        xt_enc_2 = htenclayer_2(xt_enc_1)

        debug_tensor_demucscpp(xt_enc_2, "xt_enc_2")

        htenclayer_3 = model.models[0].tencoder[3]
        xt_enc_3 = htenclayer_3(xt_enc_2)

        debug_tensor_demucscpp(xt_enc_3, "xt_enc_3")

        htenclayer_4 = model.models[0].tencoder[4]
        xt_enc_4 = htenclayer_4(xt_enc_3)

        debug_tensor_demucscpp(xt_enc_4, "xt_enc_4")

        henclayer_4 = model.models[0].encoder[4]
        x_enc_4 = henclayer_4(x_enc_3, inject=xt_enc_4)

        debug_tensor_demucscpp(x_enc_4, "x_enc_4")

        henclayer_5 = model.models[0].encoder[5]
        x_shared_enc_5 = henclayer_5(x_enc_4)

        debug_tensor_demucscpp(x_shared_enc_5, "x_shared_enc_5")

    if test_name == "all" or test_name == "decoder01":
        x_fake_shared_enc_5 = torch.ones((1, 1536, 168))
        skip_fake_dec_4 = torch.ones((768, 1, 336))

        # set even index values to -1
        x_fake_shared_enc_5[..., ::2] = -1

        # for the skip, set even index values to 0.5, odd to -0.5
        skip_fake_dec_4[..., ::2] = 0.5
        skip_fake_dec_4[..., 1::2] = -0.5

        debug_tensor_demucscpp(x_fake_shared_enc_5, "x_fake_shared_enc_5")
        debug_tensor_demucscpp(skip_fake_dec_4, "skip_fake_dec_4")

        hdecoder_0 = model.models[0].decoder[0]
        x_empty = torch.zeros((1, 1536, 168))
        x_fake_dec_4, pre_t_unused = hdecoder_0(x_empty, x_fake_shared_enc_5, 336)

        debug_tensor_demucscpp(x_fake_dec_4, "x_fake_dec_4")
        debug_tensor_demucscpp(pre_t_unused, "pre_t_unused")

        hdecoder_1 = model.models[0].decoder[1]
        x_fake_dec_3, pre_t = hdecoder_1(x_fake_dec_4, skip_fake_dec_4, 336)

        debug_tensor_demucscpp(x_fake_dec_3, "x_fake_dec_3")
        debug_tensor_demucscpp(pre_t, "pre_t")

        tdecoder_0 = model.models[0].tdecoder[0]
        pre_t = pre_t[:, :, 0]
        debug_tensor_demucscpp(pre_t, "pre_t")
        xt_fake_dec_3, _ = tdecoder_0(pre_t, None, 1344)

        debug_tensor_demucscpp(xt_fake_dec_3, "xt_fake_dec_3")

    if test_name == "all" or test_name == "decoder1isolated":
        x_fake_dec_4 = torch.ones((1, 768, 336))
        skip_fake_dec_4 = torch.ones((768, 1, 336))

        # set even index values to -1
        x_fake_dec_4[..., ::2] = -1

        # for the skip, set even index values to 0.5, odd to -0.5
        skip_fake_dec_4[..., ::2] = 0.5
        skip_fake_dec_4[..., 1::2] = -0.5

        hdecoder_1 = model.models[0].decoder[1]
        x_fake_dec_3, pre_t = hdecoder_1(x_fake_dec_4, skip_fake_dec_4, 336)

        debug_tensor_demucscpp(x_fake_dec_3, "x_fake_dec_3")
        debug_tensor_demucscpp(pre_t, "pre_t")

        tdecoder_0 = model.models[0].tdecoder[0]
        pre_t = pre_t[:, :, 0]
        debug_tensor_demucscpp(pre_t, "pre_t")
        xt_fake_dec_3, _ = tdecoder_0(pre_t, None, 1344)

        debug_tensor_demucscpp(xt_fake_dec_3, "xt_fake_dec_3")

    if test_name == "all" or test_name == "alldecoders":
        x_fake_shared_enc_5 = torch.ones((1, 1536, 168))
        skip_fake_dec_4 = torch.ones((768, 1, 336))

        # set even index values to -1
        x_fake_shared_enc_5[..., ::2] = -1

        # for the skip, set even index values to 0.5, odd to -0.5
        skip_fake_dec_4[..., ::2] = 0.5
        skip_fake_dec_4[..., 1::2] = -0.5

        debug_tensor_demucscpp(x_fake_shared_enc_5, "x_fake_shared_enc_5")
        debug_tensor_demucscpp(skip_fake_dec_4, "skip_fake_dec_4")

        x_fake_dec_4 = torch.ones((1, 768, 336))

        # set even index values to -1
        x_fake_dec_4[..., ::2] = -1

        skip_fake_dec_3 = torch.ones((384, 8, 336))

        # 0.5, -0.5 again
        skip_fake_dec_3[..., ::2] = 0.5
        skip_fake_dec_3[..., 1::2] = -0.5

        skip_fake_dec_2 = torch.ones((192, 32, 336))

        # 0.5, -0.5 again
        skip_fake_dec_2[..., ::2] = 0.5
        skip_fake_dec_2[..., 1::2] = -0.5

        skip_fake_dec_1 = torch.ones((96, 128, 336))

        # 0.5, -0.5 again
        skip_fake_dec_1[..., ::2] = 0.5
        skip_fake_dec_1[..., 1::2] = -0.5

        skip_fake_dec_0 = torch.ones((48, 512, 336))

        # 0.5, -0.5 again
        skip_fake_dec_0[..., ::2] = 0.5
        skip_fake_dec_0[..., 1::2] = -0.5

        skip_fake_tdec_3 = torch.ones((1, 384, 1344))

        # 0.5, -0.5 again
        skip_fake_tdec_3[..., ::2] = 0.5
        skip_fake_tdec_3[..., 1::2] = -0.5

        skip_fake_tdec_2 = torch.ones((1, 192, 5375))

        # 0.5, -0.5 again
        skip_fake_tdec_2[..., ::2] = 0.5
        skip_fake_tdec_2[..., 1::2] = -0.5

        skip_fake_tdec_1 = torch.ones((1, 96, 21499))

        # 0.5, -0.5 again
        skip_fake_tdec_1[..., ::2] = 0.5
        skip_fake_tdec_1[..., 1::2] = -0.5

        skip_fake_tdec_0 = torch.ones((1, 48, 85995))

        # 0.5, -0.5 again
        skip_fake_tdec_0[..., ::2] = 0.5
        skip_fake_tdec_0[..., 1::2] = -0.5

        hdecoder_0 = model.models[0].decoder[0]
        x_empty = torch.zeros((1, 1536, 168))
        x_fake_dec_4, pre_t_unused = hdecoder_0(x_empty, x_fake_shared_enc_5, 336)

        debug_tensor_demucscpp(x_fake_dec_4, "x_fake_dec_4")
        debug_tensor_demucscpp(pre_t_unused, "pre_t_unused")

        hdecoder_1 = model.models[0].decoder[1]
        x_fake_dec_3, pre_t = hdecoder_1(x_fake_dec_4, skip_fake_dec_4, 336)

        debug_tensor_demucscpp(x_fake_dec_3, "x_fake_dec_3")
        debug_tensor_demucscpp(pre_t, "pre_t")

        hdecoder_1 = model.models[0].decoder[1]
        x_fake_dec_3, pre_t = hdecoder_1(x_fake_dec_4, skip_fake_dec_4, 336)

        debug_tensor_demucscpp(x_fake_dec_3, "x_fake_dec_3")
        debug_tensor_demucscpp(pre_t, "pre_t")

        tdecoder_0 = model.models[0].tdecoder[0]
        pre_t = pre_t[:, :, 0]
        debug_tensor_demucscpp(pre_t, "pre_t")
        xt_fake_dec_3, _ = tdecoder_0(pre_t, None, 1344)

        debug_tensor_demucscpp(xt_fake_dec_3, "xt_fake_dec_3")

        hdecoder_2 = model.models[0].decoder[2]
        x_fake_dec_2, _ = hdecoder_2(x_fake_dec_3, skip_fake_dec_3, 1344)

        tdecoder_1 = model.models[0].tdecoder[1]
        xt_fake_dec_2, _ = tdecoder_1(xt_fake_dec_3, skip_fake_tdec_3, 5375)

        debug_tensor_demucscpp(x_fake_dec_2, "x_fake_dec_2")
        debug_tensor_demucscpp(xt_fake_dec_2, "xt_fake_dec_2")

    if test_name == "all" or test_name == "end2end":
        # get the henclayer
        henclayer_0 = model.models[0].encoder[0]

        # create a fake tensor of shape (1, 4, 2048, 336)
        x = torch.ones((1, 4, 2048, 336))

        # set alternating odd index values to -1
        x[..., ::2] = -1

        debug_tensor_demucscpp(x, "x")

        x_enc_0 = henclayer_0(x)

        debug_tensor_demucscpp(x_enc_0, "x_enc_0")

        # continue for the rest of the encoder layers
        # generate tensors for each layer
        # shapes are:
        #    (96, 128, 336) -> (192, 32, 336) -> (384, 8, 336)
        # continue with x_enc_1,2,3

        henclayer_1 = model.models[0].encoder[1]
        x_enc_1 = henclayer_1(x_enc_0)

        debug_tensor_demucscpp(x_enc_1, "x_enc_1")

        henclayer_2 = model.models[0].encoder[2]
        x_enc_2 = henclayer_2(x_enc_1)

        debug_tensor_demucscpp(x_enc_2, "x_enc_2")

        henclayer_3 = model.models[0].encoder[3]
        x_enc_3 = henclayer_3(x_enc_2)

        debug_tensor_demucscpp(x_enc_3, "x_enc_3")

        # create fake xt tensor of shape (1, 2, 343980)
        xt = torch.ones((1, 2, 343980))
        xt[..., ::2] = -1

        htenclayer_0 = model.models[0].tencoder[0]

        debug_tensor_demucscpp(xt, "xt")

        xt_enc_0 = htenclayer_0(xt)

        debug_tensor_demucscpp(xt_enc_0, "xt_enc_0")

        htenclayer_1 = model.models[0].tencoder[1]
        xt_enc_1 = htenclayer_1(xt_enc_0)

        debug_tensor_demucscpp(xt_enc_1, "xt_enc_1")

        htenclayer_2 = model.models[0].tencoder[2]
        xt_enc_2 = htenclayer_2(xt_enc_1)

        debug_tensor_demucscpp(xt_enc_2, "xt_enc_2")

        htenclayer_3 = model.models[0].tencoder[3]
        xt_enc_3 = htenclayer_3(xt_enc_2)

        debug_tensor_demucscpp(xt_enc_3, "xt_enc_3")

        htenclayer_4 = model.models[0].tencoder[4]
        xt_enc_4 = htenclayer_4(xt_enc_3)

        debug_tensor_demucscpp(xt_enc_4, "xt_enc_4")

        henclayer_4 = model.models[0].encoder[4]
        x_enc_4 = henclayer_4(x_enc_3, inject=xt_enc_4)

        debug_tensor_demucscpp(x_enc_4, "x_enc_4")

        henclayer_5 = model.models[0].encoder[5]
        x_shared_enc_5 = henclayer_5(x_enc_4, inject=None)

        debug_tensor_demucscpp(x_shared_enc_5, "x_shared_enc_5")

        skip_fake_dec_4 = torch.ones((768, 1, 336))

        # for the skip, set even index values to 0.5, odd to -0.5
        skip_fake_dec_4[..., ::2] = 0.5
        skip_fake_dec_4[..., 1::2] = -0.5

        debug_tensor_demucscpp(skip_fake_dec_4, "skip_fake_dec_4")

        x_fake_dec_4 = torch.ones((1, 768, 336))

        # set even index values to -1
        x_fake_dec_4[..., ::2] = -1

        skip_fake_dec_3 = torch.ones((384, 8, 336))

        # 0.5, -0.5 again
        skip_fake_dec_3[..., ::2] = 0.5
        skip_fake_dec_3[..., 1::2] = -0.5

        skip_fake_dec_2 = torch.ones((192, 32, 336))

        # 0.5, -0.5 again
        skip_fake_dec_2[..., ::2] = 0.5
        skip_fake_dec_2[..., 1::2] = -0.5

        skip_fake_dec_1 = torch.ones((96, 128, 336))

        # 0.5, -0.5 again
        skip_fake_dec_1[..., ::2] = 0.5
        skip_fake_dec_1[..., 1::2] = -0.5

        skip_fake_dec_0 = torch.ones((48, 512, 336))

        # 0.5, -0.5 again
        skip_fake_dec_0[..., ::2] = 0.5
        skip_fake_dec_0[..., 1::2] = -0.5

        skip_fake_tdec_3 = torch.ones((1, 384, 1344))

        # 0.5, -0.5 again
        skip_fake_tdec_3[..., ::2] = 0.5
        skip_fake_tdec_3[..., 1::2] = -0.5

        skip_fake_tdec_2 = torch.ones((1, 192, 5375))

        # 0.5, -0.5 again
        skip_fake_tdec_2[..., ::2] = 0.5
        skip_fake_tdec_2[..., 1::2] = -0.5

        skip_fake_tdec_1 = torch.ones((1, 96, 21499))

        # 0.5, -0.5 again
        skip_fake_tdec_1[..., ::2] = 0.5
        skip_fake_tdec_1[..., 1::2] = -0.5

        skip_fake_tdec_0 = torch.ones((1, 48, 85995))

        # 0.5, -0.5 again
        skip_fake_tdec_0[..., ::2] = 0.5
        skip_fake_tdec_0[..., 1::2] = -0.5

        hdecoder_0 = model.models[0].decoder[0]
        x_empty = torch.zeros((1, 1536, 168))
        x_fake_dec_4, pre_t_unused = hdecoder_0(x_empty, x_shared_enc_5, 336)

        debug_tensor_demucscpp(x_fake_dec_4, "x_fake_dec_4")
        debug_tensor_demucscpp(pre_t_unused, "pre_t_unused")

        hdecoder_1 = model.models[0].decoder[1]
        x_fake_dec_3, pre_t = hdecoder_1(x_fake_dec_4, skip_fake_dec_4, 336)

        debug_tensor_demucscpp(x_fake_dec_3, "x_fake_dec_3")
        debug_tensor_demucscpp(pre_t, "pre_t")

        tdecoder_1 = model.models[0].tdecoder[0]
        pre_t = pre_t[:, :, 0]
        debug_tensor_demucscpp(pre_t, "pre_t")
        xt_fake_dec_3, _ = tdecoder_1(pre_t, None, 1344)

        debug_tensor_demucscpp(xt_fake_dec_3, "xt_fake_dec_3")

        hdecoder_2 = model.models[0].decoder[2]
        x_fake_dec_2, _ = hdecoder_2(x_fake_dec_3, skip_fake_dec_3, 336)

        tdecoder_2 = model.models[0].tdecoder[1]
        xt_fake_dec_2, _ = tdecoder_2(xt_fake_dec_3, skip_fake_tdec_3, 5375)

        debug_tensor_demucscpp(x_fake_dec_2, "x_fake_dec_2")
        debug_tensor_demucscpp(xt_fake_dec_2, "xt_fake_dec_2")

        hdecoder_3 = model.models[0].decoder[3]
        x_fake_dec_1, _ = hdecoder_3(x_fake_dec_2, skip_fake_dec_2, 336)

        tdecoder_3 = model.models[0].tdecoder[2]
        xt_fake_dec_1, _ = tdecoder_3(xt_fake_dec_2, skip_fake_tdec_2, 21499)

        debug_tensor_demucscpp(x_fake_dec_1, "x_fake_dec_1")
        debug_tensor_demucscpp(xt_fake_dec_1, "xt_fake_dec_1")

        hdecoder_4 = model.models[0].decoder[4]
        x_fake_dec_0, _ = hdecoder_4(x_fake_dec_1, skip_fake_dec_1, 336)

        tdecoder_4 = model.models[0].tdecoder[3]
        xt_fake_dec_0, _ = tdecoder_4(xt_fake_dec_1, skip_fake_tdec_1, 85995)

        debug_tensor_demucscpp(x_fake_dec_0, "x_fake_dec_0")
        debug_tensor_demucscpp(xt_fake_dec_0, "xt_fake_dec_0")

        hdecoder_5 = model.models[0].decoder[5]
        x_out, _ = hdecoder_5(x_fake_dec_0, skip_fake_dec_0, 336)


        tdecoder_5 = model.models[0].tdecoder[4]
        xt_out, _ = tdecoder_5(xt_fake_dec_0, skip_fake_tdec_0, 343980)

        debug_tensor_demucscpp(x_out, "x_out")
        debug_tensor_demucscpp(xt_out, "xt_out")
