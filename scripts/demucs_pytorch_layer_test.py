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
from demucs.utils import debug_tensor_demucscpp


if __name__ == '__main__':
    #input_file = "./test/data/gspi_stereo_short.wav"

    ## load audio file and resample to 44100 Hz
    #metadata = torchaudio.info(input_file)
    #print(metadata)
    #audio, rate = torchaudio.load(input_file)
    #print(rate)

    # demucs v4 hybrid transformer
    model = get_model('htdemucs')
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

        x_enc_0 = henclayer_0(x)

        debug_tensor_demucscpp(x, "x")
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

    if test_name == "all" or test_name == "freq-dec":
        hdeclayer_0 = model.models[0].decoder[0]

        x_dec_0 = torch.ones((1, 384, 8, 336))
        x_dec_0[..., ::2] = -1

        skip_dec_0 = torch.ones((1, 384, 8, 336))*0.5
        skip_dec_0[..., 1::2] = -0.5

        skip_dec_1 = torch.ones((1, 192, 32, 336))*0.5
        skip_dec_1[..., 1::2] = -0.5

        skip_dec_2 = torch.ones((1, 96, 128, 336))*0.5
        skip_dec_2[..., 1::2] = -0.5

        skip_dec_3 = torch.ones((1, 48, 512, 336))*0.5
        skip_dec_3[..., 1::2] = -0.5

        x_dec_1, _ = hdeclayer_0(x_dec_0, skip=skip_dec_0, length=x_dec_0.shape[-1])

        debug_tensor_demucscpp(x_dec_0, "x_dec_0")
        debug_tensor_demucscpp(x_dec_1, "x_dec_1")

        hdeclayer_1 = model.models[0].decoder[1]
        x_dec_2, _ = hdeclayer_1(x_dec_1, skip=skip_dec_1, length=x_dec_1.shape[-1])

        debug_tensor_demucscpp(x_dec_2, "x_dec_2")

        hdeclayer_2 = model.models[0].decoder[2]
        x_dec_3, _ = hdeclayer_2(x_dec_2, skip=skip_dec_2, length=x_dec_2.shape[-1])

        debug_tensor_demucscpp(x_dec_3, "x_dec_3")

        hdeclayer_3 = model.models[0].decoder[3]
        x_dec_4, _ = hdeclayer_3(x_dec_3, skip=skip_dec_3, length=x_dec_3.shape[-1])

        debug_tensor_demucscpp(x_dec_4, "x_dec_4")

    if test_name == "all" or test_name == "time-enc":
        # create fake xt tensor of shape (1, 2, 343980)
        xt = torch.ones((1, 2, 343980))
        xt[..., ::2] = -1

        htenclayer_0 = model.models[0].tencoder[0]
        xt_enc_0 = htenclayer_0(xt)

        debug_tensor_demucscpp(xt, "xt")
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

    if test_name == "all" or test_name == "time-dec":
        htdeclayer_0 = model.models[0].tdecoder[0]

        xt_dec_0 = torch.ones((1, 384, 1344))
        xt_dec_0[..., ::2] = -1

        skip_tdec_0 = torch.ones((1, 384, 1344))*0.5
        skip_tdec_0[..., 1::2] = -0.5

        skip_tdec_1 = torch.ones((1, 192, 5375))*0.5
        skip_tdec_1[..., 1::2] = -0.5

        skip_tdec_2 = torch.ones((1, 96, 21499))*0.5
        skip_tdec_2[..., 1::2] = -0.5

        skip_tdec_3 = torch.ones((1, 48, 85995))*0.5
        skip_tdec_3[..., 1::2] = -0.5

        xt_dec_1, _ = htdeclayer_0(xt_dec_0, skip=skip_tdec_0, length=skip_tdec_1.shape[-1])

        debug_tensor_demucscpp(xt_dec_0, "xt_dec_0")
        debug_tensor_demucscpp(xt_dec_1, "xt_dec_1")

        htdeclayer_1 = model.models[0].tdecoder[1]
        xt_dec_2, _ = htdeclayer_1(xt_dec_1, skip=skip_tdec_1, length=skip_tdec_2.shape[-1])

        debug_tensor_demucscpp(xt_dec_2, "xt_dec_2")

        htdeclayer_2 = model.models[0].tdecoder[2]
        xt_dec_3, _ = htdeclayer_2(xt_dec_2, skip=skip_tdec_2, length=skip_tdec_3.shape[-1])

        debug_tensor_demucscpp(xt_dec_3, "xt_dec_3")

        htdeclayer_3 = model.models[0].tdecoder[3]
        xt_dec_4, _ = htdeclayer_3(xt_dec_3, skip=skip_tdec_3, length=343980)

        debug_tensor_demucscpp(xt_dec_4, "xt_dec_4")

    if test_name == "all" or test_name == "full-crosstransformer":
        x = torch.ones((1, 384, 8, 336))
        x[..., ::2] = -1

        xt = torch.ones((1, 384, 1344))
        xt[..., ::2] = -1

        debug_tensor_demucscpp(x, "x")
        debug_tensor_demucscpp(xt, "xt")

        b, c, f, t = x.shape
        x = rearrange(x, "b c f t-> b c (f t)")
        x_upsampled = model.models[0].channel_upsampler(x)
        x_upsampled = rearrange(x_upsampled, "b c (f t)-> b c f t", f=f)

        xt_upsampled = model.models[0].channel_upsampler_t(xt)

        debug_tensor_demucscpp(x_upsampled, "x pre-crosstransformer")
        debug_tensor_demucscpp(xt_upsampled, "xt pre-crosstransformer")

        x_crosstrans, xt_crosstrans = model.models[0].crosstransformer(x_upsampled, xt_upsampled)
        debug_tensor_demucscpp(x_crosstrans, "x post-crosstransformer")
        debug_tensor_demucscpp(xt_crosstrans, "xt post-crosstransformer")

        x_crosstrans = rearrange(x_crosstrans, "b c f t-> b c (f t)")
        x_downsampled = model.models[0].channel_downsampler(x_crosstrans)
        x_downsampled = rearrange(x_downsampled, "b c (f t)-> b c f t", f=f)

        xt_downsampled = model.models[0].channel_downsampler_t(xt_crosstrans)

        debug_tensor_demucscpp(x_downsampled, "x post-crosstransformer")
        debug_tensor_demucscpp(xt_downsampled, "xt post-crosstransformer")

    if test_name == "all" or test_name == "crosstransformer":
        x_2 = torch.ones((1, 512, 8, 336))
        x_2[..., ::2] = -1

        xt_2 = torch.ones((1, 512, 1344))
        xt_2[..., ::2] = -1

        debug_tensor_demucscpp(x_2, "x pre-crosstransformer")
        debug_tensor_demucscpp(xt_2, "xt pre-crosstransformer")

        x_crosstrans, xt_crosstrans = model.models[0].crosstransformer(x_2, xt_2)

        debug_tensor_demucscpp(x_crosstrans, "x post-crosstransformer")
        debug_tensor_demucscpp(xt_crosstrans, "xt post-crosstransformer")

    if test_name == "all" or test_name == "upsamplers":
        x_3 = torch.ones((1, 384, 8, 336))
        x_3[..., ::2] = -1

        xt_3 = torch.ones((1, 384, 1344))
        xt_3[..., ::2] = -1

        debug_tensor_demucscpp(x_3, "x_3")
        debug_tensor_demucscpp(xt_3, "xt_3")

        b, c, f, t = x_3.shape
        x_3 = rearrange(x_3, "b c f t-> b c (f t)")
        x_3_upsampled = model.models[0].channel_upsampler(x_3)
        x_3_upsampled = rearrange(x_3_upsampled, "b c (f t)-> b c f t", f=f)

        xt_3_upsampled = model.models[0].channel_upsampler_t(xt_3)

        debug_tensor_demucscpp(x_3_upsampled, "x channel upsampled")
        debug_tensor_demucscpp(xt_3_upsampled, "xt channel upsampled")

        x_3_upsampled = rearrange(x_3_upsampled, "b c f t-> b c (f t)")
        x_3_downsampled = model.models[0].channel_downsampler(x_3_upsampled)
        x_3_downsampled = rearrange(x_3_downsampled, "b c (f t)-> b c f t", f=f)

        xt_3_downsampled = model.models[0].channel_downsampler_t(xt_3_upsampled)

        debug_tensor_demucscpp(x_3_downsampled, "x channel downsampled")
        debug_tensor_demucscpp(xt_3_downsampled, "xt channel downsampled")

    if test_name == "all" or test_name == "ct-layer":
        x_2 = torch.ones((1, 2688, 512))
        x_2[..., ::2] = -1

        xt_2 = torch.ones((1, 1344, 512))
        xt_2[..., ::2] = -1

        debug_tensor_demucscpp(x_2, "x pre-crosstransformer")
        debug_tensor_demucscpp(xt_2, "xt pre-crosstransformer")

        x_layer_0 = model.models[0].crosstransformer.layers[0](x_2)
        xt_layer_0 = model.models[0].crosstransformer.layers_t[0](xt_2)

        debug_tensor_demucscpp(x_layer_0, "x crosstran-layer-0")
        debug_tensor_demucscpp(xt_layer_0, "xt crosstran-tlayer-0")

        debug_tensor_demucscpp(model.models[0].crosstransformer.norm_in.weight, "x norm-in weight")
        debug_tensor_demucscpp(model.models[0].crosstransformer.norm_in.bias, "x norm-in bias")
        debug_tensor_demucscpp(model.models[0].crosstransformer.norm_in_t.weight, "xt norm-in-t weight")
        debug_tensor_demucscpp(model.models[0].crosstransformer.norm_in_t.bias, "xt norm-in-t bias")

        x_norm_in = model.models[0].crosstransformer.norm_in(x_2)
        xt_norm_in = model.models[0].crosstransformer.norm_in_t(xt_2)
        x_norm_in_t = model.models[0].crosstransformer.norm_in(xt_2)
        xt_norm_in_f = model.models[0].crosstransformer.norm_in_t(x_2)

        #x_norm_in = torch.nn.functional.layer_norm(
        #    x_2,
        #    (x_2.shape[-1],),
        #    weight=model.models[0].crosstransformer.norm_in.weight,
        #    bias=None,
        #    eps=1e-5
        #)
        #xt_norm_in = torch.nn.functional.layer_norm(
        #    xt_2,
        #    (xt_2.shape[-1],),
        #    weight=model.models[0].crosstransformer.norm_in_t.weight,
        #    bias=None,
        #    eps=1e-5
        #)
        #x_norm_in_t = torch.nn.functional.layer_norm(
        #    xt_2,
        #    (xt_2.shape[-1],),
        #    weight=model.models[0].crosstransformer.norm_in.weight,
        #    bias=None,
        #    eps=1e-5
        #)
        #xt_norm_in_f = torch.nn.functional.layer_norm(
        #    x_2,
        #    (x_2.shape[-1],),
        #    weight=model.models[0].crosstransformer.norm_in_t.weight,
        #    bias=None,
        #    eps=1e-5
        #)

        debug_tensor_demucscpp(x_norm_in, "x norm-in")
        debug_tensor_demucscpp(xt_norm_in, "xt norm-in-t")
        debug_tensor_demucscpp(x_norm_in_t, "x norm-in_t")
        debug_tensor_demucscpp(xt_norm_in_f, "xt norm-in-t_f")

    if test_name == "all" or test_name == "layer-norm-basic":
        x = torch.ones((1, 2, 3))
        w = torch.ones((3))
        b = torch.ones((3))

        x[0, 0, 0] = 1.0
        x[0, 0, 1] = 2.0
        x[0, 0, 2] = 3.0
        x[0, 1, 0] = 4.0
        x[0, 1, 1] = 5.0
        x[0, 1, 2] = 6.0

        w[0] = 0.75
        w[1] = -0.5
        w[2] = -1.35

        b[0] = 0.5
        b[1] = -0.25
        b[2] = 0.75

        debug_tensor_demucscpp(x, "x")
        debug_tensor_demucscpp(w, "w")
        debug_tensor_demucscpp(b, "b")

        x_out = torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            weight=w,
            bias=b,
            eps=1e-5
        )

        debug_tensor_demucscpp(x_out, "x_out")

    if test_name == "all" or test_name == "layer-norm-bigger":
        x = torch.ones((1, 2688, 512))
        w = torch.ones((512))
        b = torch.ones((512))

        x[..., ::2] = -1

        for i in range(512):
            if i % 2 == 0:
                w[i] = -0.25 + i*0.03
                b[i] = 0.5
            else:
                w[i] = 0.25
                b[i] = -0.5 + i*0.57

        debug_tensor_demucscpp(x, "x")
        debug_tensor_demucscpp(w, "w")
        debug_tensor_demucscpp(b, "b")

        x_out = torch.nn.functional.layer_norm(
            x,
            (x.shape[-1],),
            weight=w,
            bias=b,
            eps=1e-5
        )

        debug_tensor_demucscpp(x_out, "x_out")
