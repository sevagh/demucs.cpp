#!/usr/bin/env python

import io
import sys
import torch
import numpy as np
from demucs.pretrained import get_model
import struct
import argparse
from pathlib import Path


DEMUCS_MODEL = "htdemucs"
DEMUCS_MODEL_6S = "htdemucs_6s"
DEMUCS_MODEL_FT = "htdemucs_ft"
DEMUCS_MODEL_FT_DRUMS = "htdemucs_ft_drums"
DEMUCS_MODEL_FT_BASS = "htdemucs_ft_bass"
DEMUCS_MODEL_FT_OTHER = "htdemucs_ft_other"
DEMUCS_MODEL_FT_VOCALS = "htdemucs_ft_vocals"

HT_HUB_PATH = "955717e8-8726e21a.th"
HT_HUB_PATH_6S = "5c90dfd2-34c22ccb.th"
HT_HUB_PATH_FT_DRUMS = "f7e0c4bc-ba3fe64a.th"
HT_HUB_PATH_FT_BASS = "d12395a8-e57c48e6.th"
HT_HUB_PATH_FT_OTHER = "92cfc3b6-ef3bcb9c.th"
HT_HUB_PATH_FT_VOCALS = "04573f0d-f3cf25b2.th"

LAYERS_TO_SKIP = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to GGML')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")
    parser.add_argument("--six-source", default=False, action="store_true", help="convert 6s model (default: 4s)")
    parser.add_argument("--ft-drums", default=False, action="store_true", help="convert fine-tuned drum model")
    parser.add_argument("--ft-bass", default=False, action="store_true", help="convert fine-tuned bass model")
    parser.add_argument("--ft-other", default=False, action="store_true", help="convert fine-tuned other model")
    parser.add_argument("--ft-vocals", default=False, action="store_true", help="convert fine-tuned vocals model")

    args = parser.parse_args()

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # use the demucsht v4 hybrid transformer model
    model = get_model(DEMUCS_MODEL)
    model_name = DEMUCS_MODEL
    if args.six_source:
        model = get_model(DEMUCS_MODEL_6S)
        model_name = DEMUCS_MODEL
    elif (args.ft_drums or args.ft_bass or args.ft_other or args.ft_vocals):
        model = get_model(DEMUCS_MODEL_FT)
        model_name = DEMUCS_MODEL_FT_DRUMS
        if args.ft_bass:
            model_name = DEMUCS_MODEL_FT_BASS
        elif args.ft_other:
            model_name = DEMUCS_MODEL_FT_OTHER
        elif args.ft_vocals:
            model_name = DEMUCS_MODEL_FT_VOCALS

    print(model)

    # get torchub path
    torchhub_path = Path(torch.hub.get_dir()) / "checkpoints"

    suffix = "-6s" if args.six_source else "-4s"
    dest_name = dir_out / f"ggml-model-{model_name}{suffix}-f16.bin"

    fname_inp = torchhub_path / HT_HUB_PATH
    if args.six_source:
        fname_inp = torchhub_path / HT_HUB_PATH_6S
    elif args.ft_drums:
        fname_inp = torchhub_path / HT_HUB_PATH_FT_DRUMS
    elif args.ft_bass:
        fname_inp = torchhub_path / HT_HUB_PATH_FT_BASS
    elif args.ft_other:
        fname_inp = torchhub_path / HT_HUB_PATH_FT_OTHER
    elif args.ft_vocals:
        fname_inp = torchhub_path / HT_HUB_PATH_FT_VOCALS

    # try to load PyTorch binary data
    # even though we loaded it above to print its info
    # we need to load it again ggml/whisper.cpp-style
    try:
        model_bytes = open(fname_inp, "rb").read()
        with io.BytesIO(model_bytes) as fp:
            checkpoint = torch.load(fp, map_location="cpu")
    except Exception:
        print("Error: failed to load PyTorch model file:" , fname_inp)
        sys.exit(1)

    checkpoint = checkpoint["state"]

    print(checkpoint.keys())

    # copied from ggerganov/whisper.cpp convert-pt-to-ggml.py
    fout = dest_name.open("wb")

    # dmc4 or dmc6 in hex
    magic = 0x646d6334
    if args.six_source:
        magic = 0x646d6336

    # fine-tuned has same magic

    fout.write(struct.pack("i", magic))

    # write layers
    for name in checkpoint.keys():
        if name in LAYERS_TO_SKIP:
            print(f"Skipping layer {name}")
            continue
        data = checkpoint[name].squeeze().numpy()
        print("Processing variable: " , name ,  " with shape: ", data.shape, " , dtype: ", data.dtype)

        n_dims = len(data.shape)

        # header
        str_ = name.encode('utf-8')
        fout.write(struct.pack("ii", n_dims, len(str_)))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[i]))
        fout.write(str_)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: " , dest_name)
    print("")
