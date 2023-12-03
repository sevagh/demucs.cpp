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
HT_HUB_PATH = "955717e8-8726e21a.th"
HT_HUB_PATH_6S = "5c90dfd2-34c22ccb.th"

LAYERS_TO_SKIP = [
    #"stft.window",
    ##"input_mean",
    ##"input_scale",
    ##"output_scale",
    ##"output_mean",
    #"sample_rate",
    #"transform.0.window",
    ##"bn1.running_mean",
    ##"bn1.running_var",
    #"bn1.num_batches_tracked",
    ##"bn2.running_mean",
    ##"bn2.running_var",
    #"bn2.num_batches_tracked",
    ##"bn3.running_mean",
    ##"bn3.running_var",
    #"bn3.num_batches_tracked",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to GGML')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")
    parser.add_argument("--six-source", default=False, action="store_true", help="convert 6s model (default: 4s)")

    args = parser.parse_args()

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load model from torchhub, implicitly expecting them to land in the torchhub cache path
    #    root path for torchhub: /home/sevagh/.cache/torch/hub/checkpoints/
    #    hq paths: [vocals-b62c91ce.pth, drums-9619578f.pth, bass-8d85a5bd.pth, other-b52fbbf7.pth]
    #    xl paths: [vocals-bccbd9aa.pth, drums-69e0ebd4.pth, bass-2ca1ce51.pth, other-c8c5b3e6.pth]
    # using the spectrogram model since we don't need the surrounding stft/isft + wiener

    # use the demucsht v4 hybrid transformer model
    model = get_model(DEMUCS_MODEL) if not args.six_source else get_model(DEMUCS_MODEL_6S)
    print(model)

    # get torchub path
    torchhub_path = Path(torch.hub.get_dir()) / "checkpoints"

    suffix = "-6s" if args.six_source else "-4s"
    dest_name = dir_out / f"ggml-model-{DEMUCS_MODEL}{suffix}-f16.bin"

    fname_inp = torchhub_path / HT_HUB_PATH if not args.six_source else torchhub_path / HT_HUB_PATH_6S

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
    magic = 0x646d6334 if not args.six_source else 0x646d6336

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
