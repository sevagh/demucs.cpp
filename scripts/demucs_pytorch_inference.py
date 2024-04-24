#!/usr/bin/env python
from demucs.apply import apply_model
from demucs.utils import debug_tensor_demucscpp
from demucs.pretrained import get_model
from demucs.pretrained import SOURCES
import torch
import torchaudio.backend.sox_io_backend
import torchaudio
import argparse
import numpy as np
import os

# sorted order of demucs return sources
target_digit_map = {i: source for i, source in enumerate(SOURCES)}
# drums, bass, other, vocals


if __name__ == '__main__':
    # set up argparse with input wav file as positional argument
    parser = argparse.ArgumentParser(description='Demucs')
    parser.add_argument('input_file', type=str, help='path to input wav file')
    parser.add_argument('--dest-dir', type=str, default=None, help='path to write output files')
    parser.add_argument("--six-source", default=False, action="store_true", help="use 6s model (default: 4s)")
    parser.add_argument("--fine-tuned", default=False, action="store_true", help="use ft model (default: 4s)")
    parser.add_argument("--v3", default=False, action="store_true", help="use v3 (hdemucs_mmi) model (default: 4s)")

    args = parser.parse_args()

    # load audio file and resample to 44100 Hz
    metadata = torchaudio.info(args.input_file)
    print(metadata)
    audio, rate = torchaudio.load(args.input_file)
    print(rate)

    model_name = 'htdemucs'
    if args.six_source:
        model_name += '_6s'
    elif args.fine_tuned:
        model_name = 'htdemucs_ft'
    elif args.v3:
        model_name = 'hdemucs_mmi'

    # demucs v4 hybrid transformer
    model = get_model(model_name)
    nb_out_sources = 6 if args.six_source else 4
    print(model)

    debug_tensor_demucscpp(audio, "input audio")

    ref = audio.mean(0)
    audio = (audio - ref.mean()) / ref.std()

    debug_tensor_demucscpp(audio, "audio post-normalization")

    sources = apply_model(model, audio[None])[0]
    sources = sources * ref.std() + ref.mean()
    print(sources.shape)

    for target_idx in range(4):
        target_name = target_digit_map[target_idx]
        print(f"Saving target {target_name}")
        out_audio = sources[target_idx]

        debug_tensor_demucscpp(out_audio, f"target {target_name}")

        # write to file in directory
        if args.dest_dir is not None:
            os.makedirs(args.dest_dir, exist_ok=True)
            torchaudio.save(os.path.join(args.dest_dir, f'target_{target_idx}_{target_name}.wav'), out_audio, sample_rate=44100)

    print("Skipped 2 extra sources (guitar, piano)")
    print("Goodbye!")
