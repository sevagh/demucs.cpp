#!/usr/bin/env python3

import io
import sys
import struct
import argparse
from pathlib import Path

import torch
import numpy as np

# If there are layers you explicitly don't want to export, add their
# full state_dict names here.
LAYERS_TO_SKIP = []


def load_state_dict(checkpoint_path: Path) -> dict:
    """
    Load a Demucs checkpoint and return the state_dict that matches
    what demucs.cpp expects (a flat dict of tensors).
    """
    try:
        model_bytes = checkpoint_path.read_bytes()
    except Exception as e:
        print(f"Error: failed to read checkpoint: {checkpoint_path} ({e})")
        sys.exit(1)

    try:
        with io.BytesIO(model_bytes) as fp:
            ckpt = torch.load(fp, map_location="cpu")
    except Exception as e:
        print(f"Error: failed to load PyTorch model file: {checkpoint_path} ({e})")
        sys.exit(1)

    # Typical Demucs checkpoints have one of these keys
    if isinstance(ckpt, dict):
        for key in ("state", "state_dict", "model_state", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

    # Fallback: maybe the checkpoint *is* already a state_dict
    if isinstance(ckpt, dict):
        # weak heuristic: all values tensors or dicts
        if all(isinstance(v, (torch.Tensor, dict)) for v in ckpt.values()):
            return ckpt

    print("Error: could not locate a state dict in the checkpoint.")
    print("Keys found at top level:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a 2-stem Demucs PyTorch checkpoint to demucs.cpp GGML format"
    )
    parser.add_argument(
        "dest_dir",
        type=str,
        help="Destination directory for the converted model",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to your trained Demucs checkpoint (.th)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="htdemucs_2s",
        help="Model name to embed in the output filename (default: htdemucs_2s)",
    )
    parser.add_argument(
        "--magic",
        type=lambda x: int(x, 0),
        default=0x646d6332,  # 'dmc2' in ASCII: 0x64 0x6d 0x63 0x32
        help=(
            "Magic number to write into the header. "
            "Default 0x646d6332 (ASCII 'dmc2' = 2-stem Demucs). "
            "Must match what your demucs.cpp loader expects."
        ),
    )

    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load the flat state dict
    checkpoint = load_state_dict(ckpt_path)

    print("Loaded checkpoint from:", ckpt_path)
    print("Top-level tensor keys:")
    for k in list(checkpoint.keys())[:20]:
        print("  ", k)
    if len(checkpoint) > 20:
        print(f"  ... ({len(checkpoint)} total entries)")

    # Output file name
    suffix = "-2s"
    dest_name = dest_dir / f"ggml-model-{args.name}{suffix}-f16.bin"

    # Open output file
    try:
        fout = dest_name.open("wb")
    except Exception as e:
        print(f"Error: could not open output file {dest_name} ({e})")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Header: single 32-bit magic, as in the original script
    #
    # Original:
    #   - 0x646d6334: 'dmc4' (4-stem Demucs)
    #   - 0x646d6336: 'dmc6' (6-stem Demucs)
    #   - 0x646d6333: 'dmc3' (v3/mmi)
    #
    # Here we use:
    #   - 0x646d6332: 'dmc2' (2-stem custom Demucs: dialogue/background)
    #
    # Make sure demucs.cpp understands this new magic.
    # ------------------------------------------------------------------
    magic = args.magic
    fout.write(struct.pack("i", magic))

    # ------------------------------------------------------------------
    # Write layers: for each tensor in the state dict:
    #   int32 n_dims
    #   int32 name_length
    #   int32 shape[ndims]
    #   char   name[name_length]
    #   raw   data (as in original converter)
    # ------------------------------------------------------------------
    n_written = 0

    for name, value in checkpoint.items():
        if name in LAYERS_TO_SKIP:
            print(f"Skipping layer {name}")
            continue

        if not isinstance(value, torch.Tensor):
            print(f"Skipping non-tensor entry: {name} ({type(value)})")
            continue

        # move to CPU, remove size-1 dims (as original script did)
        tensor = value.detach().cpu().squeeze()
        data = tensor.numpy()

        print(
            "Processing variable:",
            name,
            "shape:",
            data.shape,
            ", dtype:",
            data.dtype,
        )

        # Number of dimensions
        n_dims = data.ndim

        # Encode name
        name_bytes = name.encode("utf-8")
        name_len = len(name_bytes)

        # Header: n_dims, name_len
        fout.write(struct.pack("ii", n_dims, name_len))

        # Shape (one int32 per dimension)
        for dim in range(n_dims):
            fout.write(struct.pack("i", data.shape[dim]))

        # Name
        fout.write(name_bytes)

        # Data
        # Ensure C-contiguous before writing
        data = np.ascontiguousarray(data)
        data.tofile(fout)

        n_written += 1

    fout.close()

    print("")
    print(f"Done. Wrote {n_written} tensors.")
    print("Output file:", dest_name)
    print("Magic used: 0x{0:08x}".format(magic))


if __name__ == "__main__":
    main()
