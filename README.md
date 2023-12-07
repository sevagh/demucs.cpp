# demucs.cpp

C++17 implementation of the [Demucs v4 hybrid transformer](https://github.com/facebookresearch/demucs), a PyTorch neural network for music demixing. Similar project to [umx.cpp](https://github.com/sevagh/umx.cpp).

It uses [libnyquist](https://github.com/ddiakopoulos/libnyquist) to load audio files, the [ggml](https://github.com/ggerganov/ggml) file format to serialize the PyTorch weights of `htdemucs_4s` to a binary file format, and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (+ OpenMP) to implement the inference.

Track 'Zeno - Signs' from MUSDB18-HQ test set

PyTorch CLI inference (output of `demucs /path/to/track` from [this commit of demucs v4](https://github.com/facebookresearch/demucs@2496b8f7f12b01c8dd0187c040000c46e175b44d)):
```
vocals          ==> SDR:   8.264  SIR:  18.353  ISR:  15.794  SAR:   8.303
drums           ==> SDR:  10.111  SIR:  18.503  ISR:  17.089  SAR:  10.746
bass            ==> SDR:   4.222  SIR:  12.615  ISR:   6.973  SAR:   2.974
other           ==> SDR:   7.397  SIR:  11.317  ISR:  14.303  SAR:   8.137
```
PyTorch custom inference in [my script](./scripts/demucs_pytorch_inference.py):
```
vocals          ==> SDR:   8.339  SIR:  18.274  ISR:  15.835  SAR:   8.354
drums           ==> SDR:  10.058  SIR:  18.598  ISR:  17.023  SAR:  10.812
bass            ==> SDR:   3.926  SIR:  12.414  ISR:   6.941  SAR:   3.202
other           ==> SDR:   7.421  SIR:  11.289  ISR:  14.241  SAR:   8.179
```
CPP inference (this codebase):
```
vocals          ==> SDR:   8.339  SIR:  18.276  ISR:  15.836  SAR:   8.346
drums           ==> SDR:  10.058  SIR:  18.596  ISR:  17.019  SAR:  10.810
bass            ==> SDR:   3.919  SIR:  12.436  ISR:   6.931  SAR:   3.182
other           ==> SDR:   7.421  SIR:  11.286  ISR:  14.252  SAR:   8.183
```
WASM inference (freemusicdemixer, based on this codebase):
```
vocals          ==> SDR:   8.339  SIR:  18.274  ISR:  15.836  SAR:   8.347
drums           ==> SDR:  10.058  SIR:  18.596  ISR:  17.017  SAR:  10.813
bass            ==> SDR:   3.920  SIR:  12.432  ISR:   6.933  SAR:   3.188
other           ==> SDR:   7.420  SIR:  11.287  ISR:  14.250  SAR:   8.184

```

*n.b.* for the above results, the random shift in the beginning of the song was fixed to 1337 in both PyTorch and C++.

## Instructions

0. Clone the repo

Make sure you clone with submodules:
```
$ git clone --recurse-submodules https://github.com/sevagh/demucs.cpp
```

Eigen is vendored as a git submodule.

1. Set up Python

The first step is to create a Python environment (however you like; I'm a fan of [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)) and install the `requirements.txt` file:
```
$ mamba create --name demucscpp python=3.11
$ mamba activate demucscpp
$ python -m pip install -r ./scripts/requirements.txt
```

2. Dump Demucs weights to ggml file:
```
$ python ./scripts/convert-pth-to-ggml.py ./ggml-demucs
...
Processing variable:  crosstransformer.layers_t.4.norm2.bias  with shape:  (512,)  , dtype:  float16
Processing variable:  crosstransformer.layers_t.4.norm_out.weight  with shape:  (512,)  , dtype:  float16
Processing variable:  crosstransformer.layers_t.4.norm_out.bias  with shape:  (512,)  , dtype:  float16
Processing variable:  crosstransformer.layers_t.4.gamma_1.scale  with shape:  (512,)  , dtype:  float16
Processing variable:  crosstransformer.layers_t.4.gamma_2.scale  with shape:  (512,)  , dtype:  float16
Done. Output file:  ggml-demucs/ggml-model-htdemucs-4s-f16.bin
```

3. Install C++ dependencies, e.g. CMake, gcc, C++/g++, OpenMP for your OS - my instructions are for Pop!\_OS 22.04:
```
$ sudo apt-get install gcc g++ cmake clang-tools
```

4. Compile with CMake:
```
$ mkdir -p build && cd build && cmake .. && make
```

5. Run inference on your track:
```
$ ./demucs.cpp.main ../ggml-demucs/ggml-model-htdemucs-4s-f16.bin /path/to/my/track.wav  ./demucs-out-cpp/
...
Loading tensor crosstransformer.layers_t.4.gamma_2.scale with shape [512, 1, 1, 1]
crosstransformer.layers_t.4.gamma_2.scale: [  512], type = float,   0.00 MB
Loaded model (533 tensors,  80.08 MB) in 0.167395 s
demucs_model_load returned true
Starting demucs inference
...
Freq: decoder 3
Time: decoder 3
Mask + istft
mix: 2, 343980
mix: 2, 343980
mix: 2, 343980
mix: 2, 343980
returned!
Writing wav file "./demucs-out-cpp/target_0.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_1.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_2.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_3.wav"
Encoder Status: 0
```

Note: I have only tested this on my Linux-based computer (Pop!\_OS 22.04), and you may need to figure out how to get the dependencies on your own.

## Dev tips

* make lint
* Valgrind memory error test: `valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./demucs.cpp.main ../ggml-demucs/ggml-model-htdemucs-f16.bin ../test/data/gspi_stereo.wav  ./demucs-out-cpp/`
* Callgrind + KCachegrind: `valgrind --tool=callgrind ./demucs.cpp.test --gtest_filter='*FreqDec*'`
