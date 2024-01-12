# demucs.cpp

C++17 implementation of the [Demucs v4 hybrid transformer](https://github.com/facebookresearch/demucs), a PyTorch neural network for music demixing. Similar project to [umx.cpp](https://github.com/sevagh/umx.cpp). This code powers my site <https://freemusicdemixer.com>.

It uses [libnyquist](https://github.com/ddiakopoulos/libnyquist) to load audio files, the [ggml](https://github.com/ggerganov/ggml) file format to serialize the PyTorch weights of `htdemucs`, `htdemucs_6s`, and `htdemucs_ft` (4-source, 6-source, fine-tuned) to a binary file format, and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (+ OpenMP) to implement the inference.

**All Hybrid-Transformer weights** (4-source, 6-source, fine-tuned) are supported. See the [Convert weights](#convert-weights) section below. Demixing quality is nearly identical to PyTorch as shown in the [SDR scores doc](./.github/SDR_scores.md).

### Directory structure

`src` contains the library for Demucs inference, and `cli-apps` contains two driver programs, which compile to:
1. `demucs.cpp.main`: run a single model (4s, 6s, or a single fine-tuned model)
2. `demucs_ft.cpp.main`: run all 4 fine-tuned models for `htdemucs_ft` inference, same as the BagOfModels idea of PyTorch Demucs

### Multi-core, OpenMP, BLAS, etc.

:warning: `demucs.cpp` library code in `./src` **should not use any threading (e.g. pthread or OpenMP) except through the BLAS interface.** This is because demucs.cpp is compiled to a single-threaded WebAssembly module in <https://freemusicdemixer.com>.

If you have OpenMP and OpenBLAS installed, OpenBLAS might automatically use all of the threads on your machine, which doesn't always run the fastest. Use the `OMP_NUM_THREADS` environment variable to limit this. On my 16c/32t machine, I found `OMP_NUM_THREADS=16` to be the fastest. This matches the [Eigen recommendation](https://eigen.tuxfamily.org/dox/TopicMultiThreading.html) to use the same number of threads as physical cores:
>On most OS it is very important to limit the number of threads to the number of physical cores, otherwise significant slowdowns are expected, especially for operations involving dense matrices.

See the [BLAS benchmarks doc](./.github/BLAS_benchmarks.md) for more details.

## Instructions

### Build C++ code

Clone the repo

Make sure you clone with submodules to get all vendored libraries (e.g. Eigen):
```
$ git clone --recurse-submodules https://github.com/sevagh/demucs.cpp
```

Install C++ dependencies, e.g. CMake, gcc, C++/g++, OpenBLAS for your OS (my instructions are for Pop!\_OS 22.04):
```
$ sudo apt-get install gcc g++ cmake clang-tools libopenblas0-openmp libopenblas-openmp-dev
```

Compile with CMake:
```
$ mkdir -p build && cd build && cmake .. && make -j16
libdemucs.cpp.lib.a <--- library
demucs.cpp.main     <--- single-model (4s, 6s, ft)
demucs_ft.cpp.main  <--- bag of ft models
demucs.cpp.test     <--- unit tests
```

### Convert weights

Set up a Python env

The first step is to create a Python environment (however you like; I'm a fan of [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html)) and install the `requirements.txt` file:
```
$ mamba create --name demucscpp python=3.11
$ mamba activate demucscpp
$ python -m pip install -r ./scripts/requirements.txt
```

Dump Demucs weights to ggml file, with flag `--six-source` for the 6-source variant, and all of `--ft-drums, --ft-vocals, --ft-bass, --ft-other` for the fine-tuned models:
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

All supported models would look like this:
```
$ ls ../ggml-demucs/
total 133M
 81M Jan 10 22:40 ggml-model-htdemucs-4s-f16.bin
 53M Jan 10 22:41 ggml-model-htdemucs-6s-f16.bin
 81M Jan 10 22:41 ggml-model-htdemucs_ft_drums-4s-f16.bin
 81M Jan 10 22:43 ggml-model-htdemucs_ft_bass-4s-f16.bin
 81M Jan 10 22:43 ggml-model-htdemucs_ft_other-4s-f16.bin
 81M Jan 10 22:43 ggml-model-htdemucs_ft_vocals-4s-f16.bin
```

### Run demucs.cpp

Run C++ inference on your track with the built binaries:
```
# build is the cmake build dir from above
$ ./build/demucs.cpp.main ../ggml-demucs/ggml-model-htdemucs-4s-f16.bin /path/to/my/track.wav  ./demucs-out-cpp/
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
Writing wav file "./demucs-out-cpp/target_0_drums.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_1_bass.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_2_other.wav"
Encoder Status: 0
Writing wav file "./demucs-out-cpp/target_3_vocals.wav"
Encoder Status: 0
```

For the 6-source model, additional targets 4 and 5 correspond to guitar and piano.

## Dev tips

* make lint
* Valgrind memory error test: `valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./demucs.cpp.main ../ggml-demucs/ggml-model-htdemucs-f16.bin ../test/data/gspi_stereo.wav  ./demucs-out-cpp/`
* Callgrind + KCachegrind: `valgrind --tool=callgrind ./demucs.cpp.test --gtest_filter='*FreqDec*'`
