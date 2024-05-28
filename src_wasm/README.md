# Demucs.cpp WebAssembly guide

Demucs.cpp is used to build <https://freemusicdemixer.com>

This document gives a brief overview of how it works

## Set up emsdk

Set up the Emscripten SDK:

    * <https://emscripten.org/docs/tools_reference/emsdk.html>
    * <https://github.com/emscripten-core/emsdk>

On my computer, once it's set up, this is how it looks:
```
(system) sevagh@pop-os:~/repos/demucs.cpp$ source /home/sevagh/repos/emsdk/emsdk_env.sh
Setting up EMSDK environment (suppress these messages with EMSDK_QUIET=1)
Adding directories to PATH:
PATH += /home/sevagh/repos/emsdk
PATH += /home/sevagh/repos/emsdk/upstream/emscripten

Setting environment variables:
PATH = /home/sevagh/repos/emsdk:/home/sevagh/repos/emsdk/upstream/emscripten:/home/sevagh/.nvm/versions/node/v20.5.0/bin:/home/sevagh/mambaforge/envs/system/bin:/home/sevagh/mambaforge/condabin:/home/sevagh/.cargo/bin:/home/sevagh/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/sevagh/.local/bin:/usr/local/go/bin:/home/sevagh/go/bin:/home/sevagh/.yarn/bin
EMSDK = /home/sevagh/repos/emsdk
EMSDK_NODE = /home/sevagh/repos/emsdk/node/16.20.0_64bit/bin/node
```

This installs emcmake and emcc:
```
(system) sevagh@pop-os:~/repos/demucs.cpp$ emcc --version
emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) 3.1.51 (c0c2ca1314672a25699846b4663701bcb6f69cca)
Copyright (C) 2014 the Emscripten authors (see AUTHORS.txt)
This is free and open source software under the MIT license.
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

(system) sevagh@pop-os:~/repos/demucs.cpp$ emcmake --version
emcmake is a helper for cmake, setting various environment
variables so that emcc etc. are used. Typical usage:

  emcmake cmake [FLAGS]
```

## Build demucs.cpp for WASM

The demucs.cpp inference libraries stays as is in `src/`. There is a new directory, `src_wasm/`, containing a CMakeLists.txt file for compiling for WASM, and two new source files:

    * `demucs.cpp`, which contains the C++/Javascript boundary functions that call demucs.cpp's inference library
    * `model_load.hpp`, a copy of the model load function that's designed to take in bytes representing the loaded model file (more useful for web than loading from a filesystem path)

I build like this:
```
(system) sevagh@pop-os:~/repos/demucs.cpp$ /bin/bash -c 'source /home/sevagh/repos/emsdk/emsdk_env.sh && \
                rm -rf build-wasm && mkdir -p build-wasm && cd build-wasm \
                && emcmake cmake -DCMAKE_BUILD_TYPE=Release \
                ../src_wasm && make -j16'
```

In `build-wasm`, we can see the output artifacts:
```
(system) sevagh@pop-os:~/repos/demucs.cpp$ ls -latrh build-wasm/
total 696K
-rw-rw-r--  1 sevagh sevagh  13K May 28 07:28 CMakeCache.txt
-rw-rw-r--  1 sevagh sevagh  17K May 28 07:28 Makefile
-rw-rw-r--  1 sevagh sevagh  291 May 28 07:28 CTestTestfile.cmake
-rw-rw-r--  1 sevagh sevagh 1.6K May 28 07:28 cmake_install.cmake
-rwxrwxr-x  1 sevagh sevagh 566K May 28 07:28 demucs.wasm
-rw-rw-r--  1 sevagh sevagh  69K May 28 07:28 demucs.js
drwxrwxr-x  3 sevagh sevagh 4.0K May 28 07:28 .
drwxrwxr-x  4 sevagh sevagh 4.0K May 28 07:28 CMakeFiles
drwxrwxr-x 15 sevagh sevagh 4.0K May 28 08:42 ..
```

The important WASM files are `demucs.wasm` and `demucs.js`

You can see on the freemusicdemixer website, similar files (compiled from different source code but roughly equivalent to demucs.cpp) are committed directly to the repo:

* <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/demucs_free.js>
* <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/demucs_free.wasm>

The exposed library/API to `libdemucs` (to be used in Javascript) is defined in the CMakeLists.txt file:
```
# demucs executable
file(GLOB DEMUCS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "demucs.cpp")
add_executable(demucs ${DEMUCS_SOURCES})
set_target_properties(demucs PROPERTIES
    LINK_FLAGS "${COMMON_LINK_FLAGS} -s EXPORT_NAME='libdemucs' -s EXPORTED_FUNCTIONS=\"['_malloc', '_free', '_modelInit', '_modelDemixSegment']\""
)
```

You need `_malloc` and `_free` (built-in memory management functions) alongside the Demucs.cpp code.

## Host weights files somewhere

In my website, I download the model files from my own Cloudflare R2 bucket: <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/main.js#L299C1-L321C1>

These are model files converted to GGML binary format as per the repo of this project. For your own project, you should host this file yourself.

## Load weights files and pass to Demucs worker init

I use the WebWorker paradigm to interact between the Demucs WASM module in `worker.js` and the main functions/UI functions in `main.js`. This is my message telling the WASM module to load itself and passing the GGML weights file bytes: <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/main.js#L291-L294>

## Pass the left and right channels of audio to the Demucs worker

In this code: <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/main.js#L357C1-L366C8>

## In worker.js, call Demucs C++/WASM function

This should give an overview of how to call the compiled WASM module functions: <https://github.com/sevagh/freemusicdemixer.com/blob/main/docs/worker.js#L27-L140>

## Put it all together

The rest of it is an exercise for the reader.
