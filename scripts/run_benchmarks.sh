#!/usr/bin/env bash

echo "##########################################"
echo "# gcc, Eigen, no BLAS, -mnative, -Ofast  #"
echo "##########################################"
echo

for numthr in 1 2 4 8 16 32; do
        printf "OMP threads: %s\n" "${numthr}"
        export OMP_NUM_THREADS=$numthr
        /usr/bin/time --verbose ./build/demucs.cpp.main ./ggml-demucs/ggml-model-htdemucs-4s-f16.bin ./test/data/gspi_stereo.wav ./build/demucs-out-cpp
done

echo "##########################################"
echo "# gcc, Eigen, OpenBLAS, -mnative, -Ofast #"
echo "##########################################"
echo

for numthr in 1 2 4 8 16 32; do
        printf "OMP threads: %s\n" "${numthr}"
        export OMP_NUM_THREADS=$numthr
        /usr/bin/time --verbose ./build-openblas/demucs.cpp.main ./ggml-demucs/ggml-model-htdemucs-4s-f16.bin ./test/data/gspi_stereo.wav ./build/demucs-out-cpp
done

echo "#######################################################"
echo "# gcc, Eigen, AOCL-BLIS (+ friends), -mnative, -Ofast #"
echo "#######################################################"
echo

for numthr in 1 2 4 8 16 32; do
        printf "OMP threads: %s\n" "${numthr}"
        export OMP_NUM_THREADS=$numthr
        /usr/bin/time --verbose ./build-amd/demucs.cpp.main ./ggml-demucs/ggml-model-htdemucs-4s-f16.bin ./test/data/gspi_stereo.wav ./build-amd/demucs-out-cpp
done

echo "#######################################################"
echo "# gcc, Eigen, Intel-MKL (+ friends), -mnative, -Ofast #"
echo "#######################################################"
echo

for numthr in 1 2 4 8 16 32; do
        printf "OMP/MKL threads: %s\n" "${numthr}"
        export MKL_NUM_THREADS=$numthr
        export OMP_NUM_THREADS=$numthr
        unset MKL_DEBUG_CPU_TYPE
        /usr/bin/time --verbose ./build-intel/demucs.cpp.main ./ggml-demucs/ggml-model-htdemucs-4s-f16.bin ./test/data/gspi_stereo.wav ./build-intel/demucs-out-cpp
done

echo "#######################################################"
echo "# gcc, Eigen, Intel-MKL (+ friends), -mnative, -Ofast #"
echo "# with MKL_DEBUG_CPU_TYPE=5 to force avx2 for amd     #"
echo "#######################################################"
echo

for numthr in 1 2 4 8 16 32; do
        printf "OMP/MKL threads: %s\n" "${numthr}"
        export MKL_NUM_THREADS=$numthr
        export OMP_NUM_THREADS=$numthr
        export MKL_DEBUG_CPU_TYPE=5  # force AVX2 on amd cpus
        /usr/bin/time --verbose ./build-intel/demucs.cpp.main ./ggml-demucs/ggml-model-htdemucs-4s-f16.bin ./test/data/gspi_stereo.wav ./build-intel/demucs-out-cpp
done
