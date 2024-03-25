cmake_minimum_required(VERSION 3.0)

option(USE_OPENBLAS "Use OpenBLAS" ON)
option(USE_AMD_AOCL "Use AMD AOCL" OFF)
option(USE_INTEL_MKL "Use Intel MKL" OFF)

# Ensure USE_AMD_AOCL and USE_INTEL_MKL are not both set
if(USE_AMD_AOCL AND USE_INTEL_MKL)
    message(FATAL_ERROR "USE_AMD_AOCL and USE_INTEL_MKL cannot be used together. Please choose one.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_FAST_MATH=0 -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -fno-unsafe-math-optimizations -freciprocal-math -fno-signed-zeros")

# define a macro NDEBUG for Eigen3 release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

project(demucs.cpp)
enable_testing()

# set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (USE_AMD_AOCL)
    set(AOCL_ROOT "/home/sevagh/AMD_AOCL/aocl-linux-gcc-4.1.0")
    set(BLIS_PATH "${AOCL_ROOT}/aocl-blis-linux-gcc-4.1.0/amd-blis")
    set(LIBFLAME_PATH "${AOCL_ROOT}/aocl-libflame-linux-gcc-4.1.0/amd-libflame")
    set(LIBMEM_PATH "${AOCL_ROOT}/aocl-libmem-linux-gcc-4.1.0/amd-libmem")
    include_directories(${BLIS_PATH}/include)
    include_directories(${LIBFLAME_PATH}/include)
    include_directories(${LIBMEM_PATH}/include)
    link_directories(${BLIS_PATH}/lib/LP64)
    link_directories(${LIBFLAME_PATH}/lib/LP64)
    link_directories(${LIBMEM_PATH}/lib/shared)
    set(LIBRARIES_TO_LINK "-lblis-mt -lflame -laocl-libmem")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_BLAS")
elseif (USE_INTEL_MKL)
    set(INTEL_MKL_ROOT "/home/sevagh/intel/oneapi/mkl/latest")
    set(INTEL_COMPILER_ROOT "/home/sevagh/intel/oneapi/compiler/latest")
    include_directories("${INTEL_MKL_ROOT}/include")
    link_directories("${INTEL_MKL_ROOT}/lib/intel64")
    link_directories("${INTEL_COMPILER_ROOT}/lib")
    set(LIBRARIES_TO_LINK "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_BLAS")
elseif (USE_OPENBLAS)
    find_package(BLAS REQUIRED)
    set(LIBRARIES_TO_LINK ${BLAS_LIBRARIES})
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_BLAS")
else()
    # add openmp support to regular eigen
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        include_directories(${OpenMP_CXX_INCLUDE_DIRS})
    endif()

    if(OPENMP_FOUND)
        set(LIBRARIES_TO_LINK ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# include src/*.cpp and src/*.c as source files
file(GLOB SOURCES "src/*.cpp")

add_library(demucs.cpp.lib ${SOURCES})

# add library Eigen3
# expose source files to project linking to this one
target_include_directories(demucs.cpp.lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(demucs.cpp.lib ${LIBRARIES_TO_LINK})

# compile vendored submodule libnyquist for driver programs
set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example" FORCE)
add_subdirectory(vendor/libnyquist)

# Add target to compile demucs.cpp.main, the main driver program for demucs.cpp
add_executable(demucs.cpp.main "cli-apps/demucs.cpp")
target_include_directories(demucs.cpp.main PRIVATE vendor/libnyquist/include)
target_link_libraries(demucs.cpp.main demucs.cpp.lib libnyquist)

# Add target to compile demucs_ft.cpp.main, the fine-tuned driver program for demucs.cpp
add_executable(demucs_ft.cpp.main "cli-apps/demucs_ft.cpp")
target_include_directories(demucs_ft.cpp.main PRIVATE vendor/libnyquist/include)
target_link_libraries(demucs_ft.cpp.main demucs.cpp.lib libnyquist)

add_executable(demucs_mt.cpp.main "cli-apps/demucs_mt.cpp")
target_include_directories(demucs_mt.cpp.main PRIVATE vendor/libnyquist/include)
target_include_directories(demucs_mt.cpp.main PRIVATE cli-apps)
target_link_libraries(demucs_mt.cpp.main demucs.cpp.lib libnyquist)

add_executable(demucs_ft_mt.cpp.main "cli-apps/demucs_ft_mt.cpp")
target_include_directories(demucs_ft_mt.cpp.main PRIVATE vendor/libnyquist/include)
target_include_directories(demucs_ft_mt.cpp.main PRIVATE cli-apps)
target_link_libraries(demucs_ft_mt.cpp.main demucs.cpp.lib libnyquist)

file(GLOB SOURCES_TO_LINT "src/*.cpp" "src/*.hpp" "cli-apps/*.cpp" "cli-apps/*.hpp")

# add target to run standard lints and formatters
add_custom_target(lint
    COMMAND clang-format -i ${SOURCES_TO_LINT} --style=file
    # add clang-tidy command
    # add include dirs to clang-tidy
    COMMAND cppcheck -I"src/" -I"cli-apps/" --enable=all --suppress=missingIncludeSystem ${SOURCES_TO_LINT} --std=c++17
    COMMAND scan-build -o ${CMAKE_BINARY_DIR}/scan-build-report make -C ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# add target to run cpp tests in test/ directory with gtest

# get gtest from vendor/googletest
set(BUILD_GMOCK OFF CACHE BOOL "Disable gmock in googletest" FORCE)
add_subdirectory(vendor/googletest)

# include test/*.cpp as test files
file(GLOB TEST_SOURCES "test/*.cpp")

add_executable(demucs.cpp.test ${TEST_SOURCES})
target_link_libraries(demucs.cpp.test demucs.cpp.lib gtest gtest_main libnyquist)
add_test(NAME tests COMMAND demucs.cpp.test)
