# cmake file to compile src/
# link against included submodules libnyquist

cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DEIGEN_FAST_MATH=0 -O0")

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -fno-unsafe-math-optimizations -fassociative-math -freciprocal-math -fno-signed-zeros")

# define a macro NDEBUG for Eigen3 release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

# set EIGEN_USE_BLAS to 1 and link to OpenBLAS
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE")

project(demucs.cpp)
enable_testing()

# set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# add openmp support
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# compile vendored submodule libnyquist
set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "Disable libnyquist example")
add_subdirectory(vendor/libnyquist)

# add library Eigen3
include_directories(vendor/eigen)

# add OpenBLAS for blas + lapack
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# include vendor submodules libnyquist
include_directories(vendor/libnyquist/include)

# include src/ as include directory
include_directories(src)

# include src/*.cpp and src/*.c as source files
file(GLOB SOURCES "src/*.cpp")

# compile library, link against libnyquist
add_library(demucs.cpp.lib SHARED ${SOURCES})
target_link_libraries(demucs.cpp.lib libnyquist ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} lapacke)
if(OPENMP_FOUND)
    target_link_libraries(demucs.cpp.lib ${OpenMP_CXX_LIBRARIES})
endif()

file(GLOB SOURCES_TO_LINT "src/*.cpp" "src/*.hpp" "demucs.cpp" "test/*.cpp")

# add target to run standard lints and formatters
add_custom_target(lint
    COMMAND clang-format -i ${SOURCES_TO_LINT} --style=file
    # add clang-tidy command
    # add include dirs to clang-tidy
    COMMAND cppcheck --enable=all --suppress=missingIncludeSystem ${SOURCES_TO_LINT} --std=c++17
    COMMAND scan-build -o ${CMAKE_BINARY_DIR}/scan-build-report make -C ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# add target to compile demucs.cpp, the main driver program for demucs.cpp
add_executable(demucs.cpp.main demucs.cpp)
# link it against demucs.cpp.lib
target_link_libraries(demucs.cpp.main demucs.cpp.lib)

# add target to run cpp tests in test/ directory with gtest

# include test/*.cpp as test files
file(GLOB TEST_SOURCES "test/*.cpp")

add_executable(demucs.cpp.test ${TEST_SOURCES})
target_link_libraries(demucs.cpp.test demucs.cpp.lib gtest gtest_main libnyquist)
add_test(NAME tests COMMAND demucs.cpp.test)
