cmake_minimum_required(VERSION 3.0)

project(demucs.cpp)
enable_testing()

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Global compiler options
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -msimd128 -msse4.2 -flto -fno-exceptions -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

# Add Eigen3 and demucs.cpp include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Check if the build is for Emscripten
if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Only Emscripten build target is supported")
endif()

set(COMMON_LINK_FLAGS "-flto -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=5MB -s MODULARIZE=1 -s 'EXPORTED_RUNTIME_METHODS=[\"FS\"]'")

# demucs executable
file(GLOB DEMUCS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "demucs.cpp")
add_executable(demucs ${DEMUCS_SOURCES})
set_target_properties(demucs PROPERTIES
    LINK_FLAGS "${COMMON_LINK_FLAGS} -s EXPORT_NAME='libdemucs' -s EXPORTED_FUNCTIONS=\"['_malloc', '_free', '_modelInit', '_modelDemixSegment']\""
)
