cmake_minimum_required(VERSION 3.18)
project(cuda_int8_quantize LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

include_directories(include)

add_executable(int8_quantize_bench
    src/main.cu
    src/quantize_baseline.cu
    src/quantize_v0.cu
    src/quantize_v1.cu
    src/quantize_v2.cu
    src/quantize_v3.cu
    src/quantize_v4.cu
)
