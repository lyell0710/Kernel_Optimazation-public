cmake_minimum_required(VERSION 3.18)
project(cuda_gemm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_ARCHITECTURES 89)
include_directories(include)
add_executable(gemm_bench
    src/main.cu src/gemm_v0.cu src/gemm_v1.cu src/gemm_v2.cu
    src/gemm_v3.cu src/gemm_v4.cu src/gemm_v5.cu src/gemm_cublas.cu)
target_link_libraries(gemm_bench PRIVATE cublas)
