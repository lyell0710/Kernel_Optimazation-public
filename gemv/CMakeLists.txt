cmake_minimum_required(VERSION 3.18)
project(cuda_gemv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

include_directories(include)

add_executable(gemv_bench
    src/main.cu
    src/gemv_baseline.cu
    src/gemv_v0.cu
    src/gemv_v1.cu
    src/gemv_v2.cu
    src/gemv_v3.cu
    src/gemv_v4.cu
    src/gemv_cublas.cu
)

target_link_libraries(gemv_bench PRIVATE cublas)
