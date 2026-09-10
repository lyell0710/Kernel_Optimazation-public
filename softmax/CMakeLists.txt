cmake_minimum_required(VERSION 3.18)
project(cuda_softmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

include_directories(include)

add_executable(softmax_bench
    src/main.cu
    src/softmax_baseline.cu
    src/softmax_v0.cu
    src/softmax_v1.cu
    src/softmax_v2.cu
    src/softmax_v3.cu
    src/softmax_v4.cu
    src/softmax_cudnn.cu
    src/softmax_v4_2.cu
    src/softmax_v4_3.cu
    src/softmax_v4_4.cu
    src/softmax_cublas.cu
)

target_link_libraries(softmax_bench PRIVATE cublas)

# 标准库基准 cuDNN。为什么是 cuDNN 不是 cuBLAS:softmax 不属 BLAS 规范,
# BLAS 里没有这个算子,拿 cuBLAS 当对照是错的对照物(EXP-K04)。
#
# 路径不写死:本仓最初的开发机把头文件放在 venv 的 nvidia-cudnn 包内,
# 而一台干净的机器上它通常来自 apt 的 libcudnn9-dev-cuda-12(落在 /usr/include)。
# 写死 venv 路径会让本文件只能在那一台机器上配置成功——在采集主机上实际踩过。
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS /root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/cudnn/include
          /usr/include
          /usr/local/cuda/include)
# 系统可能只提供带版本号的 .so(无 libcudnn.so 符号链),所以两种名字都找
find_library(CUDNN_LIB NAMES cudnn libcudnn.so.9
             HINTS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIB)
    message(FATAL_ERROR
        "找不到 cuDNN(softmax 的标准库对照臂需要它)。\n"
        "  Ubuntu + NVIDIA apt 源: sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12\n"
        "  已装但仍找不到,用 -DCUDNN_INCLUDE_DIR=<dir> -DCUDNN_LIB=<path> 显式指定\n"
        "  当前: CUDNN_INCLUDE_DIR=${CUDNN_INCLUDE_DIR} CUDNN_LIB=${CUDNN_LIB}")
endif()

target_include_directories(softmax_bench PRIVATE ${CUDNN_INCLUDE_DIR})
target_link_libraries(softmax_bench PRIVATE ${CUDNN_LIB})
