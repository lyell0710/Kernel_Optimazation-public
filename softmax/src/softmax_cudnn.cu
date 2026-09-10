// 标准库对照:cuDNN softmax forward(cudnnSoftmaxForward)。
//
// 为什么是 cuDNN 而不是 cuBLAS:cuBLAS 是 BLAS(线性代数)库,规范里没有
// softmax 这个算子;深度学习标准算子(softmax/归一化/卷积)属于 cuDNN。
// 本文件是本项目唯一的 softmax 标准库基准——历史上曾用自写 kernel 冒充
// "cublas" 对照,该口径已作废(见 records/EXP-K01)。
//
// 布局映射:输入 rows x cols(行主序),按行做 softmax。
//   cuDNN 4D 张量 NCHW = (rows, cols, 1, 1);
//   CUDNN_SOFTMAX_MODE_INSTANCE 对每个 n 在 C*H*W 上归一 = 每行独立 softmax。
//   CUDNN_SOFTMAX_ACCURATE 内部先减行最大值(数值稳定),与本项目 v0..v4
//   的在线/两遍实现同一数学口径,比较才公平。
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "softmax_common.h"

namespace {
cudnnHandle_t g_handle = nullptr;
cudnnTensorDescriptor_t g_desc = nullptr;
int g_rows = 0, g_cols = 0;

void ensure(int rows, int cols) {
    if (!g_handle) cudnnCreate(&g_handle);
    if (!g_desc) cudnnCreateTensorDescriptor(&g_desc);
    if (rows != g_rows || cols != g_cols) {   // 形状变化才重建描述符,避免把
        cudnnSetTensor4dDescriptor(g_desc, CUDNN_TENSOR_NCHW,  // 描述符构造
                                   CUDNN_DATA_FLOAT, rows, cols, 1, 1);
        g_rows = rows; g_cols = cols;         // 成本计入 kernel 时间
    }
}
} // namespace

void softmax_cudnn(const float* input, float* output, int rows, int cols) {
    ensure(rows, cols);
    const float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(g_handle, CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_INSTANCE,
                        &alpha, g_desc, input, &beta, g_desc, output);
}
