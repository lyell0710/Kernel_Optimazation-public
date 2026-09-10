#pragma once

void softmax_baseline(const float* input, float* output, int rows, int cols);
void softmax_v0(const float* input, float* output, int rows, int cols);
void softmax_v1(const float* input, float* output, int rows, int cols);
void softmax_v2(const float* input, float* output, int rows, int cols);
void softmax_v3(const float* input, float* output, int rows, int cols);
void softmax_v4(const float* input, float* output, int rows, int cols);
void softmax_v4_2(const float* input, float* output, int rows, int cols);
void softmax_v4_3(const float* input, float* output, int rows, int cols);
void softmax_v4_4(const float* input, float* output, int rows, int cols);
void softmax_cublas(const float* input, float* output, int rows, int cols);  // 历史遗留:自写参照,非 cuBLAS(见 EXP-K01（四 kernel 4090 重基准）口径勘正)
void softmax_cudnn(const float* input, float* output, int rows, int cols);   // 标准库基准

void cpu_softmax(const float* input, float* output, int rows, int cols);
