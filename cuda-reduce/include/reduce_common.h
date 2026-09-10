#pragma once

void reduce_baseline(const float* data, float* output, int n);
void reduce_v0(const float* data, float* output, int n);
void reduce_v1(const float* data, float* output, int n);
void reduce_v2(const float* data, float* output, int n);
void reduce_v3(const float* data, float* output, int n);
void reduce_v4(const float* data, float* output, int n);
void reduce_v5(const float* data, float* output, int n);
void reduce_v6(const float* data, float* output, int n);
void reduce_v7(const float* data, float* output, int n);
void reduce_cublas(const float* data, float* output, int n);


float cpu_reduce(const float* data, int n);
void reduce_cub(const float* data, float* output, int n);   // 标准库基准(CUB DeviceReduce::Sum)
