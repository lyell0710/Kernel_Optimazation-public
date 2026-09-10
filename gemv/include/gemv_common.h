#pragma once

void gemv_baseline(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_v0(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_v1(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_v2(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_v3(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_v4(const float* mat, const float* vec, float* out, int rows, int cols);
void gemv_cublas(const float* mat, const float* vec, float* out, int rows, int cols);

void cpu_gemv(const float* mat, const float* vec, float* out, int rows, int cols);
