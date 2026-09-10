#pragma once

#include <cstdint>

void quantize_baseline(const float* input, const float* scales, int8_t* output, int channels, int hw);
void quantize_v0(const float* input, const float* scales, int8_t* output, int channels, int hw);
void quantize_v1(const float* input, const float* scales, int8_t* output, int channels, int hw);
void quantize_v2(const float* input, const float* scales, int8_t* output, int channels, int hw);
void quantize_v3(const float* input, const float* scales, int8_t* output, int channels, int hw);
void quantize_v4(const float* input, const float* scales, int8_t* output, int channels, int hw);

void cpu_quantize_per_channel(const float* input, const float* scales, int8_t* output, int channels, int hw);
