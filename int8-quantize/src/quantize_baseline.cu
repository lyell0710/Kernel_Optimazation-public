#include "quantize_common.h"
#include <cmath>

namespace
{
__device__ __forceinline__ int8_t quant_one(float x, float s)
{
    float q = nearbyintf(x / s);
    q = fminf(127.0f, fmaxf(-127.0f, q));
    return static_cast<int8_t>(q);
}

__global__ void quantize_baseline_kernel(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int total = channels * hw;
    for (int i = 0; i < total; ++i)
    {
        int c = i / hw;
        output[i] = quant_one(input[i], scales[c]);
    }
}
} // namespace

void cpu_quantize_per_channel(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    int total = channels * hw;
    for (int i = 0; i < total; ++i)
    {
        int c = i / hw;
        float q = std::nearbyint(input[i] / scales[c]);
        q = std::fmin(127.0f, std::fmax(-127.0f, q));
        output[i] = static_cast<int8_t>(q);
    }
}

void quantize_baseline(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    quantize_baseline_kernel<<<1, 1>>>(input, scales, output, channels, hw);
}
