#include "quantize_common.h"

namespace
{
constexpr int kBlockSize = 256;

__device__ __forceinline__ int8_t quant_one(float x, float s)
{
    float q = nearbyintf(x / s);
    q = fminf(127.0f, fmaxf(-127.0f, q));
    return static_cast<int8_t>(q);
}

// 在v3基础上增加float4读 + char4写，提升连续访存效率。
__global__ void quantize_v4_kernel(const float* __restrict__ input,
                                   const float* __restrict__ scales,
                                   int8_t* __restrict__ output,
                                   int channels,
                                   int hw)
{
    int c = blockIdx.x;
    int tid = threadIdx.x;
    if (c >= channels)
    {
        return;
    }
    float s = scales[c];
    int base = c * hw;

    for (int i = tid * 4; i < hw; i += blockDim.x * 4)
    {
        if (i + 3 < hw)
        {
            float4 v = reinterpret_cast<const float4*>(input + base + i)[0];
            char4 q;
            q.x = static_cast<signed char>(quant_one(v.x, s));
            q.y = static_cast<signed char>(quant_one(v.y, s));
            q.z = static_cast<signed char>(quant_one(v.z, s));
            q.w = static_cast<signed char>(quant_one(v.w, s));
            reinterpret_cast<char4*>(output + base + i)[0] = q;
        }
        else
        {
            for (int k = 0; k < 4 && i + k < hw; ++k)
            {
                output[base + i + k] = quant_one(input[base + i + k], s);
            }
        }
    }
}
} // namespace

void quantize_v4(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    quantize_v4_kernel<<<channels, kBlockSize>>>(input, scales, output, channels, hw);
}
