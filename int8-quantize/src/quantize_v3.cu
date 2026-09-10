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

// 一个block处理一个channel，scale只读一次到寄存器。
__global__ void quantize_v3_kernel(const float* __restrict__ input,
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
    for (int i = tid; i < hw; i += blockDim.x)
    {
        int idx = base + i;
        output[idx] = quant_one(input[idx], s);
    }
}
} // namespace

void quantize_v3(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    quantize_v3_kernel<<<channels, kBlockSize>>>(input, scales, output, channels, hw);
}
