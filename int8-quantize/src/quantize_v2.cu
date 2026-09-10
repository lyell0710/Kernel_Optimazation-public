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

// 每线程pack=4处理，提升指令吞吐；按元素取channel scale保证per-channel正确性。
__global__ void quantize_v2_kernel(const float* __restrict__ input,
                                   const float* __restrict__ scales,
                                   int8_t* __restrict__ output,
                                   int channels,
                                   int hw)
{
    int total = channels * hw;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int base = gid * 4; base < total; base += step * 4)
    {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int i = base + k;
            if (i < total)
            {
                int c = i / hw;
                output[i] = quant_one(input[i], scales[c]);
            }
        }
    }
}
} // namespace

void quantize_v2(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    int total = channels * hw;
    int grid = (total + (kBlockSize * 4 - 1)) / (kBlockSize * 4);
    quantize_v2_kernel<<<grid, kBlockSize>>>(input, scales, output, channels, hw);
}
