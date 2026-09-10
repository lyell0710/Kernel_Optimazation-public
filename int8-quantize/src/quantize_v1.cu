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

__global__ void quantize_v1_kernel(const float* __restrict__ input,
                                   const float* __restrict__ scales,
                                   int8_t* __restrict__ output,
                                   int channels,
                                   int hw)
{
    int total = channels * hw;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    // 每线程一次处理两个元素，减少循环控制开销。
    for (int i = gid * 2; i < total; i += step * 2)
    {
        int c0 = i / hw;
        output[i] = quant_one(input[i], scales[c0]);
        if (i + 1 < total)
        {
            int c1 = (i + 1) / hw;
            output[i + 1] = quant_one(input[i + 1], scales[c1]);
        }
    }
}
} // namespace

void quantize_v1(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    int total = channels * hw;
    int grid = (total + (kBlockSize * 2 - 1)) / (kBlockSize * 2);
    quantize_v1_kernel<<<grid, kBlockSize>>>(input, scales, output, channels, hw);
}
