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

__global__ void quantize_v0_kernel(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    int total = channels * hw;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int i = gid; i < total; i += step)
    {
        int c = i / hw;
        output[i] = quant_one(input[i], scales[c]);
    }
}
} // namespace

void quantize_v0(const float* input, const float* scales, int8_t* output, int channels, int hw)
{
    int total = channels * hw;
    int grid = (total + kBlockSize - 1) / kBlockSize;
    quantize_v0_kernel<<<grid, kBlockSize>>>(input, scales, output, channels, hw);
}
