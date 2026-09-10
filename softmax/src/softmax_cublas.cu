#include "softmax_common.h"
#include <cuda_runtime.h>
#include <cmath>

namespace
{

// Optimized softmax kernel (similar efficiency to v4 but using cuBLAS-like patterns)
// Each block handles one row with warp-level primitives
__global__ void softmax_cublas_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    int row_offset = row * cols;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    __shared__ float smax[32];
    __shared__ float ssum[32];

    // Initialize shared memory
    if (threadIdx.x < 32)
    {
        smax[threadIdx.x] = -1e9f;
        ssum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Step 1: Find row-wise max
    float row_max = -1e9f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        row_max = fmaxf(row_max, input[row_offset + i]);
    }

    // Warp-level reduction
    for (int mask = 16; mask >= 1; mask /= 2)
    {
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, mask));
    }

    // Block-level max reduction
    if (lane_id == 0)
    {
        smax[warp_id] = row_max;
    }
    __syncthreads();

    float global_max = smax[0];
    for (int i = 1; i < num_warps; ++i)
    {
        global_max = fmaxf(global_max, smax[i]);
    }
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float val = expf(input[row_offset + i] - global_max);
        output[row_offset + i] = val;
        local_sum += val;
    }

    // Warp-level sum reduction
    for (int mask = 16; mask >= 1; mask /= 2)
    {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
    }

    // Block-level sum reduction
    if (lane_id == 0)
    {
        ssum[warp_id] = local_sum;
    }
    __syncthreads();

    float row_sum = ssum[0];
    for (int i = 1; i < num_warps; ++i)
    {
        row_sum += ssum[i];
    }
    __syncthreads();

    // Step 3: Normalize by sum
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        output[row_offset + i] /= row_sum;
    }
}

} // namespace

void softmax_cublas(const float* input, float* output, int rows, int cols)
{
    softmax_cublas_kernel<<<rows, 256>>>(input, output, rows, cols);
}
