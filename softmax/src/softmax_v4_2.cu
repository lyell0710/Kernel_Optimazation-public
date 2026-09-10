#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

constexpr int kBlockSize = 256;
constexpr int kPackSize = 4;

struct MaxOp
{
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp
{
    __device__ float operator()(float a, float b) const { return a + b; }
};

template <typename Op>
__device__ float warp_reduce(float value, Op op)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value = op(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

// v4.2: 降低优化程度，接近 cuBLAS 的保守做法
// 主要改动：
// 1. 每线程只处理 2 个元素（而不是 4 个）
// 2. 用完整的 block-level reduction（不用 warp-level shuffle）
// 3. shared memory 用标量而不是向量
template <int blockSize, typename Op>
__device__ float block_reduce_v4_2(float value, float* smem, int tid, Op op)
{
    smem[tid] = value;
    __syncthreads();

    // 完整的 block 级归约，不做 warp 化优化
    for (int stride = blockSize >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = op(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

template <int blockSize, int packSize>
__global__ void softmax_v4_2_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int rows,
                                    int cols)
{
    __shared__ float smem[blockSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows)
    {
        return;
    }
    const int row_offset = row * cols;

    // v4.2: 每线程处理 2 个元素（而不是 4 个）
    float thread_max = -1.0e20f;
    for (int c = tid * 2; c < cols; c += blockSize * 2)
    {
        if (c < cols)
            thread_max = fmaxf(thread_max, input[row_offset + c]);
        if (c + 1 < cols)
            thread_max = fmaxf(thread_max, input[row_offset + c + 1]);
    }
    const float row_max = block_reduce_v4_2<blockSize>(thread_max, smem, tid, MaxOp{});

    float thread_sum = 0.0f;
    for (int c = tid * 2; c < cols; c += blockSize * 2)
    {
        if (c < cols)
            thread_sum += expf(input[row_offset + c] - row_max);
        if (c + 1 < cols)
            thread_sum += expf(input[row_offset + c + 1] - row_max);
    }
    const float row_sum = block_reduce_v4_2<blockSize>(thread_sum, smem, tid, SumOp{});

    for (int c = tid * 2; c < cols; c += blockSize * 2)
    {
        if (c < cols)
            output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
        if (c + 1 < cols)
            output[row_offset + c + 1] = expf(input[row_offset + c + 1] - row_max) / row_sum;
    }
}

} // namespace

void softmax_v4_2(const float* input, float* output, int rows, int cols)
{
    softmax_v4_2_kernel<kBlockSize, 2><<<rows, kBlockSize>>>(input, output, rows, cols);
}
