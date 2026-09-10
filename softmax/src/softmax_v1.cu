#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

constexpr int kBlockSize = 256;

// v1: 在 v0 的基础上，保留同样的交错归约框架，
// 但把 % 判断改成位运算判断，减少分支判断开销。
template <int blockSize>
__device__ float block_reduce_max_v1(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = 1; stride < blockSize; stride <<= 1)
    {
        if (((tid & ((stride << 1) - 1)) == 0) && (tid + stride) < blockSize)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

template <int blockSize>
__device__ float block_reduce_sum_v1(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = 1; stride < blockSize; stride <<= 1)
    {
        if (((tid & ((stride << 1) - 1)) == 0) && (tid + stride) < blockSize)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template <int blockSize>
__global__ void softmax_v1_kernel(const float* __restrict__ input,
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

    // 把行起始地址缓存下来，避免反复算 row * cols。
    const int row_offset = row * cols;

    float thread_max = -1.0e20f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }
    const float row_max = block_reduce_max_v1<blockSize>(thread_max, smem, tid);

    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_sum += expf(input[row_offset + c] - row_max);
    }
    const float row_sum = block_reduce_sum_v1<blockSize>(thread_sum, smem, tid);

    for (int c = tid; c < cols; c += blockSize)
    {
        output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
    }
}

} // namespace

// v1: 在 v0 的基础上，先做“低风险小优化”。
// 你可以参考 reduce 的 v0 -> v1 思路：先替换掉高开销细节，不改整体结构。
//
// 建议改动：
// 1) 归约阶段里把 % 运算/复杂判断改成位运算或更简单的分支（减少 ALU 开销）。
// 2) 能用 __restrict__ 的输入输出指针先加上（帮助编译器优化）。
// 3) 把重复计算的 row_offset/cache 下来，减少地址重算。
//
// 为什么做 v1：
// - 这一步主要训练“先赚确定性收益”的习惯。
// - 不改变算法结构，便于与 v0 做 A/B 对照，问题也更好定位。
//
// TODO(你手写练习):
// - 复制 v0 的 kernel 到 v1
// - 只改上述细节，保证数值与 v0 对齐
void softmax_v1(const float* input, float* output, int rows, int cols)
{
    softmax_v1_kernel<kBlockSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
