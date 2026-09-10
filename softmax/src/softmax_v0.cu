#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

constexpr int kBlockSize = 256;

// v0 的 block 内 max 归约：
// 采用“交错配对”(stride = 1,2,4...) 的经典写法，便于理解归约过程。
template <int blockSize>
__device__ float block_reduce_max_v0(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = 1; stride < blockSize; stride <<= 1)
    {
        if ((tid % (stride << 1)) == 0 && (tid + stride) < blockSize)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

// v0 的 block 内 sum 归约：结构与 max 一致，只是操作变为加法。
template <int blockSize>
__device__ float block_reduce_sum_v0(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = 1; stride < blockSize; stride <<= 1)
    {
        if ((tid % (stride << 1)) == 0 && (tid + stride) < blockSize)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

// v0：第一版并行 softmax。
// - 一个 block 处理一行
// - 每个线程以 stride 方式处理多个列元素
// - 行内做两次 block reduction（max + sum）
template <int blockSize>
__global__ void softmax_v0_kernel(const float* input, float* output, int rows, int cols)
{
    __shared__ float smem[blockSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    const int row_offset = row * cols;

    // 1) 每个线程先算 thread-local max
    float thread_max = -1.0e20f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }
    const float row_max = block_reduce_max_v0<blockSize>(thread_max, smem, tid);

    // 2) 计算 thread-local sum(exp(x - row_max))
    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_sum += expf(input[row_offset + c] - row_max);
    }
    const float row_sum = block_reduce_sum_v0<blockSize>(thread_sum, smem, tid);

    // 3) 回写归一化结果
    for (int c = tid; c < cols; c += blockSize)
    {
        output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
    }
}

} // namespace

// v0: 从 baseline 升级到“并行化第一版”时，建议你这样改：
// 1) 一个 block 负责一行（blockIdx.x -> row）。
// 2) 每个线程按 stride 读取多个列元素，先做 thread-local max。
// 3) 用 shared memory 做 block reduce 得到 row_max。
// 4) 再按同样方式做 exp(x-row_max) 的 block reduce 得到 row_sum。
// 5) 最后每个线程写回自己负责的列。
//
// 为什么先做这个版本：
// - 先把“串行 softmax”拆成“并行 max + 并行 sum + 并行写回”的基本框架。
// - 先不追求极致性能，重点是把数据流和同步点理顺。
//
// TODO(你手写练习):
// - 新建 softmax_v0_kernel
// - 用 shared memory + __syncthreads() 完成两次 block 归约
// - 把下面临时调用改为 softmax_v0_kernel<<<rows, block_size>>>
void softmax_v0(const float* input, float* output, int rows, int cols)
{
    softmax_v0_kernel<kBlockSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
