#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

constexpr int kBlockSize = 256;

// v2: 改成“对半收缩”归约（s = blockSize/2 -> 1）
// 访问模式更规整，通常更利于 shared memory 访问。
template <int blockSize>
__device__ float block_reduce_max_v2(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = blockSize >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

template <int blockSize>
__device__ float block_reduce_sum_v2(float value, float* smem, int tid)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = blockSize >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

template <int blockSize>
__global__ void softmax_v2_kernel(const float* __restrict__ input,
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

    float thread_max = -1.0e20f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }
    const float row_max = block_reduce_max_v2<blockSize>(thread_max, smem, tid);

    float thread_sum = 0.0f;
    for (int c = tid; c < cols; c += blockSize)
    {
        thread_sum += expf(input[row_offset + c] - row_max);
    }
    const float row_sum = block_reduce_sum_v2<blockSize>(thread_sum, smem, tid);

    for (int c = tid; c < cols; c += blockSize)
    {
        output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
    }
}

} // namespace

// v2: 继续沿用 reduce 的 v2 思路，重点优化 shared-memory 归约访问模式。
//
// 建议改动：
// 1) 采用“对半归约”写法：for (s = blockDim.x / 2; s > 0; s >>= 1)。
// 2) 保证线程访问 smem[tid] / smem[tid + s] 的模式连续，降低 bank conflict 风险。
// 3) 两次归约（max/sum）都统一成这一套模板，减少维护成本。
//
// 为什么做 v2：
// - softmax 的瓶颈之一就是两次归约；把这两次都做顺，收益会比较稳定。
// - 访问模式变规整后，后续展开/warp 化会更容易。
//
// TODO(你手写练习):
// - 在 v1 上改归约循环与 shared memory 访问模式
// - 保持接口与 benchmark 逻辑不变，只动 kernel 内核
void softmax_v2(const float* input, float* output, int rows, int cols)
{
    softmax_v2_kernel<kBlockSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
