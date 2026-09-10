#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

constexpr int kBlockSize = 256;
constexpr int kPackSize = 4;

template <int blockSize>
__device__ float block_reduce_max_v3(float value, float* smem, int tid)
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
__device__ float block_reduce_sum_v3(float value, float* smem, int tid)
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

// v3: 在 v2 的基础上引入向量化访问（float4）。
// 思路：每次处理 4 个连续元素，降低访存事务与指令条数。
template <int blockSize, int packSize>
__global__ void softmax_v3_kernel(const float* __restrict__ input,
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
    for (int c = tid * packSize; c < cols; c += blockSize * packSize)
    {
        if (c + packSize - 1 < cols)
        {
            // 连续 4 元素向量读取。
            const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
            thread_max = fmaxf(thread_max, v.x);
            thread_max = fmaxf(thread_max, v.y);
            thread_max = fmaxf(thread_max, v.z);
            thread_max = fmaxf(thread_max, v.w);
        }
        else
        {
            // 尾部不足 4 个时回退标量路径。
            for (int k = 0; k < packSize && (c + k) < cols; ++k)
            {
                thread_max = fmaxf(thread_max, input[row_offset + c + k]);
            }
        }
    }
    const float row_max = block_reduce_max_v3<blockSize>(thread_max, smem, tid);

    float thread_sum = 0.0f;
    for (int c = tid * packSize; c < cols; c += blockSize * packSize)
    {
        if (c + packSize - 1 < cols)
        {
            const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
            thread_sum += expf(v.x - row_max);
            thread_sum += expf(v.y - row_max);
            thread_sum += expf(v.z - row_max);
            thread_sum += expf(v.w - row_max);
        }
        else
        {
            for (int k = 0; k < packSize && (c + k) < cols; ++k)
            {
                thread_sum += expf(input[row_offset + c + k] - row_max);
            }
        }
    }
    const float row_sum = block_reduce_sum_v3<blockSize>(thread_sum, smem, tid);

    for (int c = tid * packSize; c < cols; c += blockSize * packSize)
    {
        if (c + packSize - 1 < cols)
        {
            const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
            float4 out_v;
            out_v.x = expf(v.x - row_max) / row_sum;
            out_v.y = expf(v.y - row_max) / row_sum;
            out_v.z = expf(v.z - row_max) / row_sum;
            out_v.w = expf(v.w - row_max) / row_sum;
            reinterpret_cast<float4*>(output + row_offset + c)[0] = out_v;
        }
        else
        {
            for (int k = 0; k < packSize && (c + k) < cols; ++k)
            {
                output[row_offset + c + k] = expf(input[row_offset + c + k] - row_max) / row_sum;
            }
        }
    }
}

} // namespace

// v3: 引入“每线程多元素处理 + 向量化加载”的思路（对齐 CUDATutorial softmax）。
//
// 建议改动：
// 1) 每个线程一次处理多个列（例如 pack=2/4），降低指令与访存事务开销。
// 2) 使用对齐向量读写（float2/float4 或自定义 VectorType）。
// 3) 越界列填 -INF（max 阶段）或 0（sum 阶段），保证尾部安全。
//
// 为什么做 v3：
// - 当 cols 比较大时，单元素读写会让 global memory transaction 利用率偏低。
// - 向量化通常会带来比较直接的吞吐提升，是 softmax 常见关键优化点。
//
// TODO(你手写练习):
// - 在 v2 上加入 pack 与向量化 load/store
// - 先保证 pack=1 正确，再切换到 pack=2/4
void softmax_v3(const float* input, float* output, int rows, int cols)
{
    softmax_v3_kernel<kBlockSize, kPackSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
