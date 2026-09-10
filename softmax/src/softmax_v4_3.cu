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

template <int blockSize, typename Op>
__device__ float block_reduce_v4_3(float value, float* smem, int tid, Op op)
{
    smem[tid] = value;
    __syncthreads();

    for (int stride = blockSize >> 1; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = op(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        float warp_value = smem[tid];
        if (blockSize >= 64)
        {
            warp_value = op(warp_value, smem[tid + 32]);
        }
        warp_value = warp_reduce(warp_value, op);
        if (tid == 0)
        {
            smem[0] = warp_value;
        }
    }
    __syncthreads();
    return smem[0];
}

// v4.3: float4 主循环 + 标量 tail，专门为非对齐 cols 设计。
//
// 与 v4 的关键区别：
// - v4 假设 cols 是 (blockSize * packSize) 的倍数，每次循环内部仍要分支判断
// - v4.3 把循环显式拆成两段：
//     第一段 main_cols (向下取整到 blockSize*packSize) 用 float4 全速跑
//     第二段 tail 用标量循环处理
// - 主循环里没有任何分支，编译器可以充分流水线
template <int blockSize, int packSize>
__global__ void softmax_v4_3_kernel(const float* __restrict__ input,
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

    // 向下取整到 packSize 的倍数（4 的倍数），保证 float4 加载安全。
    // 注意：这里不需要对齐到 blockSize*packSize，因为线程的 stride 循环
    // 本身就能让每个线程领到 0/1/多份 float4 数据。
    const int main_cols = (cols / packSize) * packSize;

    // ===== Phase 1: 求 max =====
    float thread_max = -1.0e20f;

    // 主循环：纯 float4，无分支
    for (int c = tid * packSize; c < main_cols; c += blockSize * packSize)
    {
        const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }

    // Tail 循环：标量处理剩余部分（cols - main_cols 个元素）
    for (int c = main_cols + tid; c < cols; c += blockSize)
    {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }

    const float row_max = block_reduce_v4_3<blockSize>(thread_max, smem, tid, MaxOp{});

    // ===== Phase 2: 求 sum =====
    float thread_sum = 0.0f;

    for (int c = tid * packSize; c < main_cols; c += blockSize * packSize)
    {
        const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
        thread_sum += expf(v.x - row_max);
        thread_sum += expf(v.y - row_max);
        thread_sum += expf(v.z - row_max);
        thread_sum += expf(v.w - row_max);
    }

    for (int c = main_cols + tid; c < cols; c += blockSize)
    {
        thread_sum += expf(input[row_offset + c] - row_max);
    }

    const float row_sum = block_reduce_v4_3<blockSize>(thread_sum, smem, tid, SumOp{});

    // ===== Phase 3: 写回 =====
    for (int c = tid * packSize; c < main_cols; c += blockSize * packSize)
    {
        const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
        float4 out_v;
        out_v.x = expf(v.x - row_max) / row_sum;
        out_v.y = expf(v.y - row_max) / row_sum;
        out_v.z = expf(v.z - row_max) / row_sum;
        out_v.w = expf(v.w - row_max) / row_sum;
        reinterpret_cast<float4*>(output + row_offset + c)[0] = out_v;
    }

    for (int c = main_cols + tid; c < cols; c += blockSize)
    {
        output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
    }
}

} // namespace

void softmax_v4_3(const float* input, float* output, int rows, int cols)
{
    softmax_v4_3_kernel<kBlockSize, kPackSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
