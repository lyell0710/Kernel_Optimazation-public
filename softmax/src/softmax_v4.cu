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
    // warp 内使用 shuffle 交换寄存器数据，不需要 __syncthreads()。
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value = op(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

template <int blockSize, typename Op>
__device__ float block_reduce_v4(float value, float* smem, int tid, Op op)
{
    smem[tid] = value;
    __syncthreads();

    // 先做 block 级归约直到只剩一个 warp。
    for (int stride = blockSize >> 1; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = op(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // 尾部归约切换到 warp 级，减少同步开销。
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

template <int blockSize, int packSize>
__global__ void softmax_v4_kernel(const float* __restrict__ input,
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
            const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
            thread_max = fmaxf(thread_max, v.x);
            thread_max = fmaxf(thread_max, v.y);
            thread_max = fmaxf(thread_max, v.z);
            thread_max = fmaxf(thread_max, v.w);
        }
        else
        {
            for (int k = 0; k < packSize && (c + k) < cols; ++k)
            {
                thread_max = fmaxf(thread_max, input[row_offset + c + k]);
            }
        }
    }
    const float row_max = block_reduce_v4<blockSize>(thread_max, smem, tid, MaxOp{});

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
    const float row_sum = block_reduce_v4<blockSize>(thread_sum, smem, tid, SumOp{});

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

// v4: 归约尾部 warp 化（参考 reduce 的 v4/v5 路线 + CUDATutorial WarpSoftmax 思路）。
//
// 建议改动：
// 1) block 归约做到 s > 32 时仍用 __syncthreads()。
// 2) 最后一个 warp 改为 warp-level reduce（__shfl_xor_sync 或手动 warp 展开）。
// 3) max 与 sum 都复用同一套 WarpReduce helper（op 参数化）。
//
// 为什么做 v4：
// - softmax 每行会做两次归约，尾部同步开销被放大。
// - warp 内通信不需要整块同步，通常能继续压低时延。
//
// TODO(你手写练习):
// - 从 v3 复制一份，先只替换 max 归约尾部
// - 验证正确后，再替换 sum 归约尾部
// - 最后把公共逻辑抽到模板 helper，避免两份重复代码
void softmax_v4(const float* input, float* output, int rows, int cols)
{
    softmax_v4_kernel<kBlockSize, kPackSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
