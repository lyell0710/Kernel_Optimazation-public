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

// v4.4: 故意制造 bank conflict 的归约
//
// 关键退化点（相对 v4.3）：
// 1) 归约 stride 从 1 起步，逐步增大（v0 / v1 的老式 stride 模式）
// 2) 条件改成 (tid % (stride*2)) == 0，让"参与归约的线程"在 warp 内稀疏分布
// 3) 没有 warp shuffle 尾部优化，全程 __syncthreads()
//
// 为什么会慢：
// - stride=1: 线程 0/2/4/... 访问 smem[0,1]+[1,2]，相邻线程访问相邻地址，触发 2-way conflict
// - stride=2: 线程 0/4/8/... 访问 smem[0,2]+[2,4]，相邻"工作线程"间隔 4 个 bank，触发 2-way conflict
// - stride=4: 同上，4-way conflict
// - ...
// - 每一轮归约的有效线程都稀疏分布在 warp 内 → 多个线程同时访问同一个 bank
// - 加上每轮都用 __syncthreads()（256 线程全等），同步开销也更大
//
// 这是一个"看起来合理但实际很慢"的反面教材：
// - main 循环还是 float4（带宽利用率高）
// - 但归约阶段被退化拖累
template <typename Op>
__device__ float bad_block_reduce(float value, float* smem, int tid, int blockSize, Op op)
{
    smem[tid] = value;
    __syncthreads();

    // 故意用 stride=1 起步的递增模式（v0 风格）
    for (int stride = 1; stride < blockSize; stride <<= 1)
    {
        // 用 modulo 判断哪些线程参与（参与线程在 warp 内稀疏分布）
        if ((tid % (stride << 1)) == 0 && (tid + stride) < blockSize)
        {
            smem[tid] = op(smem[tid], smem[tid + stride]);
        }
        __syncthreads();  // 全程 block 同步，不用 warp shuffle
    }

    return smem[0];
}

template <int blockSize, int packSize>
__global__ void softmax_v4_4_kernel(const float* __restrict__ input,
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

    const int main_cols = (cols / packSize) * packSize;

    // ===== Phase 1: max =====
    float thread_max = -1.0e20f;

    for (int c = tid * packSize; c < main_cols; c += blockSize * packSize)
    {
        const float4 v = reinterpret_cast<const float4*>(input + row_offset + c)[0];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }
    for (int c = main_cols + tid; c < cols; c += blockSize)
    {
        thread_max = fmaxf(thread_max, input[row_offset + c]);
    }
    const float row_max = bad_block_reduce(thread_max, smem, tid, blockSize, MaxOp{});

    // ===== Phase 2: sum =====
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
    const float row_sum = bad_block_reduce(thread_sum, smem, tid, blockSize, SumOp{});

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

void softmax_v4_4(const float* input, float* output, int rows, int cols)
{
    softmax_v4_4_kernel<kBlockSize, kPackSize><<<rows, kBlockSize>>>(input, output, rows, cols);
}
