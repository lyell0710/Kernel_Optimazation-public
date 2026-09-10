#include "gemv_common.h"

namespace
{
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__inline__ __device__ float warp_reduce_sum(float v)
{
    // Warp shuffle 折半归约：避免 block-level __syncthreads() 的开销
    // 所有 32 个 lane 原子性地交换数据，无竞争条件
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// 一行交给一个 warp，减少 block 级同步开销
// blockDim = (32, 4)：32 个 lane + 4 个 warp，每个 warp 处理 1 行
// 同一 warp 的 32 个 lane 协作读向量 vec，memory coalescing 对齐 128-bit
__global__ void gemv_v3_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    int lane = threadIdx.x;
    int warp_id_in_block = threadIdx.y;
    int row = blockIdx.x * blockDim.y + warp_id_in_block;
    if (row >= rows)
    {
        return;
    }

    const int row_offset = row * cols;
    float sum = 0.0f;
    // 32 个 lane 并行读向量，步长 32，提升 global memory bandwidth 利用率
    for (int c = lane; c < cols; c += kWarpSize)
    {
        sum += mat[row_offset + c] * vec[c];
    }
    // Warp-level 归约，无 __syncthreads() 开销
    sum = warp_reduce_sum(sum);
    if (lane == 0)
    {
        out[row] = sum;
    }
}
} // namespace

void gemv_v3(const float* mat, const float* vec, float* out, int rows, int cols)
{
    dim3 block(kWarpSize, kWarpsPerBlock);
    dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gemv_v3_kernel<<<grid, block>>>(mat, vec, out, rows, cols);
}
