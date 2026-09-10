#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 256;

template <int blockSize>
__global__ void gemv_v1_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    __shared__ float smem[blockSize];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    // 每个线程一次处理两个元素：降低循环迭代和索引计算开销
    // 步长 blockSize*2：同一 warp 的相邻线程访存地址对齐 64-bit，提升 memory coalescing
    float sum = 0.0f;
    const int row_offset = row * cols;
    for (int c = tid * 2; c < cols; c += blockSize * 2)
    {
        sum += mat[row_offset + c] * vec[c];
        if (c + 1 < cols)
        {
            sum += mat[row_offset + c + 1] * vec[c + 1];
        }
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockSize >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out[row] = smem[0];
    }
}
} // namespace

void gemv_v1(const float* mat, const float* vec, float* out, int rows, int cols)
{
    gemv_v1_kernel<kBlockSize><<<rows, kBlockSize>>>(mat, vec, out, rows, cols);
}
