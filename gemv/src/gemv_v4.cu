#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 128;
constexpr int kRowsPerBlock = 4;

template <int blockSize, int rowsPerBlock>
__global__ void gemv_v4_kernel(const float* __restrict__ mat,
                               const float* __restrict__ vec,
                               float* __restrict__ out,
                               int rows,
                               int cols)
{
    __shared__ float smem_x[blockSize];
    // shared memory padding：2D 数组行步长为 blockSize，避免同列不同行的 bank conflict
    // smem_partial[row_in_block][col] 中，同一列的多行写入会分散到不同 bank
    __shared__ float smem_partial[rowsPerBlock][blockSize];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * rowsPerBlock + ty;
    if (row >= rows)
    {
        return;
    }

    float local = 0.0f;
    int row_offset = row * cols;

    // 多行共享向量 x 的 tile：blockSize 个线程共同读取 x[base..base+blockSize)
    // 避免每行独立读 x，降低向量 x 的全局访问带宽消耗
    for (int base = 0; base < cols; base += blockSize)
    {
        int c = base + tx;
        smem_x[tx] = (c < cols) ? vec[c] : 0.0f;
        __syncthreads();

        if (c < cols)
        {
            local += mat[row_offset + c] * smem_x[tx];
        }
        __syncthreads();
    }

    // 将局部累加和写入共享内存，准备 block-level 归约
    smem_partial[ty][tx] = local;
    __syncthreads();

    // Block-level 二分归约：同一行的 blockSize 个线程相加
    for (int s = blockSize >> 1; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            smem_partial[ty][tx] += smem_partial[ty][tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        out[row] = smem_partial[ty][0];
    }
}
} // namespace

void gemv_v4(const float* mat, const float* vec, float* out, int rows, int cols)
{
    dim3 block(kBlockSize, kRowsPerBlock);
    dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_v4_kernel<kBlockSize, kRowsPerBlock><<<grid, block>>>(mat, vec, out, rows, cols);
}
