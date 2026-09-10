#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 256;

template <int blockSize>
__global__ void gemv_v0_kernel(const float* mat, const float* vec, float* out, int rows, int cols)
{
    __shared__ float smem[blockSize];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
    {
        return;
    }

    float sum = 0.0f;
    const int row_offset = row * cols;
    for (int c = tid; c < cols; c += blockSize)
    {
        sum += mat[row_offset + c] * vec[c];
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

void gemv_v0(const float* mat, const float* vec, float* out, int rows, int cols)
{
    gemv_v0_kernel<kBlockSize><<<rows, kBlockSize>>>(mat, vec, out, rows, cols);
}
