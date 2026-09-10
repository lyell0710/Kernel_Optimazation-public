#include "gemv_common.h"

namespace
{
constexpr int kBlockSize = 256;
constexpr int kVecWidth = 4;

template <int blockSize, int vecWidth>
__global__ void gemv_v2_kernel(const float* __restrict__ mat,
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

    // float4 向量化访存：每个线程一次加载 4 个 float，提升 memory coalescing 效率
    // 同一 warp 的 32 个线程各加载 4 个 float = 128 个 float = 512 bytes，单次 load 对齐 128-bit
    float sum = 0.0f;
    const int row_offset = row * cols;
    const int aligned_cols = (cols / vecWidth) * vecWidth;

    for (int c = tid * vecWidth; c < aligned_cols; c += blockSize * vecWidth)
    {
        float4 a = reinterpret_cast<const float4*>(mat + row_offset + c)[0];
        float4 x = reinterpret_cast<const float4*>(vec + c)[0];
        sum += a.x * x.x + a.y * x.y + a.z * x.z + a.w * x.w;
    }

    // 处理尾部不对齐的元素
    for (int c = aligned_cols + tid; c < cols; c += blockSize)
    {
        sum += mat[row_offset + c] * vec[c];
    }

    smem[tid] = sum;
    __syncthreads();

    // Block-level 二分归约
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

void gemv_v2(const float* mat, const float* vec, float* out, int rows, int cols)
{
    gemv_v2_kernel<kBlockSize, kVecWidth><<<rows, kBlockSize>>>(mat, vec, out, rows, cols);
}
