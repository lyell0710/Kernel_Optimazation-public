#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;
// v4 的最后一个 warp 归约 helper。
// 当前写法采用手动展开的方式，避免在只剩一个 warp 时继续使用 block 级 __syncthreads()。
// 这样可以减少 reduction 尾部阶段的同步开销。
template <int blockSize> __device__ void WarpSharedMemReduce(volatile float* smem, int tid)
{

    float x = smem[tid];

    if (blockSize >= 64)
    {
        x += smem[tid + 32];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
    }

    x += smem[tid + 16];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 8];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 4];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 2];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();

    x += smem[tid + 1];
    __syncwarp();
    smem[tid] = x;
    __syncwarp();
}

template <int blockSize> __global__ void reduce_v4_kernel(const float* d_in, float* d_out, int n)
{
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;

    int gtid = blockIdx.x * (2 * blockSize) + threadIdx.x;

    smem[tid] = 0.0f;

    // v3: 让每个线程多干活，处理两个元素
    if (gtid < n)
    {
        smem[tid] = d_in[gtid];
    }
    if (gtid + blockSize < n)
    {
        smem[tid] += d_in[gtid + blockSize];
    }

    __syncthreads();

    // v4 的核心优化：当 block 内归约走到最后只剩一个 warp 时，
    // 不再继续使用 __syncthreads() 做整 block 同步。
    // 前面 s > 32 的阶段仍然采用 block-level reduction，
    // 但最后 32 个线程改用 warp-level reduction 单独完成，
    // 这样可以减少尾部阶段不必要的同步开销。
    for (unsigned int index = blockSize / 2; index > 32; index >>= 1)
    {
        if (tid < index) // tid<index: 当前线程ID小于index，说明当前线程ID在index之前，需要相加
        {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        WarpSharedMemReduce<blockSize>(smem, tid);
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v4(const float* data, float* output, int n)
{
    const float* d_current = data;
    float* d_next = nullptr;

    // 标记 d_current 现在是不是“临时申请出来的显存”。
    // 第一轮时 d_current = data，不是临时的，不能 free。
    int current_n = n;

    bool current_is_temp = false;

    while (current_n > 1)
    {

        int grid = (current_n + (2 * kBlockSize - 1)) / (kBlockSize * 2); // 向上取整

        cudaMalloc(&d_next, grid * sizeof(float));

        reduce_v4_kernel<kBlockSize><<<grid, kBlockSize>>>(d_current, d_next, current_n);

        if (current_is_temp)
        {
            cudaFree((void*)d_current);
        }
        d_current = d_next;

        d_next = nullptr;

        current_n = grid;

        current_is_temp = true;
    }
    cudaMemcpy(output, d_current, sizeof(float), cudaMemcpyDeviceToDevice);
    if (current_is_temp)
    {
        cudaFree((void*)d_current);
    }
}