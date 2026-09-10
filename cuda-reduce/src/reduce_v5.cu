#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;
// v5: block 内归约循环展开（基于 v4 改写）
// 目标：减少 for 循环控制开销，保留 v4 的两元素加载与边界保护逻辑。
template <int blockSize> __device__ void BlockSharedMemReduce(float* seme, int tid)
{
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            seme[tid] += seme[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            seme[tid] += seme[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            seme[tid] += seme[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            seme[tid] += seme[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        volatile float* vshm = seme;
        if (blockDim.x >= 64)
        {
            vshm[tid] += vshm[tid + 32];
        }
        vshm[tid] += vshm[tid + 16];
        vshm[tid] += vshm[tid + 8];
        vshm[tid] += vshm[tid + 4];
        vshm[tid] += vshm[tid + 2];
        vshm[tid] += vshm[tid + 1];
    }
}

template <int blockSize> __global__ void reduce_v5_kernel(const float* d_in, float* d_out, int n)
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

    // v5 核心：展开版 block shared-memory reduction
    BlockSharedMemReduce<blockSize>(smem, tid);
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v5(const float* data, float* output, int n)
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

        reduce_v5_kernel<kBlockSize><<<grid, kBlockSize>>>(d_current, d_next, current_n);

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