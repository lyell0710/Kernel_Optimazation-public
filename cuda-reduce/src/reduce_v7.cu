#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;

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

template <int blockSize> __global__ void reduce_v7_kernel(const float* d_in, float* d_out, int n)
{
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = gtid; i < n; i += total_threads)
    {
        sum += d_in[i];
    }

    smem[tid] = sum;
    __syncthreads();

    BlockSharedMemReduce<blockSize>(smem, tid);
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v7(const float* data, float* output, int n)
{
    if (n <= 0)
    {
        cudaMemset(output, 0, sizeof(float));
        return;
    }

    // 计时口径:SM 数是设备常量,缓存一次即可。原实现每次调用都查 ——
    // cudaGetDeviceProperties 是重量级 runtime 调用(实测在 driver 570 上单次
    // 可达毫秒级),它落在计时区内会把被测的规约本身完全淹没。
    // 只有 v6/v7 有这个调用,v1-v5 没有 —— 这正是"v6/v7 慢一个数量级"的元凶。
    static int s_sm_count = 0;
    if (s_sm_count == 0)
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        s_sm_count = prop.multiProcessorCount;
    }
    cudaDeviceProp prop{};
    prop.multiProcessorCount = s_sm_count;

    int grid1 = (n + kBlockSize - 1) / kBlockSize;
    int max_grid = prop.multiProcessorCount * 8;
    if (max_grid < 1)
    {
        max_grid = 1;
    }
    if (grid1 > max_grid)
    {
        grid1 = max_grid;
    }
    if (grid1 < 1)
    {
        grid1 = 1;
    }

    // 计时口径:中间数组一次性分配、按容量复用,与 CUB 对照臂一致
    // (reduce_cub.cu 的 g_temp 同款做法)。原实现每次调用都 cudaMalloc + cudaFree,
    // 落在计时区内 —— 测的是分配器而不是规约本身。
    static float* s_partial = nullptr;
    static int    s_cap = 0;
    if (grid1 > s_cap)
    {
        if (s_partial) { cudaFree(s_partial); s_partial = nullptr; }
        cudaMalloc(&s_partial, grid1 * sizeof(float));
        s_cap = grid1;
    }
    float* d_partial = s_partial;

    reduce_v7_kernel<kBlockSize><<<grid1, kBlockSize>>>(data, d_partial, n);

    if (grid1 > 1)
    {
        reduce_v7_kernel<kBlockSize><<<1, kBlockSize>>>(d_partial, output, grid1);
    }
    else
    {
        cudaMemcpy(output, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // (不再 cudaFree:缓冲区跨调用复用,见上方计时口径说明)
}
