#include "reduce_common.h"
#include <cuda_runtime.h>
#include <iostream>

namespace
{

constexpr int kBlockSize = 256;

template <int blockSize>

__global__ void reduce_v0_kernel(const float* d_in, float* d_out, int n)
{
    __shared__ float smem[blockSize];

    int tid = threadIdx.x;

    int gtid = blockIdx.x * blockSize + threadIdx.x;

    smem[tid] = (gtid < n) ? d_in[gtid] : 0.0f;

    __syncthreads();

    for (int index = 1; index < blockSize; index *= 2)
    {
        if (tid % (2 * index) == 0 && tid + index < blockSize)
        {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = smem[0];
    }
}

} // namespace

void reduce_v0(const float* data, float* output, int n)
{
    const float* d_current = data;
    float* d_next = nullptr;

    // 标记 d_current 现在是不是“临时申请出来的显存”。
    // 第一轮时 d_current = data，不是临时的，不能 free。
    int current_n = n;

    bool current_is_temp = false;

    while (current_n > 1)
    {

        int grid = (current_n + kBlockSize - 1) / kBlockSize; // 向上取整

        cudaMalloc(&d_next, grid * sizeof(float));

        reduce_v0_kernel<kBlockSize><<<grid, kBlockSize>>>(d_current, d_next, current_n);

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