#include "reduce_common.h"
#include <cuda_runtime.h>

__global__ void reduce_baseline_kernel(const float* data, float* output, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += data[i];
    }
    *output = sum;
}

void reduce_baseline(const float* data, float* output, int n)
{
    reduce_baseline_kernel<<<1, 1>>>(data, output, n);
}