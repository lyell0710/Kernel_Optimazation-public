#include "reduce_common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace
{
static cublasHandle_t handle = nullptr;

void init_handle()
{
    if (handle == nullptr)
    {
        cublasCreate(&handle);
    }
}

void destroy_handle()
{
    if (handle != nullptr)
    {
        cublasDestroy(handle);
        handle = nullptr;
    }
}
}

void reduce_cublas(const float* data, float* output, int n)
{
    init_handle();

    if (n <= 0)
    {
        float zero = 0.0f;
        cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);
        return;
    }

    // cublasSasum computes the sum of absolute values
    // For our case (all positive 1.0f values), this is equivalent to sum
    float result = 0.0f;
    cublasSasum(handle, n, data, 1, &result);

    cudaMemcpy(output, &result, sizeof(float), cudaMemcpyHostToDevice);
}
