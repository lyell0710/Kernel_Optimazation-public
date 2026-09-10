#include "softmax_common.h"
#include <cuda_runtime.h>

namespace
{

// baseline: 一个 block 负责一行，单线程串行做 max/sum/normalize。
// 目的不是快，而是建立“稳定正确”的对照版本。
__global__ void softmax_baseline_kernel(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    if (threadIdx.x == 0)
    {
        const int row_offset = row * cols;
        float row_max = input[row_offset];
        for (int c = 1; c < cols; ++c)
        {
            row_max = fmaxf(row_max, input[row_offset + c]);
        }

        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            row_sum += expf(input[row_offset + c] - row_max);
        }

        for (int c = 0; c < cols; ++c)
        {
            output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
        }
    }
}

} // namespace

void cpu_softmax(const float* input, float* output, int rows, int cols)
{
    for (int r = 0; r < rows; ++r)
    {
        const int row_offset = r * cols;
        float row_max = input[row_offset];
        for (int c = 1; c < cols; ++c)
        {
            row_max = (input[row_offset + c] > row_max) ? input[row_offset + c] : row_max;
        }

        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            row_sum += expf(input[row_offset + c] - row_max);
        }

        for (int c = 0; c < cols; ++c)
        {
            output[row_offset + c] = expf(input[row_offset + c] - row_max) / row_sum;
        }
    }
}

void softmax_baseline(const float* input, float* output, int rows, int cols)
{
    softmax_baseline_kernel<<<rows, 1>>>(input, output, rows, cols);
}
