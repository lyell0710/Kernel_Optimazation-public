#include "gemv_common.h"

namespace
{

__global__ void gemv_baseline_kernel(const float* mat, const float* vec, float* out, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0)
    {
        return;
    }
    float sum = 0.0f;
    int row_offset = row * cols;
    for (int c = 0; c < cols; ++c)
    {
        sum += mat[row_offset + c] * vec[c];
    }
    out[row] = sum;
}

} // namespace

void cpu_gemv(const float* mat, const float* vec, float* out, int rows, int cols)
{
    for (int r = 0; r < rows; ++r)
    {
        float sum = 0.0f;
        const int row_offset = r * cols;
        for (int c = 0; c < cols; ++c)
        {
            sum += mat[row_offset + c] * vec[c];
        }
        out[r] = sum;
    }
}

void gemv_baseline(const float* mat, const float* vec, float* out, int rows, int cols)
{
    gemv_baseline_kernel<<<rows, 1>>>(mat, vec, out, rows, cols);
}
