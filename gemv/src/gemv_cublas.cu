#include "gemv_common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

namespace
{

cublasHandle_t& get_handle()
{
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr)
    {
        cublasStatus_t st = cublasCreate(&handle);
        if (st != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cublasCreate failed: " << st << std::endl;
        }
    }
    return handle;
}

} // namespace

// gemv_cublas: 调用 NVIDIA 真正的 cublasSgemv API。
//
// 我们的数据布局是 row-major：mat[rows][cols], vec[cols], out[rows]
// 等价计算：out = mat * vec
//
// cuBLAS 用 column-major：把 row-major M[rows][cols] 当作 column-major
// 看就是 M_cm = M^T，形状是 cols × rows。
// 所以要算 out = M * vec 实际上等于 out = (M_cm)^T * vec：
//   - op = CUBLAS_OP_T
//   - m = cols (M_cm 的行数)
//   - n = rows (M_cm 的列数)
//   - lda = cols (M_cm 的 leading dim)
void gemv_cublas(const float* mat, const float* vec, float* out, int rows, int cols)
{
    cublasHandle_t handle = get_handle();
    if (handle == nullptr)
    {
        return;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemv(handle,
                CUBLAS_OP_T,
                cols,   // m: M_cm 的行数
                rows,   // n: M_cm 的列数
                &alpha,
                mat,
                cols,   // lda
                vec, 1,
                &beta,
                out, 1);
}
