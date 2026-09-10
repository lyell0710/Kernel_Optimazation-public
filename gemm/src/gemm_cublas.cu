#include <cublas_v2.h>
#include "gemm_common.h"
// ============================================================================
// 对照物:真 cuBLAS(cublasGemmEx,CUBLAS_COMPUTE_32F + DEFAULT_TENSOR_OP),
// 与版本梯同口径:fp16 输入 / fp32 计算。
// 为什么强调「真」:EXP-K01（四 kernel 4090 重基准）§5 的 softmax 勘误(对照物系自写 kernel)之后
// 立的规矩——凡「vs cublas」必须先验调用点确系真实库调用,本文件即验真对象。
// 行主序技巧:cuBLAS 只认列主序;C_row = A_row·B_row 等价于列主序视角的
// C^T = B^T·A^T,故以 (N,M,K) 调用并交换 A/B 指针,零转置零拷贝。
// 性能锚:0.884±0.004 ms = 155.4±0.62 TFLOPS(4096³,4090,3 轮,EXP-K02（CUDA Tensor Core GEMM 版本梯）)。
// ============================================================================
// handle 懒建后进程内复用:cublasCreate 含上下文/workspace 初始化(百 ms 级),
// 若每次调用重建,会把初始化开销计入被测时延,对照失真。
static cublasHandle_t handle = nullptr;
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K) {
    if (!handle) cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
