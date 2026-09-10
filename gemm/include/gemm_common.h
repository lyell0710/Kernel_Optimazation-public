#pragma once
#include <cuda_fp16.h>
// ============================================================================
// GEMM 版本梯公共接口:C[M,N] = A[M,K] · B[K,N],全部行主序,
// fp16 存储 / fp32 累加(wmma 版在 accumulator fragment 内做 fp32 累加)。
// 接口契约(全版本一致,调用方 = main.cu bench):
//   - 指针均为 device 指针;launch 异步返回,错误检查由 bench 统一 sync 承担。
//   - 尺寸前置条件(kernel 内不做尾块处理,不满足即越界):
//       v0:任意尺寸(内部有 row/col guard);
//       v1:M,N,K % 32 == 0;
//       v2/v3:M,N % 64 == 0 且 K % 32 == 0;
//       v4:M,N % 128 == 0 且 K % 32 == 0。
//     bench 固定 4096³ 全部满足;通用尾块不在本梯目标内(目标是归因,不是产品化)。
// 性能锚(4096³,RTX 4090,3 轮 mean,EXP-K02（CUDA Tensor Core GEMM 版本梯）):
//   v0 5.2 → v1 6.5 → v2 89.5 → v3 95.5 → v4 133.1 TFLOPS;
//   同 harness 真 cuBLAS 155.4(v4 = 85.6%)。
// ============================================================================
void gemm_v0(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v1(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v5(const half* A, const half* B, half* C, int M, int N, int K);
// 真 cuBLAS 对照(调用点验真,见 gemm_cublas.cu)——一切「vs cublas」数字的对照物。
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K);
