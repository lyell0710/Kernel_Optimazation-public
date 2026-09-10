#include "gemm_common.h"
// ============================================================================
// v0 · naive:一线程一输出。
// 问题:版本梯需要一个「肯定对」的正确性锚与性能分母。
// 算法:每线程沿 K 串行 FMA,fp16 读入即转 fp32 累加;无任何片上复用。
// 契约:任意 M/N/K(grid 向上取整 + row/col guard);A/B/C 行主序 device 指针。
// 性能:26.369±0.472 ms = 5.2±0.12 TFLOPS = 真 cuBLAS 3.4%(4096³,4090,
// 3 轮,EXP-K02（CUDA Tensor Core GEMM 版本梯）)。
// 面试点:v0→v1 仅 +25% 而 v1→v2 有 13.8x——fp16 输入在 CUDA core 上做
// fp32 FMA,~6.5 TFLOPS 已近该路线实际上限(EXP-K02 §6):本算子的台阶在
// 指令世代,不在访存微调。v0 的价值是把这笔账立起来。
// ============================================================================
__global__ void gemm_v0_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;   // 末块越界线程直接退出:各输出互不相交,无部分和要合并
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        // B[k*N+col]:warp 内 col 连续 → 按行合并;A[row*K+k]:warp 内同 row 广播。
        // 问题不在合并度,在复用为零:同一 A/B 元素被不同 block 反复从
        // DRAM/L2 拉取——这笔带宽账引出 v1 的 smem tile。
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    C[row * N + col] = __float2half(acc);   // fp32 累加全程保精度,仅写回舍入一次
}
void gemm_v0(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);   // 256 thr/block;+15 向上取整兜住尾块
    gemm_v0_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
