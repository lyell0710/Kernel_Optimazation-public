#include <mma.h>
#include <cuda_pipeline.h>
#include "gemm_common.h"
// ============================================================================
// v3 · v2 + cp.async 双缓冲。
// 问题:v2 的「同步装载 → 计算」串行,Tensor Core 每个 K 块都要空等 DMA。
// 算法:smem 双缓冲 As/Bs[2][...],mma 消费 buf[p] 的同时 __pipeline_memcpy_async
// 向 buf[p^1] 预取下一 K 块——与 Triton num_stages=2 是同一件事的手写形。
// 契约:同 v2(M,N % 64 == 0,K % 32 == 0)。
// 性能:1.439±0.007 ms = 95.5±0.49 TFLOPS = 真 cuBLAS 61.4%,vs v2 +6.7%
// (EXP-K02（CUDA Tensor Core GEMM 版本梯）)。增量不大的原因:v2 已是 compute-bound,装载本占比有限,
// 重叠能省的就这点——对照 v4 大 tile 的 +39%:复用(算术强度)比重叠值钱。
// 资源:61 reg / 16KB smem / 128 thr → 6 block/SM(smem 限),理论 occupancy 50%。
//
// cp.async 组语义(本文件与 v4 的调度核心):
//   __pipeline_commit() 把此前发出的 memcpy_async 封成一个组(FIFO 排队);
//   __pipeline_wait_prior(N) = 等到「最新 N 组之外」的所有组完成。
// 稳态推导:进入第 t 轮时 buf[p] 的组(第 t 组)已在途;循环体先为 buf[p^1]
// commit 第 t+1 组,再 wait_prior(1)——「最新 1 组」恰是刚发的第 t+1 组,
// 被等掉的正是当前要消费的第 t 组。末轮不发新组,wait_prior(0) 清空全部。
// 正确性押在「每轮恰好 commit 一组」的节奏上:多发/漏发一次,wait_prior
// 的计数就指错组——这是 cp.async 手工调度最易错的地方。
// 面试点:① wait_prior 参数为什么是 1 不是 0(等 0 = 连预取组一起等,
// 重叠归零退化回 v2);② 双缓冲下两个 __syncthreads 各防哪条竞态(见行内)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 64, BN = 64, BK = 32;

// 协同预取一个 K 块(A 64x32 + B 32x64)并 commit 成一组。
// 16B/条:cp.async 的最大粒度,兼作 float4 对齐要求(c 为 8 的倍数,K%32==0)。
__device__ __forceinline__ void load_tile_async(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int M, int N, int K,
    int bm, int bn, int k0, int tid, int nthr) {
    for (int t = tid; t < BM * BK / 8; t += nthr) {
        int r = (t * 8) / BK, c = (t * 8) % BK;
        __pipeline_memcpy_async(&As[r][c], &A[(bm + r) * K + k0 + c], 16);
    }
    for (int t = tid; t < BK * BN / 8; t += nthr) {
        int r = (t * 8) / BN, c = (t * 8) % BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + bn + c], 16);
    }
    __pipeline_commit();   // 本 tile 封组:调用一次 = 恰好一组,组节奏见文件头
}

__global__ void gemm_v3_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[2][BM][BK], Bs[2][BK][BN];   // 双缓冲 2 x (4+4)KB = 16KB
    const int warp_id = threadIdx.x / 32;
    const int wr = warp_id / 2, wc = warp_id % 2;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    // 序幕:先发第 0 组,循环内的 wait_prior 才有对象;首轮无重叠(冷启动税只付一次)
    load_tile_async(As[0], Bs[0], A, B, M, N, K, bm, bn, 0,
                    threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                       // 末轮不预取:越界读 + 多出一组破坏 wait 计数
            load_tile_async(As[p ^ 1], Bs[p ^ 1], A, B, M, N, K,
                            bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留 1 组(刚发的预取)在途,只等当前块;末轮清空
        __syncthreads();   // cp.async 的完成只对发起线程可见,而 tile 由全 block
                           // 分片搬运:barrier 后任何 warp 才能读到别的线程搬的段(跨线程 RAW)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();   // 防下一轮把 buf[p](翻转后成为预取目标)交给 cp.async
                           // 覆盖时,慢 warp 仍在读本轮的 As[p]/Bs[p](WAR:
                           // 双缓冲只隔离「计算 vs 在途预取」,不隔离「本轮读 vs 下轮写同一 buf」)
        p ^= 1;
    }
    // 写回:同 v2(fp32 acc 逐元素转 half 后直写全局,合法性见 gemm_v2 文件头)
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 32 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(128), grd(N / BN, M / BM);
    gemm_v3_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
