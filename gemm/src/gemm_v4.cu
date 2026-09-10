#include <mma.h>
#include <cuda_pipeline.h>
#include "gemm_common.h"
// ============================================================================
// v4 · 大 tile(128x128,8 warp,每 warp 64x32 = 4x2 fragment)+ 双缓冲。
// 问题:v3 之后 Tensor Core 仍吃不饱,卡在每字节 smem 支撑的 FLOP 不足。
// 算法:tile 由 64² 扩到 128²(BK=32 不变),对标 Triton BM128/BN128 配置;
// 输出面积 x4、装载周长 x2 → 算术强度 BM·BN/(BM+BN) 翻倍;cp.async 调度
// 完全继承 v3(组语义推导见 gemm_v3.cu 文件头)。
// 数据布局(block 输出 128x128 = 8 warp 按 2x4 排布,每 warp 64x32):
//        n→   0..31   32..63   64..95   96..127
//   m  0..63  [w0]     [w1]     [w2]     [w3]
//    64..127  [w4]     [w5]     [w6]     [w7]
//   每 warp 4x2 = 8 个 accumulator fragment 常驻寄存器。
// 契约:M,N % 128 == 0 且 K % 32 == 0。
// 性能:1.033±0.007 ms = 133.1±0.97 TFLOPS = 真 cuBLAS 85.6%(155.4),
// vs v3 +39%——版本梯对 cuBLAS 差距的主要收口(EXP-K02（CUDA Tensor Core GEMM 版本梯）)。
// 资源:92 reg x 256 thr(≈23.5K/64K)+ 32KB smem → 2 block/SM(reg 限),
// 理论 occupancy 33%,全梯最低。
// 面试点:① occupancy 33% 最低却最快——Tensor Core 吞吐不靠线程数遮蔽
// 延迟,靠 fragment 级 ILP(8 个 acc 无相互依赖,可交错发射)与 smem 复用
// (af 一次 load 进 2 次 mma、bf 进 4 次);occupancy 是手段不是目标
// (EXP-K02 §6)。② 为什么不再加大 BK:算术强度与 BK 无关,BK 翻倍只让
// smem 冲到 64KB(2 block/SM → 1),纯亏。③ 剩余 14% 差距(swizzle/
// 多级流水/tile 形状)为推断级,NCU 不可用未剖(EXP-K02 §6,红线)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 128, BN = 128, BK = 32;

// 同 v3 的协同预取,只是 tile 尺寸翻倍:A 128x32 + B 32x128 = 8192 段 float4,
// 256 线程每人 32 段;每次调用恰好 commit 一组(组节奏 = wait_prior 计数正确的前提)。
__device__ __forceinline__ void load_tile_async4(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int N, int K,
    int bm, int bn, int k0, int tid, int nthr) {
    for (int t = tid; t < BM * BK / 8; t += nthr) {
        int r = (t * 8) / BK, c = (t * 8) % BK;
        __pipeline_memcpy_async(&As[r][c], &A[(bm + r) * K + k0 + c], 16);
    }
    for (int t = tid; t < BK * BN / 8; t += nthr) {
        int r = (t * 8) / BN, c = (t * 8) % BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + bn + c], 16);
    }
    __pipeline_commit();
}

__global__ void gemm_v4_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[2][BM][BK], Bs[2][BK][BN];   // 双缓冲 2 x (8+8)KB = 32KB
    const int warp_id = threadIdx.x / 32;           // 8 warp:2 行 x 4 列
    const int wr = warp_id / 4, wc = warp_id % 4;   // warp tile 64x32(见文件头小图)
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    // 4x2 = 8 个 fp32 accumulator 常驻寄存器整个 kernel 生命周期——
    // v4 的寄存器大头(92 reg/thr),也是 ILP 的来源
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    // 序幕:第 0 组先行,循环内 wait_prior 才有对象(同 v3)
    load_tile_async4(As[0], Bs[0], A, B, N, K, bm, bn, 0,
                     threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                    // 末轮不预取(越界 + 破坏组计数)
            load_tile_async4(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留预取组在途只等当前块;末轮清空(推导见 v3)
        __syncthreads();   // cp.async 完成仅发起线程可见 → barrier 后全 warp 才能读全 tile(跨线程 RAW)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // 4x2x2 微内核:6 次 load 喂 8 次 mma;af[i] 复用 2 次、bf[j] 复用 4 次
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 64 + i * 16][kk], BK);   // wr*64:每 warp 管 64 行
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();   // 防下一轮 cp.async 覆盖 buf[p] 时慢 warp 仍在读(WAR,同 v3)
        p ^= 1;
    }
    // 写回:同 v2/v3(逐元素 fp32→fp16 后直写,合法性见 gemm_v2 文件头面试点②)
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 64 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(256), grd(N / BN, M / BM);
    gemm_v4_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
