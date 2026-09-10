#include <mma.h>
#include "gemm_common.h"
// ============================================================================
// v2 · wmma(Tensor Core 入门形)。
// 问题:CUDA core 路线算力见顶(v1 6.5 TFLOPS),换指令世代。
// 算法:BM=BN=64/BK=32 分块;每 K 块 = 同步协同装载 → wmma load → mma_sync。
// 访存策略刻意保持与 v1 同级的朴素(同步装载),使 v1→v2 的差异只剩指令世代。
// 数据布局(block 输出 64x64 = 4 warp 按 2x2 排布,每 warp 32x32):
//        n→   0..31    32..63
//   m  0..31 [warp0]  [warp1]
//     32..63 [warp2]  [warp3]
//   每 warp 2x2 个 16x16x16 accumulator fragment;As[64][32]+Bs[32][64] 各 4KB。
// 契约:M,N % 64 == 0 且 K % 32 == 0(装载无 guard)。
// 性能:1.536±0.008 ms = 89.5±0.46 TFLOPS = 真 cuBLAS 57.6%,vs v1 x13.8
// ——全梯最大台阶(EXP-K02（CUDA Tensor Core GEMM 版本梯）)。资源:54 reg / 8KB smem / 128 thr →
// 9 block/SM(reg 限),理论 occupancy 75%。
// 面试点:① x13.8 的台阶来自指令世代而非访存微调(v0→v1 仅 +25% 为对照);
// ② 写回段 fp32→fp16 的逐元素转换为何合法:同 shape 的 accumulator
// fragment 在同架构上 lane→元素映射一致(wmma 唯一可依赖的对称性)——但
// 映射本身不公开,这正是 FA2 行级 softmax 做不进 fragment 的根源
// (见 flash-attn/src/fa2_v2.cu)。
// ============================================================================
using namespace nvcuda;
// BK=32 = 2 步 wmma-k:更小则 __syncthreads 频率翻倍,更大则 smem 翻倍而
// 复用不增(算术强度 BM·BN/(BM+BN) 与 BK 无关)——取同步开销与 smem 的平衡点。
constexpr int BM = 64, BN = 64, BK = 32;

__global__ void gemm_v2_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[BM][BK], Bs[BK][BN];
    const int warp_id = threadIdx.x / 32;
    const int wr = warp_id / 2, wc = warp_id % 2;      // warp 的 2x2 排布(见文件头小图)
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // [装载段] 128 线程按线性 tid 分片,每次一条 float4 = 8 half = 16B:
        // A 64x32、B 32x64 各 2048 half → 各 256 段,每线程 2 段。
        // c 恒为 8 的倍数且 K%32==0 ⇒ 全局地址 16B 对齐(float4 硬性要求)。
        for (int t = threadIdx.x; t < BM * BK / 8; t += blockDim.x) {
            int r = (t * 8) / BK, c = (t * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(bm + r) * K + k0 + c]);
        }
        for (int t = threadIdx.x; t < BK * BN / 8; t += blockDim.x) {
            int r = (t * 8) / BN, c = (t * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + bn + c]);
        }
        __syncthreads();   // 防 As/Bs 未写全即被读:装载按线性 tid 分片、
                           // 消费按 warp 分块,线程集不重合(跨线程 RAW)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // [计算段] 2x2x2 微内核:af[i]/bf[j] 各 load 1 次、各参与 2 次
            // mma_sync——fragment 级数据复用的起点(v4 扩到 4x2)。
            // af/bf 均 row_major(A、B 本就按行存),leading dim = 所在 tile 行宽。
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();   // 防快 warp 进入下一 k0 的装载覆盖 As/Bs 时,
                           // 慢 warp 仍在 load_matrix_sync 读旧 tile(WAR)
    }
    // [写回段] 各 warp 的输出 tile 两两不相交 → 直写全局即可,half 16x16
    // store 已按行合并。acc 为 fp32 fragment 而 C 为 half:先逐元素转出
    // half accumulator fragment 再 store(合法性见文件头面试点②)。
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float* cp = reinterpret_cast<float*>(As);   // 曾预留「经 smem 中转写回」的入口,
            (void)cp;                                   // 后判定直写已合并、无需 staging;留痕不启用
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> h;
            #pragma unroll
            for (int e = 0; e < h.num_elements; ++e)
                h.x[e] = __float2half(acc[i][j].x[e]);
            wmma::store_matrix_sync(
                &C[(bm + wr * 32 + i * 16) * N + bn + wc * 32 + j * 16],
                h, N, wmma::mem_row_major);
        }
}
void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(128), grd(N / BN, M / BM);
    gemm_v2_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
