#include <mma.h>
#include <cuda_pipeline.h>
#include "gemm_common.h"
// ============================================================================
// v5 · mma PTX + ldmatrix + smem 错位(padding) 重写 v4 的微内核。
// 问题:v4 用 wmma API,固定布局不暴露地址计算 → 无法 swizzle;smem 读的
// bank 布局被 wmma 隐式决定,计数器实测 gemm v4 smem 冲突波前占 77.1%
// (EXP-K07《NCU 计数器闭环》),这正是「swizzle 能消除该瓶颈」这条红线的对象。
// 算法:tile 尺寸完全对齐 v4(128x128,8 warp 2x4,每 warp 64x32,BK=32),
// 只换微内核——wmma::load_matrix_sync → ldmatrix,wmma::mma_sync →
// mma.sync.aligned.m16n8k16。这是「控制变量」:tile/流水/写回全不动,差异
// 只剩指令世代 + smem 布局控制权。
// 布局(m16n8k16 的 fragment 映射,PTX ISA §9.7.15.5.8):
//   groupID = %laneid >> 2, t = %laneid % 4。
//   A(16x16 f16, row-major) 4 个 .b32:a0=A[g][2t], a1=A[g+8][2t],
//   a2=A[g][2t+8], a3=A[g+8][2t+8]。
//   B(16x8 f16, col-major 语义) 2 个 .b32:b0=B[2t][g], b1=B[2t+8][g]。
//   C(16x8 f32) 4 个:c0=D[g][2t], c1=D[g+8][2t], c2=D[g][2t+1],
//   c3=D[g+8][2t+1]。
// ldmatrix 的 8x8 加载布局恰好 = mma 的 A fragment 布局(同一 ISA 定义),
// 故「ldmatrix.x4 加载 A 的 4 个 8x8 → 直接当 a0..a3 喂 mma」零拷贝。
// B 用 ldmatrix.x4.trans 从 row-major B_s 转置加载,结果恰好 = b0/b1。
// smem 错位:A_s 行宽 32→40、B_s 行宽 128→136(各 +8 half = +16B),
// ldmatrix 的 8 行读取由此错开 bank(padding 消冲突;真正的 XOR swizzle
// 是下一步增量,见文件尾「未做」)。
// 契约:M,N % 128 == 0 且 K % 32 == 0(同 v4)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int A_LDM = 40, B_LDM = 136;   // 32+8 / 128+8 的 smem 行宽(half)

// 一个 warp 的 fragment 常量(64x32 = 4 m16 x 4 n8,16 个 mma/ k-step)。
constexpr int FRAG_M = 4, FRAG_N = 4;

// ---- 微内核原语 -----------------------------------------------------------
// ldmatrix.x4 加载 4 个 8x8(row-major A),结果 = mma 的 a0..a3。
// 4 个块:a0=行0-7列0-7,a1=行8-15列0-7,a2=行0-7列8-15,a3=行8-15列8-15。
// 块 g(lane>>3)的行偏移 = (g&1)*8、列偏移 = (g>>1)*8。
__device__ __forceinline__ void ldmatrix_a4(uint32_t a[4], const half* As,
                                            int row, int col) {
    int lane = threadIdx.x & 31;
    int r = lane & 7, g = lane >> 3;
    const half* addr = &As[(row + (g & 1) * 8 + r) * A_LDM + col + (g >> 1) * 8];
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

// ldmatrix.x4.trans 从 row-major B_s 转置加载 4 个 8x8,得 2 个 n8 的 b0/b1。
// 返回寄存器顺序:{n0_b0, n0_b1, n1_b0, n1_b1}。
__device__ __forceinline__ void ldmatrix_b4_trans(uint32_t b[4], const half* Bs,
                                                  int krow, int ncol) {
    int lane = threadIdx.x & 31;
    int r = lane & 7, g = lane >> 3;   // 块 g:K 半块 = g&1,N 偏移 = g>>1
    const half* addr = &Bs[(krow + (g & 1) * 8 + r) * B_LDM + ncol + (g >> 1) * 8];
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(s));
}

// mma.sync.aligned.m16n8k16:f32 累加,f16 A/B。
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- cp.async 协同预取(同 v4,padding 行宽) ---------------------------------
__device__ __forceinline__ void load_tile_async5(
    half (*As)[A_LDM], half (*Bs)[B_LDM],
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

__global__ void gemm_v5_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[2][BM][A_LDM], Bs[2][BK][B_LDM];   // 双缓冲 20+17=37KB
    const int warp_id = threadIdx.x / 32;
    const int wr = warp_id / 4, wc = warp_id % 4;   // warp tile 64x32(同 v4)
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int lane = threadIdx.x & 31;

    // 16 个 accumulator(4x4),fp32 常驻整个 kernel 生命周期。
    float acc[FRAG_M][FRAG_N][4];
    #pragma unroll
    for (int m = 0; m < FRAG_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[m][n][e] = 0.f;

    load_tile_async5(As[0], Bs[0], A, B, N, K, bm, bn, 0,
                     threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)
            load_tile_async5(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[FRAG_M][4], b[FRAG_N][2];
            #pragma unroll
            for (int m = 0; m < FRAG_M; ++m)
                ldmatrix_a4(a[m], &As[p][wr * 64 + m * 16][0], 0, kk);
            #pragma unroll
            for (int np = 0; np < FRAG_N / 2; ++np) {
                uint32_t b4[4];
                ldmatrix_b4_trans(b4, &Bs[p][kk][0], 0, wc * 32 + np * 16);
                b[np * 2][0] = b4[0]; b[np * 2][1] = b4[1];
                b[np * 2 + 1][0] = b4[2]; b[np * 2 + 1][1] = b4[3];
            }
            #pragma unroll
            for (int m = 0; m < FRAG_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAG_N; ++n)
                    mma_m16n8k16(acc[m][n], a[m], b[n]);
        }
        __syncthreads();
        p ^= 1;
    }
    // 写回:mma D fragment 布局(实测验证,见 mma_test):
    //   c0=D[g][2t]  c1=D[g][2t+1]  c2=D[g+8][2t]  c3=D[g+8][2t+1]
    // 即 c0/c1 是同一 M 行的相邻 N 列,c2/c3 是 M+8 行的相邻 N 列。
    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int m = 0; m < FRAG_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            int r0 = bm + wr * 64 + m * 16 + g;
            int c0 = bn + wc * 32 + n * 8 + 2 * t;
            half* dst = &C[r0 * N + c0];
            dst[0] = __float2half(acc[m][n][0]);       // c0 = D[g][2t]
            dst[1] = __float2half(acc[m][n][1]);       // c1 = D[g][2t+1]
            dst[8 * N] = __float2half(acc[m][n][2]);   // c2 = D[g+8][2t]
            dst[8 * N + 1] = __float2half(acc[m][n][3]); // c3 = D[g+8][2t+1]
        }
}
void gemm_v5(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(256), grd(N / BN, M / BM);
    gemm_v5_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
