#include <mma.h>
#include <cuda_pipeline.h>
#include "fa_common.h"
// ============================================================================
// v5 · mma PTX + ldmatrix 重写 v4 的微内核(QK^T 与 P·V 两段),softmax 段不变。
// 问题:v4 用 wmma,smem 读布局被 wmma 隐式决定,无法 swizzle;计数器实测
// FA2 v4 short_scoreboard 50.13%、bank conflict 578M/76M(EXP-K07)——「swizzle
// 能消除该瓶颈」这条红线(fa2 侧)的对象。
// 算法:tile 结构/流水/smem 分区/softmax 完全对齐 v4(BM=64,BN=64,8 warp,
// SP 合一、K 双缓冲、V 与 QK^T 重叠),只换微内核:wmma → mma.m16n8k16 PTX +
// ldmatrix。控制变量:差异只剩指令世代 + smem 布局控制权。
// fragment 布局(gemm v5 已用 mma_test 实测钉死):
//   A(16x16 row) a0=A[g][2t] a1=A[g+8][2t] a2=A[g][2t+8] a3=A[g+8][2t+8]
//   B(16x8 col)  b0=B[2t][g] B[2t+1][g]  b1=B[2t+8][g] B[2t+9][g]
//   D(16x8 f32)  c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1]
// Q 从全局内存直接手工 load(只 load 一次,常驻寄存器,与 v4 同);K/V 用
// ldmatrix.trans(row-major smem → col-major B);S/O 手工 store/load(D 布局)。
// 契约:同 v4(D=128,S % 64 == 0)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 8;
constexpr int LDSP = 72;   // SP 行宽(half),72=64+8 消 ldmatrix 8 行 bank 冲突

constexpr int OFF_O = 0;
constexpr int OFF_ML = 32768;
constexpr int OFF_SP = 33536;
constexpr int OFF_K = 42752;
constexpr int OFF_V = 75520;
constexpr int SMEM_BYTES = 91904;

// ---- 微内核原语(行宽参数化,与 gemm_v5 同套,EXP-K10 实测验证) -------------
__device__ __forceinline__ void ldmatrix_a4(uint32_t a[4], const half* As,
                                            int row, int col, int ldm) {
    int lane = threadIdx.x & 31;
    int r = lane & 7, g = lane >> 3;
    const half* addr = &As[(row + (g & 1) * 8 + r) * ldm + col + (g >> 1) * 8];
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}

__device__ __forceinline__ void ldmatrix_b4_trans(uint32_t b[4], const half* Bs,
                                                  int krow, int ncol, int ldm) {
    int lane = threadIdx.x & 31;
    int r = lane & 7, g = lane >> 3;
    const half* addr = &Bs[(krow + (g & 1) * 8 + r) * ldm + ncol + (g >> 1) * 8];
    uint32_t s = (uint32_t)__cvta_generic_to_shared(addr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(s));
}

__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4],
                                             uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Q 从全局内存手工 load 一个 16x16 row-major A fragment(16 行 x 16 K 列)。
__device__ __forceinline__ void load_q_a4(uint32_t a[4], const half* q,
                                          int row, int col) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;
    const half* qrow = q + (size_t)(row + g) * FA_D + col;
    a[0] = *reinterpret_cast<const uint32_t*>(qrow + 2 * t);
    a[1] = *reinterpret_cast<const uint32_t*>(qrow + 8 * FA_D + 2 * t);
    a[2] = *reinterpret_cast<const uint32_t*>(qrow + 2 * t + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(qrow + 8 * FA_D + 2 * t + 8);
}

// 预取一个 64x128 half tile(16KB)并 commit 成一组(同 v4)
__device__ __forceinline__ void async_tile(half* dst, const half* src,
                                           int n0, int S, int tid, int nthr) {
    for (int t = tid; t < BN * FA_D / 8; t += nthr) {
        int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
        __pipeline_memcpy_async(&dst[r * FA_D + c],
                                &src[(size_t)(n0 + r) * FA_D + c], 16);
    }
    __pipeline_commit();
}

__global__ void fa2_v5_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* SP = (half*)(smem + OFF_SP);
    half* Ks = (half*)(smem + OFF_K);
    half* Vs = (half*)(smem + OFF_V);

    const int tid = threadIdx.x, warp = tid / 32;
    const int wr = warp / 2, wc = warp % 2;
    const int lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const int q0 = blockIdx.x * BM;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* q = Q + ((size_t)(b * Hq + h) * S + q0) * FA_D;

    for (int i = tid; i < BM * FA_D; i += blockDim.x) Osm[i] = 0.f;
    if (tid < BM) { m_s[tid] = -1e30f; l_s[tid] = 0.f; }

    // Q 常驻寄存器:每 warp 16 行(wr*16) x 128 K 列(8 个 k16 段)
    uint32_t af[8][4];
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        load_q_a4(af[kk], q, wr * 16, kk * 16);

    const float scale = rsqrtf((float)FA_D);
    const int nlimit = causal ? min(q0 + BM, S) : S;

    async_tile(Ks, k, 0, S, tid, blockDim.x);
    int p = 0;
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();
        async_tile(Vs, v, n0, S, tid, blockDim.x);
        __pipeline_wait_prior(1);
        __syncthreads();
        #pragma unroll
        for (int n = 0; n < 2; ++n) {              // 2 个 n16 块 = 4 个 n8
            const int nc = wc * 2 + n;             // S 列 n16 块索引
            float sc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                // K 存储 [S][D](row-major),mma 要 K^T=[D][S] 的 col-major B;
                // ldmatrix non-trans 读 [S][D] 得 a0..a3,其中 a0/a2 恰是
                // K^T 的 b0/b1(见文件头布局推导):b0=a0(K[g][2t]) b1=a2(K[g][2t+8])。
                uint32_t k4[4];
                ldmatrix_a4(k4, &Ks[p * BN * FA_D], nc * 16, kk * 16, FA_D);
                mma_m16n8k16(sc[0], af[kk], k4[0], k4[2]);   // n8 块(S 列 nc*16..+7)
                mma_m16n8k16(sc[1], af[kk], k4[1], k4[3]);   // n8 块(S 列 nc*16+8..+15)
            }
            // S 手工 store 到 SP(D fragment 布局,SP 存裸 QK^T,scale 留给 softmax)
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                int r0 = wr * 16 + g;
                int c0 = nc * 16 + nn * 8 + 2 * t;
                SP[r0 * LDSP + c0] = __float2half(sc[nn][0]);
                SP[r0 * LDSP + c0 + 1] = __float2half(sc[nn][1]);
                SP[(r0 + 8) * LDSP + c0] = __float2half(sc[nn][2]);
                SP[(r0 + 8) * LDSP + c0 + 1] = __float2half(sc[nn][3]);
            }
        }
        if (n0 + BN < nlimit)
            async_tile(Ks + (p ^ 1) * BN * FA_D, k, n0 + BN, S, tid, blockDim.x);
        __syncthreads();
        if (tid < 2 * BM) {                        // softmax(同 v4,标量,不受 mma 影响)
            const int row = tid / 2, hf = tid % 2;
            const int grow = q0 + row;
            const int jend = min(BN, (causal ? grow + 1 : S) - n0);
            const int j0 = hf * 32, j1 = min(j0 + 32, jend);
            float rmax = -1e30f;
            for (int j = j0; j < j1; ++j)
                rmax = fmaxf(rmax, __half2float(SP[row * LDSP + j]) * scale);
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));
            const float mn = fmaxf(m_s[row], rmax);
            const float alpha = __expf(m_s[row] - mn);
            float sum = 0.f;
            for (int j = j0; j < j0 + 32; ++j) {
                float pv = j < jend
                    ? __expf(__half2float(SP[row * LDSP + j]) * scale - mn) : 0.f;
                SP[row * LDSP + j] = __float2half(pv);
                sum += pv;
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            if (hf == 0) {
                l_s[row] = l_s[row] * alpha + sum;
                m_s[row] = mn; a_s[row] = alpha;
            }
        }
        __syncthreads();
        for (int i = tid; i < BM * FA_D; i += blockDim.x)
            Osm[i] *= a_s[i / FA_D];
        __pipeline_wait_prior(n0 + BN < nlimit ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {           // 4 个 n16 块 = 8 个 n8
            const int c = wc * 4 + c0;             // O 列 n16 块索引
            float oacc[2][4];
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {       // load 现有 O(D fragment 布局)
                int r0 = wr * 16 + g;
                int cc = c * 16 + nn * 8 + 2 * t;
                oacc[nn][0] = Osm[r0 * FA_D + cc];
                oacc[nn][1] = Osm[r0 * FA_D + cc + 1];
                oacc[nn][2] = Osm[(r0 + 8) * FA_D + cc];
                oacc[nn][3] = Osm[(r0 + 8) * FA_D + cc + 1];
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {       // 4 个 k16(P 的 K 维)
                uint32_t pa[4];
                ldmatrix_a4(pa, &SP[wr * 16 * LDSP], 0, kk * 16, LDSP);
                uint32_t v4_[4];
                ldmatrix_b4_trans(v4_, &Vs[0], kk * 16, c * 16, FA_D);
                mma_m16n8k16(oacc[0], pa, v4_[0], v4_[1]);
                mma_m16n8k16(oacc[1], pa, v4_[2], v4_[3]);
            }
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {       // store 回 Osm
                int r0 = wr * 16 + g;
                int cc = c * 16 + nn * 8 + 2 * t;
                Osm[r0 * FA_D + cc] = oacc[nn][0];
                Osm[r0 * FA_D + cc + 1] = oacc[nn][1];
                Osm[(r0 + 8) * FA_D + cc] = oacc[nn][2];
                Osm[(r0 + 8) * FA_D + cc + 1] = oacc[nn][3];
            }
        }
        p ^= 1;
    }
    __syncthreads();
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v5(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fa2_v5_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v5_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
