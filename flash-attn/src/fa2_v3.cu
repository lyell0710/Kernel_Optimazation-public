#include <mma.h>
#include "fa_common.h"
// ============================================================================
// v3 · 并行度翻倍:8 warp(行条带 x 列半区 = 4x2),softmax 每行 2 线程。
// 问题:v2 只有 128 线程,而 smem 90.75KB 决定了 1 block/SM——block 内的
// warp 数就是这台机器仅剩的延迟遮蔽来源,4 warp 吃不满。
// 算法:对 v2 的变量只有并行组织,算法与 smem 布局逐字节不变(控制变量):
//   QK^T:每 warp 16 行 x 32 列半区(v2 为 16x64);
//   softmax:每行 2 线程各扫 32 列,shfl 合并(v2 一行 1 线程扫 64);
//   P·V:每 warp 16 行 x 64 列半区(v2 为 16x128)。
// 契约:同 v2(D=128,S % 64 == 0)。
// 性能:4.229±0.012 ms = 32.5±0.10 TFLOPS,vs v2 +33%(EXP-K03（CUDA FA2 forward 简化版版本梯）);资源:
// 95 reg / 256 thr / smem 90.75KB → 仍 1 block/SM,理论 occupancy 16.7%。
// 面试点:+33% 全部来自 block 内并行度而 occupancy 只从 8.3% 到 16.7%——
// 被 smem 钉在 1 block/SM 时,加 warp 是唯一放大招;对照 gemm v4
// (occupancy 33% 全梯最低却最快)可完整讲清「occupancy 的正确读法」。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 8;

// smem 分区与 v2 完全一致(偏移逐字节相同,注释与推导见 fa2_v2.cu 的分区表)
constexpr int OFF_O = 0;
constexpr int OFF_S = 32768;
constexpr int OFF_ML = 50176;
constexpr int OFF_K = 50944;
constexpr int OFF_V = 67328;
constexpr int OFF_P = 83712;
constexpr int SMEM_BYTES = 92928;
constexpr int LDS = 68, LDP = 72;

__global__ void fa2_v3_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* Ssm = (float*)(smem + OFF_S);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* Ks = (half*)(smem + OFF_K);
    half* Vs = (half*)(smem + OFF_V);
    half* Psm = (half*)(smem + OFF_P);

    const int tid = threadIdx.x, warp = tid / 32;
    const int wr = warp / 2, wc = warp % 2;        // wr∈[0,4):16 行条带;wc∈{0,1}:32 列半区
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const int q0 = blockIdx.x * BM;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* q = Q + ((size_t)(b * Hq + h) * S + q0) * FA_D;

    for (int i = tid; i < BM * FA_D; i += blockDim.x) Osm[i] = 0.f;
    if (tid < BM) { m_s[tid] = -1e30f; l_s[tid] = 0.f; }

    // Q 常驻寄存器(同 v2);行基址换成 wr*16:同一行条带的 2 个 warp
    //(wc=0/1)各持一份相同的 af——用 8KB 寄存器冗余换掉跨 warp 共享的同步
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[8];
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        wmma::load_matrix_sync(af[kk], q + (wr * 16) * FA_D + kk * 16, FA_D);

    const float scale = rsqrtf((float)FA_D);
    const int nlimit = causal ? min(q0 + BM, S) : S;   // tile 级因果裁剪(推导见 v2)

    // 5 段相位链与 5 次 __syncthreads 同 v2,竞态对象注释详见 fa2_v2.cu;
    // 此处只注 v3 的分工差异
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();   // WAR:上一轮 ⑤ 读完 Vs/Psm 才许覆盖
        for (int t = tid; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            *(float4*)&Ks[r * FA_D + c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
            *(float4*)&Vs[r * FA_D + c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
        }
        __syncthreads();   // RAW:装载分片对 wmma 读者可见
        #pragma unroll
        for (int n = 0; n < 2; ++n) {              // [②] 本 warp 的 16x32 半区:
            const int nc = wc * 2 + n;             // 列块号 = 半区基址 wc*2 + 块内序
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;    // col_major = K^T 视图(见 v2)
                wmma::load_matrix_sync(bf, &Ks[(nc * 16) * FA_D + kk * 16], FA_D);
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            wmma::store_matrix_sync(&Ssm[(wr * 16) * LDS + nc * 16], sc,
                                    LDS, wmma::mem_row_major);
        }
        __syncthreads();   // RAW:S 全部落 smem 才许 ③ 读
        if (tid < 2 * BM) {   // [③] 每行 2 线程:tid 偶/奇 = 同行左右半区,
                              // 且为同 warp 相邻 lane → 可用 shfl 而非 smem 交换
            const int row = tid / 2, hf = tid % 2;
            const int grow = q0 + row;
            const int jend = min(BN, (causal ? grow + 1 : S) - n0);
            const int j0 = hf * 32, j1 = min(j0 + 32, jend);   // 本线程负责的 32 列窗口
            float rmax = -1e30f;
            for (int j = j0; j < j1; ++j)
                rmax = fmaxf(rmax, Ssm[row * LDS + j] * scale);
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));   // 与同行搭档合并 max
            const float mn = fmaxf(m_s[row], rmax);
            const float alpha = __expf(m_s[row] - mn);
            float sum = 0.f;
            for (int j = j0; j < j0 + 32; ++j) {   // 上界 j0+32 而非 j1:越界列写 0,
                                                   // ⑤ 的 wmma 吃整行(mask 即零填充,见 v2)
                float p = j < jend ? __expf(Ssm[row * LDS + j] * scale - mn) : 0.f;
                Psm[row * LDP + j] = __float2half(p);
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);   // 与搭档合并分母增量
            if (hf == 0) {   // 单写者:两线程算得同值,限一个写免 WAW(纪律性写法)
                l_s[row] = l_s[row] * alpha + sum;
                m_s[row] = mn; a_s[row] = alpha;
            }
        }
        __syncthreads();   // RAW:Psm/a_s 写完才许 ④/⑤ 读
        for (int i = tid; i < BM * FA_D; i += blockDim.x)
            Osm[i] *= a_s[i / FA_D];
        __syncthreads();   // RAW:重缩放完成才许 ⑤ load 旧 O
        #pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {           // [⑤] 本 warp 的 16 行 x 64 列半区
            const int c = wc * 4 + c0;             // 输出列块号(全 128 列切 8 块,每半区 4 块)
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &Psm[(wr * 16) * LDP + kk * 16], LDP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            float* optr = &Osm[(wr * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
        }
    }
    __syncthreads();   // 收尾 RAW:末轮 ⑤ 的 O 写对写回可见
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v3(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;   // 一次性 opt-in(同 v2)
    if (!configured) {
        cudaFuncSetAttribute(fa2_v3_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v3_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
