#include <mma.h>
#include <cuda_pipeline.h>
#include "fa_common.h"
// ============================================================================
// v4 · 访存重叠(组织结构继承 v3 的 8 warp):
// ① S/P 统一为一块 half 缓冲 SP(S 写出即转 fp16,exp 原位改写 S→P):
//    省掉 v2/v3 的 float S 区 17408B、砍半 softmax 的 smem 读流量;
//    精度代价过 gate(err 4.88e-04,EXP-K03（CUDA FA2 forward 简化版版本梯）§5——fp16 S/P 未推高误差);
// ② K 双缓冲 cp.async:QK^T(t) 进行时 K(t+1) 已在途;
// ③ V(t) 装载与 QK^T(t) 重叠(V 只在 PV 段用,重叠窗口 = ②③④ 三段)。
// 契约:同 v2/v3(D=128,S % 64 == 0)。
// 性能:3.949±0.012 ms = 34.8±0.12 TFLOPS,vs v3 仅 +7.1%——EXP-K03 最有
// 信息量的数字:K/V 访存能藏的都藏了,剩下的时间不在全局访存,而在每 tile
// 5 次 __syncthreads 串起来的相位链 → wmma 架构税的定量佐证(仍为推断级,
// NCU 不可用,红线见 flash-attn/README)。资源:80 reg / 256 thr /
// smem 89.75KB → 1 block/SM。
//
// 动态 smem 分区(合计 91904B = 89.75KB;16B 对齐;对比 v2:SP 合一省出的
// 17KB 拿去给 K 开双缓冲,总量反而少 1KB):
//   区    类型/形状            字节   用途
//   Osm   float[64][128]      32768   O 累加器(同 v2)
//   m/l/a float[64] x3          768   在线统计
//   SP    half [64][72]        9216   S 与 P 同址(LDSP=72 填充,理由见 v2 的 LDP)
//   Ks    half [2][64][128]   32768   K 双缓冲
//   Vs    half [64][128]      16384   V 单缓冲
//
// 两组 pipeline 交错等待的推导(本文件核心;cp.async 组按 commit 顺序排 FIFO,
// wait_prior(N) = 等「最新 N 组之外」的全部,语义见 gemm_v3.cu):
//   commit 序列:... G_K(t) → G_V(t) → G_K(t+1) → G_V(t+1) → ...
//   K、V 每轮各恰好 commit 一组、严格交替 ⇒ 任一时刻在途 ≤ 2 组,且
//   「最新 1 组」永远是刚为另一条流发出的那组:
//   - QK^T 前 wait_prior(1):在途 = {G_K(t) 老, G_V(t) 新} → 等掉的恰是
//     G_K(t);G_V(t) 继续在途,与 QK^T 重叠。
//   - PV 前 wait_prior(1):在途 = {G_V(t) 老, G_K(t+1) 新} → 等掉的恰是
//     G_V(t);G_K(t+1) 继续在途,跨过 softmax/重缩放段直到下轮。
//   末轮不发 G_K(t+1),PV 前的参数降为 0 清空队列。
//   正确性完全押在 commit 的交替节奏上:任何一处多发/漏发一组,两个
//   wait_prior 会同时指错对象——比 gemm v3 的单流双缓冲高一级危险。
// 面试点:① 上述交错推导(共用一个组计数器,两条流如何各取所需);
// ② V 为什么不做双缓冲——V 的消费点唯一且已有 ②③④ 整段重叠窗口,双缓冲
// 再花 16KB 换不来新的重叠;③ +7.1% 的解读(访存全藏后剩相位链)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 8;

constexpr int OFF_O = 0;                       // float [64][128] 32768
constexpr int OFF_ML = 32768;                  // m/l/a  768
constexpr int OFF_SP = 33536;                  // half [64][72]  9216(S 与 P 同址)
constexpr int OFF_K = 42752;                   // half [2][64][128] 32768
constexpr int OFF_V = 75520;                   // half [64][128] 16384
constexpr int SMEM_BYTES = 91904;
constexpr int LDSP = 72;

// 预取一个 64x128 half tile(16KB)并 commit 成一组;每次调用恰好一组——
// 这个不变量是文件头交错推导成立的前提
__device__ __forceinline__ void async_tile(half* dst, const half* src,
                                           int n0, int S, int tid, int nthr) {
    for (int t = tid; t < BN * FA_D / 8; t += nthr) {
        int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
        __pipeline_memcpy_async(&dst[r * FA_D + c],
                                &src[(size_t)(n0 + r) * FA_D + c], 16);
    }
    __pipeline_commit();
}

__global__ void fa2_v4_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* SP = (half*)(smem + OFF_SP);
    half* Ks = (half*)(smem + OFF_K);              // [2] 双缓冲
    half* Vs = (half*)(smem + OFF_V);

    const int tid = threadIdx.x, warp = tid / 32;
    const int wr = warp / 2, wc = warp % 2;        // 行条带 x 列半区(同 v3)
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const int q0 = blockIdx.x * BM;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* q = Q + ((size_t)(b * Hq + h) * S + q0) * FA_D;

    for (int i = tid; i < BM * FA_D; i += blockDim.x) Osm[i] = 0.f;
    if (tid < BM) { m_s[tid] = -1e30f; l_s[tid] = 0.f; }

    // Q 常驻寄存器(同 v2/v3)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[8];
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        wmma::load_matrix_sync(af[kk], q + (wr * 16) * FA_D + kk * 16, FA_D);

    const float scale = rsqrtf((float)FA_D);
    const int nlimit = causal ? min(q0 + BM, S) : S;   // tile 级因果裁剪(见 v2)

    async_tile(Ks, k, 0, S, tid, blockDim.x);      // 序幕:commit G_K(0),循环内 wait 才有对象
    int p = 0;
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();                           // WAR x2:上一轮 ⑤ 读完 Vs 才许 V(t) 覆盖;
                                                   // 读完 SP 才许 QK^T(t) 改写
        async_tile(Vs, v, n0, S, tid, blockDim.x); // commit G_V(t):V 到 ⑤ 才用,与 ②③④ 重叠
        __pipeline_wait_prior(1);                  // 等掉更老的 G_K(t)(交错推导见文件头);
                                                   // 刚发的 G_V(t) 留在途
        __syncthreads();                           // RAW:cp.async 完成仅发起线程可见,
                                                   // barrier 后全 warp 才能读全 K tile
        #pragma unroll
        for (int n = 0; n < 2; ++n) {              // [② QK^T] 分工同 v3(16x32 半区)
            const int nc = wc * 2 + n;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;    // col_major = K^T 视图(见 v2)
                wmma::load_matrix_sync(bf, &Ks[p * BN * FA_D
                                               + (nc * 16) * FA_D + kk * 16], FA_D);
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            // 与 v2/v3 的差异:S 即刻转 half 存 SP(逐元素转换合法性见
            // gemm_v2 面试点②),softmax 读时再转回 float
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> sh;
            #pragma unroll
            for (int e = 0; e < sh.num_elements; ++e)
                sh.x[e] = __float2half(sc.x[e]);
            wmma::store_matrix_sync(&SP[(wr * 16) * LDSP + nc * 16], sh,
                                    LDSP, wmma::mem_row_major);
        }
        if (n0 + BN < nlimit)                      // commit G_K(t+1) 入另一缓冲:此处发出,
                                                   // 到下轮 QK^T 前的 wait 有 ③④⑤ 整段给 DMA;
                                                   // 末轮不发(越界 + 破坏交错计数)
            async_tile(Ks + (p ^ 1) * BN * FA_D, k, n0 + BN, S, tid, blockDim.x);
        __syncthreads();                           // RAW:SP 的 wmma 写(按 warp 行条带)
                                                   // 对 ③ 的读者(每行 2 线程,跨 warp)可见
        if (tid < 2 * BM) {                        // [③ softmax] 分工同 v3:每行 2 线程 + shfl
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
            for (int j = j0; j < j0 + 32; ++j) {   // exp 原位改写 S→P:每个位置由同一线程
                                                   // 先读后写、两线程列窗口不相交 → 无竞态;
                                                   // 越界列写 0 喂 ⑤ 的 wmma(mask 即零填充)
                float pv = j < jend
                    ? __expf(__half2float(SP[row * LDSP + j]) * scale - mn) : 0.f;
                SP[row * LDSP + j] = __float2half(pv);
                sum += pv;
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            if (hf == 0) {                         // 单写者(同 v3)
                l_s[row] = l_s[row] * alpha + sum;
                m_s[row] = mn; a_s[row] = alpha;
            }
        }
        __syncthreads();                           // RAW:SP(已成 P)/a_s 写完才许 ④ 读 a_s、⑤ 读 SP
        for (int i = tid; i < BM * FA_D; i += blockDim.x)   // [④ O xα]
            Osm[i] *= a_s[i / FA_D];
        __pipeline_wait_prior(n0 + BN < nlimit ? 1 : 0);   // 等掉更老的 G_V(t)(留 G_K(t+1) 在途);
                                                           // 末轮在途只剩 G_V(t),0 = 清空
        __syncthreads();                           // RAW x2:④ 的重缩放写对 ⑤ 的 oacc 读可见;
                                                   // V tile 的 cp.async 分片对全 warp 可见
        #pragma unroll
        for (int c0 = 0; c0 < 4; ++c0) {           // [⑤ P·V] 分工同 v3(16 行 x 64 列半区)
            const int c = wc * 4 + c0;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &SP[(wr * 16) * LDSP + kk * 16], LDSP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            float* optr = &Osm[(wr * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
        }
        p ^= 1;                                    // K 双缓冲翻面
    }
    __syncthreads();   // 收尾 RAW:末轮 ⑤ 的 O 写对写回可见
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v4(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;   // 一次性 opt-in(同 v2)
    if (!configured) {
        cudaFuncSetAttribute(fa2_v4_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v4_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
