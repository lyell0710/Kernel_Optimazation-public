#include "fa_common.h"
// ============================================================================
// v0 · warp-per-query-row:FA2 前向的最小可信实现。
// 算法:1 warp 管 1 行 q,顺序流过全部 K/V;在线 softmax 三件套最直白形态:
//   m = 运行 max,l = 运行分母,α = exp(m_old - m_new) 把历史部分和折算到新基。
// 数据布局:每 lane 持 q 的 4 维(D=128/32);K/V 复用完全交给 L2——
// 单 kv-head 的 K 仅 S·D·2B = 1MB(S=4096)≪ 4090 的 72MB L2,这个事实
// 决定了 v1 的 smem 化只有 +11%(EXP-K03（CUDA FA2 forward 简化版版本梯）§6)。
// 契约:任意 S/causal/GQA(Hq % Hkv == 0)。
// 性能:27.795±0.047 ms = 4.9±0.06 TFLOPS(S=4096 协议点,EXP-K03)。
// 面试点:α 修正为什么数学恒等而数值必要——exp(x-m_old)·exp(m_old-m_new)
// = exp(x-m_new) 精确成立;但若不随行减去当前 max,|s|~30 时 fp32 exp 即
// 上溢。m 单调不减 ⇒ α ≤ 1、p ≤ 1,全程无上溢风险。
// ============================================================================
__global__ void fa2_v0_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    const int row = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int kvh = h / (Hq / Hkv);           // GQA:连续 Hq/Hkv 个 q-head 共享一个 kv-head
    const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;

    float qr[4];                                  // 每 lane 持 4 维
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    const float scale = rsqrtf((float)FA_D);
    float m = -1e30f, l = 0.f, acc[4] = {0, 0, 0, 0};   // m 用 -inf 哨兵起步

    const int nend = causal ? row + 1 : S;    // row+1:causal 含对角(token 可见自身)
    for (int n = 0; n < nend; ++n) {
        float s = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            s += qr[j] * __half2float(k[(size_t)n * FA_D + lane * 4 + j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)    // 蝶式规约 → 全 lane 同值:
            s += __shfl_xor_sync(0xffffffff, s, off);   // 每 lane 都要拿 s 缩放自己的 acc[4]
        s *= scale;
        // 在线更新:新 max → 折算因子 α → 本键权重 p;先乘 α 再累加,
        // 等价于把此前的部分和整体换到新基准
        float mn = fmaxf(m, s), alpha = __expf(m - mn), p = __expf(s - mn);
        l = l * alpha + p;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[j] = acc[j] * alpha
                     + p * __half2float(v[(size_t)n * FA_D + lane * 4 + j]);
        m = mn;
    }
    half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
    #pragma unroll
    for (int j = 0; j < 4; ++j) o[lane * 4 + j] = __float2half(acc[j] / l);   // 分母延迟到最后一除
}
void fa2_v0(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    fa2_v0_kernel<<<dim3(S, Hq, B), 32>>>(Q, K, V, O, Hq, Hkv, S, causal);
}
