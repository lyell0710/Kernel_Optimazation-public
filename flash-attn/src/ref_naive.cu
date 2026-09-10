#include "fa_common.h"
// ============================================================================
// fp32 精算参考:正确性 gate 的对照物。
// 算法:warp-per-row 两遍法——pass0 扫全行求 max,pass1 定基 exp/求和/加权。
// 为什么两遍而不是在线:max 先定死,无 m/l/α 的 rescale 环节,数值与控制
// 路径都与被测的在线单遍法独立——两套实现同点出错的概率极小,gate 才可信。
// 契约:任意 S/causal/GQA;fp32 累加;只求对不求快(每行重算全部 QK)。
// ============================================================================
__global__ void ref_kernel(const half* Q, const half* K, const half* V,
                           half* O, int Hq, int Hkv, int S, bool causal) {
    const int row = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x;
    const int kvh = h / (Hq / Hkv);            // GQA:连续 Hq/Hkv 个 q-head 共享一个 kv-head
    const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    float qr[4];                               // D=128 / 32 lane = 每 lane 持 4 维
    #pragma unroll
    for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    const float scale = rsqrtf((float)FA_D);   // 1/sqrt(D) 标准缩放
    const int nend = causal ? row + 1 : S;     // row+1:causal 含对角(token 可见自身)
    float m = -1e30f;                          // -inf 哨兵;跨两个 pass 存活
    for (int pass = 0; pass < 2; ++pass) {
        float l = 0.f, acc[4] = {0, 0, 0, 0};
        for (int n = 0; n < nend; ++n) {
            float s = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                s += qr[j] * __half2float(k[(size_t)n * FA_D + lane * 4 + j]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)   // 蝶式规约:全 lane 得同一 s,
                s += __shfl_xor_sync(0xffffffff, s, off);   // pass1 每 lane 都要用它
            s *= scale;
            if (pass == 0) { m = fmaxf(m, s); continue; }   // pass0 只收集 max
            float p = __expf(s - m);           // max 已定 → 指数参数恒 <= 0,无上溢
            l += p;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[j] += p * __half2float(v[(size_t)n * FA_D + lane * 4 + j]);
        }
        if (pass == 1) {
            half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                o[lane * 4 + j] = __float2half(acc[j] / l);   // 分母只在最后除一次
        }
    }
}
void attn_ref_fp32(const half* Q, const half* K, const half* V, half* O,
                   int B, int Hq, int Hkv, int S, bool causal) {
    ref_kernel<<<dim3(S, Hq, B), 32>>>(Q, K, V, O, Hq, Hkv, S, causal);
}
