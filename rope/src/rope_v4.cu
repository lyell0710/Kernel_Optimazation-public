// ============================================================================
// v4 —— 免表:在 kernel 内用 __sincosf 现算角度,不读 cos/sin 表
//
// 【变量隔离】本版是在 v3(向量化)之上只改一件事:去掉两路表访存,换成
// 现算。若像最初那样在标量版上做免表,v3->v4 就同时变了「向量化」与
// 「表/现算」两个变量,快慢无法归因 —— 版本梯的每一级必须只改一件事,
// 否则整条梯子讲不出因果。
//
// 改了什么(相对 v3):
//   theta = (pos_offset + token) * inv_freq[i]
//   (c, s) = (cos theta, sin theta)
// inv_freq 只有 D/2 个 float(head_dim=128 时 256 字节),常驻 L1。
//
// 字节账 3 -> 2(每元素):读 2 + 写 2,除以 2 个元素 = 2。
//
// 【这一级是一个可能失败的赌注,预测写在跑之前】用算力换访存,在
// memory-bound 算子上通常是对的。但 cos/sin 表本身很小([T,D],
// head_dim=128、T=32768 时只有 8MB),大概率整份常驻 4090 的 72MB L2 ——
// 也就是说 v3 那两路表访存根本没走到 HBM,v4 省掉的是 L2 命中而不是
// HBM 访问,而代价是 8 次 SFU 调用。预测:v4 ≈ v3(±5%),不排除更慢。
//
// 精度:__sincosf 是硬件近似(SFU),相对误差约 1e-6,远小于 bf16 的尾数
// 分辨率 2^-8 ≈ 4e-3,对本算子无影响。但参数规约有边界:
// theta = pos * inv_freq,最低频 inv_freq≈1、pos 可达 32K 时 theta 到 3e4,
// 硬件规约仍准确;上下文再长一个量级(>1e5 token)需要先手工规约,
// 否则精度会塌。这是免表方案的真实适用边界,不是可以忽略的细节。
// ============================================================================
#include "rope.h"

struct alignas(16) BF16x8 {
    // `alignas(16)` 只保证地址对齐,**不强制向量化访存**:nvcc 按成员类型
    // (__nv_bfloat162,4 B)逐个生成访存,编出来是 4 条 32 位 LDG 而非一条 LDG.E.128。
    // union 给出 float4 视图 + 显式拷贝语义,让整体赋值走 raw 这条 128 位通路,
    // 调用点无须改写。根因、验证与收益见 records/EXP-K08。
    union { float4 raw; __nv_bfloat162 h[4]; };
    __device__ __forceinline__ BF16x8() {}
    __device__ __forceinline__ BF16x8(const BF16x8& o) { raw = o.raw; }
    __device__ __forceinline__ BF16x8& operator=(const BF16x8& o) { raw = o.raw; return *this; }
};

__global__ void v4_kernel(__nv_bfloat16* __restrict__ q,
                          __nv_bfloat16* __restrict__ k,
                          const float* __restrict__ inv_freq,
                          int pos_offset, int HQ, int HK, int D,
                          long long nq_vec, long long total_vec) {
    const int half = D >> 1;
    const int hv   = half >> 3;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total_vec; g += (long long)gridDim.x * blockDim.x) {
        __nv_bfloat16* t;
        long long idx;
        int Hh;
        if (g < nq_vec) { t = q; idx = g;           Hh = HQ; }
        else            { t = k; idx = g - nq_vec;  Hh = HK; }

        const int vi = (int)(idx % hv);
        const long long hp = idx / hv;
        const int tok = (int)(hp / Hh);
        const float pos = (float)(pos_offset + tok);

        BF16x8* p1 = reinterpret_cast<BF16x8*>(t + hp * D) + vi;
        BF16x8* p2 = reinterpret_cast<BF16x8*>(t + hp * D + half) + vi;
        BF16x8 a = *p1, b = *p2;
        BF16x8 o1, o2;

        // inv_freq 的 8 个元素:一次 float4 x2 读完(32B),同 head 的所有
        // token 都读同一段,L1 命中率接近 1。
        const float4* ivf = reinterpret_cast<const float4*>(inv_freq + vi * 8);
        const float4 f0 = ivf[0], f1 = ivf[1];
        const float fr[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 x1 = __bfloat1622float2(a.h[j]);
            float2 x2 = __bfloat1622float2(b.h[j]);
            float c0, s0, c1, s1;
            __sincosf(pos * fr[2 * j],     &s0, &c0);
            __sincosf(pos * fr[2 * j + 1], &s1, &c1);
            o1.h[j] = __float22bfloat162_rn(
                make_float2(x1.x * c0 - x2.x * s0, x1.y * c1 - x2.y * s1));
            o2.h[j] = __float22bfloat162_rn(
                make_float2(x2.x * c0 + x1.x * s0, x2.y * c1 + x1.y * s1));
        }
        *p1 = o1;
        *p2 = o2;
    }
}

void rope_v4(__nv_bfloat16* q, __nv_bfloat16* k, const float* inv_freq,
             int pos_offset, int T, int HQ, int HK, int D, cudaStream_t st) {
    const int hv = (D / 2) / 8;
    const long long nq = (long long)T * HQ * hv;
    const long long nk = (long long)T * HK * hv;
    int blocks = (int)min((nq + nk + 255) / 256, 4096LL);
    v4_kernel<<<blocks, 256, 0, st>>>(q, k, inv_freq, pos_offset, HQ, HK, D,
                                      nq, nq + nk);
}
