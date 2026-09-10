// ============================================================================
// v3 —— 向量化:一个线程处理 8 对(= 16 字节 x 4 路)
//
// 改了什么:只动访存宽度,字节账仍是 3/元素,配对与合并 launch 沿用 v2。
// 每个线程一次搬 4 个 16B 块:前半 8 个 bf16、后半 8 个 bf16、cos 8 个、
// sin 8 个;算完写回 2 个 16B 块。
//
// 为什么有用(同 fused-norm v3 的理由):标量 bf16 时一个 warp 一次只覆盖
// 32*2=64B,不足一个 128B 事务,一半带宽被丢弃;16B/lane 后一次请求
// 32*16=512B = 4 个满事务。
//
// 形状约束:需要 D % 16 == 0(即 half % 8 == 0)。Qwen3 的 head_dim=128 满足,
// llama 系的 64/128 也满足。不满足时 binding 层回落 v2 —— 这是手写 kernel
// 的形状敏感性,写死在校验里而不是让它悄悄算错。
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

__global__ void v3_kernel(__nv_bfloat16* __restrict__ q,
                          __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ cosb,
                          const __nv_bfloat16* __restrict__ sinb,
                          int HQ, int HK, int D,
                          long long nq_vec, long long total_vec) {
    const int half = D >> 1;
    const int hv   = half >> 3;          // 每个 head 的前半有多少个 16B 块
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total_vec; g += (long long)gridDim.x * blockDim.x) {
        __nv_bfloat16* t;
        long long idx;
        int Hh;
        if (g < nq_vec) { t = q; idx = g;           Hh = HQ; }
        else            { t = k; idx = g - nq_vec;  Hh = HK; }

        const int vi  = (int)(idx % hv);             // head 内第几个 16B 块
        const long long hp = idx / hv;               // 第几个 (token, head)
        const int tok = (int)(hp / Hh);

        // 两路数据指针:前半块与后半块;两路表指针:cos 与 sin
        BF16x8* p1 = reinterpret_cast<BF16x8*>(t + hp * D) + vi;
        BF16x8* p2 = reinterpret_cast<BF16x8*>(t + hp * D + half) + vi;
        const BF16x8* pc = reinterpret_cast<const BF16x8*>(cosb + (long long)tok * D) + vi;
        const BF16x8* ps = reinterpret_cast<const BF16x8*>(sinb + (long long)tok * D) + vi;

        BF16x8 a = *p1, b = *p2, cv = *pc, sv = *ps;
        BF16x8 o1, o2;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 x1 = __bfloat1622float2(a.h[j]);
            float2 x2 = __bfloat1622float2(b.h[j]);
            float2 c  = __bfloat1622float2(cv.h[j]);
            float2 s  = __bfloat1622float2(sv.h[j]);
            o1.h[j] = __float22bfloat162_rn(
                make_float2(x1.x * c.x - x2.x * s.x, x1.y * c.y - x2.y * s.y));
            o2.h[j] = __float22bfloat162_rn(
                make_float2(x2.x * c.x + x1.x * s.x, x2.y * c.y + x1.y * s.y));
        }
        *p1 = o1;
        *p2 = o2;
    }
}

void rope_v3(__nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* cosb,
             const __nv_bfloat16* sinb, int T, int HQ, int HK, int D,
             cudaStream_t st) {
    const int hv = (D / 2) / 8;
    const long long nq = (long long)T * HQ * hv;
    const long long nk = (long long)T * HK * hv;
    int blocks = (int)min((nq + nk + 255) / 256, 4096LL);
    v3_kernel<<<blocks, 256, 0, st>>>(q, k, cosb, sinb, HQ, HK, D, nq, nq + nk);
}
