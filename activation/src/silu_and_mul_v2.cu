// ============================================================================
// v2 —— 向量化:每线程一次搬 16 字节(8 个 bf16)
//
// 改了什么:只动访存宽度,字节账仍是 3/元素。
// 标量 bf16 时一个 warp 一次只覆盖 32*2=64B,不足一个 128B 事务,
// 一半带宽被丢弃;16B/lane 后一次请求 32*16=512B = 4 个满事务。
// (同 fused-norm v3 的理由;在那个算子上这一级因为 v2 已贴带宽墙而零收益,
//  这里 v1 是标量版、离墙还远,所以本级应当有实收益 —— 同一手法在两条
//  梯子上收益不同,取决于上一级离墙多远。)
//
// 形状约束:n % 8 == 0。intermediate_size 都是 128 的倍数,天然满足;
// 不满足时 binding 回落 v1。
// ============================================================================
#include "activation.h"

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

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void v2_kernel(__nv_bfloat16* __restrict__ out,
                          const __nv_bfloat16* __restrict__ gate,
                          const __nv_bfloat16* __restrict__ up,
                          long long nv) {
    BF16x8* o = reinterpret_cast<BF16x8*>(out);
    const BF16x8* g = reinterpret_cast<const BF16x8*>(gate);
    const BF16x8* u = reinterpret_cast<const BF16x8*>(up);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < nv; i += (long long)gridDim.x * blockDim.x) {
        BF16x8 gv = g[i], uv = u[i], ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 a = __bfloat1622float2(gv.h[j]);
            float2 b = __bfloat1622float2(uv.h[j]);
            ov.h[j] = __float22bfloat162_rn(
                make_float2(silu(a.x) * b.x, silu(a.y) * b.y));
        }
        o[i] = ov;
    }
}

void silu_and_mul_v2(__nv_bfloat16* out, const __nv_bfloat16* gate,
                     const __nv_bfloat16* up, long long n, cudaStream_t st) {
    const long long nv = n / 8;
    int blocks = (int)min((nv + 255) / 256, 4096LL);
    v2_kernel<<<blocks, 256, 0, st>>>(out, gate, up, nv);
}
