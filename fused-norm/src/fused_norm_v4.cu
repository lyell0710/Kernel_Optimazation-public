// ============================================================================
// v4 —— 寄存器缓存:消掉第二遍的全局重读
//
// 改了什么:第一遍算出的 s = residual + x 舍成 bf16 后,除了写回显存,还留
// 一份在寄存器里;第二遍直接用寄存器里的副本归一化,不再回显存读。
//
// 字节账 5 -> 4(每元素):
//   读 x(1) + 读 res(1) + 写 res(1) + 写 out(1)
// 理论加速 5/4 = 1.25x —— 这是本算子在不改变语义前提下的访存下界:
// 两个输入必须读、两个输出必须写,一字节都省不掉了。
//
// 缓存的是「舍回 bf16 之后」的值而不是 fp32 的 s:
//   ① 与 v3 的重读值逐位相同,两版结果必须完全一致(这是 v4 的正确性自检:
//      maxrel 应当是 0 而不是「很小」);
//   ② 寄存器占用减半(bf16x8 = 4 个 32 位寄存器 / wave,fp32 要 8 个)。
//
// 代价:每线程要为它负责的每一「波」留 4 个寄存器。波数 = ceil(H/8/blockDim),
// 用模板参数固定成编译期常量,数组下标才能被完全展开到寄存器;若写成运行期
// 变量下标,nvcc 会把 cache[] 放进 local memory(其实就是显存),
// 优化不仅归零还会更慢 —— 这是寄存器缓存最常见的翻车方式,
// 在 ncu 里表现为 local load/store 计数非零。
// ============================================================================
#include "fused_norm.h"

struct alignas(16) BF16x8 {
    // 为什么不是裸的 `__nv_bfloat162 h[4]`:那样写 nvcc 按**成员类型**逐个生成访存,
    // 编出来是 4 条 32 位 LDG,不是一条 LDG.E.128——alignas(16) 只保证地址对齐,
    // 不强制向量化。实测(RTX 4090 / CUDA 12.8):裸结构体 4×LDG.E + 4×STG.E,
    // 本写法 1×LDG.E.128 + 1×STG.E.128,活指令 56→48。
    // union 给出 float4 视图,显式拷贝语义让整体赋值走 raw 这一条 128 位通路,
    // 于是所有 `BF16x8 v = p[i];` / `p[i] = v;` 的调用点一行都不用改。
    union { float4 raw; __nv_bfloat162 h[4]; };
    __device__ __forceinline__ BF16x8() {}
    __device__ __forceinline__ BF16x8(const BF16x8& o) { raw = o.raw; }
    __device__ __forceinline__ BF16x8& operator=(const BF16x8& o) { raw = o.raw; return *this; }
};

template <int MAX_WAVES>
__global__ void v4_kernel(__nv_bfloat16* __restrict__ out,
                          __nv_bfloat16* __restrict__ residual,
                          const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w,
                          int H, float eps) {
    __shared__ float smem[32];
    const int HV = H >> 3;
    const long long off = (long long)blockIdx.x * HV;

    BF16x8*       res  = reinterpret_cast<BF16x8*>(residual) + off;
    const BF16x8* xr   = reinterpret_cast<const BF16x8*>(x) + off;
    BF16x8*       orow = reinterpret_cast<BF16x8*>(out) + off;
    const BF16x8* wv   = reinterpret_cast<const BF16x8*>(w);

    BF16x8 cache[MAX_WAVES];        // 常驻寄存器:MAX_WAVES * 4 个 32 位寄存器
    float acc = 0.f;

    // ---- 第一遍:读两路 + 写 residual + 攒平方和 + 留副本 ----
    // 用「固定波数 + 越界守卫」而不是 while 循环:循环次数必须是编译期常量,
    // #pragma unroll 才能把 cache[k] 的下标常量化。
#pragma unroll
    for (int k = 0; k < MAX_WAVES; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < HV) {
            BF16x8 rv = res[i];
            BF16x8 av = xr[i];
            BF16x8 sv;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 r = __bfloat1622float2(rv.h[j]);
                float2 a = __bfloat1622float2(av.h[j]);
                float2 s = make_float2(r.x + a.x, r.y + a.y);
                acc += s.x * s.x + s.y * s.y;
                sv.h[j] = __float22bfloat162_rn(s);
            }
            res[i]   = sv;      // 写回显存:下一层的残差流要用
            cache[k] = sv;      // 同一份值留在寄存器:第二遍不再回显存取
        }
    }

    const float rstd = rsqrtf(block_reduce_sum(acc, smem) / H + eps);

    // ---- 第二遍:零全局读,只写 ----
#pragma unroll
    for (int k = 0; k < MAX_WAVES; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < HV) {
            BF16x8 wl = wv[i];      // 权重仍要读,但它只有 H 个元素、跨行复用,
                                    // 命中 L2/L1 概率极高,不计入行级字节账
            BF16x8 ov;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 r = __bfloat1622float2(cache[k].h[j]);
                __nv_bfloat162 n = __float22bfloat162_rn(
                    make_float2(r.x * rstd, r.y * rstd));
                ov.h[j] = __hmul2(n, wl.h[j]);
            }
            orow[i] = ov;
        }
    }
}

void fused_add_rmsnorm_v4(__nv_bfloat16* out, __nv_bfloat16* residual,
                          const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int T, int H, float eps, cudaStream_t stream) {
    const int HV = H / 8;
    int bs = ((HV + 31) / 32) * 32;
    bs = max(64, min(1024, bs));
    const int waves = (HV + bs - 1) / bs;

    // 只实例化 2 的幂个波数:多留的波在 kernel 内被越界守卫跳过,
    // 代价是几条永假的比较指令,换来模板实例从 8 个降到 4 个(编译时间/指令缓存)。
    if (waves <= 1)
        v4_kernel<1><<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
    else if (waves <= 2)
        v4_kernel<2><<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
    else if (waves <= 4)
        v4_kernel<4><<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
    else
        // H > 8*1024*8 = 65536 才会走到这里;当前 LLM 的 hidden 都远小于它。
        v4_kernel<8><<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
}
