// ============================================================================
// v3 —— 向量化访存:标量 bf16 -> 每线程一次搬 16 字节(8 个 bf16)
//
// 改了什么:只动访存宽度,字节账仍是 5,归约仍是 v2 的 warp shuffle。
//
// 为什么这是本梯最大的一级台阶:
//   GPU 的全局访存以 sector(32B)为最小单位,warp 的 32 条 lane 请求会被
//   合并成若干 128B 事务。标量 bf16 时,一个 warp 一次只覆盖 32*2=64B,
//   即半个事务 —— 硬件仍按整事务的粒度动作,一半带宽被丢弃。改成每 lane
//   16B 后,一个 warp 一次请求 32*16=512B = 4 个满事务,事务利用率 100%。
//   同时每元素的访存指令数降到 1/8,LSU 发射压力与 MIO 队列排队一起下来
//   (这正是 Kernel_Optimazation/gemm EXP-K02（CUDA Tensor Core GEMM 版本梯）里 NCU 给出的
//    "use wider loads" 建议在逐元素算子上的对应做法)。
//
// 预期:v2 -> v3 是主要加速来源(标量版通常只能跑到 HBM 峰值的 40-60%)。
//
// 代价与边界:要求 H % 8 == 0 且行首 16B 对齐。torch 张量基址是 256B 对齐,
// 行首偏移 = H*2 字节,H 为 8 的倍数即满足。Qwen3 的 hidden(1024/4096/8192)
// 与 head_dim(128)全部满足;若不满足,binding 层会拒绝并回落 v2 ——
// 「手写 kernel 的形状敏感性」在这里是硬约束,不是调优选项。
// ============================================================================
#include "fused_norm.h"

// 16 字节 = 8 个 bf16 = 4 个 bf16x2。
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

__global__ void v3_kernel(__nv_bfloat16* __restrict__ out,
                          __nv_bfloat16* __restrict__ residual,
                          const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w,
                          int H, float eps) {
    __shared__ float smem[32];
    const int HV = H >> 3;                       // 行内的 16B 组数
    const long long off = (long long)blockIdx.x * HV;

    BF16x8*       res  = reinterpret_cast<BF16x8*>(residual) + off;
    const BF16x8* xr   = reinterpret_cast<const BF16x8*>(x) + off;
    BF16x8*       orow = reinterpret_cast<BF16x8*>(out) + off;
    const BF16x8* wv   = reinterpret_cast<const BF16x8*>(w);   // 权重按行复用,无 off

    float acc = 0.f;
    for (int i = threadIdx.x; i < HV; i += blockDim.x) {
        BF16x8 rv = res[i];
        BF16x8 xv = xr[i];
        BF16x8 sv;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // 成对转换:__bfloat1622float2 是一条指令处理两个元素,
            // 比拆成两次 __bfloat162float 少一半转换指令。
            float2 r = __bfloat1622float2(rv.h[j]);
            float2 a = __bfloat1622float2(xv.h[j]);
            float2 s = make_float2(r.x + a.x, r.y + a.y);
            acc += s.x * s.x + s.y * s.y;
            sv.h[j] = __float22bfloat162_rn(s);
        }
        res[i] = sv;                              // 一次 16B store
    }

    const float rstd = rsqrtf(block_reduce_sum(acc, smem) / H + eps);

    for (int i = threadIdx.x; i < HV; i += blockDim.x) {
        BF16x8 rv = res[i];                       // 第二遍重读(v4 消掉的就是它)
        BF16x8 wl = wv[i];
        BF16x8 ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 r = __bfloat1622float2(rv.h[j]);
            // 舍入顺序与 v0-v2、vLLM 保持一致:先舍到 bf16 再乘 bf16 权重。
            __nv_bfloat162 n = __float22bfloat162_rn(
                make_float2(r.x * rstd, r.y * rstd));
            ov.h[j] = __hmul2(n, wl.h[j]);        // 一条指令算两个元素
        }
        orow[i] = ov;
    }
}

void fused_add_rmsnorm_v3(__nv_bfloat16* out, __nv_bfloat16* residual,
                          const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int T, int H, float eps, cudaStream_t stream) {
    // blockDim 按「向量组数」而非元素数取:H=4096 时只需 512 线程,
    // 比标量版少 8 倍线程做同样的事 —— 每线程搬得多、指令少,
    // 这正是 thread coarsening 在访存型算子上的形态。
    int bs = ((H / 8 + 31) / 32) * 32;
    bs = max(64, min(1024, bs));
    v3_kernel<<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
}
