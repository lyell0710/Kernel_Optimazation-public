// ============================================================================
// v1 —— 融合成单 kernel(仍是标量访存 + smem 树形归约)
//
// 改了什么:把 v0 的两个 kernel 合成一个。加法的结果 s = residual + x 在
// 写回显存的同时,顺手在寄存器里累加了 s*s —— 平方和不再需要单独读一遍。
//
// 字节账 6 -> 5(每元素):
//   读 x(1) + 读 res(1) + 写 res(1) + 【第二遍】读 res(1) + 写 out(1)
// 省掉的正是 v0 里 rmsnorm_kernel 的第一遍读。理论加速 6/5 = 1.20x。
//
// 顺带省掉的还有一次 kernel launch(~3-5 us)。在 T 很小(decode,T=1..64)
// 时 launch 占比可以超过 kernel 本身 —— 这也是 triton-kernels#EXP-T03（三件套移植 + torch 绑定）
// 「融合数比单核快慢更重要」在本算子上的对应面。
//
// 为什么第二遍还要重读 res:平方和必须等整行扫完才知道,而 rstd 又是归一化
// 的必要输入。要么重读(v1-v3),要么把整行留在片上(v4)。这是所有单遍归约
// 类算子的同一个岔路口:再读一次显存,还是花寄存器/smem 存下来。
// ============================================================================
#include "fused_norm.h"

__global__ void v1_kernel(__nv_bfloat16* __restrict__ out,
                          __nv_bfloat16* __restrict__ residual,
                          const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w,
                          int H, float eps) {
    extern __shared__ float smem[];
    const long long off = (long long)blockIdx.x * H;
    __nv_bfloat16* res = residual + off;
    const __nv_bfloat16* xr = x + off;
    __nv_bfloat16* orow = out + off;

    // ---- 第一遍:加法 + 写回 + 平方和,三件事一次读完成 ----
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float s = __bfloat162float(res[i]) + __bfloat162float(xr[i]);
        res[i] = __float2bfloat16(s);   // 立刻写回:下一层要拿它当残差
        // 注意这里累加的是 fp32 的 s,而不是舍回 bf16 再读出来的值。
        // 与 vLLM 一致。若改成累加 __bfloat162float(res[i]) 的回读值,
        // 结果会多一次舍入误差,且强制了写后读依赖,编译器无法把 store
        // 与后续计算重叠 —— 正确性差异极小,性能差异可观。
        acc += s * s;
    }

    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(smem[0] / H + eps);
    __syncthreads();

    // ---- 第二遍:重读 residual,归一化 ----
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        __nv_bfloat16 n = __float2bfloat16(__bfloat162float(res[i]) * rstd);
        orow[i] = __hmul(n, w[i]);
    }
}

void fused_add_rmsnorm_v1(__nv_bfloat16* out, __nv_bfloat16* residual,
                          const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int T, int H, float eps, cudaStream_t stream) {
    int bs = ((H + 31) / 32) * 32;
    bs = max(128, min(1024, bs));
    v1_kernel<<<T, bs, bs * sizeof(float), stream>>>(out, residual, x, w, H, eps);
}
