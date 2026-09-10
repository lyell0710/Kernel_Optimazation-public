// ============================================================================
// W8A8 linear —— 权重 int8(离线 per-channel)+ 激活 int8(在线 per-token)
//
// 完整链路(这是本子项目存在的理由):
//     ① 量化   x[T,H] bf16  ->  xq[T,H] int8 + x_scale[T] fp32   (per-token 动态)
//     ② GEMM   acc[T,O] int32 = xq @ wq^T                        (INT8 Tensor Core)
//     ③ 反量化 y[t,o] = acc[t,o] * x_scale[t] * w_scale[o] -> bf16
//
// 【为什么必须是完整链路】只把量化算子插进引擎、后面仍走 bf16 GEMM,等于量化完
// 立刻反量化,白白多两次显存往返,端到端必然更慢。W8A8 的收益全部来自 ②:
// INT8 Tensor Core 的吞吐与权重带宽减半。单独测量化算子跑得多快没有意义。
//
// 【per-token 而非 per-tensor】LLM 的激活存在少数离群通道,幅度可比其余大一到两个
// 数量级。整张量共用一个 scale 会让绝大多数元素挤进 int8 的低几位,信息塌掉。
// 按 token 一行一个 scale,把离群的影响限制在它所在的那一行内。
// (权重侧则按输出通道 per-channel,离线算好,不进本算子的热路径。)
//
// 【对称量化,零点固定为 0】scale = max|x| / 127,q = round(x/scale)。
// 不做非对称(zero-point)是因为 INT8 Tensor Core 的累加是纯整数点积,
// 带 zero-point 要额外补一项 sum(w)*zp,得多一次归约与一次广播加法;
// 对称量化让反量化退化成两个标量相乘,可以完全融进 epilogue。
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ---- ① per-token 动态量化 -------------------------------------------------
// x[T,H] bf16 -> q[T,H] int8, scale[T] fp32(scale = rowmax(|x|)/127)
using QuantFn = void (*)(int8_t* /*q*/, float* /*scale*/,
                         const __nv_bfloat16* /*x*/,
                         int /*T*/, int /*H*/, cudaStream_t);

void quant_per_token_v0(int8_t*, float*, const __nv_bfloat16*, int, int, cudaStream_t);
void quant_per_token_v1(int8_t*, float*, const __nv_bfloat16*, int, int, cudaStream_t);
void quant_per_token_v2(int8_t*, float*, const __nv_bfloat16*, int, int, cudaStream_t);

// ---- ③ 反量化 epilogue ----------------------------------------------------
// y[T,O] bf16 = acc[T,O] int32 * x_scale[T] * w_scale[O]
using DequantFn = void (*)(__nv_bfloat16* /*y*/, const int32_t* /*acc*/,
                           const float* /*x_scale*/, const float* /*w_scale*/,
                           int /*T*/, int /*O*/, cudaStream_t);

void dequant_v0(__nv_bfloat16*, const int32_t*, const float*, const float*,
                int, int, cudaStream_t);
void dequant_v1(__nv_bfloat16*, const int32_t*, const float*, const float*,
                int, int, cudaStream_t);

// ---- decode 用的 int8 GEMV(cuBLASLt 的 IMMA 要求 M>16,decode 走不通)------
// y[O] bf16 = (xq[H] . wq[O,H]) * x_scale * w_scale[O]
// x_scale 传设备指针而非主机标量:后者会在每次调用插入一次隐式同步,
// decode 路径上被逐层放大(见 src/int8_gemv.cu 的说明)。
void int8_gemv_v0(__nv_bfloat16*, const int8_t*, const int8_t*, const float*,
                  const float* /*x_scale*/, int /*O*/, int /*H*/, cudaStream_t);
void int8_gemv_v1(__nv_bfloat16*, const int8_t*, const int8_t*, const float*,
                  const float* /*x_scale*/, int /*O*/, int /*H*/, cudaStream_t);

#if defined(__CUDACC__)
// warp 内求最大值:与 fused-norm 的求和版同构,只换算子。
// 用 xor 蝶形而非 down:结束后 32 条 lane 各自持有全局最大值,省一次广播。
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    // 补 0 而非补 -INF:本算子归约的是绝对值,恒 >= 0,补 0 安全;
    // 若换成一般的 max 归约(可能有负数),这里必须补 -INFINITY。
    v = (threadIdx.x < nwarp) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

// bf16 -> int8 的舍入与饱和。
// rintf 用「四舍六入五成双」(banker's rounding),与 PyTorch 的 round 一致;
// 用 truncf 会引入系统性的向零偏置,量化误差的均值不再为 0,逐层累积后
// 表现为 logits 整体偏移 —— 这是量化实现里最隐蔽的一类 bug。
__device__ __forceinline__ int8_t quant_one(float x, float inv_scale) {
    float q = rintf(x * inv_scale);
    // 饱和到 [-127, 127] 而非 [-128, 127]:留出对称区间,避免 -128 无对应正值
    // 造成的轻微不对称(与 vLLM / SmoothQuant 的常规做法一致)。
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return (int8_t)q;
}
#endif  // __CUDACC__
