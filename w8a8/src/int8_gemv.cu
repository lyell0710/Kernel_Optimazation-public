// ============================================================================
// int8 GEMV —— W8A8 在 decode 阶段的必需件
//
// 为什么必须自己写:cuBLASLt 的 IMMA 路径(torch._int_mm)要求 M > 16,
// 而 decode 时 batch=1、一次一个 token,M 恒为 1,直接不可用。
// 这不是性能问题,是硬约束 —— 它也解释了为什么 vLLM 这类框架的 W8A8 要基于
// CUTLASS 自己写 GEMM/GEMV,而不是调库。
//
// 语义:y[o] = (sum_h xq[h] * wq[o,h]) * x_scale * w_scale[o]
//   xq : [H]     int8  (per-token 动态量化的激活,T=1)
//   wq : [O,H]   int8  行主序(每个输出通道的权重连续)
//   ws : [O]     fp32  per-channel 权重 scale
//
// 【这是访存主导算子,收益来自权重字节减半】
// 字节账:读权重 O*H 字节(int8)。bf16 版要读 2*O*H。所以理论上限就是 2x,
// 与"int8 Tensor Core 吞吐更高"无关 —— decode 的 GEMV 根本用不满算力,
// 是在等权重从显存搬过来。这一点与 prefill 的 GEMM 完全不同,
// 同一个 W8A8 在两个阶段的收益来源不是一回事。
//
// 用 __dp4a:一条指令做 4 个 int8 乘加并累进 int32。它是 Ada 的整数点积指令,
// 不占用 Tensor Core。这里用它主要不是为了算力,是为了让每条访存指令
// 对应更多的有效计算,避免发射瓶颈。
// ============================================================================
#include "w8a8.h"

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---- v0:一个 warp 负责一个输出通道,直接从全局读激活 ----------------------
template <int WARPS>
__global__ void gemv_v0_kernel(__nv_bfloat16* __restrict__ y,
                               const int8_t* __restrict__ xq,
                               const int8_t* __restrict__ wq,
                               const float* __restrict__ ws,
                               const float* __restrict__ xs, int O, int H) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int o = blockIdx.x * WARPS + wid;
    if (o >= O) return;

    const int HV = H >> 4;                      // 每行 16 字节块数
    const int4* w4 = reinterpret_cast<const int4*>(wq + (long long)o * H);
    const int4* x4 = reinterpret_cast<const int4*>(xq);

    int acc = 0;
    for (int i = lane; i < HV; i += 32) {
        int4 wv = w4[i];
        int4 xv = x4[i];        // 激活只有 H 字节,被同 block 的所有 warp 复用,
                                // 几乎必然命中 L1;v1 会把它显式搬进 smem
        // 每个 int 装 4 个 int8:一条 dp4a 完成 4 次乘加并累进 int32
        acc = __dp4a(wv.x, xv.x, acc);
        acc = __dp4a(wv.y, xv.y, acc);
        acc = __dp4a(wv.z, xv.z, acc);
        acc = __dp4a(wv.w, xv.w, acc);
    }
    acc = warp_reduce_sum_i32(acc);
    if (lane == 0)
        y[o] = __float2bfloat16((float)acc * xs[0] * ws[o]);
}

// ---- v1:激活先搬进共享内存,block 内所有 warp 共用 -------------------------
// 一个 block 里 WARPS 个 warp 各算一个输出通道,但它们读的是【同一份】激活。
// v0 里这份激活被读了 WARPS 遍(靠 L1 兜住),v1 显式搬进 smem 后只读一遍全局,
// 且后续访问不再占用 L1 带宽。H=4096 时 smem 只需 4 KB。
template <int WARPS>
__global__ void gemv_v1_kernel(__nv_bfloat16* __restrict__ y,
                               const int8_t* __restrict__ xq,
                               const int8_t* __restrict__ wq,
                               const float* __restrict__ ws,
                               const float* __restrict__ xs, int O, int H) {
    extern __shared__ int4 sx[];
    const int HV = H >> 4;
    for (int i = threadIdx.x; i < HV; i += blockDim.x)
        sx[i] = reinterpret_cast<const int4*>(xq)[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int o = blockIdx.x * WARPS + wid;
    // 注意:__syncthreads 必须在这个 return 之前 —— 越界的 warp 也要参与
    // 上面的协作搬运与同步,提前 return 会让 block 内其余 warp 永远等不到它。
    // 这是 smem 协作里最容易写错的一处。
    if (o >= O) return;

    const int4* w4 = reinterpret_cast<const int4*>(wq + (long long)o * H);
    int acc = 0;
    for (int i = lane; i < HV; i += 32) {
        int4 wv = w4[i];
        int4 xv = sx[i];
        acc = __dp4a(wv.x, xv.x, acc);
        acc = __dp4a(wv.y, xv.y, acc);
        acc = __dp4a(wv.z, xv.z, acc);
        acc = __dp4a(wv.w, xv.w, acc);
    }
    acc = warp_reduce_sum_i32(acc);
    if (lane == 0)
        y[o] = __float2bfloat16((float)acc * xs[0] * ws[o]);
}

// x_scale 以【设备指针】传入而不是主机标量。
// 这不是风格问题:量化 kernel 刚把 scale 算在显存里,若在主机侧用 .item() 取回来
// 再当参数传,每次调用都会插入一次设备到主机的隐式同步。decode 时每层每个线性层
// 各同步一次(8B 上是 36 层 x 7 个线性层 = 252 次/token),整条流水被切碎 ——
// 实测这一处让 decode 从应赢约 2x 变成净亏 19%。
// 「顺手 .item() 一下」是量化/归一化这类"算出一个标量再用"的算子里最常见的性能陷阱。
void int8_gemv_v0(__nv_bfloat16* y, const int8_t* xq, const int8_t* wq,
                  const float* ws, const float* xs, int O, int H, cudaStream_t st) {
    constexpr int WARPS = 8;                    // 256 线程/block
    gemv_v0_kernel<WARPS><<<(O + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(
        y, xq, wq, ws, xs, O, H);
}

void int8_gemv_v1(__nv_bfloat16* y, const int8_t* xq, const int8_t* wq,
                  const float* ws, const float* xs, int O, int H, cudaStream_t st) {
    constexpr int WARPS = 8;
    const size_t smem = (size_t)(H / 16) * sizeof(int4);
    gemv_v1_kernel<WARPS><<<(O + WARPS - 1) / WARPS, WARPS * 32, smem, st>>>(
        y, xq, wq, ws, xs, O, H);
}
