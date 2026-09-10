// ============================================================================
// fused_add_rmsnorm —— 共用声明与 device 侧归约原语
//
// 算子语义(与 vLLM ops.fused_add_rms_norm、llm-engine src/model.py 一致):
//     residual <- residual + x            (就地,下一层的残差流)
//     out      <- rmsnorm(residual) * w   (本层分支的归一化输入)
// 两个输出都要,是 pre-norm Transformer 每层出现两次的固定组合
// (attention 后一次、MLP 后一次),所以它是 LLM 前向里调用次数第二多的
// 访存型算子(仅次于逐元素激活)。
//
// 为什么值得单独写一个 kernel:朴素实现是「一个 add kernel + 一个 rmsnorm
// kernel」,中间结果 residual 要完整写回显存再完整读回来。本算子完全是
// memory-bound(每元素 ~1 次乘加,却要搬 8-10 字节),优化的唯一杠杆就是
// 减少每元素的全局访存字节数 —— 版本梯 v0->v4 就是这条字节账的下降史。
//
// 精度约定:存储 bf16,归约与归一化在 fp32。bf16 尾数只有 8 位,H=4096 个
// 平方直接在 bf16 上累加会把和的低位全部吃掉(相对误差可达 1e-2 量级),
// 这是 llm-engine src/layers.py rmsnorm() 注释里同一条理由。
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 五个版本共用同一签名,便于 bench 用函数指针表统一调度、吃同一份输入。
// residual 是 in-out:调用后被就地改写为 residual + x(与 vLLM 一致)。
using FusedNormFn = void (*)(__nv_bfloat16* /*out*/,
                             __nv_bfloat16* /*residual*/,
                             const __nv_bfloat16* /*x*/,
                             const __nv_bfloat16* /*w*/,
                             int /*T*/, int /*H*/, float /*eps*/,
                             cudaStream_t);

void fused_add_rmsnorm_v0(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, int, int, float, cudaStream_t);
void fused_add_rmsnorm_v1(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, int, int, float, cudaStream_t);
void fused_add_rmsnorm_v2(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, int, int, float, cudaStream_t);
void fused_add_rmsnorm_v3(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, int, int, float, cudaStream_t);
void fused_add_rmsnorm_v4(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, int, int, float, cudaStream_t);

// ===== 以下 device 侧原语只在 nvcc 的设备编译阶段可见 =====================
// binding.cpp 由主机编译器(c++)编译,它只需要上面的函数声明;若不隔开,
// 主机编译器会在 __shfl_xor_sync / threadIdx 这些设备内建上报「未声明」。
// 这是「一个头文件同时被 .cu 和 .cpp 包含」时的标准做法。
#if defined(__CUDACC__)

// ---------------------------------------------------------------------------
// warp 内求和:5 轮 __shfl_xor_sync 把 32 个 lane 的值折叠成同一个值。
//
// 为什么用 xor 而不是 down:xor 是「蝶形」交换,结束后 32 个 lane 各自都持有
// 全和,后续不需要再从 lane0 广播一次;down 只有 lane0 拿到全和(见
// cuda-reduce/src/reduce_v6.cu 的用法)。这里下游每个线程都要用 rstd,
// xor 版正好省掉广播。
//
// mask 用 0xffffffff 而非 __activemask():本函数只在「整 warp 都活着」的
// 上下文调用(行长 H 由循环 stride 覆盖,不做 if 提前 return)。若在分支内
// 调用,失活 lane 不参与 shuffle 会读到未定义值 —— 这是 warp 级原语最常见
// 的错用方式,症状是偶发的、随 H 变化的错误结果。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// block 内求和:先 warp 内折叠,每 warp 的和写进 smem[32],再由第 0 个 warp
// 折叠一次。两级结构把 __syncthreads 从 log2(blockDim) 次降到 1 次。
//
// smem 只用 32 个 float(=128B,恰好一个 bank 轮回),各 warp 写 smem[wid]
// 时天然一个 warp 一个 bank,无 bank 冲突。
//
// 尾部处理:lane >= 活跃 warp 数时补 0 —— 若不补,读到的是上一次调用残留的
// 脏值,表现为「行长不是 1024 倍数时结果偶尔偏大」。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    v = (threadIdx.x < nwarp) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    // 只有 warp0 手里是全和,广播给全 block:再借一次 smem,比让每个 warp
    // 都做一遍第二级归约省 31/32 的指令。
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

#endif  // __CUDACC__
