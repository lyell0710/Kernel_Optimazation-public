// ============================================================================
// silu_and_mul —— SwiGLU 的逐元素部分
//
// 算子语义(与 vLLM ops.silu_and_mul、llm-engine src/model.py 的
// `F.silu(g) * u` 一致):
//     out = silu(gate) * up,  silu(x) = x * sigmoid(x)
// 其中 gate = x @ W_gate,up = x @ W_up,两者形状同为 [T, I]。
//
// 它在每层 MLP 里出现一次,是 LLM 前向中张量最大的逐元素算子:
// intermediate_size 通常是 hidden 的 2.5-4 倍(Qwen3-8B:4096 -> 12288),
// 所以单次调用要搬的字节数比 RMSNorm 还多。
//
// 为什么是访存主导:每个输出元素只做一次 sigmoid + 两次乘法,却要读
// gate、读 up、写 out 三次显存。算法下界 = 3 次访存 = 6 字节/输出元素。
//
// 两种布局(版本梯 v3 要测的就是它们的差别):
//   分离(separate):gate 与 up 是两张独立张量 —— 对应 gate_proj / up_proj
//                  两次独立 GEMM,是 HF transformers 与本引擎当前的做法。
//   打包(packed) :gate 与 up 是同一张量 [T, 2I] 的前后两半 —— 对应把两个
//                  投影合并成一次 gate_up_proj GEMM,是 vLLM 的做法。
//                  算子这一层字节数完全相同,真正的收益在上游少一次 GEMM,
//                  只有接进引擎才量得到(见 llm-engine 的接入实验)。
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 分离布局:gate/up 两个指针
using ActFn = void (*)(__nv_bfloat16* /*out*/, const __nv_bfloat16* /*gate*/,
                       const __nv_bfloat16* /*up*/, long long /*n*/, cudaStream_t);

void silu_and_mul_v0(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
                     long long, cudaStream_t);
void silu_and_mul_v1(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
                     long long, cudaStream_t);
void silu_and_mul_v2(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
                     long long, cudaStream_t);
// 打包布局:一个指针 [T, 2I],前半 gate 后半 up
void silu_and_mul_v3(__nv_bfloat16* /*out*/, const __nv_bfloat16* /*gate_up*/,
                     int /*T*/, int /*I*/, cudaStream_t);
