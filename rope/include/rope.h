// ============================================================================
// RoPE(旋转位置编码)—— 共用声明
//
// 算子语义(HF llama/Qwen 约定,即 vLLM rotary_embedding.cu 的 is_neox=true 分支):
// 把 head_dim 的前后两半 (x1, x2) 视作 D/2 个复数的实部与虚部,整体乘以
// e^{i*theta_pos}:
//     out[i]       = x1[i] * cos[i] - x2[i] * sin[i]
//     out[i + D/2] = x2[i] * cos[i] + x1[i] * sin[i]
// 其中 cos/sin 表的前后两半是重复的同一组频率(见 llm-engine
// src/layers.py precompute_rope 的 cat((freqs, freqs))),所以上式右侧
// 两行共用同一个 cos[i]/sin[i] —— 这是本算子能把访存砍半的结构前提。
//
// 【命名陷阱,面试高频】vLLM 把这种「前后半段」布局叫 is_neox=true,
// 而 GPT-J 式的「奇偶交错」叫 is_neox=false。HF 的 llama/Qwen 用的是
// 前后半段,对应 vLLM 的 neox=true。两种布局互换不会报错、不会 NaN,
// 只会让模型输出变成乱码 —— llm-engine EXP-D05（RoPE 对齐）的对拍防的就是这一处。
//
// 为什么是访存主导:每个元素只做 2 次乘加,却要读自己、读配对元素、
// 读 cos、读 sin,再写回。版本梯 v0->v4 就是这条访存账的下降史。
//
// 就地更新(与 vLLM 一致):q/k 被原地旋转,不额外开输出缓冲。
// 旋转是保范变换,反复调用不会让数值发散 —— 这让 bench 可以安全地在
// 同一份缓冲上重复迭代计时。
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// q: [T, HQ, D]  k: [T, HK, D]  cos/sin: [T, D](只用前 D/2 列)
using RopeFn = void (*)(__nv_bfloat16* /*q*/, __nv_bfloat16* /*k*/,
                        const __nv_bfloat16* /*cos*/, const __nv_bfloat16* /*sin*/,
                        int /*T*/, int /*HQ*/, int /*HK*/, int /*D*/, cudaStream_t);

void rope_v0(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
             const __nv_bfloat16*, int, int, int, int, cudaStream_t);
void rope_v1(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
             const __nv_bfloat16*, int, int, int, int, cudaStream_t);
void rope_v2(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
             const __nv_bfloat16*, int, int, int, int, cudaStream_t);
void rope_v3(__nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*,
             const __nv_bfloat16*, int, int, int, int, cudaStream_t);
// v4 额外需要 inv_freq 表(D/2 个 float)与本批起始位置,用于现算角度
void rope_v4(__nv_bfloat16*, __nv_bfloat16*, const float* /*inv_freq*/,
             int /*pos_offset*/, int, int, int, int, cudaStream_t);
