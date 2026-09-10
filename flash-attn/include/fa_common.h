#pragma once
#include <cuda_fp16.h>
// ============================================================================
// FA2 forward 简化版版本梯公共接口。
// 问题:单 kernel 融合 QK^T → 在线 softmax → P·V,不物化 S×S 注意力矩阵
// (FlashAttention-2 前向)。
// 布局:Q/O [B,Hq,S,D]、K/V [B,Hkv,S,D],row-major;fp16 存储 / fp32 在线
// 累加;D = 128 固定(FA_D);GQA:要求 Hq % Hkv == 0,query head h 使用
// kv head h / (Hq/Hkv)。
// 契约:v0/v1 任意 S;v2/v3/v4 额外要求 S % 64 == 0(bench 形状均满足,
// 通用尾块由 v0/v1 兜底,EXP-K03（CUDA FA2 forward 简化版版本梯）§7)。指针为 device 指针,launch 异步。
// 正确性:全版本全 shape max_abs_err < 2e-2 vs fp32 两遍参考
// (实测 4.88e-04,EXP-K03 §5)。
// 性能锚(S=4096 协议点:B=1,Hq=32,Hkv=8,causal;4090,3 轮,EXP-K03):
//   v0 4.9 → v1 5.5 → v2 24.4 → v3 32.5 → v4 34.8 TFLOPS
//   = 自家 Triton 版(mma+寄存器驻留)的 28%(跨 harness,推断级)——
//   wmma 架构税的定量测量,机理见 fa2_v2.cu 文件头。
// ============================================================================
void fa2_v0(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v1(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v2(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v4(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v5(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
void fa2_v3(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);
// fp32 精算参考(非在线,两遍法,正确性 gate 专用,慢)
void attn_ref_fp32(const half* Q, const half* K, const half* V, half* O,
                   int B, int Hq, int Hkv, int S, bool causal);
constexpr int FA_D = 128;
