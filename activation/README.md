# silu_and_mul(SwiGLU 逐元素部分)

*把 `silu(gate) * up` 的中间张量往返消掉*

`out = silu(gate) * up`，其中 `silu(x) = x·sigmoid(x)`。它在每层 MLP 出现一次，是 LLM 前向中张量最大的逐元素算子——intermediate_size 通常是 hidden 的 2.5-4 倍（Qwen3-8B：4096 → 12288），单次调用要搬的字节数比 RMSNorm 还多。

## 性能结果

RTX 4090，bf16，I=12288,3 轮 mean±std。有效带宽按算法下界（读 gate + 读 up + 写 out = 6 B/输出元素）计，超 100% 即数据落在 L2。本页数字统一取自 [project-proof/data/derived_activation_vec-after_stability.csv](project-proof/data/derived_activation_vec-after_stability.csv)（EXP-K05《LLM 融合逐元素算子三件套》、EXP-K08《BF16x8 向量化未兑现的定位与修复》）。

| 版本 | 改动 | HBM(T=8192,600 MB) | L2(T=256,19 MB) | decode(T=1) |
|---|---|---|---|---|
| v0 | 未融合，两个 kernel | 540.4±0.6 GB/s (53.6%) | 1502.8 GB/s | 11.21 us |
| v1 | 融合成单 kernel | 907.8±0.0 GB/s (90.1%) | 2408.1 GB/s | 7.60 us |
| v2 | 16 B 向量化 | 919.0±0.9 GB/s (91.2%) | 2402.2 GB/s | 7.57 us |
| v3 | 打包布局（vLLM 风格） | **928.3±0.3 GB/s (92.1%)** | **2553.0 GB/s** | **7.22 us** |
| PyTorch eager | — | 555.5±0.0 GB/s (55.1%) | 1027.6 GB/s | 17.43 us |
| torch.compile | — | 925.9±0.4 GB/s (91.9%) | 344.4 GB/s | 57.37 us |
| Triton | — | 927.8±0.4 GB/s (92.0%) | 1052.0 GB/s | 17.98 us |

HBM 区间手写 v3 相对 PyTorch eager **1.67x**，相对 torch.compile 与 Triton 均打平； L2 区间相对 torch.compile **7.41x**。

## 关键发现

**融合这一级的实测与字节账预测精确吻合。** 未融合要搬 5 次（读 gate、写 tmp、读 tmp、读 up、写 out），融合后 3 次，理论 5/3 = 1.667x，实测 **1.680x**。被消掉的是一整份与输入等大的中间张量往返——T=8192、I=12288 时是 200 MB 写 + 200 MB 读。

**v0 与 PyTorch eager 同速（540.4 vs 555.5），这是基线的自检条件。** v0 是"手写但不优化"，复刻的正是 eager 的执行方式（两个 kernel、一份临时显存）；若两者差距很大， 说明基线写错了而不是优化有效。这条自检让 1.68x 的加速可信。

**打包布局在算子层零收益（+1.0%），而收益不在被改的地方。** 打包与分离在 HBM 层面搬的字节数完全一样。vLLM 用打包是因为它意味着 gate_proj 与 up_proj 可以合并成一次 gate_up_proj GEMM——少一次 launch、GEMM 的 N 维翻倍从而 tile 利用率更高、权重只需一次读取。这些全部发生在 GEMM 那一侧，算子级 bench 看不到。

## 代码导览

```mermaid
flowchart LR
    v0["v0 两 kernel<br>540 GB/s"] -->|"融合,消中间张量<br>5/3 = 1.67x"| v1["v1 融合<br>908"]
    v1 -->|"16B 向量化"| v2["v2 向量化<br>919"]
    v2 -->|"打包布局(+1%)"| v3["v3 packed<br>928 = 92.1% 峰值"]
```

融合后 silu 的结果留在寄存器直接乘 `up`，不再落显存（摘自 [src/silu_and_mul_v1.cu](src/silu_and_mul_v1.cu)）：

```cuda
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(silu(g) * u);
```

`silu` 的 sigmoid 在 fp32 里算（与 PyTorch 对 bf16 的 opmath 一致）：bf16 只有 8 位尾数，直接在 bf16 上算 exp，输入误差会被放大成输出的相对误差。

- 未融合基线（含为什么它不是稻草人）见 [src/silu_and_mul_v0.cu](src/silu_and_mul_v0.cu)
- 打包布局与「收益为什么不在这里」见 [src/silu_and_mul_v3.cu](src/silu_and_mul_v3.cu)

## 快速开始

```bash
export CUDA_HOME=/usr/local/cuda
python bench.py
```

## 测量方法

同 [../fused-norm/README.md](../fused-norm/README.md) 的「测量方法」节。注意本算子每 token 的工作集是 `3·I` 而不是 RMSNorm 的 `4·H`，所以同样叫 prefill 的形状在这里已经落在 HBM 区间——**每个算子都要单独判定区间，不能套用别的算子的结论**。
