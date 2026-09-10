#!/usr/bin/env python3
"""silu_and_mul(SwiGLU 逐元素部分)版本梯 bench —— 七条臂,同一 harness。

七条臂:v0(未融合) v1(融合标量) v2(向量化) v3(打包布局)
        pytorch_eager torch_compile triton

形状取 Qwen3-8B 的 MLP 中间维(I=12288),四个区间:
  decode  T=1     —— 逐 token 解码,launch 主导
  l2      T=256   —— 工作集 ~19MB,L2 常驻
  prefill T=2048  —— 工作集 ~150MB,已超 4090 的 72MB L2
  hbm     T=8192  —— 工作集 ~600MB,确定落 HBM
注意本算子的每 token 工作集是 3*I 而不是 RMSNorm 的 4*H,所以「prefill」
这一档在这里已经是 HBM 区间了 —— 同样叫 prefill,不同算子落在不同区间,
这正是为什么每个算子都要单独判定区间而不能套用别的算子的结论。

用法: BENCH_OUT=project-proof/data/<UTC>_activation_stability_r1.csv python bench.py
"""
import os, pathlib, sys
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from bench_common import (PEAK_GBPS, build_ext, timeit, open_csv,
                          load_triton, rel_err)


def eager(gate, up):
    """参考实现 = llm-engine src/model.py 的 `F.silu(g) * u`。

    这条臂会物化 silu(g) 的中间张量(与 gate 等大),再做一次逐元素乘 ——
    两个 kernel、一份临时显存。手写 kernel 省掉的正是这份往返。
    """
    return F.silu(gate) * up


def main():
    ext = build_ext("activation_ext", HERE,
                    ["binding.cpp", "silu_and_mul_v0.cu", "silu_and_mul_v1.cu",
                     "silu_and_mul_v2.cu", "silu_and_mul_v3.cu"])
    dev = "cuda"
    I = 12288                       # Qwen3-8B 的 intermediate_size
    iters_env = int(os.environ.get("BENCH_ITERS", "0"))
    shapes = [("decode", 1, 2000), ("l2", 256, 1000),
              ("prefill", 2048, 200), ("hbm", 8192, 60)]

    compiled = torch.compile(eager, dynamic=False)
    triton_act = load_triton("silu_and_mul")

    f, out_path = open_csv("project-proof/data/benchmark_results.csv", HERE,
                           f"BENCH_ITERS={iters_env} python bench.py",
                           "regime,version,T,I,latency_ms,eff_bw_GBps,pct_peak,"
                           "speedup_vs_v0,max_rel_err,correctness_pass")

    for tag, T, iters in shapes:
        if iters_env:
            iters = iters_env
        g = torch.Generator(device=dev).manual_seed(42)
        gate = torch.randn(T, I, generator=g, device=dev, dtype=torch.bfloat16)
        up = torch.randn(T, I, generator=g, device=dev, dtype=torch.bfloat16)
        # 打包布局的输入:同一张量的前后两半,内容与分离布局完全一致,
        # 保证 v3 与 v0-v2 算的是同一道题。
        gate_up = torch.cat([gate, up], dim=1).contiguous()
        out = torch.empty_like(gate)
        ref = eager(gate, up)

        plan = []
        for name in ("v0", "v1", "v2"):
            fn = getattr(ext, name)
            plan.append((name,
                         (lambda fn=fn: (fn(out, gate, up), out)[1]),
                         (lambda fn=fn: (lambda: fn(out, gate, up)))))
        plan.append(("v3",
                     (lambda: (ext.v3(out, gate_up), out)[1]),
                     (lambda: (lambda: ext.v3(out, gate_up)))))
        plan.append(("pytorch_eager", (lambda: eager(gate, up)),
                     (lambda: (lambda: eager(gate, up)))))
        plan.append(("torch_compile", (lambda: compiled(gate, up)),
                     (lambda: (lambda: compiled(gate, up)))))
        if triton_act is not None:
            plan.append(("triton", (lambda: triton_act(gate, up, out).clone()),
                         (lambda: (lambda: triton_act(gate, up, out)))))

        v0_ms = None
        for name, once, loop_setup in plan:
            err = rel_err(once(), ref)
            ms = timeit(loop_setup(), iters)
            if name == "v0":
                v0_ms = ms
            # 算法下界:读 gate + 读 up + 写 out = 3 * 2B = 6 字节/输出元素
            bw = 6 * T * I / (ms * 1e-3) / 1e9
            f.write(f"{tag},{name},{T},{I},{ms:.6f},{bw:.1f},"
                    f"{bw / PEAK_GBPS * 100:.1f},{v0_ms / ms:.3f},{err:.3e},"
                    f"{'true' if err < 2e-2 else 'false'}\n")
            print(f"{tag:8s} {name:14s} {ms:9.5f} ms {bw:7.1f} GB/s "
                  f"({bw / PEAK_GBPS * 100:5.1f}%) {v0_ms / ms:6.3f}x err={err:.2e}")
        print()
    f.close()
    print("written:", out_path)


if __name__ == "__main__":
    main()
