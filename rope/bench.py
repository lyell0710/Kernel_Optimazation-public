#!/usr/bin/env python3
"""RoPE 版本梯 bench —— 八条臂,同一 harness。

八条臂:v0(朴素一线程一元素) v1(配对) v2(q/k 合并 launch) v3(向量化)
        v4(免表现算) pytorch_eager torch_compile triton

形状取自 Qwen3-8B 的注意力布局(HQ=32, HK=8 的 GQA, head_dim=128),
按四个区间给:
  decode   T=1      —— 引擎逐 token 解码,一次 launch 就能主导总时间
  decode64 T=64     —— 连续批处理下的典型 batch
  prefill  T=2048   —— 2K prefill,总工作集 ~21MB,L2 常驻
  hbm      T=32768  —— 总工作集 ~336MB,确定落 HBM
(两个区间都测的理由见 EXP-K04（标准库基准补齐与两区间重测）:memory-bound 算子只测小尺寸会量到 L2 带宽。)

用法: BENCH_OUT=project-proof/data/<UTC>_rope_stability_r1.csv python bench.py
"""
import math, os, pathlib, sys
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from bench_common import (PEAK_GBPS, build_ext, timeit, open_csv,
                          load_triton, rel_err)

ROPE_THETA = 10000.0     # Qwen3/llama 的默认基频(llm-engine src/config.py)


def precompute(T, D, device, pos_offset=0):
    """生成 cos/sin 表与 inv_freq,与 llm-engine src/layers.py precompute_rope 同式。

    表在 fp32 里生成再降 bf16:theta^(-2i/d) 跨多个数量级,直接在 bf16 里
    算高频项会丢精度。cat((freqs, freqs)) 的「前后半重复」必须与
    rotate_half 的「前后半交换」配对 —— 换成 GPT-J 的奇偶交错要两处一起改,
    错配不报错、只输出乱码。
    """
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, D, 2, dtype=torch.float32,
                                             device=device) / D))
    t = torch.arange(pos_offset, pos_offset + T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)                       # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)           # [T, D]
    return emb.cos().bfloat16(), emb.sin().bfloat16(), inv


def eager_rope(q, k, cosb, sinb):
    """参考实现 —— 与 llm-engine src/layers.py apply_rope 逐行对应,
    只是把 [B,H,S,D] 换成本 bench 的 [T,H,D] 布局。

    这条臂会物化 rotate_half 的中间张量(cat 出一份与 q 等大的新张量),
    再做两次乘法一次加法 —— 四个 kernel、三份临时显存。手写 kernel 省掉的
    正是这些中间量,不是省算力。
    """
    def rot(x):
        c = cosb.unsqueeze(1)      # [T,1,D] 广播到 [T,H,D]
        s = sinb.unsqueeze(1)
        half = x.shape[-1] // 2
        x2 = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * c + x2 * s
    return rot(q), rot(k)


def main():
    ext = build_ext("rope_ext", HERE,
                    ["binding.cpp", "rope_v0.cu", "rope_v1.cu",
                     "rope_v2.cu", "rope_v3.cu", "rope_v4.cu"])
    dev = "cuda"
    HQ, HK, D = 32, 8, 128          # Qwen3-8B 的 GQA 布局
    iters_env = int(os.environ.get("BENCH_ITERS", "0"))
    shapes = [("decode", 1, 2000), ("decode64", 64, 2000),
              ("prefill", 2048, 300), ("hbm", 32768, 30)]

    compiled = torch.compile(eager_rope, dynamic=False)
    triton_rope = load_triton("rope")

    f, out_path = open_csv("project-proof/data/benchmark_results.csv", HERE,
                           f"BENCH_ITERS={iters_env} python bench.py",
                           "regime,version,T,HQ,HK,D,latency_ms,eff_bw_GBps,"
                           "pct_peak,speedup_vs_v0,max_rel_err,correctness_pass")

    for tag, T, iters in shapes:
        if iters_env:
            iters = iters_env
        g = torch.Generator(device=dev).manual_seed(42)
        q0 = torch.randn(T, HQ, D, generator=g, device=dev, dtype=torch.bfloat16)
        k0 = torch.randn(T, HK, D, generator=g, device=dev, dtype=torch.bfloat16)
        cosb, sinb, inv = precompute(T, D, dev)

        # 参考输出:eager 是 out-of-place,直接拿 q0/k0 算,不会污染输入
        rq, rk = eager_rope(q0, k0, cosb, sinb)

        def mk_inplace_arm(call):
            """就地算子的臂(手写 v0-v4 与 Triton 共用)。

            关键:clone 必须在计时闭包【外面】。第一版把 clone 写进了被计时的
            函数里,Triton 臂每次迭代白白多搬一遍 q+k(hbm 形状下 320MB),
            测出来「Triton 慢 2 倍」——单独测那个 kernel 其实有 907 GB/s,
            与手写 CUDA 持平。就地算子的 bench 最容易在这里翻车:
            正确性需要干净副本,时延不需要,两者必须分开。
            """
            def once():
                q, k = q0.clone(), k0.clone()      # 正确性:必须在干净副本上判
                call(q, k)
                return q, k
            def loop_setup():
                q, k = q0.clone(), k0.clone()       # clone 一次,在计时之外
                # 计时循环里反复旋转同一份缓冲:旋转是保范变换,数值不会发散,
                # 访存字节数与指令数也完全不变,计时口径不受影响。
                return lambda: call(q, k)
            return once, loop_setup

        def kernel_call(name):
            fn = getattr(ext, name)
            return ((lambda q, k: fn(q, k, inv, 0)) if name == "v4"
                    else (lambda q, k: fn(q, k, cosb, sinb)))

        def mk_torch_arm(fn):
            def once():
                return fn(q0, k0, cosb, sinb)
            def loop_setup():
                return lambda: fn(q0, k0, cosb, sinb)
            return once, loop_setup

        plan = [(n,) + mk_inplace_arm(kernel_call(n))
                for n in ("v0", "v1", "v2", "v3", "v4")]
        plan.append(("pytorch_eager",) + mk_torch_arm(eager_rope))
        plan.append(("torch_compile",) + mk_torch_arm(compiled))
        if triton_rope is not None:
            # Triton 版同样是就地算子,必须走 mk_inplace_arm,与手写版同口径
            plan.append(("triton",) + mk_inplace_arm(
                lambda q, k: triton_rope(q, k, cosb, sinb)))

        v0_ms = None
        for name, once, loop_setup in plan:
            oq, ok_ = once()
            err = max(rel_err(oq, rq), rel_err(ok_, rk))
            ms = timeit(loop_setup(), iters)
            if name == "v0":
                v0_ms = ms
            # 算法下界字节数:q/k 各读一次写一次 = 4 * T*(HQ+HK)*D 字节。
            # cos/sin 表只有 T*D 个元素、跨 head 复用 40 次,不计入行级字节账。
            bw = 4 * T * (HQ + HK) * D / (ms * 1e-3) / 1e9
            f.write(f"{tag},{name},{T},{HQ},{HK},{D},{ms:.6f},{bw:.1f},"
                    f"{bw / PEAK_GBPS * 100:.1f},{v0_ms / ms:.3f},{err:.3e},"
                    f"{'true' if err < 2e-2 else 'false'}\n")
            print(f"{tag:9s} {name:14s} {ms:9.5f} ms {bw:7.1f} GB/s "
                  f"({bw / PEAK_GBPS * 100:5.1f}%) {v0_ms / ms:6.3f}x err={err:.2e}")
        print()
    f.close()
    print("written:", out_path)


if __name__ == "__main__":
    main()
