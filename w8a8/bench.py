#!/usr/bin/env python3
"""W8A8 linear 完整链路 bench —— 量化 + INT8 GEMM + 反量化,对照 bf16 cuBLAS。

本 bench 要回答三个问题,而不是"我的 kernel 多快":
  Q1 整条 W8A8 链路相对 bf16 linear 到底快多少?(单算子数字没有意义)
  Q2 三步各占多少?(决定下一步该优化哪儿)
  Q3 权重布局对 INT8 GEMM 的影响有多大?(这一条是本实验的头条发现)

形状取 Qwen3-8B 的线性层(hidden=4096, intermediate=12288),按 prefill 的
token 数分档。decode(T=1)单列:torch._int_mm 要求 M>16,走不通 —— 这是
W8A8 落地的硬约束,必须量出来而不是绕过去。

用法: BENCH_OUT=project-proof/data/<UTC>_w8a8_stability_r1.csv python bench.py
"""
import os, pathlib, sys
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from bench_common import build_ext, timeit, open_csv, rel_err


def quantize_weight(w):
    """离线 per-channel(按输出通道)对称量化。

    权重量化在离线做,不进热路径 —— 这是 W8A8 与「动态量化激活」的关键区别:
    权重是静态的,scale 可以一次算好存下来。
    """
    s = w.abs().amax(dim=1).float().clamp_min(1e-12) / 127.0        # [O]
    q = (w.float() / s[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, s


def main():
    ext = build_ext("w8a8_ext", HERE,
                    ["binding.cpp", "quant_per_token.cu", "dequant.cu",
                     "int8_gemv.cu"])
    dev = "cuda"
    iters_env = int(os.environ.get("BENCH_ITERS", "0"))
    H = 4096
    # (标签, T, O, iters)。O=4096 对应 o_proj,O=12288 对应 gate/up_proj。
    # 【测量效度】同 decode 段的账:4090 的 L2 是 72 MB,int8 权重是 bf16 的一半,
    # 存在一个危险区间——同一个 O 下 int8 权重塞得进 L2 而 bf16 塞不进,
    # 两条臂根本不在同一存储层级上比。H=4096 时:
    #   O=12288 → int8 50 MB(L2 内) vs bf16 100 MB(HBM)  ← 对外报的 2.161x 正落在这里
    #   O=32768 → int8 134 MB       vs bf16 268 MB       ← 两边都超 L2,可外推
    # decode 段早已补了 O=32768,prefill 段一直没有;对外数字出自 prefill,故补齐。
    shapes = [("T512_O4096", 512, 4096, 200),
              ("T512_O12288", 512, 12288, 200),
              ("T2048_O12288", 2048, 12288, 100),
              ("T8192_O12288", 8192, 12288, 50),
              ("T2048_O32768", 2048, 32768, 30)]

    f, out_path = open_csv("project-proof/data/benchmark_results.csv", HERE,
                           f"BENCH_ITERS={iters_env} python bench.py",
                           "shape,arm,T,H,O,latency_ms,tflops_or_tops,"
                           "speedup_vs_bf16,max_rel_err,correctness_pass")

    for tag, T, O, iters in shapes:
        if iters_env:
            iters = iters_env
        g = torch.Generator(device=dev).manual_seed(42)
        x = torch.randn(T, H, generator=g, device=dev, dtype=torch.bfloat16)
        w = torch.randn(O, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.05
        wq, ws = quantize_weight(w)

        # 两种权重布局,内容完全相同,只是 stride 不同。
        # F.linear(x, w) 算的是 x @ w.T;w 是 [O,H] 行主序,w.t() 就是 [H,O] 列主序,
        # 恰好是 INT8 Tensor Core 想要的 NT 布局 —— 这是"免费"的,前提是别在中间
        # 做一次 contiguous() 把它打回行主序。wq_rowmajorT 就是那个反面教材。
        wqT_col = wq.t()                                  # [H,O] 列主序(NT)
        wqT_row = wq.t().contiguous()                     # [H,O] 行主序(被"整理"过)

        xq = torch.empty(T, H, device=dev, dtype=torch.int8)
        xs = torch.empty(T, device=dev, dtype=torch.float32)
        acc = torch.empty(T, O, device=dev, dtype=torch.int32)
        y = torch.empty(T, O, device=dev, dtype=torch.bfloat16)

        ref = torch.nn.functional.linear(x, w)
        flops = 2.0 * T * H * O

        def w8a8(quant, dequant, wt):
            """完整链路:量化 -> INT8 GEMM -> 反量化。

            直接用 _int_mm 返回的张量,不再 copy_ 到预分配缓冲 ——
            第一版写成了 acc.copy_(torch._int_mm(...)),白白多搬一遍
            T x O 的 int32(T=2048/O=12288 时是 100 MB 读 + 100 MB 写),
            占了整条链路近四分之一的时间。分解计时之和对不上总时间,
            就是这类多余搬运的信号。
            """
            quant(xq, xs, x)
            a = torch._int_mm(xq, wt)
            dequant(y, a, xs, ws)
            return y

        arms = []
        arms.append(("bf16_cublas", lambda: torch.nn.functional.linear(x, w)))
        arms.append(("w8a8_v0(quant_v0+dequant_v0)",
                     lambda: w8a8(ext.quant_v0, ext.dequant_v0, wqT_col)))
        arms.append(("w8a8_best(quant_v2+dequant_v1)",
                     lambda: w8a8(ext.quant_v2, ext.dequant_v1, wqT_col)))
        arms.append(("w8a8_best_weight_rowmajor",
                     lambda: w8a8(ext.quant_v2, ext.dequant_v1, wqT_row)))
        # 三步分解:各步单独计时,用于回答"下一步该优化哪儿"
        arms.append(("step_quant_v2", lambda: ext.quant_v2(xq, xs, x)))
        arms.append(("step_int8gemm_col", lambda: torch._int_mm(xq, wqT_col)))
        arms.append(("step_int8gemm_row", lambda: torch._int_mm(xq, wqT_row)))
        arms.append(("step_dequant_v1", lambda: ext.dequant_v1(y, acc, xs, ws)))

        base_ms = None
        for name, fn in arms:
            out = fn()
            # 只有完整链路与 bf16 基线才有可比的输出;分解步骤不判正确性
            err = rel_err(out, ref) if (name.startswith("w8a8") or name == "bf16_cublas") else float("nan")
            ms = timeit(fn, iters)
            if name == "bf16_cublas":
                base_ms = ms
            tf = flops / (ms * 1e-3) / 1e12
            # int8 量化误差远大于 bf16,门槛按量化算子的常规界放到 1e-1;
            # 真正的判据是接进引擎后的 logits 与 token 序列(见引擎侧实验)。
            ok = "" if err != err else ("true" if err < 1e-1 else "false")
            f.write(f"{tag},{name},{T},{H},{O},{ms:.6f},{tf:.1f},"
                    f"{base_ms / ms:.3f},{err:.3e},{ok}\n")
            print(f"{tag:14s} {name:32s} {ms:9.5f} ms {tf:7.1f} "
                  f"{base_ms / ms:6.3f}x err={err:.2e}")
        print()

    # ---- decode(T=1):库路径走不通,手写 int8 GEMV 顶上 ----------------------
    # 这一段单列,是因为 decode 的瓶颈与 prefill 完全不同:prefill 是算力,
    # decode 是权重带宽。同一个 W8A8,两个阶段的收益来源不是一回事。
    # 【测量效度先算账】4090 的 L2 是 72 MB。int8 权重是 bf16 的一半,
    # 存在一个危险区间:同一个 O 下 int8 权重能塞进 L2、bf16 塞不进 ——
    # 两条臂根本不在同一个存储层级上比,int8 的领先会被严重高估。
    # O=12288/H=4096 正是这样:int8 50 MB(L2 内)vs bf16 100 MB(HBM)。
    # 所以必须补一档两边都超 L2 的形状(O=32768:int8 128 MB / bf16 256 MB),
    # 那一档的数字才是可外推的。这是 EXP-K04 教训在 GEMV 上的重演。
    print("=== decode(T=1)===")
    L2_MB = 72
    for O in (4096, 12288, 32768):
        g = torch.Generator(device=dev).manual_seed(7)
        x1 = torch.randn(1, H, generator=g, device=dev, dtype=torch.bfloat16)
        w = torch.randn(O, H, generator=g, device=dev, dtype=torch.bfloat16) * 0.05
        wq, ws = quantize_weight(w)
        xq1 = torch.empty(1, H, device=dev, dtype=torch.int8)
        xs1 = torch.empty(1, device=dev, dtype=torch.float32)
        ext.quant_v2(xq1, xs1, x1)
        # scale 保持在设备上,不 .item()(见 int8_gemv.cu 的说明)
        y1 = torch.empty(O, device=dev, dtype=torch.bfloat16)
        ref1 = torch.nn.functional.linear(x1, w)[0]

        # 库路径先验一次:失败也要记下来,这是硬约束而非性能问题
        try:
            torch._int_mm(xq1, wq.t())
            note = "supported"
        except Exception as e:
            note = f"{type(e).__name__}:{str(e)[:60]}"
        print(f"  O={O} torch._int_mm(M=1) -> {note}")
        f.write(f"decode_T1_O{O},int8gemm_lib,1,{H},{O},nan,nan,nan,nan,{note}\n")

        arms1 = [("bf16_cublas_gemv", lambda: torch.nn.functional.linear(x1, w)[0]),
                 ("int8_gemv_v0", lambda: (ext.gemv_v0(y1, xq1, wq, ws, xs1), y1)[1]),
                 ("int8_gemv_v1_smem", lambda: (ext.gemv_v1(y1, xq1, wq, ws, xs1), y1)[1])]
        base1 = None
        for name, fn in arms1:
            out = fn()
            err = rel_err(out, ref1)
            ms = timeit(fn, iters_env or 2000)
            if base1 is None:
                base1 = ms
            # GEMV 是访存主导:报有效带宽比报 TOPS 有意义。
            # bf16 读 2*O*H 字节,int8 读 O*H —— 分母不同,所以这里统一按
            # 各自实际读的权重字节数算,量的是"离带宽墙多远"。
            wbytes = O * H * (2 if name.startswith("bf16") else 1)
            bw = wbytes / (ms * 1e-3) / 1e9
            regime = "L2" if wbytes / 1e6 < L2_MB else "HBM"
            f.write(f"decode_T1_O{O}_{regime},{name},1,{H},{O},{ms:.6f},{bw:.1f},"
                    f"{base1 / ms:.3f},{err:.3e},{'true' if err < 1e-1 else 'false'}\n")
            print(f"  {name:20s} {ms:8.5f} ms  权重 {wbytes/1e6:5.0f} MB[{regime:3s}] "
                  f"带宽 {bw:7.1f} GB/s  {base1 / ms:5.3f}x  err={err:.2e}")
    f.close()
    print("written:", out_path)


if __name__ == "__main__":
    main()
