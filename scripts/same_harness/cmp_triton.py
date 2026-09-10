#!/usr/bin/env python3
"""同 harness 复测:本仓 CUDA kernel vs 姊妹仓 triton-kernels 的 Triton 实现。

为「一切 vs Triton/sdpa 数字为跨 harness,推断级」这一限制提供解除条件。
要点是消除四处不对称——**同一进程、同一份数据、同一套 warmup/计时协议、同一次会话**。
此前 CUDA 侧走 C++ bench、Triton 侧走 Python bench,四项全不同,故只能算推断级。

用法:
  TRITON_KERNELS_SRC=$HOME/triton-kernels/src python3 cmp_triton.py
"""
import os, sys, time, statistics as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TSRC = os.environ.get("TRITON_KERNELS_SRC", os.path.expanduser("~/triton-kernels/src"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
ITERS  = int(os.environ.get("ITERS", "50"))
WARMUP = int(os.environ.get("WARMUP", "20"))

def build_ext():
    from torch.utils.cpp_extension import load
    return load(
        name="same_harness_ext",
        sources=[os.path.join(HERE, "binding.cu"),
                 os.path.join(ROOT, "gemm/src/gemm_v4.cu"),
                 os.path.join(ROOT, "gemm/src/gemm_cublas.cu"),
                 os.path.join(ROOT, "flash-attn/src/fa2_v4.cu")],
        extra_include_paths=[os.path.join(ROOT, "gemm/include"),
                             os.path.join(ROOT, "flash-attn/include")],
        extra_cuda_cflags=["-O3", "-arch=sm_89", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcublas"],
        verbose=False)

def timeit(fn):
    """统一计时协议:CUDA event,warmup 后取 ITERS 次平均,重复 ROUNDS 轮。"""
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(ROUNDS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); s.record()
        for _ in range(ITERS): fn()
        e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / ITERS)
    return st.mean(out), (st.stdev(out) if len(out) > 1 else 0.0)

def main():
    dev = "cuda"
    ext = build_ext()
    sys.path.insert(0, TSRC)
    print(f"# 同 harness 对照 | rounds={ROUNDS} iters={ITERS} warmup={WARMUP}")
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    # ---------------- GEMM 4096^3 ----------------
    try:
        import gemm_pipelined as TG
    except Exception as ex:
        TG = None; print(f"# Triton GEMM 不可用: {ex}")
    M = N = K = 4096
    g = torch.Generator(device=dev).manual_seed(1234)
    A = (torch.randn(M, K, generator=g, device=dev, dtype=torch.float16) * 0.05).contiguous()
    B = (torch.randn(K, N, generator=g, device=dev, dtype=torch.float16) * 0.05).contiguous()
    C = torch.empty(M, N, device=dev, dtype=torch.float16)
    ref = (A.float() @ B.float()).half()
    flops = 2.0 * M * N * K
    print(f"\n## GEMM {M}x{N}x{K} fp16  (同一份 A/B,seed=1234)")
    print(f"  {'臂':<22}{'ms':>10}{'±std':>9}{'TFLOPS':>10}{'max_rel_err':>13}")
    rows = []
    for name, fn in [("cuda_v4(本仓)", lambda: ext.gemm_v4(A, B, C)),
                     ("cuda_cublas(本仓)", lambda: ext.gemm_cublas(A, B, C))]:
        fn(); torch.cuda.synchronize()
        err = ((C.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        m, s = timeit(fn); rows.append((name, m, s, err))
    if TG is not None:
        def tfn():
            global _TO
            _TO = TG.gemm(A, B)
        tfn(); torch.cuda.synchronize()
        err = ((_TO.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        m, s = timeit(tfn); rows.append(("triton(姊妹仓)", m, s, err))
    for n_, m, s, e in rows:
        print(f"  {n_:<22}{m:>10.4f}{s:>9.4f}{flops/(m*1e-3)/1e12:>10.1f}{e:>13.2e}")
    if len(rows) >= 3:
        cu = [r for r in rows if r[0].startswith("cuda_v4")][0][1]
        tr = [r for r in rows if r[0].startswith("triton")][0][1]
        print(f"  ==> cuda_v4 / triton = {tr/cu*100:.1f}%  (>100% 表示 CUDA 更快)")

    # ---------------- FA2 协议形状 (B=1,Hq=32,Hkv=8,S=4096,D=128,causal) ----------------
    try:
        import fa2_fwd as TF
    except Exception as ex:
        TF = None; print(f"\n# Triton FA2 不可用: {ex}")
    Bq, Hq, Hkv, S, D = 1, 32, 8, 4096, 128
    g2 = torch.Generator(device=dev).manual_seed(4321)
    q = (torch.randn(Bq, Hq, S, D, generator=g2, device=dev, dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(Bq, Hkv, S, D, generator=g2, device=dev, dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(Bq, Hkv, S, D, generator=g2, device=dev, dtype=torch.float16) * 0.1).contiguous()
    o = torch.empty(Bq, Hq, S, D, device=dev, dtype=torch.float16)
    # FA2 FLOPs(causal 约一半):4*B*Hq*S^2*D/2
    fl = 4.0 * Bq * Hq * S * S * D / 2
    print(f"\n## FA2 B={Bq} Hq={Hq} Hkv={Hkv} S={S} D={D} causal fp16  (同一份 q/k/v,seed=4321)")
    # 参考:torch sdpa(同时也是红线里点名的对照物之一)
    def sdpa():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    try:
        ref2 = sdpa().float(); sdpa_ok = True
    except Exception as ex:
        ref2 = None; sdpa_ok = False; print(f"  (sdpa 不可用: {ex})")
    print(f"  {'臂':<22}{'ms':>10}{'±std':>9}{'TFLOPS':>10}{'max_rel_err':>13}")
    rows2 = []
    def cufn(): ext.fa2_v4(q, k, v, o, Hkv, True)
    cufn(); torch.cuda.synchronize()
    e_cu = ((o.float()-ref2).abs().max()/ref2.abs().max()).item() if sdpa_ok else float('nan')
    m, sd = timeit(cufn); rows2.append(("cuda_v4(本仓)", m, sd, e_cu))
    if TF is not None:
        def tfa():
            global _TFO
            _TFO = TF.fa2_forward(q, k, v, causal=True)
        tfa(); torch.cuda.synchronize()
        e_tf = ((_TFO.float()-ref2).abs().max()/ref2.abs().max()).item() if sdpa_ok else float('nan')
        m, sd = timeit(tfa); rows2.append(("triton(姊妹仓)", m, sd, e_tf))
    if sdpa_ok:
        m, sd = timeit(lambda: sdpa()); rows2.append(("torch_sdpa", m, sd, 0.0))
    for n_, m, sd, e in rows2:
        print(f"  {n_:<22}{m:>10.4f}{sd:>9.4f}{fl/(m*1e-3)/1e12:>10.1f}{e:>13.2e}")
    d = {r[0].split('(')[0]: r[1] for r in rows2}
    if "cuda_v4" in d and "triton" in d:
        print(f"  ==> cuda_v4 / triton = {d['triton']/d['cuda_v4']*100:.1f}%")
    if "cuda_v4" in d and "torch_sdpa" in d:
        print(f"  ==> cuda_v4 / sdpa   = {d['torch_sdpa']/d['cuda_v4']*100:.1f}%")

if __name__ == "__main__":
    main()
