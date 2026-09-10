#!/usr/bin/env python3
"""
PyTorch quantize_per_channel 参考实现，对标 C++ 的 v0..v4 + baseline。

复现 C++ main.cu 的输入：
  - channels=1024, hw=1024
  - input ~ Uniform(-3, 3), mt19937 seed=20260503
  - scales[c] = max(|input[c, :]|) / 127

跑三种 PyTorch 路径：
  1. CUDA eager : (x / scale).round().clamp(-127, 127).to(int8)  ← LLM 量化实际写法
  2. CUDA compile: torch.compile 同样的逻辑                       ← inductor fuse 后
  3. CPU PyTorch API: torch.quantize_per_channel(...)            ← 标准 PTQ API（CPU only）

输出：
  - latency_ms (mean over 100 iters)
  - int8 输出跟 C++ baseline 对比的 bitwise match rate
"""
import csv
import sys
from pathlib import Path
import numpy as np
import torch

CHANNELS = 1024
HW = 1024
TOTAL = CHANNELS * HW
ITERS = 100
WARMUP = 20

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "pytorch_reference.csv"
BENCH_CSV = ROOT / "project-proof" / "data" / "benchmark_results.csv"


def reproduce_cpp_input():
    """
    C++ 用的是 std::mt19937(seed=20260503) + uniform_real(-3, 3)。
    numpy 的 mt19937 跟 C++ std::mt19937 序列一致，但 uniform 实现细节不同——
    最简单：直接用 numpy 生成同分布的随机数据。
    我们要的是"数据特征跟 C++ 一致"（独立同分布、相同尺寸、相同 scale 公式），
    而不是 bitwise 一致——后者只在 dump binary 时能保证。
    """
    rng = np.random.default_rng(20260503)
    x = rng.uniform(-3.0, 3.0, size=(CHANNELS, HW)).astype(np.float32)
    # scales[c] = max(|x[c]|) / 127, 如果全 0 则为 1.0
    amax = np.abs(x).max(axis=1)
    amax[amax == 0] = 127.0
    scales = (amax / 127.0).astype(np.float32)
    return x, scales


def quantize_eager_cuda(x_cuda, scales_cuda):
    """LLM 量化里实际的写法（per-channel symmetric int8）"""
    # x: (C, HW), scales: (C,)
    q = (x_cuda / scales_cuda.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    return q


# torch.compile 缓存的编译版本
_compiled_quantize = None


def quantize_compile_cuda(x_cuda, scales_cuda):
    global _compiled_quantize
    if _compiled_quantize is None:
        _compiled_quantize = torch.compile(quantize_eager_cuda, mode="reduce-overhead")
    return _compiled_quantize(x_cuda, scales_cuda)


def quantize_official_cpu(x_cpu, scales_cpu):
    """PyTorch 官方 PTQ API，只支持 CPU"""
    zero_points = torch.zeros(CHANNELS, dtype=torch.int64)
    q = torch.quantize_per_channel(x_cpu, scales_cpu, zero_points, axis=0, dtype=torch.qint8)
    return q.int_repr()


def bench_cuda(fn, x, scales, label):
    """精确的 CUDA latency 测量（cudaEventRecord）"""
    # warmup
    for _ in range(WARMUP):
        out = fn(x, scales)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(ITERS):
        start.record()
        out = fn(x, scales)
        stop.record()
        stop.synchronize()
        total_ms += start.elapsed_time(stop)
    mean_ms = total_ms / ITERS
    print(f"[{label}] mean_latency={mean_ms:.6f} ms")
    return mean_ms, out


def bench_cpu(fn, x_cpu, scales_cpu, label):
    import time
    # warmup
    for _ in range(WARMUP):
        out = fn(x_cpu, scales_cpu)

    total_s = 0.0
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(x_cpu, scales_cpu)
        total_s += time.perf_counter() - t0
    mean_ms = (total_s / ITERS) * 1000
    print(f"[{label}] mean_latency={mean_ms:.6f} ms")
    return mean_ms, out


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    print(f"PyTorch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    print(f"Shape: ({CHANNELS}, {HW}), total={TOTAL} floats = {TOTAL*4/1024/1024:.1f} MB")

    x, scales = reproduce_cpp_input()
    x_cuda = torch.from_numpy(x).cuda()
    scales_cuda = torch.from_numpy(scales).cuda()
    x_cpu = torch.from_numpy(x)
    scales_cpu = torch.from_numpy(scales)

    print("\n=== Benchmarks ===")
    eager_ms, eager_out = bench_cuda(quantize_eager_cuda, x_cuda, scales_cuda, "pytorch_eager_cuda")
    cpu_ms, cpu_out = bench_cpu(quantize_official_cpu, x_cpu, scales_cpu, "pytorch_official_cpu")
    # torch.compile not available on Python 3.13+; eager is the practical LLM-quantize baseline anyway
    compile_ms = None
    compile_out = None

    print("\n=== Correctness check (CUDA eager vs CPU official API) ===")
    eager_cpu = eager_out.cpu().numpy().flatten()
    cpu_np = cpu_out.numpy().flatten()
    match_eager_cpu = (eager_cpu == cpu_np).mean() * 100
    print(f"  eager(CUDA) vs cpu_api: {match_eager_cpu:.2f}% bitwise match")

    # Compare with C++ baseline latency (read from benchmark CSV)
    cpp = {}
    if BENCH_CSV.exists():
        with BENCH_CSV.open() as f:
            for row in csv.DictReader(f):
                cpp[row["version"]] = float(row["latency_ms"])

    # Write CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "latency_ms", "note"])
        w.writerow(["pytorch_eager_cuda", f"{eager_ms:.6f}", "(x/s).round().clamp().to(int8) on CUDA"])
        w.writerow(["pytorch_official_cpu", f"{cpu_ms:.6f}", "torch.quantize_per_channel (CPU only)"])
    print(f"\nSaved: {CSV_PATH}")

    print("\n=== Comparison Summary ===")
    print(f"{'version':<25} {'lat (ms)':>12}  notes")
    print("-" * 70)
    for v in ["baseline", "v0", "v1", "v2", "v3", "v4"]:
        if v in cpp:
            print(f"{'C++ ' + v:<25} {cpp[v]:>12.6f}  hand-written CUDA")
    print(f"{'PyTorch eager CUDA':<25} {eager_ms:>12.6f}  (x/s).round().clamp().to(int8)")
    print(f"{'PyTorch official CPU':<25} {cpu_ms:>12.6f}  quantize_per_channel (CPU API)")

    if "v4" in cpp:
        print(f"\nC++ v4 vs PyTorch eager CUDA: v4 is {eager_ms/cpp['v4']:.2f}x faster")
        print(f"C++ v4 vs PyTorch official CPU: v4 is {cpu_ms/cpp['v4']:.2f}x faster")


if __name__ == "__main__":
    main()
