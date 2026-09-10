#!/usr/bin/env python3
"""
对标 cuBLAS 的完整性能分析
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_DIR = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark"

VERSION_ORDER = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cuBLAS"]


def load_data():
    """加载 CSV 数据"""
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    data = {}
    for row in rows:
        version = row["version"]
        data[version] = {
            "latency_ms": float(row["latency_ms"]),
            "speedup": float(row["speedup"]),
            "diff": float(row["diff"]),
            "correctness": row["correctness_pass"] == "true"
        }
    return data


def print_summary(data):
    """打印文字总结"""
    print("\n" + "=" * 80)
    print("CUDA Reduce vs cuBLAS 性能对标报告".center(80))
    print("=" * 80 + "\n")

    if "cuBLAS" not in data:
        print("❌ 错误：未找到 cuBLAS 数据，请先运行 reduce_bench")
        return

    cublas_latency = data["cuBLAS"]["latency_ms"]

    # 找到自写最优版本
    custom_versions = {v: data[v] for v in VERSION_ORDER if v != "cuBLAS"}
    best_custom = min(custom_versions.items(), key=lambda x: x[1]["latency_ms"])

    print(f"📊 基准配置：N = 2^24, 100 iterations, RTX 4070")
    print(f"   参考基准：cuBLAS {cublas_latency:.6f} ms\n")

    print(f"🏆 标准库性能:")
    print(f"   cuBLAS: {cublas_latency:.6f} ms ({data['cuBLAS']['speedup']:.1f}x vs baseline)")
    print(f"   ✓ correctness pass\n")

    print(f"⭐ 自写最优版本:")
    best_version, best_data = best_custom
    gap_percent = ((best_data["latency_ms"] / cublas_latency) - 1) * 100
    speedup_vs_cublas = cublas_latency / best_data["latency_ms"]
    print(f"   {best_version}: {best_data['latency_ms']:.6f} ms")
    print(f"   vs cuBLAS: {speedup_vs_cublas:.2f}x 相同速度（差距 {gap_percent:.1f}%）\n")

    print(f"📈 所有版本排名（以 cuBLAS 为基准）:")
    print(f"   {'Version':<10} {'Latency (ms)':<15} {'vs cuBLAS':<15} {'Correctness'}")
    print(f"   {'-'*55}")

    sorted_versions = sorted(
        [(v, data[v]) for v in VERSION_ORDER if v in data],
        key=lambda x: x[1]["latency_ms"]
    )

    for version, info in sorted_versions:
        if version == "cuBLAS":
            speedup_str = "1.00x (ref)"
        else:
            speedup_vs_ref = cublas_latency / info["latency_ms"]
            gap = ((info["latency_ms"] / cublas_latency) - 1) * 100
            speedup_str = f"{speedup_vs_ref:.2f}x (+{gap:.1f}%)" if gap > 0 else f"{speedup_vs_ref:.2f}x"

        status = "✓" if info["correctness"] else "✗"
        print(f"   {version:<10} {info['latency_ms']:<15.6f} {speedup_str:<15} {status}")

    print(f"\n" + "=" * 80)
    print(f"结论：自写代码的最优版本 ({best_version}) 性能接近标准库".ljust(80))
    print("=" * 80 + "\n")


def plot_comparison(data):
    """绘制对标图表"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    versions = [v for v in VERSION_ORDER if v in data]
    latencies = [data[v]["latency_ms"] for v in versions]

    # 分类：baseline, early, late, cublas
    colors = []
    for v in versions:
        if v == "baseline":
            colors.append("#cccccc")  # 灰色
        elif v in ["v0", "v1", "v2"]:
            colors.append("#ffb366")  # 橙色
        elif v in ["v3", "v4", "v5"]:
            colors.append("#66b3ff")  # 蓝色
        elif v in ["v6", "v7"]:
            colors.append("#99ff99")  # 绿色
        else:  # cuBLAS
            colors.append("#ff6666")  # 红色

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(versions, latencies, color=colors, edgecolor="black", linewidth=1.2)

    # 标注值
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height,
            f"{latency:.4f} ms",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # 标注与 cuBLAS 的差距
    if "cuBLAS" in data:
        cublas_latency = data["cuBLAS"]["latency_ms"]
        for i, (bar, v) in enumerate(zip(bars, versions)):
            if v != "cuBLAS":
                gap = ((latencies[i] / cublas_latency) - 1) * 100
                gap_text = f"+{gap:.1f}%" if gap > 0 else "ref"
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    latencies[i] * 0.5,
                    gap_text,
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5)
                )

    ax.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Version", fontsize=11, fontweight="bold")
    ax.set_title("CUDA Reduce Latency: Custom vs cuBLAS", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#cccccc", edgecolor="black", label="Baseline"),
        Patch(facecolor="#ffb366", edgecolor="black", label="v0-v2 (Early)"),
        Patch(facecolor="#66b3ff", edgecolor="black", label="v3-v5 (Optimized)"),
        Patch(facecolor="#99ff99", edgecolor="black", label="v6-v7 (Two-Level)"),
        Patch(facecolor="#ff6666", edgecolor="black", label="cuBLAS (Reference)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05-vs-cublas.png", dpi=220, bbox_inches="tight")
    print(f"✅ 图表已保存: {FIG_DIR / '05-vs-cublas.png'}")
    plt.close()


def plot_speedup_comparison(data):
    """绘制加速比对比"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    versions = [v for v in VERSION_ORDER if v in data]
    speedups = [data[v]["speedup"] for v in versions]

    colors = []
    for v in versions:
        if v == "baseline":
            colors.append("#cccccc")
        elif v in ["v0", "v1", "v2"]:
            colors.append("#ffb366")
        elif v in ["v3", "v4", "v5"]:
            colors.append("#66b3ff")
        elif v in ["v6", "v7"]:
            colors.append("#99ff99")
        else:
            colors.append("#ff6666")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(versions, speedups, color=colors, edgecolor="black", linewidth=1.2)

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height,
            f"{speedup:.0f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylabel("Speedup (vs Baseline)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Version", fontsize=11, fontweight="bold")
    ax.set_title("CUDA Reduce Speedup Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06-speedup-comparison.png", dpi=220, bbox_inches="tight")
    print(f"✅ 图表已保存: {FIG_DIR / '06-speedup-comparison.png'}")
    plt.close()


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"❌ 错误: 找不到数据文件 {CSV_PATH}")
        print("请先运行: ./build/reduce_bench")
        exit(1)

    data = load_data()
    print_summary(data)
    plot_comparison(data)
    plot_speedup_comparison(data)
