import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "03-latency-zoom.png"
VERSION_ORDER = ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cuBLAS")


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_rows_by_version(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["version"], []).append(row)

    aggregated = []
    for version, samples in grouped.items():
        latency_values = [float(r["latency_ms"]) for r in samples]
        cpu_values = [float(r["cpu_result"]) for r in samples]
        gpu_values = [float(r["gpu_result"]) for r in samples]
        diff_values = [float(r["diff"]) for r in samples]
        correctness_values = [str(r["correctness_pass"]).lower() == "true" for r in samples]
        aggregated.append(
            {
                "version": version,
                "latency_ms": f"{sum(latency_values) / len(latency_values):.6f}",
                "cpu_result": f"{sum(cpu_values) / len(cpu_values):.6e}",
                "gpu_result": f"{sum(gpu_values) / len(gpu_values):.6e}",
                "diff": f"{sum(diff_values) / len(diff_values):.6e}",
                "correctness_pass": str(all(correctness_values)).lower(),
            }
        )
    return aggregated


rows = aggregate_rows_by_version(load_benchmark_rows())
row_by_version = {row["version"]: row for row in rows}
# 移除 baseline
row_by_version.pop("baseline", None)
ordered_rows = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra_rows = [row for row in rows if row["version"] not in VERSION_ORDER and row["version"] != "baseline"]
plot_rows = ordered_rows + extra_rows

labels = [row["version"] for row in plot_rows]
latency_ms = [float(row["latency_ms"]) for row in plot_rows]
# 以 cuBLAS 为基准（如果没有则以 v0）
if "cuBLAS" in row_by_version:
    baseline_latency = float(row_by_version["cuBLAS"]["latency_ms"])
    ref_name = "cuBLAS"
elif "v0" in row_by_version:
    baseline_latency = float(row_by_version["v0"]["latency_ms"])
    ref_name = "v0"
else:
    baseline_latency = max(latency_ms) if latency_ms else 1.0
    ref_name = "ref"
speedups = [baseline_latency / value for value in latency_ms]
x = np.arange(len(labels))

fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.8), width_ratios=[1.15, 1.0])

# 左图：保留全局趋势，但改成 log 轴，避免 baseline 压扁前半段
ax_all.plot(x, latency_ms, marker="o", linewidth=2.1, color="#4C78A8")
ax_all.set_xticks(x, labels)
ax_all.set_yscale("log")
ax_all.set_title("Full Latency Trend (Log Scale)")
ax_all.set_ylabel("Latency (ms)")
ax_all.set_xlabel("Version")
ax_all.grid(True, axis="y", linestyle="--", alpha=0.35)

# 只标前半段，减少文字拥挤
for i, (name, value, sp) in enumerate(zip(labels, latency_ms, speedups)):
    if name not in {"v0", "v1", "v2"}:
        continue
    ax_all.annotate(
        f"{value:.6f} ms\n({sp:.1f}x)",
        (i, value),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
    )

# 右图：后段放大比较（v3~v7+cuBLAS）
focus_labels = [v for v in ("v3", "v4", "v5", "v6", "v7", "cuBLAS") if v in row_by_version]
focus_latency = [float(row_by_version[v]["latency_ms"]) for v in focus_labels]
focus_speedups = [baseline_latency / v for v in focus_latency]
focus_x = np.arange(len(focus_labels))

ax_zoom.plot(focus_x, focus_latency, marker="o", linewidth=2.2, color="#E15759")
ax_zoom.set_xticks(focus_x, focus_labels)
ax_zoom.set_title("Zoomed Optimization Focus (v3~v7)")
ax_zoom.set_ylabel("Latency (ms)")
ax_zoom.set_xlabel("Version")
ax_zoom.grid(True, axis="y", linestyle="--", alpha=0.35)

if focus_latency:
    y_min = min(focus_latency)
    y_max = max(focus_latency)
    pad = (y_max - y_min) * 0.28 if y_max > y_min else y_max * 0.1
    ax_zoom.set_ylim(max(0.0, y_min - pad), y_max + pad)

for xi, value, speedup in zip(focus_x, focus_latency, focus_speedups):
    ax_zoom.annotate(
        f"{value:.6f} ms\n({speedup:.1f}x)",
        (xi, value),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
    )

ax_zoom.annotate(
    f"{ref_name} = {baseline_latency:.3f} ms (ref)",
    (0, focus_latency[0] if focus_latency else 0.0),
    textcoords="offset points",
    xytext=(5, -28),
    ha="left",
    fontsize=8,
    color="#555555",
)

fig.suptitle("CUDA Reduction Latency Comparison", fontsize=13)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=200)
plt.close()
