import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "02-latency-log.png"
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


def pick_colors(count: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(count)]


rows = aggregate_rows_by_version(load_benchmark_rows())
row_by_version = {row["version"]: row for row in rows}
# 移除 baseline
row_by_version.pop("baseline", None)
ordered_rows = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra_rows = [row for row in rows if row["version"] not in VERSION_ORDER and row["version"] != "baseline"]
plot_rows = ordered_rows + extra_rows

labels = [row["version"] for row in plot_rows]
latency_ms = [float(row["latency_ms"]) for row in plot_rows]
colors = pick_colors(len(labels))

plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, latency_ms, color=colors)
plt.yscale("log")

for bar, value in zip(bars, latency_ms):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.6f} ms",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("CUDA Reduction Latency Comparison (Log Scale)")
plt.ylabel("Latency (ms, log scale)")
plt.xlabel("Version")
plt.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, dpi=200)
plt.close()
