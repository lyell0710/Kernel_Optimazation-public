import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NCU_TEXT_DIR = ROOT / "profiling" / "ncu" / "text"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "02-profiling" / "01-ncu-kernel-compare.png"

INPUT_FILES = {
    "v5": NCU_TEXT_DIR / "reduce_v5_once_after_v7_raw.csv",
    "v6": NCU_TEXT_DIR / "reduce_v6_once_raw.csv",
    "v7": NCU_TEXT_DIR / "reduce_v7_once_raw.csv",
}

METRICS = {
    "Kernel Duration (us)": "gpu__time_duration.sum",
    "SM Throughput (%)": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM Throughput (%)": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "Issue Active (%)": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "Waves/SM": "launch__waves_per_multiprocessor",
}


def load_first_kernel_row(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Process ID", "").strip():
                return row
    raise RuntimeError(f"No kernel row found in {csv_path}")


def to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


data = {name: load_first_kernel_row(path) for name, path in INPUT_FILES.items()}

versions = ["v5", "v6", "v7"]
metric_names = list(METRICS.keys())
x = np.arange(len(metric_names))
width = 0.23

fig, ax = plt.subplots(figsize=(11.2, 5.2))
colors = {"v5": "#4C78A8", "v6": "#E15759", "v7": "#59A14F"}

for idx, version in enumerate(versions):
    values = [to_float(data[version][METRICS[m]]) for m in metric_names]
    bars = ax.bar(x + (idx - 1) * width, values, width=width, label=version, color=colors[version])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x, metric_names)
ax.set_title("NCU Kernel Metrics Comparison (v5 vs v6 vs v7)")
ax.set_ylabel("Raw Metric Value")
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.legend()

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=220)
plt.close(fig)
