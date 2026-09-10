import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "03-speedup-vs-v0.png"
VERSION_ORDER = ("v0", "v1", "v2", "v3", "v4", "v4.2", "cublas", "v5", "v6", "v7")


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def ordered_rows(rows):
    row_by_version = {r["version"]: r for r in rows}
    ordered = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
    extra = [r for r in rows if r["version"] not in VERSION_ORDER]
    return ordered + extra


rows = ordered_rows(load_rows())
labels = [r["version"] for r in rows]
latency = [float(r["latency_ms"]) for r in rows]
v0_latency = latency[0]  # v0 is first in VERSION_ORDER
speedup = [v0_latency / v for v in latency]

plt.figure(figsize=(9.6, 4.8))
colors = plt.get_cmap("tab10").colors[: len(labels)]
bars = plt.bar(labels, speedup, color=colors)

# Highlight v4 and cublas with different styling
for i, (bar, label) in enumerate(zip(bars, labels)):
    value = speedup[i]
    if label == "v4":
        bar.set_edgecolor("darkred")
        bar.set_linewidth(2.5)
    elif label == "cublas":
        bar.set_edgecolor("darkblue")
        bar.set_linewidth(2.5)
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.title("CUDA Softmax Speedup vs v0")
plt.xlabel("Version")
plt.ylabel("Speedup (x)")
plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
plt.grid(True, axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, dpi=220)
plt.close()

print(f"Saved: {FIG_PATH}")
