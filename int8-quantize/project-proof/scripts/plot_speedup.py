import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "03-speedup-vs-baseline.png"
ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")

rows = list(csv.DictReader(CSV_PATH.open()))
by = {r["version"]: r for r in rows}
rows = [by[v] for v in ORDER if v in by] + [r for r in rows if r["version"] not in ORDER]
labels = [r["version"] for r in rows]
lat = [float(r["latency_ms"]) for r in rows]
baseline = lat[0]
speedup = [baseline / v for v in lat]

plt.figure(figsize=(9.6, 4.8))
bars = plt.bar(labels, speedup, color=plt.get_cmap("tab10").colors[: len(labels)])
for b, v in zip(bars, speedup):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
plt.title("CUDA INT8 Quantize Speedup vs Baseline")
plt.ylabel("Speedup (x)")
plt.xlabel("Version")
plt.grid(True, axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, dpi=220)
plt.close()
