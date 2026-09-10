import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
API_CSV_PATH = ROOT / "profiling" / "nsys" / "cuda_api_sum.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "02-profiling" / "02-nsys-api-breakdown.png"


def load_api_rows():
    lines = API_CSV_PATH.read_text().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Cannot find CSV header in {API_CSV_PATH}")

    parsed_lines = lines[header_idx:]
    reader = csv.DictReader(parsed_lines)
    return list(reader)


rows = load_api_rows()
rows = sorted(rows, key=lambda r: float(r["Total Time (ns)"]), reverse=True)

top_k = 6
top_rows = rows[:top_k]
other_rows = rows[top_k:]

labels = [r["Name"] for r in top_rows]
values_ns = [float(r["Total Time (ns)"]) for r in top_rows]

if other_rows:
    labels.append("others")
    values_ns.append(sum(float(r["Total Time (ns)"]) for r in other_rows))

values_ms = [v / 1e6 for v in values_ns]
total_ms = sum(values_ms)

plt.figure(figsize=(10.2, 4.8))
bars = plt.bar(labels, values_ms, color=plt.get_cmap("tab10").colors[: len(labels)])
plt.title("Nsight Systems CUDA API Time Breakdown")
plt.ylabel("Total API Time (ms)")
plt.xticks(rotation=18, ha="right")
plt.grid(True, axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, values_ms):
    pct = (value / total_ms * 100.0) if total_ms > 0 else 0.0
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f} ms\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=220)
plt.close()
