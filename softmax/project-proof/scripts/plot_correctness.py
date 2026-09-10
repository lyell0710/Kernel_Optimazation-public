import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "04-correctness.png"
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
diffs = [abs(float(r["max_diff"])) for r in rows]
status = [str(r["correctness_pass"]).lower() == "true" for r in rows]

fig, ax = plt.subplots(figsize=(9.2, 3.8))
ax.axis("off")
ax.text(0.5, 0.90, "CUDA Softmax Correctness Summary", ha="center", va="center", fontsize=14, weight="bold")
ax.text(
    0.5,
    0.80,
    f"Threshold=1e-4 | Pass={sum(status)}/{len(status)} | MaxDiff={max(diffs):.2e}",
    ha="center",
    va="center",
    fontsize=10,
)

table_rows = [[v, f"{d:.2e}", "PASS" if ok else "FAIL"] for v, d, ok in zip(labels, diffs, status)]
table = ax.table(
    cellText=table_rows,
    colLabels=["Version", "Abs Max Diff", "Status"],
    cellLoc="center",
    colLoc="center",
    bbox=[0.08, 0.08, 0.84, 0.62],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#E8EEF7")
        cell.set_text_props(weight="bold")
    elif c == 2:
        if cell.get_text().get_text() == "PASS":
            cell.set_facecolor("#D9F2D9")
        else:
            cell.set_facecolor("#F8D7DA")

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=220)
plt.close(fig)

print(f"Saved: {FIG_PATH}")
