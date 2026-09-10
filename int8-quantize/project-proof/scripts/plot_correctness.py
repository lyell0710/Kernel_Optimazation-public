import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "04-correctness.png"
ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")

rows = list(csv.DictReader(CSV_PATH.open()))
by = {r["version"]: r for r in rows}
rows = [by[v] for v in ORDER if v in by] + [r for r in rows if r["version"] not in ORDER]
labels = [r["version"] for r in rows]
errs = [abs(float(r["max_abs_err"])) for r in rows]
flags = [str(r["correctness_pass"]).lower() == "true" for r in rows]

fig, ax = plt.subplots(figsize=(9.2, 3.8))
ax.axis("off")
ax.text(0.5, 0.90, "CUDA INT8 Quantize Correctness", ha="center", va="center", fontsize=14, weight="bold")
ax.text(0.5, 0.80, f"Pass={sum(flags)}/{len(flags)} | MaxAbsErr={max(errs):.2e}", ha="center", va="center", fontsize=10)
table = ax.table(
    cellText=[[v, f"{e:.2e}", "PASS" if ok else "FAIL"] for v, e, ok in zip(labels, errs, flags)],
    colLabels=["Version", "Max Abs Error", "Status"],
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
        cell.set_facecolor("#D9F2D9" if cell.get_text().get_text() == "PASS" else "#F8D7DA")
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=220)
plt.close(fig)
