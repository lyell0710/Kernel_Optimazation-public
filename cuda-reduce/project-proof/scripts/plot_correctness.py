import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "06-correctness.png"
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
diff_values = [abs(float(row["diff"])) for row in plot_rows]
correct_values = [str(row["correctness_pass"]).lower() == "true" for row in plot_rows]
threshold = 1e-4
pass_count = sum(correct_values)
fail_count = len(correct_values) - pass_count
max_diff = max(diff_values) if diff_values else 0.0

# 改成摘要卡片+表格，避免“全绿色柱子没信息”
fig, ax = plt.subplots(figsize=(9.2, 3.8))
ax.axis("off")

title = "CUDA Reduction Correctness Summary"
summary = (
    f"Threshold: {threshold:.0e}    "
    f"Pass: {pass_count}/{len(correct_values)}    "
    f"Fail: {fail_count}    "
    f"Max Diff: {max_diff:.1e}"
)
ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=14, weight="bold")
ax.text(0.5, 0.84, summary, ha="center", va="center", fontsize=10, color="#333333")

table_rows = [[v, f"{d:.1e}", "PASS" if ok else "FAIL"] for v, d, ok in zip(labels, diff_values, correct_values)]
table = ax.table(
    cellText=table_rows,
    colLabels=["Version", "Abs Diff", "Status"],
    cellLoc="center",
    colLoc="center",
    bbox=[0.08, 0.10, 0.84, 0.62],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#E8EEF7")
    elif c == 2:
        status_text = cell.get_text().get_text()
        if status_text == "PASS":
            cell.set_facecolor("#D9F2D9")
        else:
            cell.set_facecolor("#F8D7DA")

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=220)
plt.close(fig)
