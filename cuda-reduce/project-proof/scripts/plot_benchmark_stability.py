import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "profiling" / "bench"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "05-stability-3runs.png"
CSV_OUT = ROOT / "project-proof" / "data" / "benchmark_stability.csv"
VERSION_ORDER = ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cuBLAS")
PATTERN = re.compile(r"^\[mean over 100 iters\]\s+(\w+):\s+([0-9.eE+-]+)\s+ms$")


def parse_run(path: Path):
    values = {}
    for line in path.read_text().splitlines():
        match = PATTERN.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


run_files = sorted(RUN_DIR.glob("run_*.txt"))
if not run_files:
    raise RuntimeError(f"No run files found in {RUN_DIR}")

parsed = [parse_run(p) for p in run_files]
versions = [v for v in VERSION_ORDER if all(v in r for r in parsed)]

means = []
stds = []
for v in versions:
    samples = np.array([r[v] for r in parsed], dtype=float)
    means.append(float(samples.mean()))
    stds.append(float(samples.std(ddof=0)))

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with CSV_OUT.open("w") as f:
    f.write("version,mean_latency_ms,std_latency_ms\n")
    for v, m, s in zip(versions, means, stds):
        f.write(f"{v},{m:.6f},{s:.6f}\n")

x = np.arange(len(versions))
plt.figure(figsize=(10.2, 4.8))
plt.errorbar(x, means, yerr=stds, fmt="o-", capsize=4, linewidth=1.8, markersize=5, color="#4C78A8")
plt.xticks(x, versions)
plt.yscale("log")
plt.title("Benchmark Stability (3 Runs, Mean ± Std)")
plt.ylabel("Latency (ms, log scale)")
plt.xlabel("Version")
plt.grid(True, axis="y", linestyle="--", alpha=0.35)

for xi, m, s in zip(x, means, stds):
    plt.text(xi, m, f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=7)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=220)
plt.close()
