import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NCU_DIR = ROOT / "project-proof" / "profiling" / "ncu"
FIG_DIR = ROOT / "project-proof" / "docs" / "figures" / "02-profiling"
VERSION_ORDER = ["baseline"] + [f"v{i}" for i in range(8)]

# 与 scripts/ncu_metrics.inc.sh 中 NCU_CSV_METRICS 对齐；图表仍主要使用前四个核心指标。
METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Throughput (%)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM Throughput (%)",
    "smsp__warps_active.avg.pct_of_peak_sustained_active": "Active Warps (%)",
    "smsp__inst_executed.sum": "Inst Executed",
    "lts__t_request_hit_rate": "L2 Request Hit Rate",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "Shmem Bank Conflicts (LD)",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "Shmem Bank Conflicts (ST)",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld": "Avg Sectors / Global LD Req",
    "tpc__average_registers_per_thread": "Avg Registers / Thread",
    "sm__sass_data_bytes_mem_shared": "SASS Shmem Bytes (est.)",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active": "Stall Long Scoreboard",
    "smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active": "Stall L1TEX Long Scoreboard",
    "smsp__warp_issue_stalled_barrier_per_warp_active": "Stall Barrier",
    "smsp__warp_issue_stalled_membar_per_warp_active": "Stall Membar",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active": "Stall Short Scoreboard",
}


def load_ncu_csv() -> tuple[pd.DataFrame | None, str]:
    path = NCU_DIR / "reduce_ncu.csv"
    if not path.is_file():
        return None, f"No NCU CSV found at {path}. Run RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh first."
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "ERR_NVGPUCTRPERM" in text:
        return None, "NCU blocked by ERR_NVGPUCTRPERM (GPU counter permission)."
    if "No kernels were profiled" in text:
        return None, "NCU ran but no kernels were profiled."

    filtered_lines = [ln for ln in text.splitlines() if not ln.startswith("==") and ln.strip()]
    if not filtered_lines:
        return None, "NCU CSV has no metric rows."
    try:
        df = pd.read_csv(io.StringIO("\n".join(filtered_lines)))
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to parse NCU CSV: {e}"
    return df, ""


def parse_metric_value(v) -> float:
    s = str(v).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def infer_version(kernel_name: str) -> str:
    low = kernel_name.lower()
    m = re.search(r"baseline|v\d+", low)
    return m.group(0) if m else "other"


def classify_bound(sm_val: float, dram_val: float) -> str:
    if dram_val >= 60.0 and dram_val > sm_val * 1.1:
        return "Memory-bound"
    if sm_val >= 60.0 and sm_val > dram_val * 1.1:
        return "Compute-bound"
    return "Latency/Mixed"


def save_unavailable_figure(msg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.62, "NCU Profiling Data Unavailable", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, msg, ha="center", va="center", fontsize=11)
    fig.savefig(FIG_DIR / "00-ncu-unavailable.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df, err = load_ncu_csv()
    if df is None:
        save_unavailable_figure(err)
        print(err)
        return

    needed = {"Kernel Name", "Metric Name", "Metric Value"}
    if not needed.issubset(set(df.columns)):
        save_unavailable_figure("Missing required columns in NCU CSV.")
        print("Missing required columns in NCU CSV.")
        return

    df = df[df["Metric Name"].isin(METRICS.keys())].copy()
    if df.empty:
        save_unavailable_figure("No target metrics found in NCU CSV.")
        print("No target metrics found in NCU CSV.")
        return

    df["version"] = df["Kernel Name"].map(infer_version)
    df["value"] = df["Metric Value"].map(parse_metric_value)

    grouped = (
        df.groupby(["version", "Metric Name"], as_index=False)["value"]
        .mean()
        .pivot(index="version", columns="Metric Name", values="value")
        .fillna(0.0)
    )

    idx = [v for v in VERSION_ORDER if v in grouped.index] + [v for v in grouped.index if v not in VERSION_ORDER]
    grouped = grouped.loc[idx]

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x = range(len(grouped.index))
    sm_vals = grouped.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", pd.Series([0.0] * len(grouped))).values
    dram_vals = grouped.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", pd.Series([0.0] * len(grouped))).values

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bar_w = 0.38
    ax.bar([i - bar_w / 2 for i in x], sm_vals, width=bar_w, label="SM Throughput (%)")
    ax.bar([i + bar_w / 2 for i in x], dram_vals, width=bar_w, label="DRAM Throughput (%)")
    ax.set_xticks(list(x), grouped.index, rotation=0)
    ax.set_ylabel("Percent of Peak Sustained")
    ax.set_title("NCU Throughput Summary (Reduce)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01-ncu-throughput.png", dpi=220)
    plt.close(fig)

    occ_vals = grouped.get("smsp__warps_active.avg.pct_of_peak_sustained_active", pd.Series([0.0] * len(grouped))).values
    inst_vals = grouped.get("smsp__inst_executed.sum", pd.Series([0.0] * len(grouped))).values

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(list(x), occ_vals, label="Active Warps (%)", color="#4C78A8")
    ax1.set_xticks(list(x), grouped.index, rotation=0)
    ax1.set_ylabel("Active Warps (%)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(list(x), inst_vals, color="#F58518", marker="o", label="Inst Executed")
    ax2.set_ylabel("Inst Executed", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")

    ax1.set_title("NCU Occupancy Proxy and Instructions (Reduce)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02-ncu-occupancy-inst.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for i, ver in enumerate(grouped.index):
        smv = float(sm_vals[i])
        dmv = float(dram_vals[i])
        label = classify_bound(smv, dmv)
        color = {"Compute-bound": "#4C78A8", "Memory-bound": "#F58518", "Latency/Mixed": "#54A24B"}[label]
        ax.scatter(smv, dmv, color=color, s=70)
        ax.text(smv + 0.5, dmv + 0.5, f"{ver} ({label})", fontsize=8)
    ax.set_xlabel("SM Throughput (%)")
    ax.set_ylabel("DRAM Throughput (%)")
    ax.set_title("NCU Bound Classification (Heuristic, Reduce)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03-ncu-bound-scatter.png", dpi=220)
    plt.close(fig)

    stall_cols = [
        "smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active",
        "smsp__warp_issue_stalled_barrier_per_warp_active",
        "smsp__warp_issue_stalled_membar_per_warp_active",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active",
    ]
    if all(c in grouped.columns for c in stall_cols):
        fig, ax = plt.subplots(figsize=(11, 5.0))
        w = 0.2
        for si, col in enumerate(stall_cols):
            offs = [i + (si - 1.5) * w for i in x]
            short = col.split("stalled_")[1].split("_per")[0] if "stalled_" in col else col
            ax.bar(offs, grouped[col].values, width=w, label=short)
        ax.set_xticks(list(x), grouped.index, rotation=0)
        ax.set_ylabel("Stall (per warp active ratio)")
        ax.set_title("NCU Warp Stall Breakdown (Reduce)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "04-ncu-warp-stall.png", dpi=220)
        plt.close(fig)

    print(f"Saved NCU figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
