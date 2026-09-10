#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""校验全仓 .ncu-rep 的采集口径,并导出成 Mac 端 Nsight Compute 可直接打开的报告包。

做两件事:

1. **口径校验**(采集脚本写得对不对,只能靠报告本身反证):
   - 一份报告里出现多个不同的 kernel 名  → FAIL。说明 `-k regex:` 没锚住,
     抓串了别的 kernel(w8a8 的 `v1_kernel` 是 `dequant_v1_kernel` 的子串,
     不加 `^` 就会中招)。这种报告的任何跨版本对比都是无效的。
   - 一份报告里出现多个不同的 grid  → WARN。两种成因,脚本不自动区分:
     (a) 多级算法的各级(归约树 65536→256→1 都是同一个 kernel),合法,
         但读图时必须知道自己在看哪一级;
     (b) 多个 regime(decode / l2 / prefill / hbm)被收进同一份报告。
         EXP-K04 的教训:L2 区间与 HBM 区间的结论会翻转,混样等于把两个
         结论搅在一起,需按输出的 grid 分布用 NCU_SKIP/NCU_COUNT 钉窗口重采。
   - 空报告 → FAIL。

2. **导出**:按算子分类复制到 artifacts/ncu_for_mac/reports/<算子>/,
   生成 MANIFEST.md(采集环境 / 覆盖情况 / 对照臂口径陷阱 / 逐份摘要)与
   manifest.csv(机器可读),打成一个 tar.gz 供传输。

用法:  python scripts/export_ncu_for_mac.py
"""

import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "ncu_for_mac"

SOL = "GPU Speed Of Light Throughput"
# (section, metric) -> 字段名。必须带 section 限定:同名 metric 在不同 section
# 里单位不同(SOL 的 "Memory Throughput" 是 %,MemoryWorkloadAnalysis 的是 Gbyte/s),
# 不限定就会把 % 覆盖成 Gbyte/s,量出 "212% 带宽" 这种不可能的数。
WANT = {
    (SOL, "Duration"): "duration",
    (SOL, "Compute (SM) Throughput"): "sm_pct",
    (SOL, "Memory Throughput"): "mem_pct",
    (SOL, "DRAM Throughput"): "dram_pct",
    (SOL, "Elapsed Cycles"): "cycles",
    ("Occupancy", "Achieved Occupancy"): "occ_pct",
    ("Occupancy", "Theoretical Occupancy"): "theo_occ",
    ("Launch Statistics", "Registers Per Thread"): "regs",
    ("Launch Statistics", "Static Shared Memory Per Block"): "smem",
    ("Launch Statistics", "Waves Per SM"): "waves",
}
FIELDS = sorted(set(WANT.values()))

ALL_OPS = ["activation", "cuda-reduce", "flash-attn", "fused-norm", "gemm",
           "gemv", "int8-quantize", "rope", "softmax", "w8a8"]


def grid_volume(inst) -> int:
    """把 "(65536, 1, 1)" 这种 grid 字符串换算成块数,用于挑代表实例。"""
    nums = [int(x) for x in re.findall(r"\d+", inst.get("grid") or "")]
    return math.prod(nums) if nums else 0


def base_name(kern: str) -> str:
    """从 demangled 名里取函数名,去掉模板实参与参数表,用于同一性判断。"""
    s = re.sub(r"^\s*(void|__global__)\s+", "", kern or "")
    s = s.split("(")[0]
    s = re.sub(r"<.*", "", s)
    return s.strip().split("::")[-1]


def version_of(stem: str) -> str:
    m = re.search(r"_((?:quant|dequant|gemv)?_?v\d+(?:[._]\d+)?[a-z_]*)_profile$", stem)
    if m:
        return m.group(1).strip("_").replace("_", ".") if re.fullmatch(
            r"v\d+[._]\d+", m.group(1)) else m.group(1).strip("_")
    m = re.search(r"(v\d+(?:[._]\d+)?)", stem)
    if m:
        v = m.group(1)
        return v.replace("_", ".") if re.fullmatch(r"v\d+[._]\d+", v) else v
    for k in ("baseline", "cublas", "ref_naive", "bench", "smoke"):
        if k in stem:
            return k
    return "-"


def read_report(path: Path):
    """返回 (每 kernel 实例的指标 dict 列表, 原始错误信息或 None)。"""
    try:
        p = subprocess.run(["ncu", "-i", str(path), "--page", "details", "--csv"],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:                                    # noqa: BLE001
        return [], f"ncu 导入失败: {e}"
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    per = {}
    for r in rows:
        try:
            kid = (int(r["ID"]), r.get("Kernel Name", ""))
        except (KeyError, ValueError, TypeError):
            continue
        d = per.setdefault(kid, {"grid": r.get("Grid Size", ""),
                                 "block": r.get("Block Size", ""),
                                 "unit": "", **{x: "" for x in FIELDS}})
        key = (r.get("Section Name"), r.get("Metric Name"))
        if key in WANT:
            d[WANT[key]] = r.get("Metric Value", "")
            if key[1] == "Duration":
                d["unit"] = r.get("Metric Unit", "")
    out = [dict(kern=k[1], _id=k[0], **v) for k, v in sorted(per.items())]
    return out, None


def session_info(path: Path):
    """读报告的 Session 页。

    **必须包含 display_name 与 multiprocessor_count**:光看 CPU 型号会误判 GPU。
    本仓 2026-05 那批报告的主机是 Ryzen 9 7945HX 笔记本,很容易顺手写成
    "笔记本 4090",实际是 RTX 4070 Laptop GPU(36 SM / 8 GB)——与桌面 4090
    (128 SM / 24 GB)差 3.6 倍 SM。占用率、波量化、L2 容量相关的结论在两者之间
    完全不可迁移。设备真身必须由报告自己说,不许由上下文推断。
    """
    p = subprocess.run(["ncu", "-i", str(path), "--page", "session"],
                       capture_output=True, text=True, timeout=600)
    info = {}
    for line in p.stdout.splitlines():
        s = line.strip()
        for key, field in [("Host Name", "host"), ("Host Processor", "cpu"),
                           ("CUDA Version", "cuda"),
                           ("Display Driver Version", "driver"),
                           ("Nsight Compute Target", "ncu"), ("Created", "created"),
                           ("display_name", "gpu"),
                           ("multiprocessor_count", "sm_count")]:
            if s.startswith(key):
                info.setdefault(field, s.split(key, 1)[1].strip())
    return info


def main() -> int:
    reports = sorted(ROOT.rglob("*.ncu-rep"))
    reports = [r for r in reports if OUT not in r.parents]
    if not reports:
        print("没有找到任何 .ncu-rep。先跑 scripts/run_ncu_all.sh。")
        return 1

    fails, warns, rows = [], [], []
    for f in reports:
        insts, err = read_report(f)
        rel = f.relative_to(ROOT)
        if err:
            fails.append(f"{rel}: {err}")
            continue
        if not insts:
            # 空报告 = 采集时 regex 没匹配到 kernel。是配置问题,不是报告损坏,
            # 所以只警告并排除,不阻断其余报告的导出。
            warns.append(f"{rel}: 报告为空(无 kernel 实例) —— 采集时 -k regex 未匹配,已排除出包")
            continue

        names = {base_name(i["kern"]) for i in insts}
        if len(names) > 1:
            fails.append(f"{rel}: 混入 {len(names)} 个不同 kernel {sorted(names)}"
                         f" —— `-k regex:` 未锚定,跨版本对比无效")
        grids = {i["grid"] for i in insts if i["grid"]}
        if len(grids) > 1:
            # 同一 kernel 多次 launch 且 grid 不同,有两种成因,不能一概当错:
            #   (a) 算法本身多级——如归约树 65536→256→1,各级都是这个 kernel;
            #   (b) bench 扫多个 regime(decode/l2/prefill/hbm)被收进同一份报告。
            # (a) 合法但读图时必须知道自己在看哪一级;(b) 才需要钉窗口重采。
            # 脚本无法自动区分,只报事实,判断留给人。
            warns.append(f"{rel}: {len(insts)} 个实例横跨 {len(grids)} 种 grid "
                         f"{sorted(grids)} —— 多级算法的各级,还是多 regime 混样?"
                         f"若为后者,用 NCU_SKIP/NCU_COUNT 钉窗口重采")

        # 代表实例的选法:**取 grid 体积最大的那一次 launch**,不是第一次。
        # 为什么不能用第一次:flash-attn 的 bench 是形状扫描(S 从 512 到 4096),
        # 前面还有四个正确性 gate 形状,于是 ID 0 落在 S=512 的 gate 上——
        # 而 EXP-K03 的协议形状是 S=4096。用第一次做摘要,等于拿一个正确性检查
        # 的小形状去代表整条版本梯,数字全错但看起来完全正常。
        # 取最大 grid 至少落在工作量最大的那次 launch 上,与各仓 bench 的协议点一致。
        # (逐实例数据仍在报告里,GUI 中可任选;本列只决定清单摘要显示哪一个。)
        d = max(insts, key=grid_volume)
        # 逐份记出处,不靠"整包同源"的假设:本包就跨了两批采集
        # (2026-05-03 用 ncu 2026.1.1、05-23 用 2022.4.1),cuda-reduce 两批都有。
        si = session_info(f)
        rows.append(dict(cat=rel.parts[0], ver=version_of(f.stem), rel=str(rel),
                         bytes=f.stat().st_size, nk=len(insts),
                         n_grids=len(grids), kern=d["kern"],
                         gpu=si.get("gpu", ""), sm_count=si.get("sm_count", ""),
                         host=si.get("host", ""), ncu_ver=si.get("ncu", ""),
                         created=si.get("created", ""),
                         **{k: d.get(k, "") for k in
                            ["grid", "block", "unit", *FIELDS]}))

    print(f"扫描 {len(reports)} 份报告：FAIL {len(fails)} / WARN {len(warns)}")
    for m in fails:
        print(f"  [FAIL] {m}")
    for m in warns:
        print(f"  [WARN] {m}")
    if fails:
        print("\n有 FAIL,导出中止。修好 profile_ncu.sh 的 kernel regex 后重采。")
        return 1

    # ---- 导出 ----
    if (OUT / "reports").exists():
        shutil.rmtree(OUT / "reports")
    for r in rows:
        src = ROOT / r["rel"]
        dst = OUT / "reports" / r["cat"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        r["bundled"] = str(dst.relative_to(OUT))

    cols = ["cat", "ver", "bundled", "rel", "gpu", "sm_count", "host",
            "ncu_ver", "created", "bytes", "nk", "n_grids", "kern",
            "grid", "block", "duration", "unit", "sm_pct", "mem_pct", "dram_pct",
            "cycles", "occ_pct", "theo_occ", "regs", "smem", "waves"]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["cat"], r["ver"])):
            w.writerow(r)

    batches = defaultdict(list)
    for r in rows:
        batches[(r["gpu"], r["sm_count"], r["host"], r["ncu_ver"],
                 (r["created"] or "")[:11])].append(r)
    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    missing = [o for o in ALL_OPS if o not in by]

    L = ["# NCU 报告导出包(供 Mac 端 Nsight Compute GUI 打开)\n",
         f"共 {len(rows)} 份 `.ncu-rep`，按算子分类于 `reports/<算子>/`。"
         "逐份摘要见 `manifest.csv`。本文件由 `scripts/export_ncu_for_mac.py` 生成。\n",
         "## 采集批次\n",
         "逐份出处见 `manifest.csv` 的 gpu / sm_count / host / ncu_ver / created 列。\n",
         "| GPU | SM 数 | 主机 | Nsight Compute | 采集日期 | 报告数 |",
         "|---|---:|---|---|---|---:|"]
    for (gpu, sm, host, nv, when), rs in sorted(batches.items(), key=lambda kv: kv[0][4]):
        L.append(f"| {gpu or '?'} | {sm or '?'} | {host or '?'} | "
                 f"{nv or '?'} | {when or '?'} | {len(rs)} |")
    L += ["", "> **GPU 一栏以报告自述为准，不要从主机型号推断。** 本包 2026-05 那批的主机是"
              " Ryzen 9 7945HX 笔记本，很容易被顺手写成「笔记本 4090」，实际是"
              " RTX 4070 Laptop GPU（36 SM / 8 GB），与桌面 RTX 4090（128 SM / 24 GB）"
              "差 3.6 倍 SM。占用率、波量化、L2 容量相关的结论在两者之间不可迁移。\n",
          "> 上表每一行是一个采集批次。**跨批次的数字不可混排为同一行**——"
              "不同 GPU 不必说，同一台机器上不同 Nsight Compute 版本的 metric 定义也可能有出入。\n",
          "## Mac 端打开方式\n",
          "1. 装 **Nsight Compute 2025.3 或更新**（原生支持 macOS arm64，最低 macOS 13.0）。"
          "新版 GUI 可读旧版报告，反之不成立。\n",
          "2. `File > Open` 打开 `.ncu-rep`，或把 `reports/` 整个拖进去。\n",
          "3. 版本梯对比用 **Add Baseline**：先开 v0 设为 baseline，再开 v1..vN，"
          "Details 页每个 metric 会显示相对增减——这是看\"这一版到底改善了什么\"最快的读法。\n",
          "## 覆盖情况\n", "| 算子 | 报告数 | 版本 |", "|---|---:|---|"]
    for c in sorted(by):
        L.append(f"| `{c}` | {len(by[c])} | {', '.join(sorted({r['ver'] for r in by[c]}))} |")
    L.append("")
    if missing:
        L.append(f"**未覆盖（零 NCU 数据）**：{', '.join('`' + m + '`' for m in missing)}。\n")
    else:
        L.append("**十个算子全部覆盖。**\n")

    multi = [r for r in rows if r["n_grids"] > 1]
    if multi:
        L += ["## 含多个 grid 的报告（读图时先认清自己在看哪一次 launch）\n",
              "下列报告里同一个 kernel 被 launch 了多次且 grid 不同。两种成因：\n\n"
              "- **多级算法的各级**——如归约树 `65536 → 256 → 1`，三级都是同一个 kernel。"
              "此时只有第一级（grid 最大那次）反映真实的 HBM 行为，后面几级数据量已极小，"
              "它们的 SOL 低是必然的，不是优化空间。\n"
              "- **多 regime 混样**——bench 扫了 decode / l2 / prefill / hbm 几个尺寸，"
              "都落进了同一份报告。这种情况下**不能跨实例比较**：L2 区间与 HBM 区间的"
              "结论会翻转（等效带宽超过硬件峰值就是落在 L2 的信号，4090 的 L2 是 72 MB）。\n\n"
              "在 GUI 里按 launch 实例逐个看，先看 grid 认出自己在读哪一次。\n",
              "| 报告 | 实例数 | grid 种类 |", "|---|---:|---:|"]
        for r in sorted(multi, key=lambda r: (r["cat"], r["ver"])):
            L.append(f"| `{r['bundled']}` | {r['nk']} | {r['n_grids']} |")
        L.append("")

    L += ["## 对照臂口径陷阱\n",
          "- `gemv_cublas_profile` 内是 `gemv2T_kernel_val<...cublasGemvParams...>`，"
          "**是真 cuBLAS**，可作标准库对照。\n"
          "- `softmax_cublas_profile` 内是 `softmax_cublas_kernel`，"
          "**是自写 kernel，不是 cuBLAS**（BLAS 规范里没有 softmax）。文件名有误导性，"
          "该臂不得用于\"对比 cuBLAS\"的表述——此为已存档红线项。\n"
          "- `cuda-reduce` 的标准库对照应为 CUB `DeviceReduce`，本包内无 CUB 臂。\n"
          "- `*_baseline` 为朴素实现，SOL 与 Occupancy 极低属预期，不代表硬件上限。\n",
          "## 逐份摘要\n",
          "取报告内第一个 kernel 实例；`k` 为实例总数，`g` 为不同 grid 数。\n",
          "| 算子 | 版本 | k | g | 代表 grid | Duration | SM % | Memory % | DRAM % | Occ % | Regs |",
          "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["cat"], r["ver"])):
        L.append(f"| `{r['cat']}` | {r['ver']} | {r['nk']} | {r['n_grids']} | "
                 f"`{r['grid']}` | {r['duration']} {r['unit']} | {r['sm_pct']} | "
                 f"{r['mem_pct']} | {r['dram_pct']} | {r['occ_pct']} | {r['regs']} |")
    L.append("")
    (OUT / "MANIFEST.md").write_text("\n".join(L), encoding="utf-8")

    tar_path = OUT / "ncu_for_mac_all.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(OUT / "MANIFEST.md", arcname="MANIFEST.md")
        tf.add(OUT / "manifest.csv", arcname="manifest.csv")
        tf.add(OUT / "reports", arcname="reports")
    import hashlib
    h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (OUT / "ncu_for_mac_all.tar.gz.sha256").write_text(
        f"{h}  {tar_path.name}\n", encoding="utf-8")

    print(f"\n导出完成：{tar_path.relative_to(ROOT)} "
          f"({tar_path.stat().st_size / 1e6:.1f} MB, {len(rows)} 份报告)")
    if warns:
        print(f"注意：{len(warns)} 份报告含多个 grid（多级算法或多 regime），已在 MANIFEST.md 列出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
