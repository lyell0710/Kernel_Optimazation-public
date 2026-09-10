#!/usr/bin/env python3
"""把 N 轮 stability CSV 归并成 mean±std 的 derived 表。

为什么单独一个脚本而不是在 bench 里直接算:铁律 3「raw 不可变」——
raw 是每轮独立落盘的原始观测,derived 是可随时从 raw 重算的派生物。
两者分开,任何时候都能用新口径重算而不必重跑 GPU。

用法: python scripts/derive_stability.py <out.csv> <r1.csv> <r2.csv> ...
      键 = 除数值列外的全部标识列(regime/version/T/H 之类),自动识别。
"""
import csv, sys, statistics, pathlib, datetime

def read(path):
    prov, rows = None, []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                prov = prov or line.rstrip("\n")
                continue
            rows.append(line)
    rdr = csv.DictReader(rows)
    return prov, list(rdr), rdr.fieldnames

def main():
    out, srcs = sys.argv[1], sys.argv[2:]
    provs, tables, header = [], [], None
    for s in srcs:
        p, t, h = read(s)
        provs.append((pathlib.Path(s).name, p)); tables.append(t); header = h

    # 数值列 = 能被 float() 解析的列;其余列作为分组键。
    num_cols = [c for c in header
                if all(_isnum(r[c]) for t in tables for r in t)]
    key_cols = [c for c in header if c not in num_cols]

    agg = {}
    for t in tables:
        for r in t:
            k = tuple(r[c] for c in key_cols)
            agg.setdefault(k, {c: [] for c in num_cols})
            for c in num_cols:
                agg[k][c].append(float(r[c]))

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out, "w") as f:
        f.write(f"# derived: n_rounds={len(srcs)} date={ts}\n")
        for name, p in provs:
            f.write(f"# source: {name} :: {p[len('# provenance: '):] if p else 'NA'}\n")
        cols = key_cols + [x for c in num_cols for x in (f"{c}_mean", f"{c}_std")]
        f.write(",".join(cols) + "\n")
        for k, v in agg.items():
            row = list(k)
            for c in num_cols:
                # 剔除 nan:分解步骤那类臂不判正确性,err 列写的是 nan,
                # 直接喂给 statistics 会抛 AttributeError(nan 没有 numerator)。
                vals = [x for x in v[c] if x == x]
                if not vals:
                    row.append("nan"); row.append("nan"); continue
                row.append(f"{statistics.mean(vals):.6g}")
                row.append(f"{statistics.stdev(vals):.4g}" if len(vals) > 1 else "0")
            f.write(",".join(row) + "\n")
    print("written:", out, f"({len(agg)} rows, {len(srcs)} rounds)")

def _isnum(s):
    try:
        float(s); return True      # 'nan' 也能被 float() 解析,按数值列处理
    except (TypeError, ValueError):
        return False

if __name__ == "__main__":
    main()
