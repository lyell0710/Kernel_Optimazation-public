#!/usr/bin/env bash
# reduce 双区间 3 轮重测（计时口径修正后）。
#
# 为什么要重测:EXP-K04 的「L2 区间 CUB 快 33.3%」是在**口径不对称**下测的 ——
#   v1–v7 每次调用都在计时区内 cudaMalloc/cudaFree(v6/v7 还多一个
#   cudaGetDeviceProperties),而 CUB/cuBLAS/baseline 侧都没有。
#   偏置只加在手写侧:对 v7(约 30 μs)占四分之一,对 v1(约 118 μs)只占 6%,
#   于是版本梯越往后被罚越重、与 CUB 的差距被系统性高估。
#   口径已修(分配与设备属性提到计时外,与 CUB 的 g_temp 同款),故须重测。
#
# 协议与 stability_rebench.sh 一致:二进制写死 benchmark_results.csv(trunc),
#   先把既有文件挪到安全位,跑完 mv 成 UTC 名并补 provenance 首行,最后原样复位。
#
# 两个区间:
#   l2   N=1<<24      = 67.1 MB  < 72 MB L2(含 baseline,顺带给出端到端口径)
#   hbm  N=268435456  = 1.07 GB >> 72 MB L2(跳过 baseline:单轮 1.6 s×100 迭代太慢,
#                                            且端到端口径不取自该区间)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/cuda-reduce"
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CSV=project-proof/data/benchmark_results.csv
SAFE=""; [ -e "$CSV" ] && { SAFE="${CSV}.orig_hold"; mv "$CSV" "$SAFE"; }
trap '[ -n "${SAFE:-}" ] && [ -e "$SAFE" ] && mv "$SAFE" "$CSV"' EXIT

run_regime() {   # $1=标签 $2=N $3=是否跳 baseline
  for r in 1 2 3; do
    TS=$(date -u +%Y%m%dT%H%M%S)
    if [ "$3" = skip ]; then REDUCE_N=$2 REDUCE_SKIP_BASELINE=1 ./build/reduce_bench >/dev/null
    else                     REDUCE_N=$2                        ./build/reduce_bench >/dev/null; fi
    OUT="project-proof/data/${TS}_cuda-reduce_${1}_N${2}_calfix_r${r}.csv"
    { echo "# provenance: env=4090-main-container sha=$SHA cmd=\"REDUCE_N=$2 ./build/reduce_bench (calfix $1 r$r)\" date=$(date -u +%FT%TZ) gpu=\"NVIDIA GeForce RTX 4090\" driver=$DRV"
      cat "$CSV"; } > "$OUT"
    rm -f "$CSV"; echo "  $1 r$r -> $OUT"
  done
}
echo "== L2 常驻区间 (N=16777216, 67.1 MB)"; run_regime l2  16777216  keep
echo "== HBM-bound 区间 (N=268435456, 1.07 GB)"; run_regime hbm 268435456 skip
echo REDUCE_REGIME_REBENCH_DONE
