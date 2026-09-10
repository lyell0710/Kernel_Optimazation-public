#!/usr/bin/env bash
# EXP-K01（四 kernel 4090 重基准）§7 整改:softmax/gemv/int8-quantize 各 3 轮,UTC 前缀落盘,不动既有 raw。
# 原理:二进制写死 project-proof/data/benchmark_results.csv(trunc)——
#   先把既有文件挪到安全位,跑完把输出 mv 成 UTC 名,最后原样放回既有文件。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
declare -A BIN=( [softmax]=softmax_bench [gemv]=gemv_bench [int8-quantize]=int8_quantize_bench [cuda-reduce]=reduce_bench )
for proj in softmax gemv int8-quantize cuda-reduce; do
  cd "$ROOT/$proj"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j >/dev/null
  CSV=project-proof/data/benchmark_results.csv
  SAFE=""
  [ -e "$CSV" ] && { SAFE="${CSV}.orig_hold"; mv "$CSV" "$SAFE"; }
  for r in 1 2 3; do
    TS=$(date -u +%Y%m%dT%H%M%S)
    ./build/${BIN[$proj]} > /dev/null
    OUT="project-proof/data/${TS}_${proj}_stability_r${r}.csv"
    { echo "# provenance: env=4090-container sha=$SHA cmd=\"./build/${BIN[$proj]} (stability r$r)\" date=$(date -u +%FT%TZ) gpu=\"NVIDIA GeForce RTX 4090\" driver=$DRV"; cat "$CSV"; } > "$OUT"
    rm -f "$CSV"
    echo "$proj r$r -> $OUT"
  done
  [ -n "$SAFE" ] && mv "$SAFE" "$CSV"   # 既有 raw 原样复位
done
echo STABILITY_REBENCH_DONE
