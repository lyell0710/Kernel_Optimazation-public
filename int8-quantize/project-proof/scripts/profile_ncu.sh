#!/usr/bin/env bash
# 按 int8 quantize 版本分别采集完整 NCU Section（与 ncu_metrics.inc.sh 一致）。
# 用法（在 int8-quantize 目录已编译 build/int8_quantize_bench 后）：
#   bash project-proof/scripts/profile_ncu.sh
# 环境变量：
#   BENCH_ITERS    默认 1
#   RUN_NCU_CSV    若设为 1，额外导出 int8_quantize_ncu.csv（全版本一次跑）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=/dev/null
source "$KO_ROOT/scripts/ncu_metrics.inc.sh"

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_PATH="$ROOT_DIR/build/int8_quantize_bench"
OUT_DIR="$ROOT_DIR/project-proof/profiling/ncu"
CSV_BASE="$OUT_DIR/int8_quantize_ncu"

mkdir -p "$OUT_DIR"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found. Please install Nsight Compute first."
  exit 1
fi

if [ ! -x "$BIN_PATH" ]; then
  echo "binary not found: $BIN_PATH"
  echo "run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

BENCH_ITERS="${BENCH_ITERS:-1}"

section_args=()
for s in "${NCU_PROFILE_SECTIONS[@]}"; do
  section_args+=(--section "$s")
done

PROFILE_TARGETS=(
  "baseline:quantize_baseline_kernel"
  "v0:quantize_v0_kernel"
  "v1:quantize_v1_kernel"
  "v2:quantize_v2_kernel"
  "v3:quantize_v3_kernel"
  "v4:quantize_v4_kernel"
)

run_ncu_rep() {
  local tag="$1"
  local kernel="$2"
  local out_prefix="$OUT_DIR/int8_quantize_${tag}_profile"
  local err
  err="$(mktemp)"
  set +e
  ncu \
    -f \
    --target-processes all \
    -k "$kernel" \
    "${section_args[@]}" \
    -o "$out_prefix" \
    env BENCH_ITERS="$BENCH_ITERS" QUANTIZE_PROFILE_ONLY="$tag" "$BIN_PATH" 2>"$err"
  local status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    if grep -q ERR_NVGPUCTRPERM "$err" 2>/dev/null; then
      cat "$err"
      rm -f "$err"
      echo "NCU permission blocked by ERR_NVGPUCTRPERM."
      exit 0
    fi
    cat "$err"
    rm -f "$err"
    echo "NCU profiling failed for kernel=${kernel} tag=${tag} (exit ${status})"
    exit "$status"
  fi
  rm -f "$err"
  echo "NCU report: ${out_prefix}.ncu-rep"
}

for entry in "${PROFILE_TARGETS[@]}"; do
  tag="${entry%%:*}"
  kernel="${entry#*:}"
  echo "== Profiling ${tag} (${kernel}) -> int8_quantize_${tag}_profile.ncu-rep"
  run_ncu_rep "$tag" "$kernel"
done

if [ "${RUN_NCU_CSV:-0}" = "1" ]; then
  echo "== RUN_NCU_CSV=1: exporting extended metrics CSV"
  set +e
  ncu \
    -f \
    --target-processes all \
    --metrics "$NCU_CSV_METRICS" \
    --csv \
    --log-file "${CSV_BASE}.csv" \
    env BENCH_ITERS="$BENCH_ITERS" "$BIN_PATH"
  status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    if awk '/ERR_NVGPUCTRPERM/ {found=1} END{exit found?0:1}' "${CSV_BASE}.csv" 2>/dev/null; then
      echo "NCU CSV export blocked by ERR_NVGPUCTRPERM."
      exit 0
    fi
    echo "NCU CSV export failed with exit code ${status}"
    exit "$status"
  fi
  echo "NCU CSV saved: ${CSV_BASE}.csv"
fi

echo "Done. Reports under: ${OUT_DIR}/int8_quantize_*_profile.ncu-rep"
