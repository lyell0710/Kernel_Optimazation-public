#!/usr/bin/env bash
# 按 softmax 版本分别采集 NCU 报告（.ncu-rep），便于 ncu-ui / Compare。
# 用法：在 softmax 目录已编译 build/softmax_bench 后执行：
#   bash project-proof/scripts/profile_ncu.sh
# 环境变量：
#   BENCH_ITERS   默认 1
#   RUN_NCU_CSV   若设为 1，在循环结束后额外导出 softmax_ncu.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=/dev/null
source "$KO_ROOT/scripts/ncu_metrics.inc.sh"

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_PATH="$ROOT_DIR/build/softmax_bench"
OUT_DIR="$ROOT_DIR/project-proof/profiling/ncu"
CSV_BASE="$OUT_DIR/softmax_ncu"

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

# v0–v4 为模板实例，使用 regex 匹配 NCU 中的 demangled 名
PROFILE_TARGETS=(
  "baseline:softmax_baseline_kernel"
  "v0:regex:softmax_v0_kernel"
  "v1:regex:softmax_v1_kernel"
  "v2:regex:softmax_v2_kernel"
  "v3:regex:softmax_v3_kernel"
  "v4:regex:softmax_v4_kernel"
  "v4.2:regex:softmax_v4_2_kernel"
  "v4.3:regex:softmax_v4_3_kernel"
  "v4.4:regex:softmax_v4_4_kernel"
  "cublas:regex:softmax_cublas_kernel"
)

run_ncu_rep() {
  local tag="$1"
  local kernel="$2"
  local out_prefix="$OUT_DIR/softmax_${tag}_profile"
  local err
  err="$(mktemp)"
  set +e
  ncu \
    -f \
    --target-processes all \
    -k "$kernel" \
    "${section_args[@]}" \
    -o "$out_prefix" \
    env BENCH_ITERS="$BENCH_ITERS" SOFTMAX_PROFILE_ONLY="$tag" "$BIN_PATH" 2>"$err"
  local status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    if grep -q ERR_NVGPUCTRPERM "$err" 2>/dev/null; then
      cat "$err"
      rm -f "$err"
      echo "NCU permission blocked by ERR_NVGPUCTRPERM."
      echo "Please enable GPU performance counter access, then rerun."
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
  echo "== Profiling ${tag} (${kernel}) -> softmax_${tag}_profile.ncu-rep"
  run_ncu_rep "$tag" "$kernel"
done

if [ "${RUN_NCU_CSV:-0}" = "1" ]; then
  echo "== RUN_NCU_CSV=1: exporting extended metrics CSV for plot_ncu_summary.py"
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

echo "Done. Reports under: ${OUT_DIR}/softmax_*_profile.ncu-rep"
