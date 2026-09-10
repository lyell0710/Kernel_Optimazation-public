#!/usr/bin/env bash
# 按版本分别采集 flash-attn 的 NCU 报告(.ncu-rep),供 ncu-ui 的 Add Baseline 做版本梯对比。
# 用法(在 flash-attn/ 已编译 build/fa2_bench 后):
#   bash project-proof/scripts/profile_ncu.sh
# 环境变量:
#   BENCH_ITERS  默认 1(采集只需一次计时循环)
#   NCU_SKIP / NCU_COUNT  限定 launch 窗口(见 scripts/ncu_profile_lib.inc.sh 注释)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$(cd "$SCRIPT_DIR/../../.." && pwd)/scripts/ncu_profile_lib.inc.sh"
ncu_lib_init "$SCRIPT_DIR/../../.."

BIN_PATH="$ROOT_DIR/build/fa2_bench"
OUT_DIR="$ROOT_DIR/project-proof/profiling/ncu"

if [ ! -x "$BIN_PATH" ]; then
  echo "找不到可执行文件: $BIN_PATH"
  echo "先构建: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
  exit 1
fi

BENCH_ITERS="${BENCH_ITERS:-1}"

PROFILE_TARGETS=(
  "v0_warp_row:fa2_v0_kernel"
  "v1_smem_tile:fa2_v1_kernel"
  "v2_wmma:fa2_v2_kernel"
  "v3_8warp:fa2_v3_kernel"
  "v4_overlap:fa2_v4_kernel"
  "ref_naive:ref_kernel"
)

# profiler 隔离(CORE):本算子没有 <OP>_PROFILE_ONLY 早退分支,bench 跑完必写 CSV。
# 两道保险:① BENCH_OUT 指到 /dev/null(本算子读这个变量);
# ② 在一次性沙箱 cwd 里跑——CSV 路径是相对 cwd 的,即便哪天不再读 BENCH_OUT 也写不进仓。
# 少了这两道,profiler 口径的时延会静默覆盖 project-proof/data/ 的权威数据。
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT
cd "$SANDBOX"

ncu_profile_all "$OUT_DIR" "fa2" PROFILE_TARGETS \
  env BENCH_ITERS="$BENCH_ITERS" BENCH_OUT=/dev/null "$BIN_PATH"

ncu_export_csv "$OUT_DIR" "fa2" \
  env BENCH_ITERS="$BENCH_ITERS" BENCH_OUT=/dev/null "$BIN_PATH"
