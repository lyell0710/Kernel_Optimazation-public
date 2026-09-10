#!/usr/bin/env bash
# 按版本分别采集 rope 的 NCU 报告(.ncu-rep),供 ncu-ui 的 Add Baseline 做版本梯对比。
# 用法(在 rope/ 下):
#   bash project-proof/scripts/profile_ncu.sh
# 环境变量:
#   BENCH_ITERS  默认 1
#   NCU_SKIP / NCU_COUNT  限定 launch 窗口
#
# 注意:bench.py 逐 regime 跑同一个 kernel,timeit 又是 warmup=10 + iters,
# 所以一个 kernel 在一次进程里会 launch 多次且横跨多个访存区间。默认全采,
# 采完必须跑 scripts/verify_ncu_reports.py 看每份报告里的 grid 分布,
# 再用 NCU_SKIP/NCU_COUNT 钉住想要的那个 regime 重采——
# 不这么做就会把 L2 区间和 HBM 区间混进同一份报告(EXP-K04 的教训)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$(cd "$SCRIPT_DIR/../../.." && pwd)/scripts/ncu_profile_lib.inc.sh"
ncu_lib_init "$SCRIPT_DIR/../../.."

OUT_DIR="$ROOT_DIR/project-proof/profiling/ncu"
PY="${PYTHON:-python3}"
BENCH_ITERS="${BENCH_ITERS:-1}"

cd "$ROOT_DIR"

# 先在 ncu 之外跑一遍:torch cpp_extension 首编 60-90 秒、Triton 还有 JIT,
# 让它们在采集之外完成,否则测到的是编译而不是 kernel(EXP-T02 的 JIT 伪影教训)。
echo "== 预热:ncu 之外先构建扩展并跑通 bench.py"
BENCH_ITERS=1 BENCH_OUT=/dev/null "$PY" bench.py >/dev/null

PROFILE_TARGETS=(
  "v0:v0_kernel_safe"
  "v1:v1_kernel"
  "v2:v2_kernel"
  "v3:v3_kernel"
  "v4:v4_kernel"
)

ncu_profile_all "$OUT_DIR" "rope" PROFILE_TARGETS \
  env BENCH_ITERS="$BENCH_ITERS" BENCH_OUT=/dev/null "$PY" bench.py
