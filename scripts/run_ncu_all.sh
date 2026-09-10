#!/usr/bin/env bash
set -euo pipefail

# 全算子 NCU 采集总入口。逐算子调用 <op>/project-proof/scripts/profile_ncu.sh,
# 有 plot_ncu_summary.py 的算子顺带出图,最后统一校验 + 导出 Mac 用的报告包。
#
# 前提:本机有 GPU 性能计数器权限。容器默认没有(ERR_NVGPUCTRPERM),需宿主设
#   options nvidia NVreg_RestrictProfilingToAdminUsers=0
# 或以 --cap-add=SYS_ADMIN 起容器。详见 artifacts/ncu_for_mac/MANIFEST.md。
#
# 只采不出图:      RUN_NCU_CSV=0 bash scripts/run_ncu_all.sh
# 只跑部分算子:    NCU_OPS="gemm flash-attn" bash scripts/run_ncu_all.sh
# 采完不重建导出包:NCU_EXPORT=0 bash scripts/run_ncu_all.sh

RUN_NCU_CSV="${RUN_NCU_CSV:-1}"
NCU_EXPORT="${NCU_EXPORT:-1}"
export RUN_NCU_CSV

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found in PATH."
  echo "Please install Nsight Compute and ensure 'ncu --version' works."
  exit 1
fi

# 先探一次权限,别等跑到第七个算子才发现采不了
probe="$(mktemp -d)"
if ! ncu --version >/dev/null 2>&1; then
  echo "ncu 无法运行。"; exit 1
fi
rm -rf "$probe"

DEFAULT_OPS="softmax gemv int8-quantize cuda-reduce gemm flash-attn fused-norm rope activation w8a8"
OPS="${NCU_OPS:-$DEFAULT_OPS}"

run_one_ncu() {
  local proj="$1"
  echo
  echo "========== [$proj] ncu profile =========="
  if [ ! -f "$ROOT_DIR/$proj/project-proof/scripts/profile_ncu.sh" ]; then
    echo "[$proj] 无 profile_ncu.sh,跳过"
    return 0
  fi
  pushd "$ROOT_DIR/$proj" >/dev/null
  bash project-proof/scripts/profile_ncu.sh
  # 出图脚本只有早期四个算子有;新增算子暂无,不当作错误。
  # 出图失败也不中断采集:新机器上可能没装 matplotlib,而报告本身才是目的,
  # 图随时可以回本机补出。set -e 下必须显式吞掉返回码。
  if [ -f project-proof/scripts/plot_ncu_summary.py ]; then
    echo "========== [$proj] ncu plots =========="
    python3 project-proof/scripts/plot_ncu_summary.py || \
      echo "  (出图跳过:$? —— 不影响 .ncu-rep,回本机再补)"
  fi
  popd >/dev/null
}

for op in $OPS; do
  run_one_ncu "$op"
done

if [ "$NCU_EXPORT" = "1" ]; then
  echo
  echo "========== 校验 + 导出 Mac 报告包 =========="
  python3 "$ROOT_DIR/scripts/export_ncu_for_mac.py"
fi

# profiler 隔离硬 gate:采集不得改动权威数据(CORE 铁律3)。
# 静默污染比报错危险——覆盖后文件看起来完全正常,只有数字悄悄换成了 profiler 口径。
dirty="$(cd "$ROOT_DIR" && git status --porcelain -- '*/project-proof/data/' 2>/dev/null)"
if [ -n "$dirty" ]; then
  echo
  echo "!! profiler 隔离被破坏,以下权威数据被改动:"
  printf '%s\n' "$dirty" | sed 's/^/    /'
  echo "   恢复: git checkout -- <上列文件>"
  echo "   这些文件在恢复前不可用作时延权威。"
  exit 1
fi

echo
echo "All NCU profiling jobs completed.  (project-proof/data/ 未被触碰)"
