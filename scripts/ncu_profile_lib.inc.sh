# 各算子 project-proof/scripts/profile_ncu.sh 的共用采集逻辑。
# shellcheck shell=bash
#
# 设计要点(与 softmax/gemv/int8-quantize/cuda-reduce 的旧脚本口径一致):
#   1. section 组合来自 ncu_metrics.inc.sh,与 2026-05 那批 Laptop 报告完全相同,
#      否则新旧报告在 GUI 里无法逐 metric 对照。
#   2. kernel 选择用 `-k regex:` + `^` 锚定。本仓 kernel 不在匿名 namespace 里,
#      demangled 名直接以函数名开头,所以 `^` 有效;不锚定会出事:
#      w8a8 的 `v1_kernel` 是 `dequant_v1_kernel` / `gemv_v1_kernel` 的子串,
#      不锚定会把三个不同算子的 kernel 混进同一份报告。
#   3. launch 窗口(NCU_SKIP/NCU_COUNT)可覆盖。为什么需要它:
#      python 家族的 bench.py 逐 regime(decode/l2/prefill/hbm)跑同一个 kernel,
#      timeit 又是 warmup=10 + iters,于是一个 kernel 在一次进程里被 launch 几十次、
#      横跨好几个访存区间。把它们收进同一份报告 = 把 L2 区间和 HBM 区间混为一谈,
#      正是 EXP-K04 那条"结论会因区间翻转"的坑。默认先全采并由
#      verify_ncu_reports.py 报出实际分布,再按分布钉窗口重采。

set -euo pipefail

ncu_lib_init() {
  KO_ROOT="$(cd "$1" && pwd)"
  # shellcheck source=/dev/null
  source "$KO_ROOT/scripts/ncu_metrics.inc.sh"

  if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found in PATH. 装 Nsight Compute 后重试。"
    exit 1
  fi

  NCU_SECTION_ARGS=()
  for s in "${NCU_PROFILE_SECTIONS[@]}"; do
    NCU_SECTION_ARGS+=(--section "$s")
  done

  # 默认不限窗口:先全采,看清 launch 分布再钉。
  NCU_SKIP="${NCU_SKIP:-}"
  NCU_COUNT="${NCU_COUNT:-}"
}

# ncu_run_one <out_prefix> <kernel_regex> <cmd...>
# 返回 0=成功,10=权限被拒(调用方应整体放弃并提示),其余=真失败
ncu_run_one() {
  local out_prefix="$1"; shift
  local kernel="$1"; shift

  local window=()
  # 用 :- 兜底:这两个是「钉采样窗口」的可选环境变量
  # (见 docs/archive/2026-08-29_profiling_host_tasks.md §4,原 PROFILING_HOST_TASKS.md)。
  # 它们的默认值由上面的初始化函数设置,但那形成了一个隐式前置依赖——
  # 任何先调 ncu_run_one 而没走过初始化的调用方,都会在 set -u 下直接崩,
  # 且 "unbound variable" 完全指不到真正的原因。就地兜底比依赖调用顺序稳。
  [ -n "${NCU_SKIP:-}" ]  && window+=(-s "${NCU_SKIP}")
  [ -n "${NCU_COUNT:-}" ] && window+=(-c "${NCU_COUNT}")

  local err status
  err="$(mktemp)"
  set +e
  ncu -f \
    --target-processes all \
    -k "regex:$kernel" \
    "${window[@]}" \
    "${NCU_SECTION_ARGS[@]}" \
    -o "$out_prefix" \
    "$@" >/dev/null 2>"$err"
  status=$?
  set -e

  if [ "$status" -ne 0 ]; then
    if grep -q ERR_NVGPUCTRPERM "$err" 2>/dev/null; then
      echo
      echo "==================================================================="
      echo " ERR_NVGPUCTRPERM:本机没有 GPU 性能计数器权限,采集无法进行。"
      echo " 容器内无解,需二选一(见 artifacts/ncu_for_mac/MANIFEST.md):"
      echo "   宿主: /etc/modprobe.d/nvidia-profiler.conf"
      echo "         → options nvidia NVreg_RestrictProfilingToAdminUsers=0"
      echo "   容器: docker run --cap-add=SYS_ADMIN"
      echo "==================================================================="
      rm -f "$err"
      return 10
    fi
    echo "NCU 采集失败 kernel=${kernel} (exit ${status}):"
    cat "$err"
    rm -f "$err"
    return "$status"
  fi
  rm -f "$err"

  # ncu 对「regex 没匹配到任何 kernel」是静默的:退出码 0,但报告里一个实例都没有。
  # 这种空报告下游用不了,而且会掩盖 regex 写错(gemm 的 cuBLAS 臂就是待验证的猜测)。
  # 但它**不该中止整轮采集**——机器在计费,不能因为一个臂没抓到就丢掉其余几十份。
  # 两种"没抓到"的形态,都必须放过而不是中止:
  #   (a) ncu 压根不生成报告文件(实测:regex 无匹配时就是这样,只在 stdout 打
  #       "No kernels were profiled");
  #   (b) 文件生成了但零实例。
  # 原实现只处理了 (b),且用 `ncu -i` 去读一个不存在的文件——它把错误打在
  # **stdout** 上,于是 `| sed 1d | wc -l` 得到 1 而非 0,判空恒不成立;
  # 更糟的是 `set -euo pipefail` 的 pipefail 让该管道的非零退出码直接中止整轮,
  # 后面所有臂全部丢失(w8a8 因此只采到 2/8)。故先判文件、再判内容,并给管道兜底。
  if [ ! -s "${out_prefix}.ncu-rep" ]; then
    echo "  !! 空报告(regex 未匹配到任何 kernel,ncu 未生成文件),已跳过:$kernel"
    rm -f "${out_prefix}.ncu-rep"
    return 0
  fi
  local n
  n=$(ncu -i "${out_prefix}.ncu-rep" --page details --csv 2>/dev/null | sed '1d' | wc -l || true)
  if [ "${n:-0}" -eq 0 ]; then
    echo "  !! 空报告(零实例),已删除并跳过:$kernel"
    rm -f "${out_prefix}.ncu-rep"
    return 0
  fi
  echo "  -> ${out_prefix}.ncu-rep"
  return 0
}

# ncu_profile_all <out_dir> <prefix> <targets_nameref> <cmd...>
# targets 数组元素格式: "<tag>:<kernel_regex_without_anchor>"
ncu_profile_all() {
  local out_dir="$1"; shift
  local prefix="$1"; shift
  local -n _targets="$1"; shift

  mkdir -p "$out_dir"
  local entry tag kernel rc
  for entry in "${_targets[@]}"; do
    tag="${entry%%:*}"
    kernel="${entry#*:}"
    echo "== [${prefix}] ${tag}  (kernel ^${kernel})"
    set +e
    ncu_run_one "${out_dir}/${prefix}_${tag}_profile" "^${kernel}" "$@"
    rc=$?
    set -e
    if [ "$rc" -eq 10 ]; then
      echo "采集中止。权限恢复后重跑本脚本即可。"
      exit 0
    elif [ "$rc" -ne 0 ]; then
      exit "$rc"
    fi
  done
  echo "完成。报告位于 ${out_dir}/${prefix}_*_profile.ncu-rep"
  echo "下一步:python3 scripts/export_ncu_for_mac.py  (口径校验 + 导出)"
}

# ncu_export_csv <out_dir> <prefix> <cmd...>
# 导出 NCU_CSV_METRICS(含分管线利用率)。section 给不出 Tensor/FMA/ALU 的分管线占用,
# 只能走 --metrics;而"wmma 有没有真的用上 Tensor Core"恰恰要它。
# 仅在 RUN_NCU_CSV=1 时执行(默认关:多一个 pass,且新机器上未必需要)。
ncu_export_csv() {
  [ "${RUN_NCU_CSV:-0}" = "1" ] || return 0
  local out_dir="$1"; shift
  local prefix="$1"; shift
  local csv="${out_dir}/${prefix}_ncu.csv"
  echo "== RUN_NCU_CSV=1: 导出扩展指标 -> $(basename "$csv")"
  set +e
  ncu -f --target-processes all --metrics "$NCU_CSV_METRICS" --csv --log-file "$csv" "$@" >/dev/null 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ] || ! [ -s "$csv" ]; then
    echo "  !! 扩展指标导出失败(rc=$rc),不影响已采的 .ncu-rep"
    return 0
  fi
  echo "  -> $csv"
}
