#!/usr/bin/env bash
set -uo pipefail

# 全算子 nsys 采集。与 run_ncu_all.sh 互补:
#
#   NCU  = 单 kernel 内部的显微镜。它默认**序列化 kernel**,因此把 kernel 之间的
#          间隙、launch 开销、并发全部抹掉了。且本机无计数器权限时它什么都给不了。
#   nsys = 时间线。它答的恰恰是 NCU 答不了的那类问题:
#          launch 次数与开销、kernel 间隙、host/device 重叠、CUDA Graph 塌缩。
#          **不需要任何特殊权限**,容器里就能跑。
#
# 已验证的一例(rope):v1 走 run_one() 对 q/k 各调一次 = 248 次 launch,
# v2 合并成一次 = 124 次,精确 2:1。这解释了 README 里 v2「HBM 区间慢 1.1%
# 但 decode 快 30%」的两面性——kernel 内多了分支(HBM 吃亏),launch 砍半(decode 大赚)。
# 这个结论 NCU 给不出来。
#
# CORE 铁律:profiler 下跑 bench 一律不写 data/(全部 BENCH_OUT=/dev/null),
# profiler 环境的时延数字永不进 benchmark 表。本脚本只产出 .nsys-rep。
#
# 用法:   bash scripts/run_nsys_all.sh
# 只跑部分:NSYS_OPS="rope gemm" bash scripts/run_nsys_all.sh
# 迭代数:  BENCH_ITERS=20 bash scripts/run_nsys_all.sh   (默认 20,够看 launch 模式即可)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITERS="${BENCH_ITERS:-20}"
PY="${PYTHON:-python3}"

command -v nsys >/dev/null 2>&1 || { echo "nsys 不在 PATH"; exit 1; }

# preflight:本机是共享双卡,外来 GPU 进程会污染时间线也会抢显存。
foreign="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if [ "$foreign" -gt 0 ]; then
  echo "检测到 $foreign 个外来 GPU 进程,中止(时间线会被污染)。"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 1
fi

# op:kind:target   kind=bin 用 build/ 下的可执行文件;kind=py 用 bench.py
TARGETS=(
  "gemm:bin:build/gemm_bench"
  "flash-attn:bin:build/fa2_bench"
  "softmax:bin:build/softmax_bench"
  "gemv:bin:build/gemv_bench"
  "int8-quantize:bin:build/int8_quantize_bench"
  "cuda-reduce:bin:build/reduce_bench"
  "fused-norm:py:bench.py"
  "rope:py:bench.py"
  "activation:py:bench.py"
  "w8a8:py:bench.py"
)

OPS="${NSYS_OPS:-}"
ok=0; skip=0; fail=0

for entry in "${TARGETS[@]}"; do
  op="${entry%%:*}"; rest="${entry#*:}"; kind="${rest%%:*}"; tgt="${rest#*:}"
  if [ -n "$OPS" ] && ! printf '%s\n' $OPS | grep -qx "$op"; then continue; fi

  d="$ROOT_DIR/$op"
  out_dir="$d/project-proof/profiling/nsys"
  out="$out_dir/${op}_timeline"

  echo
  echo "========== [$op] nsys =========="
  if [ ! -e "$d/$tgt" ]; then
    echo "  跳过:找不到 $d/$tgt"
    [ "$kind" = bin ] && echo "  先构建: cd $op && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    skip=$((skip+1)); continue
  fi
  mkdir -p "$out_dir"

  # CORE 铁律「profiler 隔离」的实现:在**一次性沙箱 cwd** 里跑被测程序。
  # 为什么不能只靠 BENCH_OUT=/dev/null:十个 bench 里只有 gemm 与 flash-attn 读这个
  # 环境变量,另外四个 C++ bench 把路径写死成 const char* kCsvPath =
  # "project-proof/data/benchmark_results.csv"(相对 cwd)。在算子目录里跑 profiler
  # 会把带 profiler 开销的时延直接覆盖掉权威数据——已经发生过一次,靠 git checkout 才救回来。
  # 相对路径 + 沙箱 cwd = 无论程序读不读 BENCH_OUT 都写不到仓里。
  sandbox="$(mktemp -d)"
  if [ "$kind" = bin ]; then
    ( cd "$sandbox" && BENCH_ITERS="$ITERS" BENCH_OUT=/dev/null \
      nsys profile --force-overwrite true -o "$out" --trace=cuda --stats=false "$d/$tgt" ) \
      >"$out_dir/${op}_nsys.log" 2>&1
  else
    # python 家族要在算子目录下才能找到 src/,但它们都认 BENCH_OUT
    ( cd "$d" && BENCH_ITERS="$ITERS" BENCH_OUT=/dev/null \
      nsys profile --force-overwrite true -o "$out" --trace=cuda --stats=false "$PY" "$tgt" ) \
      >"$out_dir/${op}_nsys.log" 2>&1
  fi
  rm -rf "$sandbox"

  if [ -f "$out.nsys-rep" ]; then
    echo "  -> $(realpath --relative-to="$ROOT_DIR" "$out.nsys-rep")"
    # 立刻导出 kernel 汇总:launch 次数是本次采集的主要目的,存成 csv 方便比对
    # nsys stats 会先吐两行进度提示再吐 CSV,直接重定向会污染表头。
    # 从真表头(以 "Time (%)" 开头那行)起截断。
    # --force-export:重跑时 .sqlite 会残留,nsys 见到旧导出会直接报 usage 错误退出,
    # 表现为「csv 空文件」而不是明确报错——静默失败,必须显式强制重导。
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$out.nsys-rep" 2>/dev/null \
      | awk '/^Time \(%\)/{p=1} p' > "$out_dir/${op}_kern_sum.csv" \
      && echo "     kernel 汇总 -> ${op}_kern_sum.csv"
    ok=$((ok+1))
  else
    echo "  失败,日志尾部:"; tail -8 "$out_dir/${op}_nsys.log" | sed 's/^/    /'
    fail=$((fail+1))
  fi
done

# 硬 gate:profiler 跑完,project-proof/data/ 下不允许有任何改动。
# 静默污染比报错危险——覆盖后文件看起来完全正常,只有数字悄悄换成了 profiler 口径。
dirty="$(cd "$ROOT_DIR" && git status --porcelain -- '*/project-proof/data/' 2>/dev/null)"
if [ -n "$dirty" ]; then
  echo
  echo "!! profiler 隔离被破坏:以下权威数据文件被改动"
  printf '%s\n' "$dirty" | sed 's/^/    /'
  echo "   恢复: git checkout -- <上列文件>"
  exit 1
fi

echo
echo "完成: 成功 $ok / 跳过 $skip / 失败 $fail  (project-proof/data/ 未被触碰)"
echo "下一步:比对各版本的 Instances 列——launch 次数的变化是融合与合并类优化的直接证据。"
