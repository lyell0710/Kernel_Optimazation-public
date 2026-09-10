#!/usr/bin/env bash
# 回归测试:空报告(regex 抓不到 kernel)不得中止整轮采集。
#
# 为什么有这个测试:`ncu_profile_lib.inc.sh` 的空报告拦截曾**声称**生效
# (代码注释 + 采集主机任务书 §5 都写着"不中止整轮"),但实测从未生效——
# (任务书已归档:docs/archive/2026-08-29_profiling_host_tasks.md)
# w8a8 因此只采到 2/8 个臂就整轮退出,而日志里看不出任何原因。见 EXP-K09 §7.1。
# 这类"静默丢数据"的 bug 修完不加测试,一定会再回来。
#
# 关键设计:**坏臂必须夹在两个好臂中间**。若把它放最后,即使拦截失效也测不出来
# ——后面本来就没有臂了。中间位置才能证明"坏臂没有打断后续采集"。
#
# 用法: bash scripts/test_ncu_empty_report_guard.sh
# 需要:nvcc + ncu + 计数器权限(RmProfilingAdminOnly=0 或 root)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

skip() { echo "SKIP: $*"; exit 77; }
command -v nvcc >/dev/null 2>&1 || skip "无 nvcc"
command -v ncu  >/dev/null 2>&1 || skip "无 ncu"
flag="$(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -dc '0-9' | tail -c1)"
[ "${flag:-1}" = "1" ] && [ "$(id -u)" != "0" ] && skip "无计数器权限(RmProfilingAdminOnly=1 且非 root)"

cat > "$TMP/t.cu" <<'CU'
__global__ void alpha_kernel(float* o, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=i;}
__global__ void omega_kernel(float* o, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=i*2.f;}
int main(){float*d; cudaMalloc(&d,1<<20);
  alpha_kernel<<<64,256>>>(d,1<<16); omega_kernel<<<64,256>>>(d,1<<16);
  cudaDeviceSynchronize(); return 0;}
CU
nvcc -arch=sm_89 -o "$TMP/t" "$TMP/t.cu" 2>"$TMP/nvcc.err" \
  || nvcc -o "$TMP/t" "$TMP/t.cu" 2>>"$TMP/nvcc.err" \
  || { echo "FAIL: 测试程序编译失败"; cat "$TMP/nvcc.err"; exit 1; }

# shellcheck source=/dev/null
source "$ROOT/scripts/ncu_metrics.inc.sh"
# shellcheck source=/dev/null
source "$ROOT/scripts/ncu_profile_lib.inc.sh"

# 好 / 坏 / 好 —— 坏臂在中间是本测试的要害
TARGETS=(
  "good_first:alpha_kernel"
  "bad_middle:__definitely_no_such_kernel__"
  "good_last:omega_kernel"
)

echo "== 采集 3 个臂(中间那个 regex 故意抓不到) =="
set +e
ncu_profile_all "$TMP/out" "guardtest" TARGETS "$TMP/t"
rc=$?
set -e

fail=0
chk() { if [ "$2" = "$3" ]; then echo "  PASS: $1"; else echo "  FAIL: $1 (期望 $3,实得 $2)"; fail=1; fi; }

echo "== 断言 =="
chk "整轮退出码为 0(未被空报告中止)" "$rc" "0"
chk "好臂1 产出报告" "$([ -s "$TMP/out/guardtest_good_first_profile.ncu-rep" ] && echo yes || echo no)" "yes"
chk "坏臂  未留下报告文件"  "$([ -e "$TMP/out/guardtest_bad_middle_profile.ncu-rep" ] && echo yes || echo no)" "no"
chk "好臂2 产出报告(证明未被中断)" "$([ -s "$TMP/out/guardtest_good_last_profile.ncu-rep" ] && echo yes || echo no)" "yes"

if [ "$fail" -eq 0 ]; then echo; echo "OK: 空报告拦截正常,坏臂未打断后续采集"; exit 0
else echo; echo "REGRESSION: 空报告拦截失效 —— 见 EXP-K09 §7.1"; exit 1; fi
