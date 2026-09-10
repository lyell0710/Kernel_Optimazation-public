#!/usr/bin/env bash
# 将四个子项目已生成的 *_profile.ncu-rep 打成 tar.gz，便于 scp 到 Mac。
# 用法（在 Kernel_Optimazation 根目录）：
#   bash scripts/pack_ncu_reps_for_mac.sh
# 输出目录：./artifacts/ncu_for_mac/
# Mac 拉取示例（替换为你的 Tailscale / 局域网 IP）：
#   mkdir -p ~/Desktop/CUDA/NCU-report && scp "$RHOST":~/path/to/Kernel_Optimazation/artifacts/ncu_for_mac/*.tar.gz ~/Desktop/CUDA/NCU-report/
set -euo pipefail

KO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$KO_ROOT/artifacts/ncu_for_mac"
mkdir -p "$OUT_DIR"

pack_dir() {
  local name="$1"
  local dir="$2"
  local pattern="$3"
  local out="$OUT_DIR/${name}_ncu_profiles.tar.gz"
  if compgen -G "$dir/$pattern" > /dev/null; then
    (cd "$dir" && tar czvf "$out" $pattern)
    echo "Wrote $out"
  else
    echo "Skip $name (no files matching $dir/$pattern)"
  fi
}

pack_dir "cuda-reduce" "$KO_ROOT/cuda-reduce/project-proof/profiling/ncu" 'reduce_*_profile.ncu-rep'
pack_dir "softmax" "$KO_ROOT/softmax/project-proof/profiling/ncu" 'softmax_*_profile.ncu-rep'
pack_dir "gemv" "$KO_ROOT/gemv/project-proof/profiling/ncu" 'gemv_*_profile.ncu-rep'
pack_dir "int8-quantize" "$KO_ROOT/int8-quantize/project-proof/profiling/ncu" 'int8_quantize_*_profile.ncu-rep'

echo ""
echo "== 在 Mac 终端执行（路径 / IP 请按需改）："
MAC_BASE="${MAC_BASE:-$HOME/Desktop/CUDA/NCU-report}"
RHOST="${RHOST:-user@host}"
ART_REMOTE="$RHOST:$OUT_DIR"
echo "mkdir -p \"$MAC_BASE/reduce\" \"$MAC_BASE/softmax\" \"$MAC_BASE/gemv\" \"$MAC_BASE/int8-quantize\""
echo "scp \"$ART_REMOTE/cuda-reduce_ncu_profiles.tar.gz\" \"$MAC_BASE/reduce/\" && tar xzf \"$MAC_BASE/reduce/cuda-reduce_ncu_profiles.tar.gz\" -C \"$MAC_BASE/reduce\""
echo "scp \"$ART_REMOTE/softmax_ncu_profiles.tar.gz\" \"$MAC_BASE/softmax/\" && tar xzf \"$MAC_BASE/softmax/softmax_ncu_profiles.tar.gz\" -C \"$MAC_BASE/softmax\""
echo "scp \"$ART_REMOTE/gemv_ncu_profiles.tar.gz\" \"$MAC_BASE/gemv/\" && tar xzf \"$MAC_BASE/gemv/gemv_ncu_profiles.tar.gz\" -C \"$MAC_BASE/gemv\""
echo "scp \"$ART_REMOTE/int8-quantize_ncu_profiles.tar.gz\" \"$MAC_BASE/int8-quantize/\" && tar xzf \"$MAC_BASE/int8-quantize/int8-quantize_ncu_profiles.tar.gz\" -C \"$MAC_BASE/int8-quantize\""
