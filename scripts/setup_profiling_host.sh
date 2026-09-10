#!/usr/bin/env bash
set -uo pipefail

# 在一台**有性能计数器权限**的机器上,一条命令走完:环境自检 → 构建 → 全算子 NCU 采集 → 导出。
#
#   git clone <repo> && cd Kernel_Optimazation
#   bash scripts/setup_profiling_host.sh              # 六个 C++ 算子
#   bash scripts/setup_profiling_host.sh --with-python # 再加四个 torch 扩展算子(要先装 torch)
#
# 为什么需要这个脚本:本项目的主力机(容器)`RmProfilingAdminOnly=1` 且无 CAP_SYS_ADMIN,
# NCU 采不到计数器。所以采集要在另一台机器上做,而那台机器是干净的——
# 没有构建产物、可能没有 cmake、CUDA 版本也可能不同。这个脚本把这些差异一次性抹平。
#
# 关键的一件事:**显式 -DCMAKE_CUDA_ARCHITECTURES=89**。
# softmax / gemv / int8-quantize / cuda-reduce 四个算子的 CMakeLists 没写编译目标,
# 用了 CMake 默认值,实测编出来是 sm_75(Turing)。在 4090(sm_89)上跑的是驱动 JIT 出来的码,
# 而 cuobjdump 看到的是 sm_75 SASS——不是实际执行的码。寄存器分配与占用率也因此不可比。
# 这里统一按 sm_89 原生编译。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WITH_PYTHON=0
[ "${1:-}" = "--with-python" ] && WITH_PYTHON=1

CPP_OPS="gemm flash-attn softmax gemv int8-quantize cuda-reduce"
PY_OPS="fused-norm rope activation w8a8"
ARCH="${CUDA_ARCH:-89}"

die() { echo; echo "!! $*"; exit 1; }

# ---------------------------------------------------------------- 1. 环境自检
echo "=========== 1. 环境自检 ==========="
command -v nvidia-smi >/dev/null 2>&1 || die "没有 nvidia-smi"
nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader | sed 's/^/  /'

cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
if [ "$cc" != "89" ]; then
  echo "  ! 本机 compute_cap 不是 8.9(读到 $cc)。默认按 sm_$ARCH 编,"
  echo "    如需改用 CUDA_ARCH=<xx> 重跑本脚本。跨架构的结论迁移性要另行评估。"
fi

for t in ncu nvcc cuobjdump; do
  command -v $t >/dev/null 2>&1 || die "缺 $t。装 CUDA Toolkit:见 README 或 scripts/ncu_check_portable.sh 的提示"
done
command -v cmake >/dev/null 2>&1 || die "缺 cmake。先: sudo apt install -y cmake"
echo "  ncu   $(ncu --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1)"
echo "  nvcc  $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"

# 计数器权限是本脚本存在的全部理由,采集前必须确认,别构建完才发现采不了
flag="$(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -dc '0-9' | tail -c1)"
echo "  RmProfilingAdminOnly = ${flag:-未暴露}"
if [ "$flag" = "1" ]; then
  echo
  echo "  ! 计数器对普通用户是关的。若本机是虚机/裸金属且你有 root:"
  echo "      echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\"
  echo "        | sudo tee /etc/modprobe.d/zz-nvidia-profiler.conf"
  echo "      sudo update-initramfs -u && sudo reboot"
  echo "    重启后重跑本脚本。(容器内无解,得让宿主改。)"
  die "计数器权限未开,中止"
fi

# ---------------------------------------------------------------- 2. 构建
echo
echo "=========== 2. 构建六个 C++ 算子(sm_$ARCH) ==========="
built=0; failed=""
for d in $CPP_OPS; do
  [ -f "$d/CMakeLists.txt" ] || { echo "  跳过 $d(无 CMakeLists.txt)"; continue; }
  printf "  %-15s " "$d"
  if cmake -S "$d" -B "$d/build" -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >"$d/build_cfg.log" 2>&1 \
     && cmake --build "$d/build" -j"$(nproc)" >"$d/build.log" 2>&1; then
    echo "OK"; built=$((built+1))
  else
    # 配置阶段失败写 build_cfg.log、编译阶段失败写 build.log。
    # 只 tail 后者会在配置失败时报 "No such file",把真正的错误藏起来。
    echo "失败:"
    for lg in "$d/build_cfg.log" "$d/build.log"; do
      [ -s "$lg" ] || continue
      echo "      --- $(basename "$lg") 末尾 ---"
      tail -8 "$lg" | sed 's/^/      /'
    done
    failed="$failed $d"
  fi
done
[ -n "$failed" ] && echo "  构建失败:$failed"
[ "$built" -eq 0 ] && die "一个都没构建成功"

# ---------------------------------------------------------------- 3. 编译目标核对
echo
echo "=========== 3. 编译目标核对(必须是 sm_$ARCH) ==========="
bad=""
for d in $CPP_OPS; do
  b=$(ls "$d"/build/*bench* 2>/dev/null | head -1)
  [ -n "$b" ] || continue
  got=$(cuobjdump -lelf "$b" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | paste -sd',')
  printf "  %-15s %s\n" "$d" "${got:-?}"
  [ "$got" = "sm_$ARCH" ] || bad="$bad $d"
done
if [ -n "$bad" ]; then
  echo "  ! 以下算子的目标不是 sm_$ARCH:$bad"
  echo "    继续采集也可以,但这些算子的 SASS 与 occupancy 结论要标注实际目标。"
fi

# ---------------------------------------------------------------- 4. 采集
echo
echo "=========== 4. NCU 采集 ==========="
OPS="$CPP_OPS"
if [ "$WITH_PYTHON" = "1" ]; then
  if python3 -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
    OPS="$OPS $PY_OPS"
    echo "  含四个 torch 扩展算子"
  else
    echo "  ! 没有可用的 PyTorch,跳过 $PY_OPS"
    echo "    装法: pip3 install torch --index-url https://download.pytorch.org/whl/cu128"
  fi
fi
# 采集本身不出图(新机器可能没 matplotlib),导出在下一步单独做
NCU_OPS="$OPS" NCU_EXPORT=0 RUN_NCU_CSV=0 bash scripts/run_ncu_all.sh

# ---------------------------------------------------------------- 5. 导出 + 铁律自检
echo
echo "=========== 5. 校验与导出 ==========="
python3 scripts/export_ncu_for_mac.py

# profiler 隔离硬 gate:采集不得改动权威数据(CORE 铁律3 raw 不可变)
dirty="$(git status --porcelain -- '*/project-proof/data/' 2>/dev/null)"
if [ -n "$dirty" ]; then
  echo
  echo "  !! profiler 隔离被破坏,以下权威数据被改动:"
  printf '%s\n' "$dirty" | sed 's/^/      /'
  echo "     恢复: git checkout -- <上列文件>"
fi

echo
echo "=========== 完成 ==========="
n=$(find . -name "*.ncu-rep" -newermt "-1 day" 2>/dev/null | wc -l)
echo "  本轮新增报告约 $n 份"
echo "  导出包: artifacts/ncu_for_mac/ncu_for_mac_all.tar.gz"
echo
echo "  取回本地(在你的 Mac 上执行):"
echo "    scp <user>@<本机IP>:$ROOT_DIR/artifacts/ncu_for_mac/ncu_for_mac_all.tar.gz ."
echo "  然后用 Nsight Compute 2025.3+ 打开 reports/ 里的 .ncu-rep"
[ "$WITH_PYTHON" = "0" ] && {
  echo
  echo "  另外四个 torch 扩展算子($PY_OPS)还没采,装好 torch 后:"
  echo "    bash scripts/setup_profiling_host.sh --with-python"
}
