#!/usr/bin/env bash
# ============================================================
# 新开的 GPU 虚机 / 裸金属:先跑这个,三分钟决定留还是销毁。
#   bash vm_gpu_check.sh
# ============================================================
# 判断四件事:
#   1. 是不是真虚机(能自己控制驱动) —— 决定你能不能自己开性能计数器
#   2. GPU 型号 —— 是不是 sm_89(4090 / L40S / RTX 6000 Ada 同一族,代码不用改)
#   3. 驱动版本 —— CUDA 13.x 要 >= 580.65.06
#   4. 性能计数器现在开没开;没开的话,你有没有权限自己开
#
# 全程只读,不改任何东西。要改的命令它只打印不执行。

set -uo pipefail
say() { printf '%s\n' "$*"; }
line(){ printf '%s\n' "------------------------------------------------------------"; }

say "=== 1. 形态:虚机还是容器 ==="
init="$(cat /proc/1/comm 2>/dev/null)"
IS_VM=0
if [ "$init" = "systemd" ] || [ "$init" = "init" ]; then
  say "  init = $init  → 虚机/裸金属"; IS_VM=1
else
  say "  init = $init  → 容器(驱动参数由宿主控制,你改不了)"
fi
if command -v systemd-detect-virt >/dev/null 2>&1; then
  say "  虚拟化    $(systemd-detect-virt 2>/dev/null || echo 未知)"
fi
# 决定性的一条:nvidia 内核模块是不是在"本机"加载的。
# 虚机里 lsmod 能看到 nvidia 且 /sys/module/nvidia/parameters 可见 = 驱动归你管。
if [ -d /sys/module/nvidia/parameters ]; then
  say "  /sys/module/nvidia/parameters 可见 → nvidia 模块在本机加载,参数归你管"
else
  say "  /sys/module/nvidia/parameters 不可见 → 驱动不在本机加载"
fi
say "  当前用户  $(id -un) (uid=$(id -u))"
cap="$(awk '/^CapEff/{print $2}' /proc/self/status 2>/dev/null)"
if [ -n "$cap" ] && [ $(( 0x$cap >> 21 & 1 )) -eq 1 ]; then
  say "  CAP_SYS_ADMIN  有  ← 这个是关键,有它才可能采计数器"; HAS_CAP=1
else
  say "  CAP_SYS_ADMIN  无"; HAS_CAP=0
fi

say
say "=== 2. GPU 与驱动 ==="
command -v nvidia-smi >/dev/null 2>&1 || { say "  没有 nvidia-smi,驱动没装"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /'
drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
say
if [ "$cc" = "8.9" ]; then
  say "  compute_cap 8.9 = sm_89,与 RTX 4090 同族(L40S / RTX 6000 Ada 也是)"
  say "  → 你的 CMAKE_CUDA_ARCHITECTURES 89 / -arch=sm_89 不用改"
else
  say "  compute_cap $cc ≠ 8.9 → 编译目标要改,结论迁移性也要重新评估"
fi
maj="${drv%%.*}"
if [ -n "$maj" ] && [ "$maj" -ge 580 ] 2>/dev/null; then
  say "  驱动 $drv >= 580 → CUDA 13.x 可用(你现在的工具链是 13.2)"
else
  say "  驱动 $drv < 580 → 跑不了 CUDA 13.x。要么让平台升驱动,"
  say "     要么用该机自带的 CUDA 12.x 重新构建(sm_89 一样支持,代码不用改)"
fi

say
say "=== 3. 性能计数器 ==="
FLAG=""
if [ -r /proc/driver/nvidia/params ]; then
  FLAG="$(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -dc '0-9' | tail -c1)"
  say "  RmProfilingAdminOnly = ${FLAG:-未暴露}"
fi
if command -v ncu >/dev/null 2>&1; then
  say "  ncu       $(ncu --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1)"
else
  say "  ncu       未安装(随 CUDA Toolkit 提供,或单独装 Nsight Compute)"
fi

line
if [ "$FLAG" = "0" ] && [ "$HAS_CAP" = "1" ]; then
  say "结论:计数器应该是开的。装上 ncu 直接实采验证即可。"
elif [ "$IS_VM" = "1" ] && [ "$(id -u)" = "0" ]; then
  say "结论:现在没开,但**这台是虚机且你是 root,可以自己开**。"
  say
  say "  执行(会重启,先确认没有在跑的任务):"
  say "    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\"
  say "      | sudo tee /etc/modprobe.d/nvidia-profiler.conf"
  say "    sudo update-initramfs -u      # Debian/Ubuntu;RHEL 系: sudo dracut -f"
  say "    sudo reboot"
  say
  say "  重启后重跑本脚本,第 3 节应变成 RmProfilingAdminOnly = 0。"
else
  say "结论:没开,而且这台看起来是容器 / 你不是 root —— 自己开不了,只能找平台。"
fi
