#!/usr/bin/env bash
# ============================================================
# 这台机器能不能用 Nsight Compute 采 GPU 性能计数器?
# 自包含,不装任何东西,不改任何配置,不影响别人正在跑的任务。
# 用法:  bash ncu_check.sh
# ============================================================
# 说明:
#  - 全程只读探测 + 一个几毫秒的玩具 kernel,不占显存不占算力。
#  - 显式带 --clock-control none:Nsight Compute 默认会把 GPU 时钟锁到 base clock,
#    在共享机器上会拖慢同卡上别人的任务。加了这个就不动时钟。
#  - 不写任何系统目录,临时文件在 /tmp 且退出即删。

set -uo pipefail
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
line() { printf '%s\n' "------------------------------------------------------------"; }

echo "=== 1. 硬件 ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /'
else
  echo "  没有 nvidia-smi —— 这台机器上看不到 NVIDIA 驱动"; exit 1
fi

echo
echo "=== 2. 工具链 ==="
if command -v ncu >/dev/null 2>&1; then
  echo "  ncu       $(ncu --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1)  ($(command -v ncu))"
else
  echo "  ncu       未安装"
  echo "            (它随 CUDA Toolkit 提供,通常在 /usr/local/cuda/bin/ncu"
  echo "             或 /opt/nvidia/nsight-compute/*/ncu;先 find 一下再说没有)"
fi
command -v nsys >/dev/null 2>&1 && echo "  nsys      $(nsys --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1)"
command -v nvcc >/dev/null 2>&1 && echo "  nvcc      $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"

echo
echo "=== 3. 运行环境 ==="
if [ -f /.dockerenv ] || grep -qaE 'docker|kubepods|containerd' /proc/1/cgroup 2>/dev/null \
   || [ "$(cat /proc/1/comm 2>/dev/null)" != "systemd" ]; then
  echo "  形态      容器 / Pod (init=$(cat /proc/1/comm 2>/dev/null))"
  echo "            容器里改不了驱动参数,权限得由宿主给"
else
  echo "  形态      裸机 / 虚拟机 (init=systemd) —— 有 root 的话自己就能开"
fi
echo "  用户      $(id -un) (uid=$(id -u))"
if [ -r /proc/driver/nvidia/params ]; then
  v="$(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | tr -d ' ')"
  echo "  ${v:-RmProfilingAdminOnly:(未暴露)}"
  echo "            1 = 只有管理员能读计数器 / 0 = 所有用户都能读"
fi
if [ -r /proc/self/status ]; then
  cap="$(awk '/^CapEff/{print $2}' /proc/self/status)"
  if [ -n "$cap" ] && [ $(( 0x$cap >> 21 & 1 )) -eq 1 ]; then
    echo "  CAP_SYS_ADMIN  有"
  else
    echo "  CAP_SYS_ADMIN  无"
  fi
fi

echo
echo "=== 4. 实采烟测(决定性) ==="
command -v ncu >/dev/null 2>&1 || { echo "  跳过:没有 ncu"; line; echo "结论:装了 ncu 再测"; exit 1; }

RUN=()          # 用数组:被测命令含空格与分号,字符串+无引号展开会被切碎
if command -v nvcc >/dev/null 2>&1; then
  cat > "$TMP/p.cu" <<'CU'
__global__ void probe_kernel(float* o, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = i * 1.0f;
}
int main(){ float* d; cudaMalloc(&d, 1<<20);
    probe_kernel<<<256,256>>>(d, 1<<18); cudaDeviceSynchronize(); return 0; }
CU
  if nvcc -o "$TMP/p" "$TMP/p.cu" 2>"$TMP/nvcc.err"; then
    RUN=("$TMP/p"); echo "  被测程序:自编译的玩具 kernel"
  else
    echo "  nvcc 编译失败,改用 PyTorch 路径"
  fi
fi
if [ ${#RUN[@]} -eq 0 ]; then
  for PY in python3 python; do
    if command -v $PY >/dev/null 2>&1 && $PY -c "import torch;assert torch.cuda.is_available()" 2>/dev/null; then
      RUN=("$PY" -c "import torch
a = torch.randn(512, 512, device='cuda')
(a @ a)
torch.cuda.synchronize()")
      echo "  被测程序:PyTorch 512x512 matmul"; break
    fi
  done
fi
[ ${#RUN[@]} -gt 0 ] || { echo "  跳过:既没有 nvcc 也没有可用的 PyTorch"; line
  echo "结论:无法实测。看第 3 节两个信号——RmProfilingAdminOnly=0 或有 CAP_SYS_ADMIN 通常就能采"; exit 1; }

# --clock-control none:不锁时钟,不影响同卡上别人的任务
ncu --clock-control none --section SpeedOfLight -f -o "$TMP/rep" "${RUN[@]}" >"$TMP/out" 2>&1

line
if grep -q ERR_NVGPUCTRPERM "$TMP/out"; then
  echo "结论:不支持 —— ERR_NVGPUCTRPERM(没有性能计数器权限)"
  echo
  echo "如果这台是你能管的裸机/虚拟机,开启办法(需 root,改完要重载驱动或重启):"
  echo "  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \\"
  echo "    | sudo tee /etc/modprobe.d/nvidia-profiler.conf"
  echo "  sudo update-initramfs -u   # Debian/Ubuntu;RHEL 系用 dracut -f"
  echo "  # 然后重启,或卸载并重新加载 nvidia 模块(会中断该机所有 GPU 任务)"
  echo
  echo "如果这台是容器/集群节点:容器内无解,得让管理员在宿主上设同样的参数,"
  echo "或者用 --cap-add=SYS_ADMIN 起容器。注意这两个都是全局的,会影响同机所有人。"
  exit 2
fi
if [ ! -f "$TMP/rep.ncu-rep" ]; then
  echo "结论:失败,但不是权限问题。原始输出:"
  sed 's/^/  /' "$TMP/out" | head -25
  exit 3
fi
dur="$(ncu -i "$TMP/rep.ncu-rep" --page details --csv 2>/dev/null \
       | awk -F'","' '$13=="Duration"{print $15" "$14; exit}')"
echo "结论:支持 —— 成功采到计数器并回读(烟测 kernel Duration = ${dur:-?})"
echo
echo "请把上面第 1、2 节的内容发回来(GPU 型号 / 显存 / 驱动 / ncu 版本)。"
exit 0
