#!/usr/bin/env bash
# 探针:这台机器能不能采 GPU 性能计数器?
# 换机器、换实例、换容器之后先跑它,10 秒出结论,不要等跑完整套采集才发现采不了。
#
#   bash scripts/probe_ncu_permission.sh
#
# 退出码: 0=可采  1=工具链缺失  2=权限被拒(ERR_NVGPUCTRPERM)  3=其它失败

set -uo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 环境"
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "  没有 nvidia-smi"; exit 1; fi
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/  /'

for t in ncu nvcc; do
  if ! command -v $t >/dev/null 2>&1; then echo "  缺 $t,装 CUDA Toolkit / Nsight Compute"; exit 1; fi
done
echo "  ncu   $(ncu --version 2>/dev/null | grep -oE '[0-9]{4}\.[0-9.]+' | head -1)"
echo "  nvcc  $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1)"

# 权限相关的两个直接信号(有就打印,没有不算错:不同平台暴露程度不同)
if [ -r /proc/driver/nvidia/params ]; then
  echo "  $(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params || echo 'RmProfilingAdminOnly: (未暴露)')"
  echo "    ↑ 1 = 只有管理员可读计数器;容器里通常意味着采不了"
fi
if [ -r /proc/self/status ]; then
  cap="$(awk '/^CapEff/{print $2}' /proc/self/status)"
  # CAP_SYS_ADMIN = bit 21
  if [ -n "$cap" ] && [ $(( 0x$cap >> 21 & 1 )) -eq 1 ]; then
    echo "  CAP_SYS_ADMIN: 有"
  else
    echo "  CAP_SYS_ADMIN: 无(容器默认)"
  fi
fi

echo "== 实采烟测"
cat > "$TMP/probe.cu" <<'EOF'
__global__ void probe_kernel(float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = i * 1.0f;
}
int main() {
    float* d; cudaMalloc(&d, 1 << 20);
    probe_kernel<<<256, 256>>>(d, 1 << 18);
    cudaDeviceSynchronize();
    return 0;
}
EOF

# 不指定 -arch,让 nvcc 用默认目标,换卡也能编
if ! nvcc -o "$TMP/probe" "$TMP/probe.cu" 2>"$TMP/nvcc.err"; then
  echo "  编译失败:"; sed 's/^/    /' "$TMP/nvcc.err"; exit 1
fi

ncu --section SpeedOfLight -f -o "$TMP/probe_rep" "$TMP/probe" >"$TMP/out" 2>&1
status=$?

if grep -q ERR_NVGPUCTRPERM "$TMP/out"; then
  echo "  不可采:ERR_NVGPUCTRPERM"
  echo
  echo "  这台机器没有 GPU 性能计数器权限。容器内无解,需要其一:"
  echo "    - 宿主 /etc/modprobe.d/ 设 options nvidia NVreg_RestrictProfilingToAdminUsers=0 并重载模块"
  echo "    - 或以 --cap-add=SYS_ADMIN 起容器"
  echo "    - 或换一台独占整机/裸金属的实例"
  echo "  共享宿主上前两条通常不会被批准(改的是全局驱动参数,影响同宿主所有租户)。"
  exit 2
fi

if [ "$status" -ne 0 ] || [ ! -f "$TMP/probe_rep.ncu-rep" ]; then
  echo "  失败(非权限问题):"; sed 's/^/    /' "$TMP/out" | head -20; exit 3
fi

val="$(ncu -i "$TMP/probe_rep.ncu-rep" --page details --csv 2>/dev/null \
      | awk -F'","' '$13=="Duration"{print $15; exit}')"
echo "  可采。烟测 kernel Duration = ${val:-?} (报告已生成并成功回读)"
echo
echo "  下一步: bash scripts/run_ncu_all.sh"
exit 0
