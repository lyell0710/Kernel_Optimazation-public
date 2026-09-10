// 标准库对照:CUB DeviceReduce::Sum。
//
// 为什么 CUB 才是 reduce 的标准库基准:cuBLAS 是 BLAS 库,其中最接近的
// cublasSasum 算的是 Σ|x_i|(绝对值和),与本项目的 Σx_i 语义不同,只能算
// "同带宽量级的邻近算子"参照;CUB(CUDA Core Compute Libraries 的一部分,
// 随 toolkit 分发)的 DeviceReduce::Sum 才是同一算子的官方实现,由 NVIDIA
// 按架构调优(tile 尺寸/展开度/两阶段规约策略随 SM 版本切换)。
//
// 计时口径:temp_storage 一次性分配、按 n 复用,不落在计时区内 ——
// 否则测的是 cudaMalloc 而不是规约本身。
//
// 注:此处原注有一句"与被测 kernel 的显存分配同样在计时外",**当时并不属实**:
// v1-v7 的 cudaMalloc/cudaFree 都在计时区内,v6/v7 还多一次 cudaGetDeviceProperties。
// 这个不对称已于 EXP-K09 §5.17 修复,现在这句话才成立。
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "reduce_common.h"

namespace {
void*  g_temp = nullptr;      // CUB 需要的临时工作区:大小由 CUB 自己算(第一次
size_t g_temp_bytes = 0;      // 传 nullptr 让它回填 bytes,是 CUB 的标准两段式调用
int    g_n = 0;
} // namespace

void reduce_cub(const float* data, float* output, int n) {
    if (n <= 0) { float z = 0.0f; cudaMemcpy(output, &z, sizeof(float), cudaMemcpyHostToDevice); return; }
    if (n != g_n || g_temp == nullptr) {          // 工作区一次性分配,与被测 kernel
        if (g_temp) { cudaFree(g_temp); g_temp = nullptr; }   // 的显存分配同为计时外(K09 §5.17 后)
        g_temp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, g_temp_bytes, data, output, n);  // 两段式:先问大小
        cudaMalloc(&g_temp, g_temp_bytes);
        g_n = n;
    }
    // 结果直接落在调用方给的 device 指针上(与本项目其余版本同口径,无额外 D2H)
    cub::DeviceReduce::Sum(g_temp, g_temp_bytes, data, output, n);
}
