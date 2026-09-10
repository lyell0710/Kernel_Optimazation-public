// ============================================================================
// v0 —— 未融合基线:两个 kernel,对应 eager 的 `F.silu(g) * u`
//     kernel1: tmp = silu(gate)      -> 物化一份与 gate 等大的中间张量
//     kernel2: out = tmp * up
//
// 字节账(每输出元素,bf16=2B):
//   kernel1: 读 gate(1) + 写 tmp(1)            = 2
//   kernel2: 读 tmp(1) + 读 up(1) + 写 out(1)  = 3
//   合计 5 —— 融合后的下界是 3,所以 v0->v1 的理论上限是 5/3 ≈ 1.67x。
//
// 中间张量 tmp 是这里唯一被浪费的东西:它被写出去、立刻又读回来,
// 在 T=8192/I=12288 的形状下就是 200MB 的无谓往返。「融合免搬运」
// 这句话在这个算子上就是字面意思。
// ============================================================================
#include "activation.h"

__device__ __forceinline__ float silu(float x) {
    // sigmoid 在 fp32 里算:与 PyTorch 对 bf16 的处理一致(TensorIterator 的
    // opmath_t = float)。直接在 bf16 上算 exp,|x| 稍大就会明显失真 ——
    // bf16 只有 8 位尾数,exp 的输入误差会被放大成输出的相对误差。
    return x / (1.0f + __expf(-x));
}

__global__ void v0_silu_kernel(__nv_bfloat16* __restrict__ tmp,
                               const __nv_bfloat16* __restrict__ gate,
                               long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x)
        tmp[i] = __float2bfloat16(silu(__bfloat162float(gate[i])));
}

__global__ void v0_mul_kernel(__nv_bfloat16* __restrict__ out,
                              const __nv_bfloat16* __restrict__ tmp,
                              const __nv_bfloat16* __restrict__ up,
                              long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(__bfloat162float(tmp[i]) * __bfloat162float(up[i]));
}

// 中间缓冲静态持有,避免每次调用 cudaMalloc(那会引入隐式同步,把 bench
// 计到的时间污染成分配器时间)。容量不足才重分配。
static __nv_bfloat16* g_tmp = nullptr;
static size_t g_tmp_bytes = 0;

void silu_and_mul_v0(__nv_bfloat16* out, const __nv_bfloat16* gate,
                     const __nv_bfloat16* up, long long n, cudaStream_t st) {
    const size_t bytes = (size_t)n * sizeof(__nv_bfloat16);
    if (bytes > g_tmp_bytes) {
        if (g_tmp) cudaFree(g_tmp);
        cudaMalloc(&g_tmp, bytes);
        g_tmp_bytes = bytes;
    }
    int blocks = (int)min((n + 255) / 256, 4096LL);
    v0_silu_kernel<<<blocks, 256, 0, st>>>(g_tmp, gate, n);
    v0_mul_kernel<<<blocks, 256, 0, st>>>(out, g_tmp, up, n);
}
