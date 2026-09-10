// ============================================================================
// v1 —— 融合成单 kernel(标量访存)
//
// 改了什么:silu 的结果留在寄存器里直接乘 up,不再落显存。
// 字节账 5 -> 3(每输出元素):读 gate(1) + 读 up(1) + 写 out(1)。
// 理论加速 5/3 ≈ 1.67x,顺带省一次 kernel launch。
//
// 这是本梯的主要台阶:被消掉的是一整份与输入等大的中间张量往返
// (T=8192/I=12288 时是 200MB 写 + 200MB 读)。
// ============================================================================
#include "activation.h"

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void v1_kernel(__nv_bfloat16* __restrict__ out,
                          const __nv_bfloat16* __restrict__ gate,
                          const __nv_bfloat16* __restrict__ up,
                          long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(silu(g) * u);
    }
}

void silu_and_mul_v1(__nv_bfloat16* out, const __nv_bfloat16* gate,
                     const __nv_bfloat16* up, long long n, cudaStream_t st) {
    int blocks = (int)min((n + 255) / 256, 4096LL);
    v1_kernel<<<blocks, 256, 0, st>>>(out, gate, up, n);
}
