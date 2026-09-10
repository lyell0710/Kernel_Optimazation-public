// ============================================================================
// v3 —— 打包布局(vLLM 风格):gate 与 up 是同一张量 [T, 2I] 的前后两半
//
// 改了什么:只动输入布局,算法与访存宽度沿用 v2,字节账仍是 3/输出元素。
//
// 【这一级测的是布局本身,预测是「算子层零收益」】
// 打包与分离在 HBM 层面搬的字节数完全一样:都是读 2 份 I 宽、写 1 份 I 宽。
// 唯一差别是两路读的地址相距 I*2 字节(打包)还是分属两块显存(分离),
// 对 DRAM 的 bank/row 局部性影响很小。所以本级预期 ≈ v2(±3%)。
//
// 那为什么 vLLM 要用打包?因为收益不在这个算子里,在它上游:
// 打包意味着 gate_proj 与 up_proj 可以合并成一次 gate_up_proj GEMM ——
// 少一次 kernel launch、GEMM 的 N 维翻倍从而 tile 利用率更高、权重也
// 只需一次读取。这些全部发生在 GEMM 那一侧,算子级 bench 看不到,
// 必须接进引擎才量得到。把「收益不在被改的地方」这件事讲清楚,
// 比报一个持平的数字更重要。
// ============================================================================
#include "activation.h"

struct alignas(16) BF16x8 {
    // `alignas(16)` 只保证地址对齐,**不强制向量化访存**:nvcc 按成员类型
    // (__nv_bfloat162,4 B)逐个生成访存,编出来是 4 条 32 位 LDG 而非一条 LDG.E.128。
    // union 给出 float4 视图 + 显式拷贝语义,让整体赋值走 raw 这条 128 位通路,
    // 调用点无须改写。根因、验证与收益见 records/EXP-K08。
    union { float4 raw; __nv_bfloat162 h[4]; };
    __device__ __forceinline__ BF16x8() {}
    __device__ __forceinline__ BF16x8(const BF16x8& o) { raw = o.raw; }
    __device__ __forceinline__ BF16x8& operator=(const BF16x8& o) { raw = o.raw; return *this; }
};

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void v3_kernel(__nv_bfloat16* __restrict__ out,
                          const __nv_bfloat16* __restrict__ gate_up,
                          int Iv) {          // Iv = I/8,每行的向量组数
    const int row = blockIdx.y;
    // 同一行内:gate 在 [0, I),up 在 [I, 2I)。两路读的地址差 I 个元素,
    // 都落在同一行的连续区间,DRAM row 局部性略好于分属两块显存。
    const BF16x8* g = reinterpret_cast<const BF16x8*>(gate_up) + (long long)row * 2 * Iv;
    const BF16x8* u = g + Iv;
    BF16x8* o = reinterpret_cast<BF16x8*>(out) + (long long)row * Iv;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Iv;
         i += gridDim.x * blockDim.x) {
        BF16x8 gv = g[i], uv = u[i], ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 a = __bfloat1622float2(gv.h[j]);
            float2 b = __bfloat1622float2(uv.h[j]);
            ov.h[j] = __float22bfloat162_rn(
                make_float2(silu(a.x) * b.x, silu(a.y) * b.y));
        }
        o[i] = ov;
    }
}

void silu_and_mul_v3(__nv_bfloat16* out, const __nv_bfloat16* gate_up,
                     int T, int I, cudaStream_t st) {
    const int Iv = I / 8;
    // 二维 grid:y 维一行一 block 组,x 维覆盖行内。这样每个 block 只碰
    // 一行,gate 与 up 两路读的偏移在 block 内是常量,寻址更简单。
    int bx = min((Iv + 255) / 256, 64);
    dim3 grid(max(1, bx), T);
    v3_kernel<<<grid, 256, 0, st>>>(out, gate_up, Iv);
}
