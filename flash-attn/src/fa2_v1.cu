#include "fa_common.h"
// ============================================================================
// v1 · K/V tile 进 smem:block = 4 warp 管 4 行 q,64 键一批协同搬进 smem
// 再消费。对 v0 的唯一变量 = K/V 读取层级(L2 → smem)——控制变量设计,
// 收益大小本身就是「L2 已经把广播读扛住多少」的量度。
// 数据布局:Ks/Vs[64][128] half,各 16KB;每 warp 仍持自己行的 q 于寄存器。
// 契约:任意 S(装载与消费均带边界 guard);causal/GQA 同 v0。
// 性能:25.113±0.071 ms = 5.5±0.00 TFLOPS,仅 +11%(EXP-K03（CUDA FA2 forward 简化版版本梯）§6:单 kv-head
// K 才 1MB ≪ 72MB L2,广播读早已被 L2 扛住)。
// 面试点:+11% 不是失败而是测量——它证明本算子此层级不缺带宽,为「v2 必须
// 换指令世代」提供依据;与 gemm v0→v1(+25%,compute-bound)对照同理。
// ============================================================================
constexpr int BN = 64, WARPS = 4;   // BN=64:一批键;Ks+Vs=32KB,占用与批粒度的折中

__global__ void fa2_v1_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    __shared__ half Ks[BN][FA_D], Vs[BN][FA_D];   // 16KB + 16KB
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + warp;    // 1 warp 1 行,block 管连续 4 行
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;

    float qr[4] = {0, 0, 0, 0};
    if (row < S) {                                // 尾 block 的越界行:不读 q,但线程
        const half* q = Q + ((size_t)(b * Hq + h) * S + row) * FA_D;   // 必须活着陪跑 barrier
        #pragma unroll
        for (int j = 0; j < 4; ++j) qr[j] = __half2float(q[lane * 4 + j]);
    }
    const float scale = rsqrtf((float)FA_D);
    float m = -1e30f, l = 0.f, acc[4] = {0, 0, 0, 0};
    const int rmax = blockIdx.x * WARPS + WARPS - 1;      // block 内最大行
    // tile 循环次数必须全 block 一致(循环体内有 __syncthreads,发散到
    // barrier 是 UB)→ causal 上界取 block 最大行的可见键数;行级精确
    // 因果边界由下面的 jend 收
    const int nlimit = causal ? min(rmax + 1, S) : S;

    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();   // 防本轮装载覆盖 Ks/Vs 时,上一轮仍有 warp 在读旧 tile(WAR)
        for (int t = threadIdx.x; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            if (n0 + r < S) {                     // 尾 tile 部分行:防全局越界读
                *(float4*)&Ks[r][c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
                *(float4*)&Vs[r][c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
            }
        }
        __syncthreads();   // 装载按线性 tid 分片、消费按 warp/行——线程集不重合,
                           // barrier 后 smem 写才对消费者可见(跨线程 RAW)
        if (row >= S) continue;   // 放在两个 barrier 之后:保证所有线程经过相同 barrier 序列
        const int jend = min(BN, (causal ? row + 1 : S) - n0);   // 行级因果/尾块双重上界
        for (int j = 0; j < jend; ++j) {
            float s = 0;
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                s += qr[d] * __half2float(Ks[j][lane * 4 + d]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                s += __shfl_xor_sync(0xffffffff, s, off);
            s *= scale;
            // 在线 m/l/α 更新与 v0 逐行同构(见 fa2_v0.cu 注释)
            float mn = fmaxf(m, s), alpha = __expf(m - mn), p = __expf(s - mn);
            l = l * alpha + p;
            #pragma unroll
            for (int d = 0; d < 4; ++d)
                acc[d] = acc[d] * alpha + p * __half2float(Vs[j][lane * 4 + d]);
            m = mn;
        }
    }
    if (row < S) {
        half* o = O + ((size_t)(b * Hq + h) * S + row) * FA_D;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[lane * 4 + j] = __float2half(acc[j] / l);
    }
}
void fa2_v1(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    fa2_v1_kernel<<<dim3((S + WARPS - 1) / WARPS, Hq, B), WARPS * 32>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
