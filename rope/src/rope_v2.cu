// ============================================================================
// v2 —— q 与 k 合并进一次 launch
//
// 改了什么:只动 launch 结构,访存模式与 v1 完全一致(字节账仍是 3)。
// 用一维 grid 覆盖 q 与 k 的全部 (token, head, 频率) 组合,靠下标区间
// 判断当前线程该处理哪张张量。
//
// 为什么值得单列一级:
//   prefill 时 T 很大,kernel 本身耗时远大于 launch,这一级应当接近零收益;
//   decode 时 T=1,一次 launch(~3-5us)可能比 kernel 本身还贵,省掉一次
//   就是接近 2x。同一个改动在两个区间的收益差好几倍 —— 这正是
//   triton-kernels#EXP-T03（三件套移植 + torch 绑定）「小 kernel 的瓶颈在主机侧而非设备侧」在本算子
//   上的复现,也是「优化必须绑定工作区间来谈」的又一个实例。
//
// 代价:kernel 里多了一次分支。分支在 warp 内是发散的吗?不是 ——
// q 的线程与 k 的线程按全局下标连续切分,只有跨越边界的那一个 warp 会
// 发散,占比 1/(总 warp 数),可忽略。
// ============================================================================
#include "rope.h"

__global__ void v2_kernel(__nv_bfloat16* __restrict__ q,
                          __nv_bfloat16* __restrict__ k,
                          const __nv_bfloat16* __restrict__ cosb,
                          const __nv_bfloat16* __restrict__ sinb,
                          int T, int HQ, int HK, int D,
                          long long nq_pair, long long total_pair) {
    const int half = D >> 1;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < total_pair; g += (long long)gridDim.x * blockDim.x) {
        // 前 nq_pair 个线程处理 q,其余处理 k
        __nv_bfloat16* t;
        long long idx;
        int Hh;
        if (g < nq_pair) { t = q; idx = g;            Hh = HQ; }
        else             { t = k; idx = g - nq_pair;  Hh = HK; }

        const int i = (int)(idx % half);
        const long long hp = idx / half;
        const int tok = (int)(hp / Hh);
        const long long base = hp * D + i;

        const float c = __bfloat162float(cosb[(long long)tok * D + i]);
        const float s = __bfloat162float(sinb[(long long)tok * D + i]);
        const float x1 = __bfloat162float(t[base]);
        const float x2 = __bfloat162float(t[base + half]);
        t[base]        = __float2bfloat16(x1 * c - x2 * s);
        t[base + half] = __float2bfloat16(x2 * c + x1 * s);
    }
}

void rope_v2(__nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* cosb,
             const __nv_bfloat16* sinb, int T, int HQ, int HK, int D,
             cudaStream_t st) {
    const int half = D / 2;
    const long long nq = (long long)T * HQ * half;
    const long long nk = (long long)T * HK * half;
    int blocks = (int)min((nq + nk + 255) / 256, 4096LL);
    v2_kernel<<<blocks, 256, 0, st>>>(q, k, cosb, sinb, T, HQ, HK, D, nq, nq + nk);
}
