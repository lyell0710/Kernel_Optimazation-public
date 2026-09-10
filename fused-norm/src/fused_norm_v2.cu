// ============================================================================
// v2 —— 归约方式改造:smem 树形归约 -> warp shuffle 两级归约
//
// 改了什么:只动归约,访存模式与 v1 完全一致(字节账仍是 5)。
//   v1: log2(blockDim) 轮 smem 读改写 + 同样多次 __syncthreads
//   v2: 5 轮 __shfl_xor(纯寄存器) + 1 次 smem 落地 + 1-2 次 __syncthreads
// 对 blockDim=512,同步次数从 9 次降到 2 次,smem 事务从 ~1000 次降到 ~32 次。
//
// 【这一级是版本梯里故意留的对照】:如果本算子确实是 memory-bound,那么
// 砍掉这些片上开销应当几乎不涨速(个位数百分比)。它验证的是「优化手段
// 必须与瓶颈类型匹配」——同一手法在 compute-bound 的 reduce 上是台阶
//(cuda-reduce v5->v6),在这里只应是坡。跑出来若涨了 20%,说明我们对
// 瓶颈的判断错了,要回头查是不是同步把访存流水打断了。
//
// 预期:v1 -> v2 约 1.0x。写下这条是为了让实验可证伪(EXP 八节 §1)。
// ============================================================================
#include "fused_norm.h"

__global__ void v2_kernel(__nv_bfloat16* __restrict__ out,
                          __nv_bfloat16* __restrict__ residual,
                          const __nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ w,
                          int H, float eps) {
    // 固定 32 个 float:两级归约只需要「每 warp 一格」,与 blockDim 无关。
    // v1 的 smem 是 blockDim 个 float(512 线程 = 2KB),这里恒为 128B ——
    // smem 占用直接影响每 SM 能驻留几个 block,虽然本算子不受 occupancy
    // 限制,但省下来没有代价。
    __shared__ float smem[32];

    const long long off = (long long)blockIdx.x * H;
    __nv_bfloat16* res = residual + off;
    const __nv_bfloat16* xr = x + off;
    __nv_bfloat16* orow = out + off;

    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float s = __bfloat162float(res[i]) + __bfloat162float(xr[i]);
        res[i] = __float2bfloat16(s);
        acc += s * s;
    }

    // block_reduce_sum 内部已包含必要的 __syncthreads 与广播(见头文件)。
    // 关键前提:调用点在所有线程都到达的位置,不在任何 if 分支内 ——
    // 上面的 for 循环对 i >= H 的线程自然跳过循环体,但线程本身仍会走到这里,
    // 整个 warp 都活着,__shfl_xor_sync 的 0xffffffff mask 才成立。
    const float sumsq = block_reduce_sum(acc, smem);
    const float rstd = rsqrtf(sumsq / H + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        __nv_bfloat16 n = __float2bfloat16(__bfloat162float(res[i]) * rstd);
        orow[i] = __hmul(n, w[i]);
    }
}

void fused_add_rmsnorm_v2(__nv_bfloat16* out, __nv_bfloat16* residual,
                          const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int T, int H, float eps, cudaStream_t stream) {
    int bs = ((H + 31) / 32) * 32;
    bs = max(128, min(1024, bs));
    v2_kernel<<<T, bs, 0, stream>>>(out, residual, x, w, H, eps);
}
