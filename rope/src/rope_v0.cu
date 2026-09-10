// ============================================================================
// v0 —— 朴素基线:一线程一元素,q 与 k 分两次 launch
//
// 每个线程独立负责输出张量里的一个元素,自己去读「配对元素」与 cos/sin。
// 这是把数学公式一对一翻译成代码的最直接形态,也是本梯的字节账起点。
//
// 字节账(每元素,bf16=2B):
//   读自己(1) + 读配对元素(1) + 读 cos(1) + 读 sin(1) + 写回(1) = 5
// 其中「读配对元素」与「读 cos/sin」都是纯粹的重复劳动:i 与 i+D/2 两个
// 线程读的是同一对数据、同一个 cos/sin —— v1 把它们合并到一个线程里做掉。
//
// 注意 v0 不等于 pytorch_eager 臂:eager 还要额外物化 rotate_half 的中间
// 张量、再做两次乘法一次加法,是 4 个 kernel 4 份临时显存。v0 已经是
// 「手写但不优化」,所以 v0 应当明显快于 eager;若不是,说明 v0 写错了。
// ============================================================================
#include "rope.h"

// 【一线程一元素 + 就地更新 = 读写冲突】
// 前半的线程要读 t[g+half],后半的线程要读 t[g-half],互为对方的输入;
// 而两者又都要写自己。同一 kernel 内不同 block 之间没有执行顺序保证,
// 后执行的线程会读到已被覆盖的值。这类 bug 不会崩、不会 NaN,只是结果
// 悄悄错一半,且随 block 调度顺序随机变化 —— 属于最难查的一类。
//
// v0 的保守解法:先把整份输入拷到临时缓冲,kernel 从缓冲读、往原地写。
// 代价是多一次全量读+写。v1 让同一个线程同时持有一对元素后,依赖关系
// 被收进线程内部,冲突自然消失,临时缓冲也就不需要了 —— 这正是 v0->v1
// 的收益来源,不只是「少读一次 cos」。
__global__ void v0_kernel_safe(__nv_bfloat16* __restrict__ dst,
                               const __nv_bfloat16* __restrict__ src,
                               const __nv_bfloat16* __restrict__ cosb,
                               const __nv_bfloat16* __restrict__ sinb,
                               int T, int Hh, int D) {
    const long long n = (long long)T * Hh * D;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < n; g += (long long)gridDim.x * blockDim.x) {
        const int d = (int)(g % D);
        const int tok = (int)(g / ((long long)Hh * D));
        const int half = D >> 1;
        const int i = d % half;
        const float c = __bfloat162float(cosb[(long long)tok * D + i]);
        const float s = __bfloat162float(sinb[(long long)tok * D + i]);
        const float self = __bfloat162float(src[g]);
        const float pair = __bfloat162float(d < half ? src[g + half] : src[g - half]);
        dst[g] = __float2bfloat16(d < half ? self * c - pair * s
                                           : self * c + pair * s);
    }
}

// 临时缓冲用静态 device 指针缓存,避免每次调用都 cudaMalloc(那会引入
// 同步点、把 bench 计到的时间污染成分配器时间)。容量不足时才重分配。
static __nv_bfloat16* g_tmp = nullptr;
static size_t g_tmp_bytes = 0;

static void ensure_tmp(size_t bytes) {
    if (bytes > g_tmp_bytes) {
        if (g_tmp) cudaFree(g_tmp);
        cudaMalloc(&g_tmp, bytes);
        g_tmp_bytes = bytes;
    }
}

static void run_one(__nv_bfloat16* t, const __nv_bfloat16* cosb,
                    const __nv_bfloat16* sinb, int T, int Hh, int D,
                    cudaStream_t st) {
    if (Hh == 0) return;
    const size_t bytes = (size_t)T * Hh * D * sizeof(__nv_bfloat16);
    ensure_tmp(bytes);
    cudaMemcpyAsync(g_tmp, t, bytes, cudaMemcpyDeviceToDevice, st);
    const long long n = (long long)T * Hh * D;
    int blocks = (int)min((n + 255) / 256, 4096LL);
    v0_kernel_safe<<<blocks, 256, 0, st>>>(t, g_tmp, cosb, sinb, T, Hh, D);
}

void rope_v0(__nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* cosb,
             const __nv_bfloat16* sinb, int T, int HQ, int HK, int D,
             cudaStream_t st) {
    run_one(q, cosb, sinb, T, HQ, D, st);   // 两次独立 launch:
    run_one(k, cosb, sinb, T, HK, D, st);   // decode 时这两次 launch 就是主要开销
}
