// ============================================================================
// v1 —— 配对处理:一个线程负责一对 (i, i+D/2),读 2 写 2
//
// 改了什么:把互为输入的两个元素收进同一个线程。
//   ① 读写冲突消失:一对元素的新值都在这个线程的寄存器里算完再一起写回,
//      不再需要 v0 的整份临时缓冲(省掉一次全量 D2D 拷贝 = 一读一写);
//   ② cos/sin 只读一次而不是两次(前后半共用同一组频率,见头文件);
//   ③ 线程数减半,launch 配置更小。
//
// 字节账(每元素):v0 的 5 + 临时缓冲的 2 = 7  ->  v1 的
//   (读 2 + 写 2 + cos 1 + sin 1) / 2 元素 = 3
// 理论加速 7/3 ≈ 2.3x。这是本梯预期最大的一级。
//
// q 与 k 仍是两次 launch,合并留给 v2。
// ============================================================================
#include "rope.h"

__global__ void v1_kernel(__nv_bfloat16* __restrict__ t,
                          const __nv_bfloat16* __restrict__ cosb,
                          const __nv_bfloat16* __restrict__ sinb,
                          int T, int Hh, int D) {
    const int half = D >> 1;
    const long long npair = (long long)T * Hh * half;   // 线程数 = 元素数 / 2
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         g < npair; g += (long long)gridDim.x * blockDim.x) {
        const int i   = (int)(g % half);                // 频率下标 [0, D/2)
        const long long hp = g / half;                  // 第几个 (token, head)
        const int tok = (int)(hp / Hh);
        const long long base = hp * D + i;              // 该对的前半元素位置

        const float c = __bfloat162float(cosb[(long long)tok * D + i]);
        const float s = __bfloat162float(sinb[(long long)tok * D + i]);
        const float x1 = __bfloat162float(t[base]);
        const float x2 = __bfloat162float(t[base + half]);

        // 复数乘法 (x1 + i*x2) * (c + i*s) 的实部与虚部。
        // 两个输出都算完再写,写的顺序无关紧要 —— 依赖已被收进寄存器。
        t[base]        = __float2bfloat16(x1 * c - x2 * s);
        t[base + half] = __float2bfloat16(x2 * c + x1 * s);
    }
}

static void run_one(__nv_bfloat16* t, const __nv_bfloat16* cosb,
                    const __nv_bfloat16* sinb, int T, int Hh, int D,
                    cudaStream_t st) {
    if (Hh == 0) return;
    const long long npair = (long long)T * Hh * (D / 2);
    int blocks = (int)min((npair + 255) / 256, 4096LL);
    v1_kernel<<<blocks, 256, 0, st>>>(t, cosb, sinb, T, Hh, D);
}

void rope_v1(__nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* cosb,
             const __nv_bfloat16* sinb, int T, int HQ, int HK, int D,
             cudaStream_t st) {
    run_one(q, cosb, sinb, T, HQ, D, st);
    run_one(k, cosb, sinb, T, HK, D, st);
}
