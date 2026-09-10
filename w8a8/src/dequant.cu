// ============================================================================
// 反量化 epilogue —— y[t,o] = acc[t,o] * x_scale[t] * w_scale[o]
//
// 字节账(每输出元素):读 acc(4 B int32)+ 写 y(2 B bf16)= 6 B。
// 两路 scale 分别只有 T 个与 O 个 float,跨维度复用,不计入。
//
// 【为什么它必须单独存在,又为什么它不该单独存在】
// 单独存在:cuBLASLt 的 INT8 matmul 输出 int32 累加值,scale 只有调用方知道,
//   所以必须有一步把它变回 bf16。
// 不该单独存在:它是纯访存的一遍读写,理想情况下应当融进 GEMM 的 epilogue,
//   由 GEMM kernel 在结果还在寄存器/共享内存里时就地完成 —— 那样这 6 B/元素
//   完全消失。用 cuBLASLt 的现成接口拿不到这个位置,这正是生产框架
//   (vLLM 的 CUTLASS W8A8)要自己写 GEMM 的主要理由之一。
//   本子项目量化了这一步的代价,让「为什么要自己写 GEMM」有数字支撑。
// ============================================================================
#include "w8a8.h"

__global__ void dequant_v0_kernel(__nv_bfloat16* __restrict__ y,
                                  const int32_t* __restrict__ acc,
                                  const float* __restrict__ xs,
                                  const float* __restrict__ ws,
                                  int T, int O) {
    const long long n = (long long)T * O;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        const int o = (int)(i % O);
        const int t = (int)(i / O);
        // 两个 scale 相乘再乘累加值:先乘 scale 还是先转 float 不影响结果,
        // 但 acc 是 int32,最大可到 127*127*H ≈ 5e8(H=4096),
        // 单精度尾数 24 位只能精确表示到 1.6e7 —— 超出后 (float)acc 本身就有
        // 舍入误差。这是 int8 GEMM 的固有精度上限,不是本 kernel 的问题;
        // 若要更准需要 fp64 或分段累加,代价远大于收益。
        y[i] = __float2bfloat16((float)acc[i] * xs[t] * ws[o]);
    }
}

void dequant_v0(__nv_bfloat16* y, const int32_t* acc, const float* xs,
                const float* ws, int T, int O, cudaStream_t st) {
    const long long n = (long long)T * O;
    int blocks = (int)min((n + 255) / 256, 4096LL);
    dequant_v0_kernel<<<blocks, 256, 0, st>>>(y, acc, xs, ws, T, O);
}

// ---- v1:一行一 block,x_scale 只读一次;列方向向量化 ------------------------
struct alignas(16) I32x4 { int32_t v[4]; };
struct alignas(8)  BF16x4 { __nv_bfloat162 h[2]; };

__global__ void dequant_v1_kernel(__nv_bfloat16* __restrict__ y,
                                  const int32_t* __restrict__ acc,
                                  const float* __restrict__ xs,
                                  const float* __restrict__ ws,
                                  int O) {
    const int t = blockIdx.y;
    // x_scale 每行只读一次而不是每元素读一次:v0 里 xs[t] 虽然会命中 L1,
    // 但仍占一条 LSU 指令;提到循环外后,行内所有元素共用一个寄存器。
    const float sx = xs[t];
    const int OV = O >> 2;
    const I32x4* arow = reinterpret_cast<const I32x4*>(acc) + (long long)t * OV;
    BF16x4* yrow = reinterpret_cast<BF16x4*>(y) + (long long)t * OV;
    const float4* wsv = reinterpret_cast<const float4*>(ws);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < OV;
         i += gridDim.x * blockDim.x) {
        I32x4 a = arow[i];
        float4 w = wsv[i];       // w_scale 按输出通道,跨 token 复用,命中率极高
        BF16x4 o;
        o.h[0] = __float22bfloat162_rn(make_float2((float)a.v[0] * sx * w.x,
                                                   (float)a.v[1] * sx * w.y));
        o.h[1] = __float22bfloat162_rn(make_float2((float)a.v[2] * sx * w.z,
                                                   (float)a.v[3] * sx * w.w));
        yrow[i] = o;
    }
}

void dequant_v1(__nv_bfloat16* y, const int32_t* acc, const float* xs,
                const float* ws, int T, int O, cudaStream_t st) {
    const int OV = O / 4;
    int bx = min((OV + 255) / 256, 64);
    dim3 grid(max(1, bx), T);
    dequant_v1_kernel<<<grid, 256, 0, st>>>(y, acc, xs, ws, O);
}
