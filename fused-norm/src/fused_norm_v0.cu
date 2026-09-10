// ============================================================================
// v0 —— 未融合基线:两个 kernel,严格对应引擎里现有的两行 PyTorch
//     h = res + o                                  -> add_kernel
//     x = rmsnorm(h, w, eps)                       -> rmsnorm_kernel
// (llm-engine src/model.py 的 attention 块与 MLP 块各出现一次)
//
// 字节账(每元素,bf16=2B):
//   add_kernel   : 读 x + 读 res + 写 res            = 3
//   rmsnorm_kernel: 读 res(求平方和) + 读 res(归一化) + 写 out = 3
//   合计 6 —— 后续四级优化砍的就是这 6。
//
// 这一版故意写得「教科书朴素」:标量访存、smem 树形归约、行内两遍读全局。
// 它不是稻草人 —— 逐条对应 PyTorch eager 真实的执行方式(两次 kernel launch、
// 中间量落显存),所以 v0 与 pytorch_eager 的差距应当很小;若差距很大,
// 说明基线写错了而不是优化有效。这是版本梯第一级的自检条件。
// ============================================================================
#include "fused_norm.h"

// ---- kernel 1:逐元素加法,residual 就地累加 ----------------------------------
// grid-stride 循环而非「一线程一元素」:T*H 可达 1e8,一次 launch 的 block 数
// 有上限;grid-stride 让同一份代码对任意规模都正确,且 block 数可按 SM 数调,
// 避免尾部 wave 的负载不均。
__global__ void v0_add_kernel(__nv_bfloat16* __restrict__ residual,
                              const __nv_bfloat16* __restrict__ x,
                              long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += (long long)gridDim.x * blockDim.x) {
        // 在 fp32 里做加法再舍回 bf16:与 PyTorch 的 bf16 加法一致
        //(TensorIterator 对 bf16 的 opmath_t 就是 float),两边只舍入一次。
        // 若直接用 __hadd,舍入次数相同,但这里显式写出以对齐语义。
        float s = __bfloat162float(residual[i]) + __bfloat162float(x[i]);
        residual[i] = __float2bfloat16(s);
    }
}

// ---- kernel 2:逐行 RMSNorm --------------------------------------------------
// 一个 block 负责一行(一个 token 的 H 维),行内归约在 block 内完成。
// 为什么按行分块:RMSNorm 的归约边界就是行,跨行无依赖;一行一 block 让
// 归约完全留在片上,不需要跨 block 通信(否则要 atomic 或两趟 kernel)。
__global__ void v0_rmsnorm_kernel(__nv_bfloat16* __restrict__ out,
                                  const __nv_bfloat16* __restrict__ in,
                                  const __nv_bfloat16* __restrict__ w,
                                  int H, float eps) {
    extern __shared__ float smem[];
    const __nv_bfloat16* row = in + (long long)blockIdx.x * H;
    __nv_bfloat16* orow      = out + (long long)blockIdx.x * H;

    // ---- 第一遍:读整行,累加平方和 ----
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        acc += v * v;          // fp32 累加,理由见头文件精度约定
    }
    smem[threadIdx.x] = acc;
    __syncthreads();

    // 朴素 smem 树形归约:每轮活跃线程减半,共 log2(blockDim) 轮,
    // 每轮一次 __syncthreads。这正是 cuda-reduce 版本梯 v0/v1 的形态,
    // 后面 v2 会换成 warp shuffle 以砍掉这一串同步。
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    // rsqrtf 而非 1.f/sqrtf:前者是一条硬件指令(MUFU.RSQ),后者是
    // 平方根 + 除法两条长延迟指令。逐行只算一次,收益很小,但没有理由不用。
    const float rstd = rsqrtf(smem[0] / H + eps);
    __syncthreads();   // 保护 smem[0]:下一轮迭代(若有)会覆盖它

    // ---- 第二遍:再读一次整行,归一化后乘权重 ----
    // 这第二次全局读就是 v4 要消掉的那一次。
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        // 与 vLLM layernorm.cu 同款舍入顺序:先把归一化结果舍到 bf16,
        // 再与 bf16 权重相乘。顺序若反(在 fp32 里乘完权重再舍),数值上
        // 更准但与 HF/vLLM 逐位不一致 —— 对拍引擎输出时会看到末位差异。
        __nv_bfloat16 n = __float2bfloat16(__bfloat162float(row[i]) * rstd);
        orow[i] = __hmul(n, w[i]);
    }
}

void fused_add_rmsnorm_v0(__nv_bfloat16* out, __nv_bfloat16* residual,
                          const __nv_bfloat16* x, const __nv_bfloat16* w,
                          int T, int H, float eps, cudaStream_t stream) {
    const long long n = (long long)T * H;
    // 加法 kernel 的 grid:按元素数给,上限 4096 个 block —— 4090 有 128 个 SM,
    // 4096 block 足以填满并让调度器有余量做尾部均衡。
    int add_blocks = (int)min((n + 255) / 256, 4096LL);
    v0_add_kernel<<<add_blocks, 256, 0, stream>>>(residual, x, n);

    // 归一化 kernel 的 blockDim:向上取到 warp 整数倍并夹在 [128,1024]。
    // 太小则行内并行度不足,太大则每线程只摊到不足一个元素、归约树白跑几轮。
    int bs = ((H + 31) / 32) * 32;
    bs = max(128, min(1024, bs));
    v0_rmsnorm_kernel<<<T, bs, bs * sizeof(float), stream>>>(out, residual, w, H, eps);
}
