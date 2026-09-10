#include "gemm_common.h"
// ============================================================================
// v1 · shared memory 32x32 tile。
// 问题:v0 输入元素零复用,带宽被重复读浪费。
// 算法:K 维按 32 分块,A/B 子块协同装进 smem 后块内标量 FMA;每元素装载
// 1 次供 32 线程复用 → 全局访存量 /32。仍是 CUDA core——本级刻意只动访存
// 不动指令,把「访存问题」与「算力问题」拆成两个独立变量。
// 数据布局:
//   As[32][32] ← A 的 32 行 x 当前 K 块;Bs[32][32] ← 当前 K 块 x B 的 32 列;
//   acc += As[ty][k] * Bs[k][tx](k=0..31);smem 合计 4KB。
// 契约:M,N,K % 32 == 0(grid 不取整、装载无 guard;bench 4096 满足)。
// 性能:21.114±0.047 ms = 6.5±0.00 TFLOPS,仅 +25%(EXP-K02（CUDA Tensor Core GEMM 版本梯）H1:fp16 走
// CUDA core 的算力上限太低,tiling 省下的带宽换不来算力)。
// 面试点:同样的 smem tile 化在 memory-bound 的 gemv/reduce 是主菜,在
// compute-bound 的 GEMM 只是 +25% 的坡——先判 bound 类型再选优化手段。
// ============================================================================
constexpr int T = 32;   // 32x32=1024 线程恰为 block 上限;更大 tile 无法一线程一输出
__global__ void gemm_v1_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    __shared__ half As[T][T], Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T) {
        // 装载:两条读 tx 连续 → 全局访存均按行合并
        As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();   // 防 tile 未装满即有线程开读:内积要读整行/列,
                           // 大部分由别的线程装载(跨线程 RAW)
        #pragma unroll
        for (int k = 0; k < T; ++k)
            // As[ty][k]:warp 内同址 → smem 广播;Bs[k][tx]:tx 连续,行内顺序访问
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();   // 防快线程进入下一 k0 覆盖 As/Bs 时,慢线程仍在读旧 tile(WAR)
    }
    C[row * N + col] = __float2half(acc);
}
void gemm_v1(const half* A, const half* B, half* C, int M, int N, int K) {
    dim3 blk(T, T), grd(N / T, M / T);   // 不向上取整:依赖 M,N % 32 == 0 前置条件
    gemm_v1_kernel<<<grd, blk>>>(A, B, C, M, N, K);
}
