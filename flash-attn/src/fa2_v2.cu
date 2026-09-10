#include <mma.h>
#include "fa_common.h"
// ============================================================================
// v2 · Tensor Core 版:QK^T 与 P·V 走 wmma,softmax/在线统计走 smem 标量段。
// 问题:v0/v1 的 CUDA core 路线 5.5 TFLOPS 见顶,两个矩阵乘上 Tensor Core。
// 为什么 S/P 要 smem 往返:wmma accumulator fragment 的 lane→元素映射
// 未定义(编译器私有),行级 max/exp/α 无法在 fragment 上做——生产核
// (FA2/CUTLASS)用 mma PTX 拿到确定布局后在寄存器里做,这是 wmma→mma
// 的本质分界(与 gemm/ 记录 EXP-K02（CUDA Tensor Core GEMM 版本梯）§7 的 v5 backlog 同源)。
// 算法:每 block 管 64 行 q,流式吃 64 键 tile;每 tile 走 5 段相位链
// (段间恰 5 次 __syncthreads,每处的竞态对象见行内注释):
//   ① K/V 装载 → ② QK^T(wmma)→ S 落 smem → ③ 标量段行级在线 softmax
//   (S→P,更新 m/l/α)→ ④ O ×α 重缩放 → ⑤ P·V(wmma)累加进 O。
// 约束:D=128,S % 64 == 0(bench 形状均满足;通用尾块见 v0/v1)。
// 性能:5.635±0.017 ms = 24.4±0.06 TFLOPS,vs v1 x4.5 = FA2 梯最大台阶
// (EXP-K03（CUDA FA2 forward 简化版版本梯）);资源:72 reg / 128 thr / 动态 smem 90.75KB → 1 block/SM,
// 理论 occupancy 8.3%。v4 把 ① 全部预取重叠后仅 +7.1%(EXP-K03 §6)——
// 瓶颈是这条相位链本身:本文件就是「wmma 架构税」的实体。
// 面试点:① softmax 为什么必须出 fragment(映射私有);② O 的 ×α 重缩放
// 为什么同样做不进 fragment——α 按行,而 fragment 元素不知道自己属于哪一
// 行,故 O 只能驻 smem 按行缩放;③ P 的 padding 列写 0 = causal mask 的
// wmma 形态(⑤ 会把整 64 列乘进去,mask 即零填充)。
// ============================================================================
using namespace nvcuda;
constexpr int BM = 64, BN = 64, WARPS = 4;

// 动态 smem 分区(字节偏移,均 16B 对齐——float4 装载与 cp.async/wmma 的
// 对齐要求;合计 92928B = 90.75KB,超 48KB 静态上限,须动态 smem +
// cudaFuncSetAttribute opt-in,Ada 每 block 上限 99KB):
//   区    类型/形状          字节   用途
//   Osm   float[64][128]    32768   O 累加器:fp32 驻 smem,逐 tile ×α 重缩放
//   Ssm   float[64][68]     17408   QK^T 结果(③ 的读源)
//   m/l/a float[64] x3        768   在线统计:运行 max / 分母 / 本轮 α
//   Ks    half [64][128]    16384   K tile
//   Vs    half [64][128]    16384   V tile
//   Psm   half [64][72]      9216   exp 后的 P(⑤ 的 A 操作数,故存 half)
// LDS=68=64+4:行跨距 64 float ≡ 0 mod 32 bank,wmma 16x16 store 的列向
// 访问会全撞同 bank;+4 是保住 16B 对齐(4 float)的最小错位。
// LDP=72=64+8:同理,half 粒度下 16B 对齐(8 half)的最小错位。
constexpr int OFF_O = 0;                       // float [64][128] 32768
constexpr int OFF_S = 32768;                   // float [64][68]  17408
constexpr int OFF_ML = 50176;                  // float m[64] l[64] a[64] 768
constexpr int OFF_K = 50944;                   // half  [64][128] 16384
constexpr int OFF_V = 67328;                   // half  [64][128] 16384
constexpr int OFF_P = 83712;                   // half  [64][72]   9216
constexpr int SMEM_BYTES = 92928;
constexpr int LDS = 68, LDP = 72;

__global__ void fa2_v2_kernel(const half* Q, const half* K, const half* V,
                              half* O, int Hq, int Hkv, int S, bool causal) {
    extern __shared__ char smem[];
    float* Osm = (float*)(smem + OFF_O);
    float* Ssm = (float*)(smem + OFF_S);
    float* m_s = (float*)(smem + OFF_ML);
    float* l_s = m_s + BM;
    float* a_s = l_s + BM;
    half* Ks = (half*)(smem + OFF_K);
    half* Vs = (half*)(smem + OFF_V);
    half* Psm = (half*)(smem + OFF_P);

    const int tid = threadIdx.x, warp = tid / 32;
    const int h = blockIdx.y, b = blockIdx.z;
    const int kvh = h / (Hq / Hkv);            // GQA 映射
    const int q0 = blockIdx.x * BM;            // 本 block 的 query 行基址
    const half* k = K + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* v = V + (size_t)(b * Hkv + kvh) * S * FA_D;
    const half* q = Q + ((size_t)(b * Hq + h) * S + q0) * FA_D;

    // [初始化] O 清零;m = -1e30(-inf 哨兵)、l = 0,一行一份
    for (int i = tid; i < BM * FA_D; i += blockDim.x) Osm[i] = 0.f;
    if (tid < BM) { m_s[tid] = -1e30f; l_s[tid] = 0.f; }

    // [Q 常驻] 每 warp 的 16 行 q 沿 D=128 切 8 个 fragment,整个 kernel
    // 只 load 一次——Q 是唯一不随 tile 变化的操作数,常驻寄存器零重读
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[8];
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        wmma::load_matrix_sync(af[kk], q + (warp * 16) * FA_D + kk * 16, FA_D);

    const float scale = rsqrtf((float)FA_D);
    // causal 时 block 最大行 q0+BM-1 至多可见 q0+BM 个键 → tile 级裁剪;
    // 行级精确因果边界由 ③ 的 jend 收。循环次数全 block 一致,循环体内的
    // __syncthreads 才不会发散(发散到 barrier 是 UB)
    const int nlimit = causal ? min(q0 + BM, S) : S;

    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();   // ①前置:防新 tile 装载覆盖 Ks/Vs 时,上一轮 ⑤
                           // 仍有 warp 在读 Vs/Psm(WAR)
        // [① K/V 装载] float4 协同搬运;无边界 guard,靠 S % 64 == 0 前置条件
        for (int t = tid; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            *(float4*)&Ks[r * FA_D + c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
            *(float4*)&Vs[r * FA_D + c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
        }
        __syncthreads();   // ①→②:装载按线性 tid 分片、wmma 按 warp 读——
                           // 线程集不重合,barrier 后写才可见(跨线程 RAW)
        #pragma unroll
        for (int n = 0; n < 4; ++n) {                  // [② QK^T] 每 warp 16 行 x 全 64 列条带
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;    // K^T 的第 j 列 = K 的第 j 行:
                wmma::load_matrix_sync(bf, &Ks[(n * 16) * FA_D + kk * 16], FA_D);   // 声明 col_major 即得转置视图,免物化 K^T
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            // 架构税现场:sc 内做不了行级 max(lane→元素映射私有),
            // 只能整块 store 落 smem 交给 ③ 的标量段
            wmma::store_matrix_sync(&Ssm[(warp * 16) * LDS + n * 16], sc,
                                    LDS, wmma::mem_row_major);
        }
        __syncthreads();   // ②→③:行 r 的 S 由 warp r/16 写、由线程 tid=r
                           // (属 warp r/32)读——跨 warp RAW
        if (tid < BM) {                                // [③ 行级在线 softmax] 一行一线程
                                                       // (128 线程闲一半——v3 的改进点)
            const int row = q0 + tid;
            const int jend = min(BN, (causal ? row + 1 : S) - n0);   // row+1:causal 含对角
            float rmax = -1e30f;
            for (int j = 0; j < jend; ++j)
                rmax = fmaxf(rmax, Ssm[tid * LDS + j] * scale);   // scale 读时乘,免改写 S 一遍
            const float mn = fmaxf(m_s[tid], rmax);    // 新全局 max 单调不减 → exp 参数恒 <=0
            const float alpha = __expf(m_s[tid] - mn); // 历史部分和折算因子
            float sum = 0.f;
            for (int j = 0; j < BN; ++j) {
                // j >= jend 写 0:P 的因果/越界列必须清零——⑤ 的 wmma 把
                // 整 64 列乘进去,mask 即零填充(面试点③)
                float p = j < jend ? __expf(Ssm[tid * LDS + j] * scale - mn) : 0.f;
                Psm[tid * LDP + j] = __float2half(p);
                sum += p;
            }
            l_s[tid] = l_s[tid] * alpha + sum;         // 在线分母:旧 l 折算 + 本 tile 增量
            m_s[tid] = mn; a_s[tid] = alpha;           // α 落 smem:④ 由别的线程读
        }
        __syncthreads();   // ③→④:Psm/a_s 写完,④ 才能读 a_s、⑤ 才能读 Psm(RAW)
        for (int i = tid; i < BM * FA_D; i += blockDim.x)   // [④ O ×α] i/FA_D = 行号;
            Osm[i] *= a_s[i / FA_D];                        // 必须先于 ⑤:O_new = α·O_old + P·V
        __syncthreads();   // ④→⑤:重缩放按线性 tid 分片,⑤ 按 warp 行条带
                           // load 旧 O——线程集不重合(RAW)
        #pragma unroll
        for (int c = 0; c < 8; ++c) {                  // [⑤ P·V 累加] 每 warp 16 行 x 128 列,
                                                       // P[16x64]·V[64x128] 按 kk 4 步
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv, oacc;
            wmma::fill_fragment(pv, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                               wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::row_major> vf;
                wmma::load_matrix_sync(pf, &Psm[(warp * 16) * LDP + kk * 16], LDP);
                wmma::load_matrix_sync(vf, &Vs[(kk * 16) * FA_D + c * 16], FA_D);
                wmma::mma_sync(pv, pf, vf, pv);
            }
            // wmma 无「累加到内存」原语:load 旧 O → 逐元素加 → store 回;
            // 逐元素加合法:同 shape accumulator 映射一致(见 gemm_v2 面试点②)
            float* optr = &Osm[(warp * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
        }
    }
    __syncthreads();   // 收尾:末轮 ⑤ 的 O 写(按 warp)对写回(线性 tid 读全 Osm)可见(RAW)
    half* o = O + ((size_t)(b * Hq + h) * S + q0) * FA_D;
    // [写回] 分母 l 延迟到此处只除一次;fp32→fp16 舍入仅发生在最终输出
    for (int i = tid; i < BM * FA_D; i += blockDim.x)
        o[i] = __float2half(Osm[i] / l_s[i / FA_D]);
}

void fa2_v2(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal) {
    static bool configured = false;   // 一次性 opt-in:>48KB 动态 smem 须显式放行;
    if (!configured) {                // bench 单线程调用,无并发初始化问题
        cudaFuncSetAttribute(fa2_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v2_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
}
