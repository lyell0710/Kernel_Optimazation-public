// ============================================================================
// FA2 CUDA 版本梯 bench:协议对齐 triton-kernels/scripts/test_fa2.py
// (B=1,Hq=32,Hkv=8,D=128,causal,S=512..4096)——同协议是「跨 harness
// 对照自家 Triton 版」成立的前提(EXP-K03（CUDA FA2 forward 简化版版本梯）,推断级)。
// 两段结构:
//   1) 正确性 gate:形状族覆盖非 causal、GQA 1:1/2:1/4:1、S 512..2048,
//      阈值 max_abs_err < 2e-2 vs fp32 两遍参考(算法路径独立,见 ref_naive.cu);
//   2) benchmark:协议形状,20 warmup + 100 iters,CUDA event 计时。
// 落盘纪律同 gemm:BENCH_OUT 传 UTC 前缀新文件,首行 provenance,不覆盖历史。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "fa_common.h"


// 真实内核驱动版本(/proc/driver/nvidia/version)。cudaDriverGetVersion 返回的是
// CUDA driver-API 版本,曾被误填进 provenance 的 driver= 字段(勘误见
// project-proof/data/manifest.txt);现 driver=真实驱动,CUDA 版本另立 cuda= 字段。
static const char* nvidia_driver_version() {
    static char buf[64] = "unknown";
    std::ifstream f("/proc/driver/nvidia/version");
    std::string tok;
    while (f >> tok) {
        if (tok.find('.') == std::string::npos) continue;
        bool ok = true;
        for (char c : tok)
            if (!(c >= '0' && c <= '9') && c != '.') { ok = false; break; }
        if (ok && tok.size() < sizeof(buf)) {
            snprintf(buf, sizeof buf, "%s", tok.c_str());
            break;
        }
    }
    return buf;
}

using Fn = void (*)(const half*, const half*, const half*, half*,
                    int, int, int, int, bool);

static void fill(std::vector<half>& h) {
    for (auto& x : h)
        x = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2);
}

struct Bufs { half *Q, *K, *V, *O, *R; };
static Bufs alloc_case(int B, int Hq, int Hkv, int S) {
    std::vector<half> hq((size_t)B * Hq * S * FA_D),
        hk((size_t)B * Hkv * S * FA_D), hv((size_t)B * Hkv * S * FA_D);
    fill(hq); fill(hk); fill(hv);
    Bufs bf;
    cudaMalloc(&bf.Q, hq.size() * 2); cudaMalloc(&bf.K, hk.size() * 2);
    cudaMalloc(&bf.V, hv.size() * 2);
    cudaMalloc(&bf.O, hq.size() * 2); cudaMalloc(&bf.R, hq.size() * 2);
    cudaMemcpy(bf.Q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(bf.K, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(bf.V, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice);
    return bf;
}
static void free_case(Bufs b) {
    cudaFree(b.Q); cudaFree(b.K); cudaFree(b.V); cudaFree(b.O); cudaFree(b.R);
}

// 绝对误差口径:输出经 softmax 加权,量级 O(1),绝对阈 2e-2 与相对同阶
static float max_err(const half* a, const half* b, size_t n) {
    std::vector<half> ha(n), hb(n);
    cudaMemcpy(ha.data(), a, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, n * 2, cudaMemcpyDeviceToHost);
    float e = 0;
    for (size_t i = 0; i < n; ++i)
        e = fmaxf(e, fabsf(__half2float(ha[i]) - __half2float(hb[i])));
    return e;
}

int main() {
    srand(42);   // 固定种子:轮间/版本间输入一致,数字才可比
    struct { const char* name; Fn fn; } vs[] = {
        {"v0_warp_row", fa2_v0}, {"v1_smem_tile", fa2_v1}, {"v2_wmma", fa2_v2},
        {"v3_8warp", fa2_v3},
        {"v4_overlap", fa2_v4},
        {"v5_mmaPTX", fa2_v5}};

    // ---- 正确性 gate ----
    // 形状族刻意打散:非 causal 分支、GQA 三种比、S 覆盖 512..2048;
    // S=4096 协议点在 bench 段用同一 gate 再验一遍
    struct { int B, Hq, Hkv, S; bool causal; } cases[] = {
        {1, 8, 8, 512, true}, {1, 8, 8, 512, false},
        {1, 16, 8, 1024, true},                        // GQA 2:1
        {1, 32, 8, 2048, true}};                       // 协议形状族
    bool all_pass = true;
    printf("| ver | B | Hq | Hkv | S | causal | max_abs_err | pass |\n");
    for (auto& c : cases) {
        Bufs b = alloc_case(c.B, c.Hq, c.Hkv, c.S);
        attn_ref_fp32(b.Q, b.K, b.V, b.R, c.B, c.Hq, c.Hkv, c.S, c.causal);
        cudaDeviceSynchronize();
        for (auto& v : vs) {
            // O 先清零:防上一版本的残留输出让错误实现蒙混过 gate
            cudaMemset(b.O, 0, (size_t)c.B * c.Hq * c.S * FA_D * 2);
            v.fn(b.Q, b.K, b.V, b.O, c.B, c.Hq, c.Hkv, c.S, c.causal);
            cudaError_t err = cudaDeviceSynchronize();
            float e = err == cudaSuccess
                          ? max_err(b.O, b.R, (size_t)c.B * c.Hq * c.S * FA_D)
                          : 1e9f;                  // launch 失败记天大误差:崩溃版本不得静默跳过
            bool p = e < 2e-2f;
            all_pass &= p;
            printf("| %s | %d | %d | %d | %d | %d | %.2e | %s |\n",
                   v.name, c.B, c.Hq, c.Hkv, c.S, c.causal, e,
                   p ? "PASS" : "FAIL");
        }
        free_case(b);
    }
    printf("CORRECTNESS %s\n", all_pass ? "PASS" : "FAIL");

    // ---- benchmark(协议形状)----
    const char* outp = std::getenv("BENCH_OUT");   // CORE 铁律5:UTC 前缀新文件,不覆盖历史
    std::ofstream csv(outp ? outp : "project-proof/data/benchmark_results.csv",
                      std::ios::trunc);
    {   // 首行 provenance:每个数字能指回环境与代码版本
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        int drv = 0; cudaDriverGetVersion(&drv);
        char ts[32]; time_t t = time(nullptr);
        strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
        const char* sha = std::getenv("GIT_SHA");
        csv << "# provenance: env=4090-container sha=" << (sha ? sha : "unknown")
            << " cmd=\"fa2_bench\" date=" << ts << " gpu=\"" << prop.name
            << "\" driver=" << nvidia_driver_version()
            << " cuda=" << drv / 1000 << "." << drv % 1000 / 10 << "\n";
    }
    csv << "version,B,Hq,Hkv,S,D,causal,latency_ms,tflops,max_abs_err_vs_ref,"
           "correctness_pass\n";
    int iters = 100;
    if (const char* e = std::getenv("BENCH_ITERS")) iters = atoi(e);
    for (int S : {512, 1024, 2048, 4096}) {
        const int B = 1, Hq = 32, Hkv = 8;
        Bufs b = alloc_case(B, Hq, Hkv, S);
        attn_ref_fp32(b.Q, b.K, b.V, b.R, B, Hq, Hkv, S, true);
        cudaDeviceSynchronize();
        // FLOPs 口径:2(mul+add) x 2 个 GEMM(QK^T、PV)x B·Hq·S²·D,
        // causal ÷2(下三角)——与 Triton 版同式,跨 harness 对照的前提
        const double tf_count = 4.0 * B * Hq * (double)S * S * FA_D / 2 / 1e12;
        for (auto& v : vs) {
            v.fn(b.Q, b.K, b.V, b.O, B, Hq, Hkv, S, true);
            cudaDeviceSynchronize();
            float e = max_err(b.O, b.R, (size_t)B * Hq * S * FA_D);
            bool p = e < 2e-2f;
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int w = 0; w < 20; ++w) v.fn(b.Q, b.K, b.V, b.O, B, Hq, Hkv, S, true);   // 预热驱走冷时钟
            cudaEventRecord(e0);
            for (int i = 0; i < iters; ++i)
                v.fn(b.Q, b.K, b.V, b.O, B, Hq, Hkv, S, true);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;   // 单 event 对包整段取均值
            printf("S=%-5d %-13s %8.4f ms %7.1f TFLOPS  err=%.2e %s\n",
                   S, v.name, ms, tf_count / (ms / 1e3), e,
                   p ? "PASS" : "FAIL");
            csv << v.name << "," << B << "," << Hq << "," << Hkv << "," << S
                << "," << FA_D << ",1," << std::fixed << std::setprecision(4)
                << ms << "," << std::setprecision(1) << tf_count / (ms / 1e3)
                << "," << std::scientific << e << ","
                << (p ? "true" : "false") << "\n";
        }
        free_case(b);
    }
    return 0;
}
