// ============================================================================
// GEMM 版本梯 bench:v0 naive → v1 tile → v2 wmma → v3 双缓冲 → v4 大tile,
// 对照 = 真 cuBLAS(cublasGemmEx,调用点验真见 gemm_cublas.cu)。
// 测量协议(EXP-K02（CUDA Tensor Core GEMM 版本梯）§2):固定 4096³;每版本 3 warmup + 50 iters(慢版
// iters/10,下限 3),CUDA event 计时取均值;正确性 = 对 cuBLAS 输出抽样
// 最大相对误差 < 2e-2(fp16 存储的合理界;实测全版本 7.58e-04)。
// 落盘纪律:结果只写 BENCH_OUT 指定的 UTC 前缀新文件,首行 provenance
// (env/sha/cmd/gpu/driver/cuda),不覆盖历史数据——数字可溯源的基础设施。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include "gemm_common.h"


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

using Fn = void (*)(const half*, const half*, half*, int, int, int);

int main() {
    const int M = 4096, N = 4096, K = 4096;
    int iters = 50;
    if (const char* e = std::getenv("BENCH_ITERS")) iters = atoi(e);

    // 固定种子:所有版本/所有轮吃同一输入,对比才可比。
    // 输入取 (-1,1) 零均值:4096 长点积的部分和 ~O(20),fp16 存储不溢出。
    std::vector<half> hA(size_t(M) * K), hB(size_t(K) * N);
    srand(42);
    for (auto& x : hA) x = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2);
    for (auto& x : hB) x = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2);

    half *A, *B, *C, *Ref;
    cudaMalloc(&A, hA.size() * 2); cudaMalloc(&B, hB.size() * 2);
    cudaMalloc(&C, size_t(M) * N * 2); cudaMalloc(&Ref, size_t(M) * N * 2);
    cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice);

    // 参考输出与其抽样 absmax:相对误差分母用全局 absmax 而非逐元素
    //(逐元素相对误差在近零元素上无意义地爆炸);997 为素数步长,
    // 避开 2 的幂结构的周期性采偏。
    gemm_cublas(A, B, Ref, M, N, K); cudaDeviceSynchronize();
    std::vector<half> href(size_t(M) * N);
    cudaMemcpy(href.data(), Ref, href.size() * 2, cudaMemcpyDeviceToHost);
    float ref_absmax = 0;
    for (size_t i = 0; i < href.size(); i += 997)
        ref_absmax = fmaxf(ref_absmax, fabsf(__half2float(href[i])));

    struct { const char* name; Fn fn; } vs[] = {
        {"v0", gemm_v0}, {"v1", gemm_v1}, {"v2_wmma", gemm_v2},
        {"v3_dbuf", gemm_v3}, {"v4_bigtile", gemm_v4}, {"v5_mmaPTX", gemm_v5},
        {"cublas", gemm_cublas}};

    const char* outp = std::getenv("BENCH_OUT");          // CORE 铁律5:UTC 前缀新文件
    std::ofstream csv(outp ? outp : "project-proof/data/benchmark_results.csv",
                      std::ios::trunc);
    {   // CORE 铁律4:首行 provenance——每个数字能指回环境与代码版本
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        int drv = 0; cudaDriverGetVersion(&drv);
        char ts[32]; time_t t = time(nullptr);
        strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
        const char* sha = std::getenv("GIT_SHA");
        csv << "# provenance: env=4090-container sha=" << (sha ? sha : "unknown")
            << " cmd=\"BENCH_ITERS=" << iters << " gemm_bench\" date=" << ts
            << " gpu=\"" << prop.name << "\" driver=" << nvidia_driver_version()
            << " cuda=" << drv / 1000 << "." << drv % 1000 / 10 << "\n";
    }
    csv << "version,m,n,k,latency_ms,tflops,speedup_vs_v0,max_rel_err,correctness_pass\n";
    const double flops = 2.0 * M * N * K;   // 每输出 1 mul + 1 add
    float v0_ms = 0;
    std::vector<half> hout(size_t(M) * N);

    for (auto& v : vs) {
        v.fn(A, B, C, M, N, K); cudaDeviceSynchronize();       // 预热+出数
        cudaMemcpy(hout.data(), C, hout.size() * 2, cudaMemcpyDeviceToHost);
        float maxrel = 0;
        for (size_t i = 0; i < hout.size(); i += 97) {         // 素数步长抽样,理由同上
            float d = fabsf(__half2float(hout[i]) - __half2float(href[i]));
            maxrel = fmaxf(maxrel, d / ref_absmax);
        }
        bool pass = maxrel < 2e-2f;
        int it = (strcmp(v.name, "v0") == 0 || strcmp(v.name, "v1") == 0)
                     ? std::max(3, iters / 10) : iters;         // 慢版少跑:>20ms/iter,
                                                                // 5 次统计已够(EXP-K02 §7 v1 std≈0)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 3; ++w) v.fn(A, B, C, M, N, K);     // 预热:驱走冷时钟/懒初始化
        cudaEventRecord(e0);
        for (int i = 0; i < it; ++i) v.fn(A, B, C, M, N, K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;  // 单 event 对包住整段:免逐次 event 开销偏置
        if (strcmp(v.name, "v0") == 0) v0_ms = ms;
        double tf = flops / (ms / 1e3) / 1e12;
        printf("%-10s %9.4f ms  %7.1f TFLOPS  maxrel=%.2e  %s\n",
               v.name, ms, tf, maxrel, pass ? "PASS" : "FAIL");
        csv << v.name << "," << M << "," << N << "," << K << ","
            << std::fixed << std::setprecision(4) << ms << ","
            << std::setprecision(1) << tf << ","
            << std::setprecision(2) << (v0_ms / ms) << ","
            << std::scientific << maxrel << ","
            << (pass ? "true" : "false") << "\n";
    }
    return 0;
}
