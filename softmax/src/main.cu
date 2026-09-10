#include "softmax_common.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

bool check_close(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-4f)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::fabs(a[i] - b[i]) > eps)
        {
            std::cout << "mismatch at " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    // 形状默认 1024x1024(工作集 8.4 MB, 常驻 4090 的 72 MB L2)。
    // 可由 SOFTMAX_ROWS/SOFTMAX_COLS 覆盖,用于做 L2 常驻 vs HBM-bound 的区间对照
    // ——形状写死正是"对外数字未标注 regime"的根源,见 EXP-K09 §5.9/§6.8。
    int rows = 1024;
    int cols = 1024;
    if (const char* e = std::getenv("SOFTMAX_ROWS")) rows = std::atoi(e);
    if (const char* e = std::getenv("SOFTMAX_COLS")) cols = std::atoi(e);
    const int n = rows * cols;
    int kBenchmarkIters = 100;
    if (const char* env_iters = std::getenv("BENCH_ITERS"))
    {
        int parsed = std::atoi(env_iters);
        if (parsed > 0)
        {
            kBenchmarkIters = parsed;
        }
    }
    const char* kCsvPath = "project-proof/data/benchmark_results.csv";

    // 构造一个更接近真实分布的输入，避免“全 1”过于理想。
    std::vector<float> h_in(n, 0.0f);
    std::vector<float> h_ref(n, 0.0f);
    std::vector<float> h_out(n, 0.0f);
    std::mt19937 rng(20260502);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < n; ++i)
    {
        h_in[i] = dist(rng);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cpu_softmax(h_in.data(), h_ref.data(), rows, cols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run_and_check = [&](const std::string& name, void (*fn)(const float*, float*, int, int), float& mean_ms, float& max_diff) {
        // warmup
        fn(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        float total_ms = 0.0f;
        for (int i = 0; i < kBenchmarkIters; ++i)
        {
            cudaEventRecord(start);
            fn(d_in, d_out, rows, cols);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start, stop);
            total_ms += iter_ms;
        }
        mean_ms = total_ms / static_cast<float>(kBenchmarkIters);

        cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

        max_diff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            float diff = std::fabs(h_out[i] - h_ref[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
        bool ok = check_close(h_out, h_ref);
        std::cout << "[" << name << "] " << (ok ? "PASS" : "FAIL") << ", mean_latency=" << mean_ms << " ms, max_diff=" << max_diff
                  << std::endl;
    };

    if (const char* profile_only = std::getenv("SOFTMAX_PROFILE_ONLY"))
    {
        if (profile_only[0] != '\0')
        {
            float mean_ms = 0.0f, max_diff = 0.0f;
            if (std::strcmp(profile_only, "baseline") == 0)
            {
                run_and_check("baseline", softmax_baseline, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v0") == 0)
            {
                run_and_check("v0", softmax_v0, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v1") == 0)
            {
                run_and_check("v1", softmax_v1, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v2") == 0)
            {
                run_and_check("v2", softmax_v2, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v3") == 0)
            {
                run_and_check("v3", softmax_v3, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v4") == 0)
            {
                run_and_check("v4", softmax_v4, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v4.2") == 0)
            {
                run_and_check("v4.2", softmax_v4_2, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v4.3") == 0)
            {
                run_and_check("v4.3", softmax_v4_3, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v4.4") == 0)
            {
                run_and_check("v4.4", softmax_v4_4, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "cudnn") == 0)
            {
                run_and_check("cudnn", softmax_cudnn, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "cublas") == 0)
            {
                run_and_check("cublas", softmax_cublas, mean_ms, max_diff);
            }
            else
            {
                std::cerr << "SOFTMAX_PROFILE_ONLY must be baseline, v0..v4, v4.2/3/4, or cublas (got: " << profile_only << ")\n";
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_in);
                cudaFree(d_out);
                return 2;
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_in);
            cudaFree(d_out);
            return 0;
        }
    }

    float v0_ms = 0.0f, v0_diff = 0.0f;
    float v1_ms = 0.0f, v1_diff = 0.0f;
    float v2_ms = 0.0f, v2_diff = 0.0f;
    float v3_ms = 0.0f, v3_diff = 0.0f;
    float v4_ms = 0.0f, v4_diff = 0.0f;
    float v4_2_ms = 0.0f, v4_2_diff = 0.0f;
    float v4_4_ms = 0.0f, v4_4_diff = 0.0f;
    float cublas_ms = 0.0f, cublas_diff = 0.0f;
    float cudnn_ms = 0.0f, cudnn_diff = 0.0f;

    run_and_check("v0", softmax_v0, v0_ms, v0_diff);
    run_and_check("v1", softmax_v1, v1_ms, v1_diff);
    run_and_check("v2", softmax_v2, v2_ms, v2_diff);
    run_and_check("v3", softmax_v3, v3_ms, v3_diff);
    run_and_check("v4", softmax_v4, v4_ms, v4_diff);
    run_and_check("v4.2", softmax_v4_2, v4_2_ms, v4_2_diff);
    run_and_check("v4.4", softmax_v4_4, v4_4_ms, v4_4_diff);
    run_and_check("handwritten_ref", softmax_cublas, cublas_ms, cublas_diff);
    run_and_check("cudnn", softmax_cudnn, cudnn_ms, cudnn_diff);   // 标准库基准

    // ===== 第二组 benchmark：misaligned cols=1500 =====
    // 1500 卡在 blockSize*packSize (1024) 的"中间"：
    //   - v4 主循环步进 1024，第一轮所有 256 线程都走 float4
    //   - 第二轮 tid=0..118 走 float4 (c=1024..1496+3=1499<1500)，tid=119 起跨过 cols
    //   - 部分线程跑 1 轮、部分跑 2 轮 → warp 内出现负载不均，可能 divergence
    // 用于评估 v4 在"真正破"的尺寸下相对 v4.3 / cuBLAS 的退化幅度。
    // 非对齐形状,同样可覆盖(与主形状的 SOFTMAX_ROWS/COLS 独立)
    int rows_mis = 1024;
    int cols_mis = 1500;
    if (const char* e = std::getenv("SOFTMAX_MIS_ROWS")) rows_mis = std::atoi(e);
    if (const char* e = std::getenv("SOFTMAX_MIS_COLS")) cols_mis = std::atoi(e);
    const int n_mis = rows_mis * cols_mis;
    std::vector<float> h_in_mis(n_mis, 0.0f);
    std::vector<float> h_ref_mis(n_mis, 0.0f);
    std::vector<float> h_out_mis(n_mis, 0.0f);
    {
        std::mt19937 rng2(20260503);
        std::uniform_real_distribution<float> dist2(-2.0f, 2.0f);
        for (int i = 0; i < n_mis; ++i)
        {
            h_in_mis[i] = dist2(rng2);
        }
    }
    float* d_in_mis = nullptr;
    float* d_out_mis = nullptr;
    cudaMalloc(&d_in_mis, n_mis * sizeof(float));
    cudaMalloc(&d_out_mis, n_mis * sizeof(float));
    cudaMemcpy(d_in_mis, h_in_mis.data(), n_mis * sizeof(float), cudaMemcpyHostToDevice);
    cpu_softmax(h_in_mis.data(), h_ref_mis.data(), rows_mis, cols_mis);

    auto run_mis = [&](const std::string& name, void (*fn)(const float*, float*, int, int),
                       float& mean_ms, float& max_diff) {
        fn(d_in_mis, d_out_mis, rows_mis, cols_mis);
        cudaDeviceSynchronize();
        float total_ms = 0.0f;
        for (int i = 0; i < kBenchmarkIters; ++i)
        {
            cudaEventRecord(start);
            fn(d_in_mis, d_out_mis, rows_mis, cols_mis);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start, stop);
            total_ms += iter_ms;
        }
        mean_ms = total_ms / static_cast<float>(kBenchmarkIters);
        cudaMemcpy(h_out_mis.data(), d_out_mis, n_mis * sizeof(float), cudaMemcpyDeviceToHost);
        max_diff = 0.0f;
        for (int i = 0; i < n_mis; ++i)
        {
            float diff = std::fabs(h_out_mis[i] - h_ref_mis[i]);
            if (diff > max_diff) max_diff = diff;
        }
        bool ok = max_diff <= 1e-4f;
        std::cout << "[mis-" << name << "] " << (ok ? "PASS" : "FAIL")
                  << ", mean_latency=" << mean_ms << " ms, max_diff=" << max_diff << std::endl;
    };

    std::cout << "\n===== Misaligned benchmark (cols=1500) =====" << std::endl;
    float v4_mis_ms = 0.0f, v4_mis_diff = 0.0f;
    float v4_3_mis_ms = 0.0f, v4_3_mis_diff = 0.0f;
    float v4_4_mis_ms = 0.0f, v4_4_mis_diff = 0.0f;
    float cublas_mis_ms = 0.0f, cublas_mis_diff = 0.0f;
    float cudnn_mis_ms = 0.0f, cudnn_mis_diff = 0.0f;
    run_mis("v4", softmax_v4, v4_mis_ms, v4_mis_diff);
    run_mis("v4.3", softmax_v4_3, v4_3_mis_ms, v4_3_mis_diff);
    run_mis("v4.4", softmax_v4_4, v4_4_mis_ms, v4_4_mis_diff);
    run_mis("handwritten_ref", softmax_cublas, cublas_mis_ms, cublas_mis_diff);
    run_mis("cudnn", softmax_cudnn, cudnn_mis_ms, cudnn_mis_diff);

    cudaFree(d_in_mis);
    cudaFree(d_out_mis);

    std::ofstream csv_out(kCsvPath, std::ios::trunc);
    if (csv_out.is_open())
    {
        csv_out << "version,rows,cols,latency_ms,speedup_vs_v0,max_diff,correctness_pass\n";
        auto write_row = [&](const char* version, float latency_ms, float diff) {
            const float speedup = v0_ms / latency_ms;
            const bool pass = diff <= 1e-4f;
            csv_out << version << "," << rows << "," << cols << ",";
            csv_out << std::fixed << std::setprecision(6) << latency_ms << ",";
            csv_out << std::fixed << std::setprecision(2) << speedup << ",";
            csv_out << std::scientific << std::setprecision(4) << diff << ",";
            csv_out << (pass ? "true" : "false") << "\n";
        };
        write_row("v0", v0_ms, v0_diff);
        write_row("v1", v1_ms, v1_diff);
        write_row("v2", v2_ms, v2_diff);
        write_row("v3", v3_ms, v3_diff);
        write_row("v4", v4_ms, v4_diff);
        write_row("v4.2", v4_2_ms, v4_2_diff);
        write_row("handwritten_ref", cublas_ms, cublas_diff);
        write_row("cudnn", cudnn_ms, cudnn_diff);

        // misaligned (cols=1000) 行：单独标记，speedup 用 v4 misaligned 作为基准
        auto write_mis = [&](const char* version, float latency_ms, float diff) {
            const float speedup = v4_mis_ms / latency_ms;
            const bool pass = diff <= 1e-4f;
            csv_out << version << "," << rows_mis << "," << cols_mis << ",";
            csv_out << std::fixed << std::setprecision(6) << latency_ms << ",";
            csv_out << std::fixed << std::setprecision(2) << speedup << ",";
            csv_out << std::scientific << std::setprecision(4) << diff << ",";
            csv_out << (pass ? "true" : "false") << "\n";
        };
        write_mis("v4_mis", v4_mis_ms, v4_mis_diff);
        write_mis("v4.3_mis", v4_3_mis_ms, v4_3_mis_diff);
        write_mis("v4.4_mis", v4_4_mis_ms, v4_4_mis_diff);
        write_mis("handwritten_ref_mis", cublas_mis_ms, cublas_mis_diff);
        write_mis("cudnn_mis", cudnn_mis_ms, cudnn_mis_diff);
        std::cout << "Updated benchmark CSV: " << kCsvPath << std::endl;
    }
    else
    {
        std::cout << "WARN: failed to write CSV: " << kCsvPath << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
