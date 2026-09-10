#include "gemv_common.h"
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
bool check_close(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-3f)
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
    // 默认 4096x2048 fp32 = 33.55 MB,常驻 4090 的 72 MB L2。
    // 可由 GEMV_ROWS/GEMV_COLS 覆盖,用于 L2 常驻 vs HBM-bound 的区间对照
    // ——形状写死正是"对外数字未标注 regime"的根源(EXP-K09 §5.8/§6.8)。
    int rows = 4096;
    int cols = 2048;
    if (const char* e = std::getenv("GEMV_ROWS")) rows = std::atoi(e);
    if (const char* e = std::getenv("GEMV_COLS")) cols = std::atoi(e);
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

    std::vector<float> h_mat(rows * cols);
    std::vector<float> h_vec(cols);
    std::vector<float> h_ref(rows, 0.0f);
    std::vector<float> h_out(rows, 0.0f);

    std::mt19937 rng(20260503);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : h_mat) v = dist(rng);
    for (float& v : h_vec) v = dist(rng);

    float* d_mat = nullptr;
    float* d_vec = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_mat, h_mat.size() * sizeof(float));
    cudaMalloc(&d_vec, h_vec.size() * sizeof(float));
    cudaMalloc(&d_out, h_out.size() * sizeof(float));
    cudaMemcpy(d_mat, h_mat.data(), h_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec.data(), h_vec.size() * sizeof(float), cudaMemcpyHostToDevice);

    cpu_gemv(h_mat.data(), h_vec.data(), h_ref.data(), rows, cols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run_and_check = [&](const std::string& name, void (*fn)(const float*, const float*, float*, int, int), float& mean_ms, float& max_diff) {
        fn(d_mat, d_vec, d_out, rows, cols);
        cudaDeviceSynchronize();

        float total_ms = 0.0f;
        for (int i = 0; i < kBenchmarkIters; ++i)
        {
            cudaEventRecord(start);
            fn(d_mat, d_vec, d_out, rows, cols);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start, stop);
            total_ms += iter_ms;
        }
        mean_ms = total_ms / static_cast<float>(kBenchmarkIters);

        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        max_diff = 0.0f;
        for (int i = 0; i < rows; ++i)
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

    if (const char* profile_only = std::getenv("GEMV_PROFILE_ONLY"))
    {
        if (profile_only[0] != '\0')
        {
            float mean_ms = 0.0f, max_diff = 0.0f;
            if (std::strcmp(profile_only, "baseline") == 0)
            {
                run_and_check("baseline", gemv_baseline, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v0") == 0)
            {
                run_and_check("v0", gemv_v0, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v1") == 0)
            {
                run_and_check("v1", gemv_v1, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v2") == 0)
            {
                run_and_check("v2", gemv_v2, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v3") == 0)
            {
                run_and_check("v3", gemv_v3, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "v4") == 0)
            {
                run_and_check("v4", gemv_v4, mean_ms, max_diff);
            }
            else if (std::strcmp(profile_only, "cublas") == 0)
            {
                run_and_check("cublas", gemv_cublas, mean_ms, max_diff);
            }
            else
            {
                std::cerr << "GEMV_PROFILE_ONLY must be baseline, v0..v4, or cublas (got: " << profile_only << ")\n";
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_mat);
                cudaFree(d_vec);
                cudaFree(d_out);
                return 2;
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_mat);
            cudaFree(d_vec);
            cudaFree(d_out);
            return 0;
        }
    }

    float baseline_ms = 0.0f, baseline_diff = 0.0f;
    float v0_ms = 0.0f, v0_diff = 0.0f;
    float v1_ms = 0.0f, v1_diff = 0.0f;
    float v2_ms = 0.0f, v2_diff = 0.0f;
    float v3_ms = 0.0f, v3_diff = 0.0f;
    float v4_ms = 0.0f, v4_diff = 0.0f;
    float cublas_ms = 0.0f, cublas_diff = 0.0f;

    run_and_check("baseline", gemv_baseline, baseline_ms, baseline_diff);
    run_and_check("v0", gemv_v0, v0_ms, v0_diff);
    run_and_check("v1", gemv_v1, v1_ms, v1_diff);
    run_and_check("v2", gemv_v2, v2_ms, v2_diff);
    run_and_check("v3", gemv_v3, v3_ms, v3_diff);
    run_and_check("v4", gemv_v4, v4_ms, v4_diff);
    run_and_check("cublas", gemv_cublas, cublas_ms, cublas_diff);

    std::ofstream csv_out(kCsvPath, std::ios::trunc);
    csv_out << "version,rows,cols,latency_ms,speedup_vs_baseline,max_diff,correctness_pass\n";
    auto write_row = [&](const char* version, float latency_ms, float diff) {
        const float speedup = baseline_ms / latency_ms;
        const bool pass = diff <= 1e-3f;
        csv_out << version << "," << rows << "," << cols << ",";
        csv_out << std::fixed << std::setprecision(6) << latency_ms << ",";
        csv_out << std::fixed << std::setprecision(2) << speedup << ",";
        csv_out << std::scientific << std::setprecision(4) << diff << ",";
        csv_out << (pass ? "true" : "false") << "\n";
    };
    write_row("baseline", baseline_ms, baseline_diff);
    write_row("v0", v0_ms, v0_diff);
    write_row("v1", v1_ms, v1_diff);
    write_row("v2", v2_ms, v2_diff);
    write_row("v3", v3_ms, v3_diff);
    write_row("v4", v4_ms, v4_diff);
    write_row("cublas", cublas_ms, cublas_diff);
    std::cout << "Updated benchmark CSV: " << kCsvPath << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_out);
    return 0;
}
