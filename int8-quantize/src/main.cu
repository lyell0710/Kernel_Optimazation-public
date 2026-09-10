#include "quantize_common.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
void build_per_channel_scales(const std::vector<float>& input, std::vector<float>& scales, int channels, int hw)
{
    for (int c = 0; c < channels; ++c)
    {
        float amax = 0.0f;
        int base = c * hw;
        for (int i = 0; i < hw; ++i)
        {
            amax = std::max(amax, std::fabs(input[base + i]));
        }
        scales[c] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    }
}
} // namespace

int main()
{
    // 默认 1024x1024 fp32 输入 = 4.19 MB,远小于 72 MB L2(EXP-K09 §5.9 实测热 cache 下 DRAM 读为 0)
    int channels = 1024;
    int hw = 1024;
    if (const char* e = std::getenv("Q8_CHANNELS")) channels = std::atoi(e);
    if (const char* e = std::getenv("Q8_HW")) hw = std::atoi(e);
    const int total = channels * hw;
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

    std::vector<float> h_in(total);
    std::vector<float> h_scales(channels, 1.0f);
    std::vector<int8_t> h_ref(total, 0);
    std::vector<int8_t> h_out(total, 0);

    std::mt19937 rng(20260503);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (float& v : h_in) v = dist(rng);
    build_per_channel_scales(h_in, h_scales, channels, hw);
    cpu_quantize_per_channel(h_in.data(), h_scales.data(), h_ref.data(), channels, hw);

    float* d_in = nullptr;
    float* d_scales = nullptr;
    int8_t* d_out = nullptr;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_scales, channels * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(int8_t));
    cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), channels * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run_and_check = [&](const std::string& name,
                             void (*fn)(const float*, const float*, int8_t*, int, int),
                             float& mean_ms,
                             float& max_abs_err,
                             bool& pass_out) {
        fn(d_in, d_scales, d_out, channels, hw);
        cudaDeviceSynchronize();

        float total_ms = 0.0f;
        for (int i = 0; i < kBenchmarkIters; ++i)
        {
            cudaEventRecord(start);
            fn(d_in, d_scales, d_out, channels, hw);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start, stop);
            total_ms += iter_ms;
        }
        mean_ms = total_ms / static_cast<float>(kBenchmarkIters);

        cudaMemcpy(h_out.data(), d_out, total * sizeof(int8_t), cudaMemcpyDeviceToHost);
        bool exact_pass = true;
        max_abs_err = 0.0f;
        for (int i = 0; i < total; ++i)
        {
            if (h_out[i] != h_ref[i])
            {
                exact_pass = false;
                break;
            }
            int c = i / hw;
            float deq = static_cast<float>(h_out[i]) * h_scales[c];
            float err = std::fabs(h_in[i] - deq);
            if (err > max_abs_err)
            {
                max_abs_err = err;
            }
        }
        pass_out = exact_pass;
        std::cout << "[" << name << "] " << (exact_pass ? "PASS" : "FAIL") << ", mean_latency=" << mean_ms
                  << " ms, max_abs_err=" << max_abs_err << std::endl;
    };

    if (const char* profile_only = std::getenv("QUANTIZE_PROFILE_ONLY"))
    {
        if (profile_only[0] != '\0')
        {
            float mean_ms = 0.0f, err = 0.0f;
            bool pass = false;
            if (std::strcmp(profile_only, "baseline") == 0)
            {
                run_and_check("baseline", quantize_baseline, mean_ms, err, pass);
            }
            else if (std::strcmp(profile_only, "v0") == 0)
            {
                run_and_check("v0", quantize_v0, mean_ms, err, pass);
            }
            else if (std::strcmp(profile_only, "v1") == 0)
            {
                run_and_check("v1", quantize_v1, mean_ms, err, pass);
            }
            else if (std::strcmp(profile_only, "v2") == 0)
            {
                run_and_check("v2", quantize_v2, mean_ms, err, pass);
            }
            else if (std::strcmp(profile_only, "v3") == 0)
            {
                run_and_check("v3", quantize_v3, mean_ms, err, pass);
            }
            else if (std::strcmp(profile_only, "v4") == 0)
            {
                run_and_check("v4", quantize_v4, mean_ms, err, pass);
            }
            else
            {
                std::cerr << "QUANTIZE_PROFILE_ONLY must be baseline or v0..v4 (got: " << profile_only << ")\n";
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_in);
                cudaFree(d_scales);
                cudaFree(d_out);
                return 2;
            }
            (void)pass;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_in);
            cudaFree(d_scales);
            cudaFree(d_out);
            return 0;
        }
    }

    float baseline_ms = 0.0f, baseline_err = 0.0f;
    float v0_ms = 0.0f, v0_err = 0.0f;
    float v1_ms = 0.0f, v1_err = 0.0f;
    float v2_ms = 0.0f, v2_err = 0.0f;
    float v3_ms = 0.0f, v3_err = 0.0f;
    float v4_ms = 0.0f, v4_err = 0.0f;
    bool baseline_pass = false;
    bool v0_pass = false;
    bool v1_pass = false;
    bool v2_pass = false;
    bool v3_pass = false;
    bool v4_pass = false;

    run_and_check("baseline", quantize_baseline, baseline_ms, baseline_err, baseline_pass);
    run_and_check("v0", quantize_v0, v0_ms, v0_err, v0_pass);
    run_and_check("v1", quantize_v1, v1_ms, v1_err, v1_pass);
    run_and_check("v2", quantize_v2, v2_ms, v2_err, v2_pass);
    run_and_check("v3", quantize_v3, v3_ms, v3_err, v3_pass);
    run_and_check("v4", quantize_v4, v4_ms, v4_err, v4_pass);

    std::ofstream csv_out(kCsvPath, std::ios::trunc);
    csv_out << "version,channels,hw,latency_ms,speedup_vs_baseline,max_abs_err,correctness_pass\n";
    auto write_row = [&](const char* version, float latency_ms, float err, bool pass) {
        const float speedup = baseline_ms / latency_ms;
        csv_out << version << "," << channels << "," << hw << ",";
        csv_out << std::fixed << std::setprecision(6) << latency_ms << ",";
        csv_out << std::fixed << std::setprecision(2) << speedup << ",";
        csv_out << std::scientific << std::setprecision(4) << err << ",";
        csv_out << (pass ? "true" : "false") << "\n";
    };
    write_row("baseline", baseline_ms, baseline_err, baseline_pass);
    write_row("v0", v0_ms, v0_err, v0_pass);
    write_row("v1", v1_ms, v1_err, v1_pass);
    write_row("v2", v2_ms, v2_err, v2_pass);
    write_row("v3", v3_ms, v3_err, v3_pass);
    write_row("v4", v4_ms, v4_err, v4_pass);
    std::cout << "Updated benchmark CSV: " << kCsvPath << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_scales);
    cudaFree(d_out);
    return 0;
}
