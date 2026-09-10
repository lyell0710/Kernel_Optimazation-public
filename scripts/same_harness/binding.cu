// 把本仓的 CUDA kernel 暴露成 torch 扩展,唯一目的是让它与姊妹仓 triton-kernels
// 的 Triton 实现能在**同一个 Python 进程、同一份数据、同一套计时协议**下对照。
//
// 为什么需要它:LEDGER 红线「一切 vs Triton/sdpa 数字为跨 harness 推断级」的
// 解锁条件就是同 harness 复测。EXP-K05 的三个融合逐元素算子做到了(它们本就是
// torch 扩展),而 gemm/flash-attn 是纯 C++ bench —— 两边用各自的 harness 测,
// 计时方式、warmup、数据都不同,所以只能算推断级。这个绑定消除该不对称。
//
// 注意:这里**不复制任何 kernel 实现**,直接编译各算子的 .cu(单一事实源)。
#include <torch/extension.h>
#include <cuda_fp16.h>

void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K);
void fa2_v4(const half* Q, const half* K, const half* V, half* O,
            int B, int Hq, int Hkv, int S, bool causal);

static inline const half* ch(const torch::Tensor& t) {
    return reinterpret_cast<const half*>(t.data_ptr<at::Half>());
}
static inline half* mh(torch::Tensor& t) {
    return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}

void py_gemm_v4(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    gemm_v4(ch(A), ch(B), mh(C), A.size(0), B.size(1), A.size(1));
}
void py_gemm_cublas(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    gemm_cublas(ch(A), ch(B), mh(C), A.size(0), B.size(1), A.size(1));
}
// q/k/v/o 约定 (B, H, S, D) 连续布局,与 Triton 侧 fa2_forward 一致
void py_fa2_v4(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
               int64_t Hkv, bool causal) {
    fa2_v4(ch(Q), ch(K), ch(V), mh(O),
           Q.size(0), Q.size(1), (int)Hkv, Q.size(2), causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_v4", &py_gemm_v4);
    m.def("gemm_cublas", &py_gemm_cublas);
    m.def("fa2_v4", &py_fa2_v4);
}
