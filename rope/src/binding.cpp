// ============================================================================
// RoPE 的 torch 绑定 —— 五个手写版本与 PyTorch/Triton 臂进同一个 harness。
// 绑定方式与理由见 fused-norm/src/binding.cpp 的文件头。
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "rope.h"

namespace {

// 形状契约:q [T,HQ,D],k [T,HK,D],cos/sin [T,D]。
// v3 额外要求 D%16==0(16B 向量访存);不满足时报错而不是悄悄算错 ——
// RoPE 的错误不会崩也不会 NaN,只会让输出变成乱码,必须在入口拦住。
void check(const at::Tensor& q, const at::Tensor& k, int D, bool need_vec) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q/k must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16,
                "only bf16 supported");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q/k must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "expect [T,H,D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2), "T/D mismatch");
    TORCH_CHECK(D % 2 == 0, "head_dim must be even");
    if (need_vec)
        TORCH_CHECK(D % 16 == 0, "v3 需要 D%16==0(16B 向量访存);当前 D=", D);
}

template <RopeFn FN, bool NEED_VEC>
void run(at::Tensor q, at::Tensor k, at::Tensor cosb, at::Tensor sinb) {
    const int D = (int)q.size(2);
    check(q, k, D, NEED_VEC);
    FN(reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
       reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(cosb.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(sinb.data_ptr()),
       (int)q.size(0), (int)q.size(1), (int)k.size(1), D,
       at::cuda::getCurrentCUDAStream());
}

// v4 签名不同:吃 inv_freq(fp32)与本批起始位置,自己现算角度。
// inv_freq 必须是 fp32 —— theta^(-2i/d) 跨多个数量级,bf16 存不下
// 高频项的精度(与 llm-engine precompute_rope 保持 fp32 同一条理由)。
void run_v4(at::Tensor q, at::Tensor k, at::Tensor inv_freq, int64_t pos_offset) {
    const int D = (int)q.size(2);
    check(q, k, D, true);   // v4 继承 v3 的向量化,同样要求 D%16==0
    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat && inv_freq.is_cuda(),
                "inv_freq must be fp32 CUDA");
    TORCH_CHECK(inv_freq.numel() == D / 2, "inv_freq length must be D/2");
    rope_v4(reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
            inv_freq.data_ptr<float>(), (int)pos_offset,
            (int)q.size(0), (int)q.size(1), (int)k.size(1), D,
            at::cuda::getCurrentCUDAStream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v0", &run<rope_v0, false>, "naive: one thread per element, q/k separate");
    m.def("v1", &run<rope_v1, false>, "paired: one thread per (i, i+D/2)");
    m.def("v2", &run<rope_v2, false>, "paired + q/k in one launch");
    m.def("v3", &run<rope_v3, true>,  "vectorized 16B");
    m.def("v4", &run_v4, "v3 + table-free: __sincosf on the fly");
}
