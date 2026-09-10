// ============================================================================
// silu_and_mul 的 torch 绑定。绑定方式与理由见 fused-norm/src/binding.cpp。
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "activation.h"

namespace {

void check(const at::Tensor& out, const at::Tensor& a, bool need_vec) {
    TORCH_CHECK(out.is_cuda() && a.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 &&
                a.scalar_type() == at::kBFloat16, "only bf16 supported");
    TORCH_CHECK(out.is_contiguous() && a.is_contiguous(), "must be contiguous");
    if (need_vec)
        TORCH_CHECK(out.numel() % 8 == 0,
                    "v2/v3 需要元素数为 8 的倍数(16B 向量访存)");
}

template <ActFn FN, bool NEED_VEC>
void run(at::Tensor out, at::Tensor gate, at::Tensor up) {
    check(out, gate, NEED_VEC);
    TORCH_CHECK(gate.sizes() == up.sizes() && gate.sizes() == out.sizes(),
                "shape mismatch");
    FN(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
       out.numel(), at::cuda::getCurrentCUDAStream());
}

// 打包布局:gate_up 是 [T, 2I],out 是 [T, I]
void run_packed(at::Tensor out, at::Tensor gate_up) {
    check(out, gate_up, true);
    TORCH_CHECK(gate_up.dim() == 2 && out.dim() == 2, "expect 2-D [T, ...]");
    TORCH_CHECK(gate_up.size(0) == out.size(0), "T mismatch");
    TORCH_CHECK(gate_up.size(1) == 2 * out.size(1),
                "gate_up 的第二维必须是 out 的两倍(前半 gate、后半 up)");
    silu_and_mul_v3(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
                    (int)out.size(0), (int)out.size(1),
                    at::cuda::getCurrentCUDAStream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v0", &run<silu_and_mul_v0, false>, "unfused: silu kernel + mul kernel");
    m.def("v1", &run<silu_and_mul_v1, false>, "fused, scalar");
    m.def("v2", &run<silu_and_mul_v2, true>,  "fused, vectorized 16B");
    m.def("v3", &run_packed, "packed [T,2I] layout (vLLM style), vectorized");
}
