// ============================================================================
// W8A8 的 torch 绑定。量化与反量化两步是手写 kernel,中间的 INT8 GEMM 直接用
// torch._int_mm(底层是 cuBLASLt 的 IMMA 路径)—— 不自己写 GEMM 是有意的:
// 本子项目要回答的是「整条 W8A8 链路能不能在引擎里兑现收益」,
// 而不是「我能不能写出比 cuBLASLt 快的 int8 GEMM」。用现成库做 GEMM,
// 才能把量化/反量化这两步的真实代价单独量出来。
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include "w8a8.h"

namespace {

template <QuantFn FN, bool NEED_VEC>
void run_quant(at::Tensor q, at::Tensor scale, at::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(),
                "x must be contiguous bf16 CUDA");
    TORCH_CHECK(q.scalar_type() == at::kChar && scale.scalar_type() == at::kFloat,
                "q must be int8, scale must be fp32");
    const int H = (int)x.size(-1);
    const int T = (int)(x.numel() / H);
    TORCH_CHECK(scale.numel() == T, "scale length must equal token count");
    if (NEED_VEC)
        TORCH_CHECK(H % 8 == 0, "v2 需要 H%8==0(16B 向量读);当前 H=", H);
    FN(reinterpret_cast<int8_t*>(q.data_ptr()), scale.data_ptr<float>(),
       reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
       T, H, at::cuda::getCurrentCUDAStream());
}

template <DequantFn FN, bool NEED_VEC>
void run_dequant(at::Tensor y, at::Tensor acc, at::Tensor xs, at::Tensor ws) {
    TORCH_CHECK(acc.scalar_type() == at::kInt && acc.is_contiguous(), "acc must be int32");
    TORCH_CHECK(y.scalar_type() == at::kBFloat16, "y must be bf16");
    const int O = (int)acc.size(-1);
    const int T = (int)(acc.numel() / O);
    TORCH_CHECK(xs.numel() == T && ws.numel() == O, "scale 长度不匹配");
    if (NEED_VEC)
        TORCH_CHECK(O % 4 == 0, "v1 需要 O%4==0;当前 O=", O);
    FN(reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), acc.data_ptr<int32_t>(),
       xs.data_ptr<float>(), ws.data_ptr<float>(), T, O,
       at::cuda::getCurrentCUDAStream());
}

// decode 路径:xq[H] int8 + wq[O,H] int8 -> y[O] bf16
// x_scale 以【设备张量】传入。第一版写成主机 double,每次调用都要 .item(),
// 那是一次设备到主机的隐式同步 —— decode 逐层调用会被放大成每 token 上百次同步。
void run_gemv(at::Tensor y, at::Tensor xq, at::Tensor wq, at::Tensor ws,
              at::Tensor x_scale, bool use_smem) {
    TORCH_CHECK(xq.scalar_type() == at::kChar && wq.scalar_type() == at::kChar,
                "xq/wq must be int8");
    TORCH_CHECK(wq.dim() == 2 && wq.is_contiguous(), "wq must be contiguous [O,H]");
    const int O = (int)wq.size(0), H = (int)wq.size(1);
    TORCH_CHECK(xq.numel() == H, "xq length must equal H");
    TORCH_CHECK(H % 16 == 0, "int8 GEMV 需要 H%16==0(16B 向量读);当前 H=", H);
    (use_smem ? int8_gemv_v1 : int8_gemv_v0)(
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        reinterpret_cast<const int8_t*>(xq.data_ptr()),
        reinterpret_cast<const int8_t*>(wq.data_ptr()),
        ws.data_ptr<float>(), x_scale.data_ptr<float>(), O, H,
        at::cuda::getCurrentCUDAStream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_v0", &run_quant<quant_per_token_v0, false>, "two-kernel");
    m.def("quant_v1", &run_quant<quant_per_token_v1, false>, "fused, warp reduce");
    m.def("quant_v2", &run_quant<quant_per_token_v2, true>,  "fused, vectorized");
    m.def("dequant_v0", &run_dequant<dequant_v0, false>, "naive");
    m.def("dequant_v1", &run_dequant<dequant_v1, true>,  "row-block, vectorized");
    m.def("gemv_v0", [](at::Tensor y, at::Tensor xq, at::Tensor wq, at::Tensor ws,
                        at::Tensor s) { run_gemv(y, xq, wq, ws, s, false); },
          "int8 GEMV (dp4a), activation from global");
    m.def("gemv_v1", [](at::Tensor y, at::Tensor xq, at::Tensor wq, at::Tensor ws,
                        at::Tensor s) { run_gemv(y, xq, wq, ws, s, true); },
          "int8 GEMV (dp4a), activation staged in shared memory");
}
