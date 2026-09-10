// ============================================================================
// torch 绑定 —— 让五个手写版本与 PyTorch eager / torch.compile 在同一个
// harness、同一套计时代码下受测。
//
// 为什么必须同 harness:本仓早期「vs Triton / vs SDPA」的数字是跨 harness
// 对比(自家 C++ bench vs 别人的 Python bench),只能标注为「推断级」。
// 计时口径(是否含 launch、warmup 轮数、是否同步)一旦不同,几个百分点的
// 结论就不可信。把手写 kernel 绑进 torch 后,所有臂共用同一段
// CUDA-event 计时,差值才是算子本身的差值。
//
// 绑定方式选 pybind(cpp_extension.load)而非 TORCH_LIBRARY:本算子只需要
// 被 bench 与引擎直接调用,不需要进 dispatcher 参与 autograd/设备分发。
// TORCH_LIBRARY 那条路线(算子注册 -> OperatorEntry -> dispatchTable)
// 留给需要被 torch.compile 捕获的场景。
// ============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // getCurrentCUDAStream
#include <cuda_bf16.h>
#include "fused_norm.h"

namespace {

// 参数校验集中一处:五个版本共用。
// 校验不是形式主义 —— v3/v4 的 16B 向量访存要求 H%8==0,若不校验而是让
// kernel 越界读,症状是「大部分形状对、个别形状随机错」,极难定位。
void check(const at::Tensor& out, const at::Tensor& residual,
           const at::Tensor& x, const at::Tensor& w, bool need_vec) {
    TORCH_CHECK(out.is_cuda() && residual.is_cuda() && x.is_cuda() && w.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "only bf16 supported");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 &&
                x.scalar_type() == at::kBFloat16 &&
                w.scalar_type() == at::kBFloat16, "dtype mismatch");
    TORCH_CHECK(out.is_contiguous() && residual.is_contiguous() && x.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(x.sizes() == residual.sizes() && x.sizes() == out.sizes(),
                "shape mismatch");
    const int H = (int)x.size(-1);
    TORCH_CHECK(w.numel() == H, "weight length must equal last dim");
    if (need_vec)
        TORCH_CHECK(H % 8 == 0,
                    "v3/v4 需要 H%8==0(16B 向量访存);当前 H=", H);
}

template <FusedNormFn FN, bool NEED_VEC>
void run(at::Tensor out, at::Tensor residual, at::Tensor x, at::Tensor w, double eps) {
    check(out, residual, x, w, NEED_VEC);
    const int H = (int)x.size(-1);
    const int T = (int)(x.numel() / H);     // 前导维全部压平成 token 数:
                                            // [B,S,H] 与 [T,H] 走同一条路径
    FN(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
       reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
       reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
       T, H, (float)eps,
       // 取当前流而不是默认流:引擎里若开了 CUDA Graph 或多流,
       // 用默认流会导致捕获失败或隐式同步(triton-kernels#EXP-T05（CUDA Graph 消 launch 开销实测）的坑)。
       at::cuda::getCurrentCUDAStream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v0", &run<fused_add_rmsnorm_v0, false>, "unfused two-kernel baseline");
    m.def("v1", &run<fused_add_rmsnorm_v1, false>, "fused, scalar, smem reduce");
    m.def("v2", &run<fused_add_rmsnorm_v2, false>, "fused, warp shuffle reduce");
    m.def("v3", &run<fused_add_rmsnorm_v3, true>,  "vectorized 16B");
    m.def("v4", &run<fused_add_rmsnorm_v4, true>,  "vectorized + register cache");
}
