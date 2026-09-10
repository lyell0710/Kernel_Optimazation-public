# CUDA FA2 forward 简化版版本梯(在线 softmax,D=128,causal+GQA)

与自家 Triton 版（triton-kernels#EXP-T01《Triton FA2 forward》）同协议对照的 CUDA 原生实现， 完整记录见 ../records/EXP-K03_cuda_fa2_ladder.md。

## 版本梯(S=4096,B=1,Hq=32,Hkv=8,causal,3 轮,数据权威 = project-proof/data/)

| 版本 | 优化点 | TFLOPS | 归因 |
|---|---|---|---|
| v0 | warp-per-row 在线 softmax | 4.9 |— |
| v1 | K/V tile 进 smem | 5.5 | +11%（L2 已扛住广播读）|
| v2 | wmma QK^T/PV + smem 往返 softmax | 24.4 | Tensor Core ×4.5 |
| v3 | 8 warp 并行组织 | 32.5 | +33% |
| v4 | half S/P + cp.async K 双缓冲/V 重叠 | **34.8** | 仅 +7.1% → 瓶颈在相位链 |

一句话读法：**wmma 做 GEMM 够到 cuBLAS 86%（EXP-K02《CUDA Tensor Core GEMM 版本梯》），做 FA2 只够到自家 Triton 版的 28%**——fragment 布局不透明逼出的 smem 往返吃掉融合优势， 这就是 FA2 官方实现要用 mma/CUTLASS 的定量理由。

## 复现

```bash
cmake -S . -B build && cmake --build build -j
BENCH_OUT=project-proof/data/$(date -u +%Y%m%dT%H%M)_fa2_proto_r1.csv ./build/fa2_bench
```

## 红线(措辞)

| 红线 | 状态 | 解锁条件 |
|---|---|---|
|「CUDA FA2 达到 sdpa/Triton 水平」 | 禁用（28%）| v5 mma PTX 路线 |
|「smem 往返是差距主因」 | 推断（v4 增量 7.1% 佐证）| NCU 计数器（容器不可用）|
| vs Triton/sdpa 数字 | 跨 harness 推断级 | 同 harness 复测 |
