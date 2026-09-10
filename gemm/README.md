# CUDA Tensor Core GEMM 版本梯(fp16 in / fp32 acc,4096³)

对标真 cuBLAS（`cublasGemmEx`，调用点验真）的手写版本梯，完整记录见 ../records/EXP-K02_cuda_gemm_tc_ladder.md。

## 版本梯与关键数字(4090,3 轮,数据权威 = project-proof/data/)

| 版本 | 优化点 | TFLOPS | vs cuBLAS |
|---|---|---|---|
| v0 | naive,1 thread/输出 | 5.2 | 3.4% |
| v1 | 32×32 smem tile(CUDA core) | 6.5 | 4.2% |
| v2 | wmma 16³ fragment,4 warp | 89.5 | 57.6% |
| v3 | + cp.async 双缓冲 | 95.5 | 61.4% |
| v4 | + 128² 大 tile,8 warp | **133.1** | **85.6%** |

一句话读法：fp16 GEMM 的性能台阶是指令世代（v1→v2 ×13.8），不是访存微调（v0→v1 仅 +25%）；v4 理论 occupancy 33% 全梯最低却最快（ILP > 线程数）。

## 复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
BENCH_OUT=project-proof/data/$(date -u +%Y%m%dT%H%M)_gemm4096_r1.csv ./build/gemm_bench
```

## 红线(措辞)

| 红线 | 状态 | 解锁条件 |
|---|---|---|
|「超过/追平 cuBLAS」 | 禁用（85.6%） | 无（Triton 版数字不得挪用） |
|「swizzle/bank conflict 是剩余差距主因」 | 推断，不可当实测说 | NCU 计数器（容器内不可用） |
