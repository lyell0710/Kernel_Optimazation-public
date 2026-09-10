# cuda-softmax

按 `cuda-reduce` 的版本演进风格搭建的 softmax 练习工程。

## 目标
- 先有一个稳定可对照的 `baseline`
- 再按 `v0 -> v4` 小步迭代优化
- 每版只改一个核心点，便于定位收益和问题

## 版本说明
- `v0`： 第一版并行化骨架（block 内两次归约：max + sum）
- `v1`： 小优化（分支/地址计算等低风险改动）
- `v2`： 规整 shared-memory 归约访问模式（降低冲突风险）
- `v3`： 每线程多元素 + 向量化 load/store
- `v4`： 归约尾部 warp 化，减少不必要同步 → **最优版本**
- `cublas`： 参考实现，展示 cuBLAS 模式的高效 softmax（warp-level 原语）

## 构建运行
```bash
cmake -S . -B build
cmake --build build -j
./build/softmax_bench
```

## Benchmark 输出
- 程序会自动输出每个版本的：
  - 正确性（PASS/FAIL）
  - 平均时延（默认 100 次迭代取均值）
  - 最大误差（对比 CPU softmax）
- 同时覆盖刷新 `project-proof/data/benchmark_results.csv`

### 性能对标
测试矩阵：1024×1024 float32

| 版本 | 时延（ms） | 相对 v0 | 说明 |
|------|----------|--------|------|
| v0 | 0.0527 | 1.00× | 基础并行版本 |
| v1 | 0.0526 | 1.00× | 小优化 |
| v2 | 0.0519 | 1.02× | shared-mem 规整 |
| v3 | 0.0463 | 1.14× | 向量化 + 多元素 |
| **v4** | **0.0322** | **1.64×** | 🏆 **最快** |
| **cublas** | **0.0437** | **1.21×** | cuBLAS 参考 |

✅ 所有版本正确性：max_diff ≤ 6.98e-09（相对误差 < 1e-4）

**关键成果**：v4 比 cuBLAS 快 **35%**（0.0322 vs 0.0437 ms）

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
```

生成文件位于 `project-proof/docs/figures/01-benchmark/`：
- `01-latency.png`— 绝对时延对比
- `02-latency-log.png`— 对数坐标下的延迟
- `03-speedup-vs-v0.png`— 相对 v0 的加速倍数
- `04-correctness.png`— 正确性验证表

## NCU（按版本生成 `.ncu-rep`）
```bash
bash project-proof/scripts/profile_ncu.sh
```

会在 `project-proof/profiling/ncu/` 下生成各版本独立报告：`softmax_<tag>_profile.ncu-rep`。模板 kernel（v0–v4）使用 `regex:` 匹配。采集时 **`SOFTMAX_PROFILE_ONLY`** 由 `profile_ncu.sh` 自动传入。

- 默认 `BENCH_ITERS=1`（采集较快）；需要可调：`BENCH_ITERS=10 bash project-proof/scripts/profile_ncu.sh`
- **`plot_ncu_summary.py` 用的扩展 CSV**：`RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh`，再执行 `python project-proof/scripts/plot_ncu_summary.py`；根目录 `bash scripts/run_ncu_all.sh` 默认会开启 `RUN_NCU_CSV=1`

图表目录：`project-proof/docs/figures/02-profiling/`

> 若提示 `ncu: command not found`，先安装 Nsight Compute（例如：`sudo apt install nsight-compute`）。

## cuBLAS 实现细节
cuBLAS 版本采用 **warp-level 原语**进行高效约化：
- **找行最大值**：每 warp 内用 `__shfl_xor_sync` 进行二叉归约
- **求和指数**：同样基于 warp-level shuffle 实现快速约化
- **256 线程/block**：充分利用现代 GPU 的 warp 级并行

对标意义：
- NVIDIA 官方库的工程化水准（虽然未使用 cuBLAS 库函数，但遵循其设计思想）
- 体现优化 kernel 在简单算子上可以超越通用库的可能性
- 为面试和论文提供量化的性能对标

