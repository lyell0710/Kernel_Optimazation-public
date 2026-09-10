# cuda-reduce

手写 CUDA reduction 优化实验项目，用于学习、基准测试和可复现记录。

## 项目目标
- 从 `baseline` 逐步优化到 `v6`
- 对比不同版本的性能与正确性
- 生成结构化数据（CSV）和图表（PNG）用于文档留档

## 当前实现版本
- `baseline`： 单线程 GPU 归约基线
- `v0`: shared-memory tree reduction
- `v1` / `v2`： 逐步优化的 block 内归约策略
- `v3`： 每线程处理两个元素，减少访存轮次
- `v4`： warp 尾归约优化，减少尾部同步开销
- `v5`： 在 `v4` 基础上做 block 内循环展开
- `v6`： 在 `v5` 基础上改为 grid-stride + two-pass 归约

## 最新基准快照（baseline ~ v6）
- 输入规模：`N = 1 << 24`
- baseline: `349.310730 ms`
- v0: `0.511450 ms` (`682.98x`)
- v1: `0.614215 ms` (`568.71x`)
- v2: `0.587037 ms` (`595.04x`)
- v3: `0.379432 ms` (`920.61x`)
- v4: `0.372975 ms` (`936.55x`)
- v5: `0.373608 ms` (`934.97x`)
- v6: `0.312438 ms` (`1118.02x`)
- 正确性：所有版本与 CPU 对齐（`Diff = 0`）

## 关键结论
- 当前最优版本为 `v6`，相对 `v5` 进一步提升约 `16.37%`。
- 当前运行环境为 WSL2，`Nsight Compute` 在该环境下不支持 kernel profiling；项目以稳定 benchmark + correctness 作为主证据，后续可在原生 Linux/Windows 环境补齐 occupancy/throughput/warp 指标。

> 说明：基准结果会随设备温度、功耗策略、后台负载出现小幅波动。

## 目录结构
- `src/`： 各版本 kernel 与 benchmark 入口
- `include/`： 公共声明
- `project-proof/data/`： 基准与环境 CSV
- `project-proof/scripts/`： 画图脚本
- `project-proof/docs/`： 图表与实验总结

## 构建与运行
```bash
cmake -S . -B build
cmake --build build -j
./build/reduce_bench
```

## Nsight Compute（全指标 + 分版本 `.ncu-rep`）
- 脚本：`project-proof/scripts/profile_ncu.sh`（Section：SpeedOfLight、LaunchStats、Occupancy、WarpStateStats、MemoryWorkloadAnalysis*、SchedulerStats）
- 对每个版本生成 `project-proof/profiling/ncu/reduce_<tag>_profile.ncu-rep`；模板 kernel（v4–v7）使用 `regex:` 匹配。
- 环境变量：`BENCH_ITERS`（默认 1）、`RUN_NCU_CSV=1` 导出 `reduce_ncu.csv`（与仓库根目录 `scripts/ncu_metrics.inc.sh` 中扩展 metrics 一致）、`REDUCE_PROFILE_ONLY` 由脚本自动设置（勿手改）。
- 汇总图：`RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh` 后执行 `python project-proof/scripts/plot_ncu_summary.py`
- 全仓库一键（含本工程）：在 `Kernel_Optimazation` 根目录执行 `bash scripts/run_ncu_all.sh`（默认 `RUN_NCU_CSV=1`；仅要 `.ncu-rep` 时可 `RUN_NCU_CSV=0`）

## 基准流程说明
- `main.cu` 采用多次迭代计时取均值；默认 `100` 次。若设置环境变量 `BENCH_ITERS`（例如 NCU 采集时为 `1`），则以该值为准。
- 每次运行 `reduce_bench` 会自动覆盖刷新：`project-proof/data/benchmark_results.csv`

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_latency_line.py
python project-proof/scripts/plot_correctness.py
```

## 相关文档
- 基准数据：`project-proof/data/benchmark_results.csv`
- 实验总结：`project-proof/docs/experiment-summary.md`
