# cuda-gemv

GEMV 手写 CUDA kernel 优化项目，采用和 `cuda-reduce` / `softmax` 一致的版本化实验方式。

## 版本
- `baseline`： 单线程行内累加
- `v0`： block 并行归约
- `v1`： 每线程双元素展开 + restrict
- `v2`： float4 向量化访存
- `v3`： 一行一 warp（shuffle 归约）
- `v4`： 多行共享 x-tile，降低重复读向量

## 构建运行
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gemv_bench
```

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
```

## NCU（完整 Section + 每版本独立 `.ncu-rep`）
```bash
bash project-proof/scripts/profile_ncu.sh
RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh   # 需要 plot / 扩展 CSV 时
python project-proof/scripts/plot_ncu_summary.py
```

生成 `project-proof/profiling/ncu/gemv_<tag>_profile.ncu-rep`（baseline、v0–v4）。Section 列表见 `Kernel_Optimazation/scripts/ncu_metrics.inc.sh`。采集时通过环境变量 **`GEMV_PROFILE_ONLY`** 仅跑单版本（由脚本自动设置）。

打包拷到 Mac：在仓库根目录 `bash scripts/pack_ncu_reps_for_mac.sh`，再按脚本提示 `scp` 各 ``.ncu-rep` 原件同目录*.tar.gz`。

输出目录：`project-proof/profiling/ncu/` 图表目录：`project-proof/docs/figures/02-profiling/`

> 若提示 `ncu: command not found`，先安装 Nsight Compute（例如：`sudo apt install nsight-compute`）。
