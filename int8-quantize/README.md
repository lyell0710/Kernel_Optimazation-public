# cuda-int8-quantize

INT8 per-channel symmetric quantize CUDA kernel 优化项目，保持和 `reduce/softmax/gemv` 一致的工程组织方式。

## 版本
- `baseline`： 单线程串行量化
- `v0`： grid-stride 一线程一元素
- `v1`： 每线程 2 元素展开 + restrict
- `v2`： 每线程 4 元素展开
- `v3`： 一块一 channel，scale 寄存器复用
- `v4`： float4 读 + char4 写向量化

## 构建运行
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/int8_quantize_bench
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
RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh
python project-proof/scripts/plot_ncu_summary.py
```

生成 `project-proof/profiling/ncu/int8_quantize_<tag>_profile.ncu-rep`（baseline、v0–v4）。**`QUANTIZE_PROFILE_ONLY`** 由脚本在采集时自动设置。

打包拷到 Mac：仓库根目录执行 `bash scripts/pack_ncu_reps_for_mac.sh`。

输出目录：`project-proof/profiling/ncu/` 图表目录：`project-proof/docs/figures/02-profiling/`

> 若提示 `ncu: command not found`，先安装 Nsight Compute（例如：`sudo apt install nsight-compute`）。
