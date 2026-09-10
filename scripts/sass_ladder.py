#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""按版本梯逐 kernel 输出 SASS 判据表——编译期证据链,与运行时计数器互补。

## 为什么需要它

计数器与 SASS 答的不是同一类问题,两者互补而不互相替代:
仓内已有 51 份 RTX 4090 / CUDA 12.8 的 `.ncu-rep`(闭环记录 = EXP-K07
《采集主机 NCU 计数器闭环》),它们给运行时观测量;本脚本给编译期事实。
SASS 在三件事上比计数器更硬:

1. **它是编译期事实,不是采样。** `cuobjdump` 读的是 cubin 里已经定死的机器码,
   没有采样窗口、没有重放、没有 warm-up 差异,同一个二进制读一万次结果相同。
2. **它不需要任何权限。** `cuobjdump` / `nvdisasm` / `ptxas` 是纯静态工具,
   不初始化 CUDA、不打开设备节点、不碰 GPU。在共享机器上跑它对别人零影响。
3. **它能证伪源码注释。** 「我写了 float4 所以向量化了」是意图;
   `LDG.E.128` 出现几条是事实。本仓曾有五处「16 B 向量化」的记载被这一条判死
   (见 `docs/sass_evidence_ladder.md` §6.1;同一结论后来由计数器侧的
   L1TEX 扇区数独立复现,见 EXP-K07 §6.4)。其中 `fused-norm` 已按同一判据
   定位根因并改好,重编后 v3 的 `LDG.E.128` 由 0 变 4;剩下四处仍未兑现。
   同一条判据既能判死也能验收,验收全程静态、不碰 GPU。

## 它答不了什么

SASS 是**静态条数**,不是动态执行次数,更不是时间。以下一概答不了,
别拿本脚本的输出去回答它们:

- **时延 / 带宽 / TFLOPS**——权威在 `records/` 各 EXP 的 `data/` 指针。
- **动态指令数**——静态条数 × 循环次数才是动态量,而循环次数在运行时边界的
  循环里根本不是编译期常量。凡涉及运行时上界的循环,静态条数只反映展开体大小。
- **DRAM 字节数**——一条 LDG 是一次 LSU 请求,不是一次显存访问。warp 内合并、
  L1/L2 命中都在指令之后发生。指令条数降 32 倍不等于显存流量降 32 倍
  (`gemm` v0→v1 就是这个坑)。
- **cache 命中率 / bank conflict / sector 利用率 / stall 分解 / achieved occupancy**
  ——本质上就是计数器量,SASS 里没有承载字段。这些数在仓内 `.ncu-rep` 里,
  本机 `ncu --import` 纯读文件即可取出(不需要 profiling 权限),
  读法与口径见 `docs/sass_evidence_ladder.md` §7.2。
- **launch 次数与 launch 开销**——host 侧行为,cubin 里没有。用 nsys。

理论 occupancy 不在此列:它由编译期资源(REG/SMEM)+ 架构常数算出,
本脚本给出前者,算法见 `docs/sass_evidence_ladder.md`。

## 计数口径(读数字前必看,踩过的坑都在这)

- **按 Function 段精确切分,不裸 grep。** 本仓多处 kernel 名互为子串
  (`v1_kernel` ⊂ `dequant_v1_kernel` ⊂ `gemv_v1_kernel`),裸 `grep -c` 必然串味。
  段的右边界是下一个 `Function :` 或下一个 fatbin 段头,后者不加会把下一个
  fatbin 的内容吞进来。
- **谓词恒假的死指令单列。** ptxas 在 cp.async 代码里会插 `@!PT LDS RZ, [RZ]`
  这类占位指令,计入总条数但永不执行。`gemm` v3/v4 各 12 条、`flash-attn` v4 有 9 条。
  裸数 LDS 会得出「v3/v4 还在用 LDS」的错误结论。本脚本的所有判据只计活指令,
  死指令与 NOP 填充单列。
- **`LDS` 与 `LDSM` 必须分开。** `LDSM`(ldmatrix)以 `LDS` 为前缀,
  `grep -E '^LDS'` 会把它算进普通 shared load。
- **`BAR.SYNC` 与 `BSSY`/`BSYNC` 必须分开。** 后者是分支重收敛指令不是 barrier,
  混在一个键里会把 `__syncthreads` 的条数数错。
- **`BRA` 的后缀变体各占一个键。** `BRA.CONV` 是子程序调用点的收敛跳转
  (与 `CALL.REL.NOINC` 成对出现),`BRA.DIV` 是发散跳转,都不是算法自身的循环
  与条件分支。把它们并进 `BRA` 会凭空造出分支:`cuda-reduce` v4 并进去是 14,
  拆开是 `BRA` 2 + `BRA.CONV` 12——后者全部来自那个没被内联的 warp 归约 helper。
- **全局访存按宽度分桶,含 16 位与 8 位桶。** 本仓大量算子用 bf16/half 标量访存,
  只看 32/64/128 三档会得出「这个 kernel 不访存」的错误结论。
- **IMAD 是整个家族**(含 `IMAD.MOV.U32` 这类借 FMA 管线的搬运),不是纯整数乘加。
- **`-res-usage` 的 SHARED 只算静态 smem。** 用 `extern __shared__` 的 kernel
  这里恒为 0,真实用量在 launch 语句里,本脚本看不到,输出会标 `dyn?`。
- **arch 会骗人。** `softmax` / `gemv` / `int8-quantize` / `cuda-reduce` 四个二进制
  是 `sm_75` 编的(CMake 默认值),4090 上执行的是驱动 JIT 出来的 sm_89 码,
  不是这里读到的这份。本脚本逐段打印 arch,非 sm_89 会告警。
  `--recheck-sm89` 把内嵌 PTX 抽出来用本机 `ptxas -arch=sm_89 -O3` 重编再逐项对比,
  给出「哪些判据跨 arch 稳定、哪些不稳定」的实测清单(全程静态)。
  已知结论:访存/同步/浮点类判据逐项一致,**活指令总数、`IMAD*`、`BRA` 不一致**,
  寄存器数差别可以很大(`softmax` v4.3 是 62 对 40)。凡以后三类为论据的结论
  必须标注「基于未执行的 sm_75 码」。

用法:
    python scripts/sass_ladder.py                 # 十个算子全表
    python scripts/sass_ladder.py --op gemm       # 只看一个
    python scripts/sass_ladder.py --json          # 机器可读(含完整助记符直方图)
    python scripts/sass_ladder.py --op w8a8 --mnemonics   # 逐 kernel 打全助记符直方图
    python scripts/sass_ladder.py --recheck-sm89  # 非 sm_89 产物的重编等价性核验
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()

def torch_ext_so(name: str) -> Path:
    """定位 torch cpp_extension 编出的 .so,不写死 CUDA 版本目录。

    路径形如 ~/.cache/torch_extensions/py<pyver>_cu<cudaver>/<name>/<name>.so,
    其中 cu<ver> 跟着当前 torch 的 CUDA 版本走。写死一个版本的后果已经发生过:
    torch 从 cu132 换到 cu130 后,脚本仍在读 8-26 的旧 .so,于是"向量化未兑现"
    这个结论在源码已修好之后还在继续输出。取 mtime 最新的那个。
    """
    base = HOME / ".cache/torch_extensions"
    cands = sorted(base.glob(f"py*_cu*/{name}/{name}.so"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return cands[0] if cands else base / f"(未构建)/{name}/{name}.so"

# --------------------------------------------------------------------------
# 表驱动的产物路径与版本梯。
#
# `ladder` 里存的是**完整 mangled 名**而不是前缀:本仓多处 kernel 名互为子串,
# 前缀匹配会把 `v1_kernel` 匹到 `dequant_v1_kernel` 上。名字随重编译而变的风险
# 由 `--- 未列入版本梯 ---` 那一节兜底:二进制里出现而表里没有的 Function 会被
# 列出来,不会静默消失。
# --------------------------------------------------------------------------
OPS = {
    "gemm": dict(
        path=ROOT / "gemm/build/gemm_bench",
        build="cd gemm && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="fp16 GEMM 4096³。cuBLAS 臂是真库调用,其 kernel 不在本二进制内。",
        ladder=[
            ("v0", "朴素", "_Z14gemm_v0_kernelPK6__halfS1_PS_iii"),
            ("v1", "smem tile", "_Z14gemm_v1_kernelPK6__halfS1_PS_iii"),
            ("v2", "wmma", "_Z14gemm_v2_kernelPK6__halfS1_PS_iii"),
            ("v3", "cp.async 双缓冲", "_Z14gemm_v3_kernelPK6__halfS1_PS_iii"),
            ("v4", "128² tile", "_Z14gemm_v4_kernelPK6__halfS1_PS_iii"),
        ],
    ),
    "flash-attn": dict(
        path=ROOT / "flash-attn/build/fa2_bench",
        build="cd flash-attn && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="FA2 prototype。v2/v3/v4 用 extern __shared__,静态 SHARED 恒为 0。",
        ladder=[
            ("v0", "warp-per-row", "_Z13fa2_v0_kernelPK6__halfS1_S1_PS_iiib"),
            ("v1", "K/V 进 smem", "_Z13fa2_v1_kernelPK6__halfS1_S1_PS_iiib"),
            ("v2", "wmma QK^T/PV", "_Z13fa2_v2_kernelPK6__halfS1_S1_PS_iiib"),
            ("v3", "8 warp", "_Z13fa2_v3_kernelPK6__halfS1_S1_PS_iiib"),
            ("v4", "cp.async + half S/P", "_Z13fa2_v4_kernelPK6__halfS1_S1_PS_iiib"),
            ("ref", "fp32 两遍参考(非梯)", "_Z10ref_kernelPK6__halfS1_S1_PS_iiib"),
        ],
    ),
    "softmax": dict(
        path=ROOT / "softmax/build/softmax_bench",
        build="cd softmax && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="二进制为 sm_75(CMake 默认 arch)。`cublas` 臂是自写 kernel,不是 cuBLAS。",
        ladder=[
            ("baseline", "单线程一行", "_ZN52_GLOBAL__N__72f3ce17_19_softmax_baseline_cu_7304c94623softmax_baseline_kernelEPKfPfii"),
            ("v0", "block 归约", "_ZN46_GLOBAL__N__689e725f_13_softmax_v0_cu_15e0781f17softmax_v0_kernelILi256EEEvPKfPfii"),
            ("v1", "restrict + 位运算", "_ZN46_GLOBAL__N__d022153a_13_softmax_v1_cu_d94a788117softmax_v1_kernelILi256EEEvPKfPfii"),
            ("v2", "对半收缩归约", "_ZN46_GLOBAL__N__c297bad4_13_softmax_v2_cu_57c57f6217softmax_v2_kernelILi256EEEvPKfPfii"),
            ("v3", "float4 + 多元素", "_ZN46_GLOBAL__N__7a2bddb1_13_softmax_v3_cu_9b6f7ffc17softmax_v3_kernelILi256ELi4EEEvPKfPfii"),
            ("v4", "warp shuffle 尾部", "_ZN46_GLOBAL__N__e7fce508_13_softmax_v4_cu_91aa76e517softmax_v4_kernelILi256ELi4EEEvPKfPfii"),
            ("v4.2", "反例:退回标量+全同步", "_ZN48_GLOBAL__N__e035b4c2_15_softmax_v4_2_cu_d952f82719softmax_v4_2_kernelILi256ELi2EEEvPKfPfii"),
            ("v4.3", "探索:main+tail 分离", "_ZN48_GLOBAL__N__5889d3a7_15_softmax_v4_3_cu_15f8f8b919softmax_v4_3_kernelILi256ELi4EEEvPKfPfii"),
            ("v4.4", "反例:只破坏归约", "_ZN48_GLOBAL__N__c55eeb1e_15_softmax_v4_4_cu_1f3df1a019softmax_v4_4_kernelILi256ELi4EEEvPKfPfii"),
            ("handwritten", "误名为 cublas 的自写臂", "_ZN50_GLOBAL__N__0ca5bc4a_17_softmax_cublas_cu_d7a800bd21softmax_cublas_kernelEPKfPfii"),
        ],
    ),
    "gemv": dict(
        path=ROOT / "gemv/build/gemv_bench",
        build="cd gemv && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="二进制为 sm_75。cuBLAS 臂是真库调用(gemv2T_kernel_val),不在本二进制内。",
        ladder=[
            ("baseline", "一行一线程", "_ZN49_GLOBAL__N__cdff1bbb_16_gemv_baseline_cu_187c6a2c20gemv_baseline_kernelEPKfS1_Pfii"),
            ("v0", "block 归约", "_ZN43_GLOBAL__N__c64c7716_10_gemv_v0_cu_3ea0541014gemv_v0_kernelILi256EEEvPKfS2_Pfii"),
            ("v1", "unroll2 + restrict", "_ZN43_GLOBAL__N__7ef01073_10_gemv_v1_cu_ff2e8bd014gemv_v1_kernelILi256EEEvPKfS2_Pfii"),
            ("v2", "float4", "_ZN43_GLOBAL__N__6c45bf9d_10_gemv_v2_cu_66ccedd114gemv_v2_kernelILi256ELi4EEEvPKfS2_Pfii"),
            ("v3", "一行一 warp(shuffle)", "_ZN43_GLOBAL__N__d4f9d8f8_10_gemv_v3_cu_a742321114gemv_v3_kernelEPKfS1_Pfii"),
            ("v4", "多行共享 x-tile", "_ZN43_GLOBAL__N__492ee041_10_gemv_v4_cu_8e79279214gemv_v4_kernelILi128ELi4EEEvPKfS2_Pfii"),
        ],
    ),
    "int8-quantize": dict(
        path=ROOT / "int8-quantize/build/int8_quantize_bench",
        build="cd int8-quantize && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="二进制为 sm_75。baseline 是 <<<1,1>>> 的 CPU 级实现,倍数不可对外引用(EXP-K01 §7)。",
        ladder=[
            ("baseline", "<<<1,1>>> 串行", "_ZN53_GLOBAL__N__3230317f_20_quantize_baseline_cu_61be501f24quantize_baseline_kernelEPKfS1_Paii"),
            ("v0", "grid-stride", "_ZN47_GLOBAL__N__a11c91e7_14_quantize_v0_cu_0f0ade7e18quantize_v0_kernelEPKfS1_Paii"),
            ("v1", "unroll2 + restrict", "_ZN47_GLOBAL__N__19a0f682_14_quantize_v1_cu_ce8401be18quantize_v1_kernelEPKfS1_Paii"),
            ("v2", "unroll4", "_ZN47_GLOBAL__N__0b15596c_14_quantize_v2_cu_576667bf18quantize_v2_kernelEPKfS1_Paii"),
            ("v3", "一 block 一 channel", "_ZN47_GLOBAL__N__b3a93e09_14_quantize_v3_cu_96e8b87f18quantize_v3_kernelEPKfS1_Paii"),
            ("v4", "float4 向量化", "_ZN47_GLOBAL__N__2e7e06b0_14_quantize_v4_cu_bfd3adfc18quantize_v4_kernelEPKfS1_Paii"),
        ],
    ),
    "cuda-reduce": dict(
        path=ROOT / "cuda-reduce/build/reduce_bench",
        build="cd cuda-reduce && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j",
        note="二进制为 sm_75。CUB 臂是真库(header-only,与自家 kernel 同编在本二进制内,可逐指令对照)。"
             "自写八版全梯 LDG.128=0、SHFL=0;v6 与 v7 的 SASS 逐助记符完全相同。",
        ladder=[
            ("baseline", "单线程串行", "_Z22reduce_baseline_kernelPKfPfi"),
            ("v0", "交错寻址,tid % (2*i)", "_ZN51_GLOBAL__N__9acffe79_18_reduce_naive_v0_cu_42cf9d1716reduce_v0_kernelILi256EEEvPKfPfi"),
            ("v1", "同 v0,% 换 &", "_ZN45_GLOBAL__N__6607feac_12_reduce_v1_cu_e4b896a316reduce_v1_kernelILi256EEEvPKfPfi"),
            ("v2", "对半收缩", "_ZN45_GLOBAL__N__74b25142_12_reduce_v2_cu_d5508c3e16reduce_v2_kernelILi256EEEvPKfPfi"),
            ("v3", "首次加载即相加(2 元素)", "_ZN45_GLOBAL__N__cc0e3627_12_reduce_v3_cu_7327878a16reduce_v3_kernelILi256EEEvPKfPfi"),
            ("v4", "末 warp volatile+__syncwarp", "_ZN45_GLOBAL__N__51d90e9e_12_reduce_v4_cu_b680b90416reduce_v4_kernelILi256EEEvPKfPfi"),
            ("v5", "归约全展开", "_ZN45_GLOBAL__N__e96569fb_12_reduce_v5_cu_10f7b2b016reduce_v5_kernelILi256EEEvPKfPfi"),
            ("v6", "grid-stride + two-pass", "_ZN45_GLOBAL__N__fbd0c615_12_reduce_v6_cu_211fa82d16reduce_v6_kernelILi256EEEvPKfPfi"),
            ("v7", "源码与 v6 同构", "_ZN45_GLOBAL__N__436ca170_12_reduce_v7_cu_8768a39916reduce_v7_kernelILi256EEEvPKfPfi"),
            ("cub:tile", "库对照 SingleTile", "_ZN3cub16_V_300200_SM_7506detail6reduce28DeviceReduceSingleTileKernelINS2_10policy_hubIfjN4cuda3std3__44plusIvEEE10Policy1000EPfSC_iS9_ffNS7_8identityEEEvT0_T1_T2_T3_T4_T6_"),
            ("cub:main", "库对照 DeviceReduce", "_ZN3cub16_V_300200_SM_7506detail6reduce18DeviceReduceKernelINS2_10policy_hubIfjN4cuda3std3__44plusIvEEE10Policy1000EPKfjS9_fNS7_8identityEEEvT0_PT3_T1_NS0_13GridEvenShareISI_EET2_T4_"),
            ("cub:final", "库对照 SingleTile(收尾)", "_ZN3cub16_V_300200_SM_7506detail6reduce28DeviceReduceSingleTileKernelINS2_10policy_hubIfjN4cuda3std3__44plusIvEEE10Policy1000EPKfPfjS9_ffNS7_8identityEEEvT0_T1_T2_T3_T4_T6_"),
        ],
    ),
    "fused-norm": dict(
        path=torch_ext_so("fused_norm_ext"),
        build="cd fused-norm && python bench.py   # torch JIT 扩展,首次运行即编译(会跑 GPU)",
        note="H=4096 时 bench 只派发到 v4<1>,<2>/<4>/<8> 编进 .so 但从未执行。"
             "v3/v4 的 BF16x8 已改为 union{float4 raw;} + 显式拷贝语义,LDG.E.128 已出现;"
             "rope/activation/w8a8 的同名载体尚未改。",
        ladder=[
            ("v0.add", "两 kernel 之一", "_Z13v0_add_kernelP13__nv_bfloat16PKS_x"),
            ("v0.rms", "两 kernel 之二", "_Z17v0_rmsnorm_kernelP13__nv_bfloat16PKS_S2_if"),
            ("v1", "融合 + smem 树归约", "_Z9v1_kernelP13__nv_bfloat16S0_PKS_S2_if"),
            ("v2", "warp shuffle 归约", "_Z9v2_kernelP13__nv_bfloat16S0_PKS_S2_if"),
            ("v3", "16B 向量化(已修复)", "_Z9v3_kernelP13__nv_bfloat16S0_PKS_S2_if"),
            ("v4<1>", "寄存器缓存(唯一被跑到)", "_Z9v4_kernelILi1EEvP13__nv_bfloat16S1_PKS0_S3_if"),
            ("v4<2>", "未被跑到", "_Z9v4_kernelILi2EEvP13__nv_bfloat16S1_PKS0_S3_if"),
            ("v4<4>", "未被跑到", "_Z9v4_kernelILi4EEvP13__nv_bfloat16S1_PKS0_S3_if"),
            ("v4<8>", "未被跑到", "_Z9v4_kernelILi8EEvP13__nv_bfloat16S1_PKS0_S3_if"),
        ],
    ),
    "rope": dict(
        path=torch_ext_so("rope_ext"),
        build="cd rope && python bench.py   # torch JIT 扩展,首次运行即编译(会跑 GPU)",
        note="全梯 smem 恒为 0;每版都带两段 64 位整数除法的模拟子程序(MUFU.RCP 来源)。",
        ladder=[
            ("v0", "一线程一元素 + 临时缓冲", "_Z14v0_kernel_safeP13__nv_bfloat16PKS_S2_S2_iii"),
            ("v1", "一线程一对(就地)", "_Z9v1_kernelP13__nv_bfloat16PKS_S2_iii"),
            ("v2", "q/k 合并一次 launch", "_Z9v2_kernelP13__nv_bfloat16S0_PKS_S2_iiiixx"),
            ("v3", "自称 16B 向量化", "_Z9v3_kernelP13__nv_bfloat16S0_PKS_S2_iiixx"),
            ("v4", "免表 __sincosf 现算", "_Z9v4_kernelP13__nv_bfloat16S0_PKfiiiixx"),
        ],
    ),
    "activation": dict(
        path=torch_ext_so("activation_ext"),
        build="cd activation && python bench.py   # torch JIT 扩展,首次运行即编译(会跑 GPU)",
        note="全梯 smem 恒为 0、无 barrier。silu 里的 `/` 编成了完整 IEEE 除法慢路径。",
        ladder=[
            ("v0.silu", "两 kernel 之一", "_Z14v0_silu_kernelP13__nv_bfloat16PKS_x"),
            ("v0.mul", "两 kernel 之二", "_Z13v0_mul_kernelP13__nv_bfloat16PKS_S2_x"),
            ("v1", "融合", "_Z9v1_kernelP13__nv_bfloat16PKS_S2_x"),
            ("v2", "自称 16B 向量化", "_Z9v2_kernelP13__nv_bfloat16PKS_S2_x"),
            ("v3", "打包布局(vLLM 风格)", "_Z9v3_kernelP13__nv_bfloat16PKS_i"),
        ],
    ),
    "w8a8": dict(
        path=torch_ext_so("w8a8_ext"),
        build="cd w8a8 && python bench.py   # torch JIT 扩展,首次运行即编译(会跑 GPU)",
        note="三条链共处一个 .so,名字互为子串(v1_kernel ⊂ dequant_v1_kernel ⊂ gemv_v1_kernel),"
             "必须按 Function 段切。INT8 GEMM 走 cuBLASLt,其 kernel 不在本 .so 内。",
        ladder=[
            ("q.v0.absmax", "量化链:两 kernel 之一", "_Z16v0_absmax_kernelPfPK13__nv_bfloat16i"),
            ("q.v0.quant", "量化链:两 kernel 之二", "_Z15v0_quant_kernelPaPK13__nv_bfloat16PKfi"),
            ("q.v1", "量化链:融合 + shuffle", "_Z9v1_kernelPaPfPK13__nv_bfloat16i"),
            ("q.v2", "量化链:自称 16B 读 / 8B 写", "_Z9v2_kernelPaPfPK13__nv_bfloat16i"),
            ("dq.v0", "反量化链:逐元素", "_Z17dequant_v0_kernelP13__nv_bfloat16PKiPKfS4_ii"),
            ("dq.v1", "反量化链:一行一 block", "_Z17dequant_v1_kernelP13__nv_bfloat16PKiPKfS4_i"),
            ("gv.v0", "INT8 GEMV:dp4a", "_Z14gemv_v0_kernelILi8EEvP13__nv_bfloat16PKaS3_PKfS5_ii"),
            ("gv.v1", "INT8 GEMV:+ smem 激活", "_Z14gemv_v1_kernelILi8EEvP13__nv_bfloat16PKaS3_PKfS5_ii"),
        ],
    ),
}

# --------------------------------------------------------------------------
# 反汇编解析
# --------------------------------------------------------------------------
# 一条指令行形如:
#     /*0a30*/            @!P1 LDG.E.128 R4, desc[UR6][R2.64] ;   /* 0x... */
# 段边界:下一个 `Function :`,或下一个 fatbin 段头(不认后者会把下一个 fatbin
# 的内容吞进当前段——w8a8 的三个编译单元就是这么串味的)。
FUNC_RE = re.compile(r"^\s*Function : (\S+)\s*$")
INST_RE = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
ARCH_RE = re.compile(r"^\s*(?:code for|arch\s*=\s*)\s*(sm_\d+)")
FATBIN_RE = re.compile(r"^\s*Fatbin \w+ code:")
PRED_RE = re.compile(r"^@!?(P\d|PT)\s+")
RES_RE = re.compile(r"^\s*Function (\S+):\s*$")
RES_VAL_RE = re.compile(r"(\w+(?:\[\d+\])?):(\d+)")


def run(cmd):
    """跑一个静态工具,返回 stdout。这些工具都不碰 GPU。"""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0 and not p.stdout:
        raise RuntimeError(f"{' '.join(cmd[:2])} 失败: {p.stderr.strip()[:400]}")
    return p.stdout


def parse_sass(text):
    """把 `cuobjdump -sass` 的输出切成 {mangled: {...}}。

    每段记两个 Counter:活指令与死指令(谓词恒假 `@!PT`)的**完整助记符**直方图。
    所有判据都从这两个直方图派生,不再二次 grep 原文——这样口径只有一处,
    改口径不会出现「表里改了、notes 没改」的分叉。
    """
    segs, cur, arch = {}, None, "?"
    for line in text.splitlines():
        m = ARCH_RE.match(line)
        if m:
            arch = m.group(1)
            continue
        if FATBIN_RE.match(line):
            cur = None                      # fatbin 段头 = 当前 Function 段的右边界
            continue
        m = FUNC_RE.match(line)
        if m:
            cur = segs.setdefault(m.group(1), dict(
                arch=arch, live=Counter(), dead=Counter(), depbar=[]))
            continue
        if cur is None:
            continue
        m = INST_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        pm = PRED_RE.match(body)
        dead = bool(pm and pm.group(0).startswith("@!PT"))
        if pm:
            body = body[pm.end():]
        mn = body.split()[0] if body.split() else ""
        (cur["dead"] if dead else cur["live"])[mn] += 1
        # cp.async 的流水级数是写死在 DEPBAR 立即数里的编译期常量
        # (`__pipeline_wait_prior(n)` -> `DEPBAR.LE SB0, 0xn`),
        # 助记符直方图丢掉操作数,这里单独留一份。
        if not dead and mn.startswith("DEPBAR"):
            cur["depbar"].append(" ".join(body.split()))
    return segs


def parse_res(text):
    """把 `cuobjdump -res-usage` 的输出切成 {mangled: {REG:.., SHARED:.., ..}}。"""
    out, cur = {}, None
    for line in text.splitlines():
        m = RES_RE.match(line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        if cur is not None and ":" in line and "REG:" in line:
            for k, v in RES_VAL_RE.findall(line):
                cur[k] = int(v)
            cur = None
    return out


# --------------------------------------------------------------------------
# 判据派生
# --------------------------------------------------------------------------
def width_of(mn):
    """全局/共享访存指令的单条宽度(bit)。裸 `LDG.E` 是 32 位。"""
    parts = set(mn.split("."))
    for w in ("128", "64", "32"):
        if w in parts:
            return w
    if parts & {"U16", "S16"}:
        return "16"
    if parts & {"U8", "S8"}:
        return "8"
    return "32"


def sig_of(live):
    """从活指令直方图派生判据签名。每一项的口径都写在 docstring 的「计数口径」里。"""
    s = Counter()
    for mn, n in live.items():
        head = mn.split(".")[0]
        # --- Tensor Core / 矩阵管线 ---
        if head == "HMMA":
            s["HMMA"] += n
        elif head == "IMMA":
            s["IMMA"] += n
        elif head == "OMMA":
            s["OMMA"] += n
        elif head == "IDP" and mn.startswith("IDP.4A"):
            s["DP4A"] += n
        elif head == "LDSM":                       # ldmatrix,必须与 LDS 分开
            s["LDSM"] += n
        # --- cp.async 流水 ---
        elif head == "LDGSTS":
            s["LDGSTS"] += n
        elif head == "LDGDEPBAR":
            s["LDGDEPBAR"] += n
        elif head == "DEPBAR":
            s["DEPBAR"] += n
        # --- 访存,按宽度分桶 ---
        elif head == "LDG":
            s["LDG." + width_of(mn)] += n
        elif head == "STG":
            s["STG." + width_of(mn)] += n
        elif head == "LDS":                        # LDSM 已在上面截走
            s["LDS"] += n
        elif head == "STS":
            s["STS"] += n
        elif head in ("LDL", "STL"):               # local memory = 寄存器溢出
            s["LOCAL"] += n
        # --- 归约 / 同步 / 特殊函数 ---
        elif head == "SHFL":
            s["SHFL"] += n
        elif head == "MUFU":
            s["MUFU"] += n
        elif mn.startswith("BAR.SYNC"):            # BSSY/BSYNC 不是 barrier
            s["BAR.SYNC"] += n
        elif head in ("BSSY", "BSYNC"):
            s["BSSY"] += n
        # --- 控制流 / 算术 ---
        elif head == "BRA":
            # BRA.CONV(子程序收敛跳转)/ BRA.DIV(发散跳转)不是算法自身的分支,
            # 并进 BRA 会凭空造出分支条数,各占一个键。
            parts = mn.split(".", 1)
            s["BRA" if len(parts) == 1 else "BRA." + parts[1]] += n
        elif head in ("IMAD", "UIMAD"):
            s["IMAD*"] += n
        elif head == "FFMA":
            s["FFMA"] += n
    return s


def digest(path):
    """读一个二进制/so,返回 {mangled: 记录}。纯静态,不碰 GPU。"""
    segs = parse_sass(run(["cuobjdump", "-sass", str(path)]))
    res = parse_res(run(["cuobjdump", "-res-usage", str(path)]))
    out = {}
    for name, seg in segs.items():
        live, dead = seg["live"], seg["dead"]
        nop = live.pop("NOP", 0)               # 尾部填充,不是真工作
        r = res.get(name, {})
        out[name] = dict(
            arch=seg["arch"], regs=r.get("REG"), smem=r.get("SHARED"),
            stack=r.get("STACK"), local=r.get("LOCAL"),
            inst_live=sum(live.values()), inst_dead=sum(dead.values()), nop=nop,
            sig=dict(sig_of(live)), depbar=seg["depbar"],
            mnemonics=dict(live.most_common()),
            dead_mnemonics=dict(dead.most_common()))
    return out


# --------------------------------------------------------------------------
# sm_75 → sm_89 重编等价性核验(全程静态,不碰 GPU)
#
# 四个 C++ 产物是 CMake 默认 arch 编的 sm_75,而 4090 上执行的是驱动 JIT 出来的
# sm_89 码。这里把 fatbin 里内嵌的 PTX 抽出来(`cuobjdump -ptx`),把 `.target`
# 改写成 sm_89,用本机 `ptxas -arch=sm_89 -O3` 重编,再用同一套 sig_of() 逐项对比。
#
# 这不是「4090 上真正执行的那份码」——驱动 JIT 用的是驱动内置的 ptxas,版本与
# 优化开关都未必与本机一致。它给的是一条弱一些但可复现的结论:**哪些判据跨
# arch 稳定**。稳定的判据可以拿 sm_75 那份数直接用;不稳定的必须标注口径。
# --------------------------------------------------------------------------
PTX_HDR_RE = re.compile(r"^Fatbin\s")
PTX_TARGET_RE = re.compile(r"\.target\s+sm_\d+")


def split_embedded_ptx(text):
    """`cuobjdump -ptx` 的输出里 ptx 段与 elf 段交替,按段头切出 ptx 正文。"""
    out, cur = [], None
    for line in text.splitlines():
        if PTX_HDR_RE.match(line):
            if cur:
                out.append("\n".join(cur))
            cur = [] if line.startswith("Fatbin ptx code:") else None
            continue
        if cur is None:
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [c[c.find(".version"):] for c in out if ".version" in c]


def recompile_sm89(path):
    """抽 PTX → ptxas -arch=sm_89 -O3 → 反汇编,返回与 digest() 同构的字典。"""
    import tempfile
    chunks = split_embedded_ptx(run(["cuobjdump", "-ptx", str(path)]))
    out, failed = {}, 0
    with tempfile.TemporaryDirectory() as d:
        for i, body in enumerate(chunks):
            src = Path(d) / f"m{i}.ptx"
            cub = Path(d) / f"m{i}.cubin"
            src.write_text(PTX_TARGET_RE.sub(".target sm_89", body))
            p = subprocess.run(["ptxas", "-arch=sm_89", "-O3", "-o", str(cub), str(src)],
                               capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                failed += 1
                continue
            segs = parse_sass(run(["cuobjdump", "-sass", str(cub)]))
            res = parse_res(run(["cuobjdump", "-res-usage", str(cub)]))
            for name, seg in segs.items():
                live = seg["live"]
                live.pop("NOP", 0)
                out[name] = dict(regs=res.get(name, {}).get("REG"),
                                 smem=res.get(name, {}).get("SHARED"),
                                 inst_live=sum(live.values()), sig=dict(sig_of(live)))
    return out, failed


# 跨 arch 对比时要看的判据键。含全部访存宽度桶、同步、归约、矩阵管线与算术。
RECHECK_KEYS = [
    "HMMA", "IMMA", "OMMA", "DP4A", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR",
    "SHFL", "MUFU", "BAR.SYNC", "BSSY", "LOCAL",
    "LDG.128", "LDG.64", "LDG.32", "LDG.16", "LDG.8",
    "STG.128", "STG.64", "STG.32", "STG.16", "STG.8",
    "LDS", "STS", "BRA", "BRA.CONV", "BRA.DIV", "IMAD*", "FFMA"]


def report_recheck(op, cfg):
    """打印一个算子的 sm_75 → sm_89 重编对比表。返回 json 记录。"""
    path = cfg["path"]
    print(f"\n{'=' * 78}\n{op}  <-  {path}")
    if not path.exists():
        print(f"  [跳过] 产物不存在。构建方法:\n      {cfg['build']}")
        return dict(op=op, exists=False)

    orig = digest(path)
    archs = {r["arch"] for r in orig.values()}
    if archs == {"sm_89"}:
        print("  [跳过] 本产物已是 sm_89,无需重编核验。")
        return dict(op=op, exists=True, skipped="already sm_89")

    new, failed = recompile_sm89(path)
    if failed:
        print(f"  [注意] {failed} 个 PTX 段 ptxas 未通过(通常是不含 kernel 的编译单元)")

    rows, rec = [], []
    for tag, _short, mangled in cfg["ladder"]:
        o, n = orig.get(mangled), new.get(mangled)
        if o is None or n is None:
            print(f"  [缺失] {tag}: {'重编产物' if o else '原产物'}里没有这个 Function")
            continue
        diff = {k: (o["sig"].get(k, 0), n["sig"].get(k, 0))
                for k in RECHECK_KEYS
                if o["sig"].get(k, 0) != n["sig"].get(k, 0)}
        rows.append([tag, o["inst_live"], n["inst_live"], cell(o["regs"]), cell(n["regs"]),
                     ", ".join(f"{k} {a}->{b}" for k, (a, b) in diff.items()) or "判据全同"])
        rec.append(dict(tag=tag, mangled=mangled,
                        sm75=dict(inst_live=o["inst_live"], regs=o["regs"]),
                        sm89=dict(inst_live=n["inst_live"], regs=n["regs"]),
                        diff={k: dict(sm75=a, sm89=b) for k, (a, b) in diff.items()}))
    if rows:
        print("\n  重编对比 (左 = 仓内 sm_75 cubin,右 = 本机 ptxas -arch=sm_89 -O3 重编)")
        print("    " + fmt_table(rows, ["版本", "活指令75", "活指令89", "REG75", "REG89",
                                        "判据差异"]).replace("\n", "\n    "))
    return dict(op=op, exists=True, kernels=rec)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
COLS_A = [("REG", "regs"), ("SMEM B", "smem"), ("活指令", "inst_live"),
          ("死指令", "inst_dead"), ("NOP", "nop")]
# 判据分两张表:算什么用什么管线(A),搬多宽的数据(B)。挤成一张表在 80 列终端里没法读。
KEYS_A = ["HMMA", "IMMA", "DP4A", "LDSM", "LDGSTS", "LDGDEPBAR", "DEPBAR",
          "SHFL", "MUFU", "BAR.SYNC", "LOCAL"]
KEYS_B = ["LDG.128", "LDG.64", "LDG.32", "LDG.16", "LDG.8",
          "STG.128", "STG.64", "STG.32", "STG.16", "STG.8",
          "LDS", "STS", "BRA", "BRA.CONV", "BRA.DIV", "IMAD*", "FFMA"]


def fmt_table(rows, headers):
    """定宽表。中日文字符按 2 列计,否则表头对不齐。"""
    def w(s):
        return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))
    widths = [max(w(h), *(w(r[i]) for r in rows)) if rows else w(h)
              for i, h in enumerate(headers)]
    out = ["  ".join(h + " " * (widths[i] - w(h)) for i, h in enumerate(headers))]
    out.append("  ".join("-" * x for x in widths))
    for r in rows:
        out.append("  ".join(
            " " * (widths[i] - w(c)) + str(c) if i else
            str(c) + " " * (widths[i] - w(c))
            for i, c in enumerate(r)))
    return "\n".join(out)


def cell(v):
    return "." if not v else str(v)


def report_op(op, cfg, want_mn=False):
    """打印一个算子的版本梯判据表。返回该算子的 json 记录。"""
    path = cfg["path"]
    print(f"\n{'=' * 78}\n{op}  <-  {path}")
    if not path.exists():
        print(f"  [跳过] 产物不存在。构建方法:\n      {cfg['build']}")
        return dict(op=op, path=str(path), exists=False, build=cfg["build"])

    d = digest(path)
    archs = {r["arch"] for n, r in d.items()}
    print(f"  arch: {', '.join(sorted(archs))}"
          + ("" if archs == {"sm_89"} else
             "   [告警] 非 sm_89:4090 上执行的是驱动 JIT 出来的码,不是这一份"))
    print(f"  说明: {cfg['note']}")

    rows_a, rows_b, kj = [], [], []
    listed = set()
    for tag, short, mangled in cfg["ladder"]:
        r = d.get(mangled)
        if r is None:
            print(f"  [缺失] {tag}: 二进制里没有 {mangled}"
                  f" —— 源码或编译配置变了?重建后再跑: {cfg['build']}")
            continue
        listed.add(mangled)
        sig = r["sig"]
        smem = "dyn?" if r["smem"] == 0 else r["smem"]
        rows_a.append([f"{tag} {short}", cell(r["regs"]), cell(smem),
                       r["inst_live"], cell(r["inst_dead"]), cell(r["nop"])]
                      + [cell(sig.get(k)) for k in KEYS_A])
        rows_b.append([f"{tag}"] + [cell(sig.get(k)) for k in KEYS_B])
        kj.append(dict(tag=tag, short=short, mangled=mangled, **r))

    if rows_a:
        print("\n  指令世代与流水  (SMEM 的 dyn? = 静态 0,可能用 extern __shared__)")
        print("    " + fmt_table(rows_a, ["版本"] + [c[0] for c in COLS_A] + KEYS_A)
              .replace("\n", "\n    "))
        print("\n  访存宽度与算术  (LDG/STG 后缀是单条指令的位宽;LDS 已排除 LDSM)")
        print("    " + fmt_table(rows_b, ["版本"] + KEYS_B).replace("\n", "\n    "))

    dep = [(k["tag"], k["depbar"]) for k in kj if k["depbar"]]
    if dep:
        print("\n  cp.async 流水级数 (DEPBAR 立即数 = __pipeline_wait_prior 的实参,编译期常量)")
        for tag, ds in dep:
            print(f"    {tag}: " + " | ".join(ds))

    extra = sorted(set(d) - listed)
    if extra:
        print(f"\n  --- 未列入版本梯的 Function ({len(extra)}) ---")
        for n in extra:
            print(f"    {n}  ({d[n]['inst_live']} 条活指令)")

    if want_mn:
        print("\n  --- 完整助记符直方图 ---")
        for k in kj:
            print(f"    [{k['tag']}] " + "  ".join(
                f"{m}:{c}" for m, c in list(k["mnemonics"].items())))
            if k["dead_mnemonics"]:
                print(f"      (死指令) " + "  ".join(
                    f"{m}:{c}" for m, c in k["dead_mnemonics"].items()))

    return dict(op=op, path=str(path), exists=True, note=cfg["note"],
                archs=sorted(archs), kernels=kj,
                unlisted=[dict(mangled=n, **d[n]) for n in extra])


def main():
    ap = argparse.ArgumentParser(
        description="按版本梯逐 kernel 输出 SASS 判据表(纯静态,不碰 GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="判据的含义与本机的替代办法见 docs/sass_evidence_ladder.md;"
               "有计数器时该看什么见 docs/ncu_reading_guide.md。")
    ap.add_argument("--op", action="append", choices=sorted(OPS),
                    help="只看这个算子(可重复)")
    ap.add_argument("--json", action="store_true", help="输出 JSON(含完整助记符直方图)")
    ap.add_argument("--mnemonics", action="store_true", help="人读模式下附完整助记符直方图")
    ap.add_argument("--recheck-sm89", action="store_true",
                    help="对非 sm_89 产物做重编等价性核验(抽 PTX + ptxas -arch=sm_89,全程静态)")
    a = ap.parse_args()

    if not shutil.which("cuobjdump"):
        print("找不到 cuobjdump。它随 CUDA Toolkit 安装,通常在 /usr/local/cuda/bin。",
              file=sys.stderr)
        return 2

    ops = a.op or list(OPS)

    if a.recheck_sm89:
        if not shutil.which("ptxas"):
            print("找不到 ptxas。它随 CUDA Toolkit 安装,通常在 /usr/local/cuda/bin。",
                  file=sys.stderr)
            return 2
        if a.json:
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                res = [report_recheck(o, OPS[o]) for o in ops]
            json.dump(dict(tool="scripts/sass_ladder.py --recheck-sm89", ops=res),
                      sys.stdout, ensure_ascii=False, indent=1)
            print()
            return 0
        print("sm_75 → sm_89 重编等价性核验:抽内嵌 PTX,本机 ptxas -arch=sm_89 -O3 重编,")
        print("再用同一套判据逐项对比。全程静态,不碰 GPU。口径见 docs/sass_evidence_ladder.md §4 第 5 条。")
        for o in ops:
            report_recheck(o, OPS[o])
        print(f"\n{'=' * 78}\n重编核验完成。判据全同的项可跨 arch 直接引用;"
              "有差异的项(活指令总数 / IMAD* / BRA / REG)必须标注 sm_75 口径。")
        return 0

    if a.json:
        # JSON 模式下人读的表全部憋掉,stdout 只有一个合法 JSON 文档
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = [report_op(o, OPS[o]) for o in ops]
        json.dump(dict(tool="scripts/sass_ladder.py", ops=res),
                  sys.stdout, ensure_ascii=False, indent=1)
        print()
        return 0

    print(__doc__.split("用法:")[0].strip().splitlines()[0])
    print("纯静态分析:cuobjdump 不初始化 CUDA、不碰 GPU。")
    missing = 0
    for o in ops:
        r = report_op(o, OPS[o], want_mn=a.mnemonics)
        missing += 0 if r["exists"] else 1
    print(f"\n{'=' * 78}\n完成:{len(ops) - missing}/{len(ops)} 个算子有产物。"
          f"{'' if not missing else f' {missing} 个缺产物,按上面的构建命令补。'}")
    print("判据怎么读、答不了的怎么办:docs/sass_evidence_ladder.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
