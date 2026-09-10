"""三个 LLM 融合算子(fused-norm / rope / activation)共用的 bench 工装。

抽出来的理由不是省代码行数,而是保证三个算子的数字可以横向比较:
计时口径(warmup 轮数、event 包段方式、是否同步)一旦各写各的,
「rope 比 fused-norm 更贴带宽」这类跨算子结论就不成立了。

同理,手写 CUDA / PyTorch eager / torch.compile / Triton 四类臂也全部
走下面同一个 timeit —— 本仓早期「vs 别人」的数字是跨 harness 对比,
只能标注「推断级」;单一 harness 是把它升级为实测级的前提。
"""
import datetime
import os
import pathlib
import subprocess
import sys

import torch

PEAK_GBPS = 1008.0     # RTX 4090 GDDR6X 理论峰值(ENV.md);超过它即说明数据落在 L2


def build_ext(name, here, sources, include=None):
    """JIT 编译 torch 扩展。

    踩坑:构建被中断会在 ~/.cache/torch_extensions/<pyver_cuver>/<name>/ 下
    残留 lock 文件,之后每次 load() 都无限等锁(现象是卡住且没有编译进程)。
    这里主动清理陈旧 lock —— 只清本扩展自己的,不动别人的。
    """
    from torch.utils.cpp_extension import load, _get_build_directory
    try:
        lock = pathlib.Path(_get_build_directory(name, False)) / "lock"
        if lock.exists():
            lock.unlink()
    except Exception:
        pass
    return load(name=name,
                sources=[str(pathlib.Path(here) / "src" / f) for f in sources],
                extra_include_paths=[str(pathlib.Path(here) / "include")]
                                     + list(include or []),
                # 不开 --use_fast_math:这几个算子都是访存主导,快速数学
                # 没有性能价值,却会让超越函数的舍入偏离 PyTorch,给「与 eager
                # 的数值差异」引入无法归因的来源。与本仓其他子项目的 -O3 一致。
                extra_cuda_cflags=["-O3", "-arch=sm_89"],
                verbose=False)


def timeit(fn, iters, warmup=10):
    """CUDA-event 计时:一对 event 包住整段循环,再除以次数。

    为什么不逐次记 event:每对 event 有 ~1us 记录开销,而本组算子最快形状
    只有几微秒,逐次计时会让开销成为被测量本身。整段包住则被 iters 摊薄。
    warmup 驱走冷时钟与 JIT/懒初始化(triton-kernels#EXP-T02（流水线 GEMM）的 JIT 伪影教训:
    新 Triton config 首跑必须预热,否则测到的是编译时间)。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def provenance(here, cmd):
    """CORE 铁律 4:文本结果文件首行写清环境/代码版本/命令/硬件。

    driver 取 nvidia-smi 的真实内核驱动版本,不是 cudaDriverGetVersion
    (后者是 CUDA driver-API 版本,本仓 raw 曾因此误填,勘误在各
     project-proof/data/manifest.txt)。
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(here), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    try:
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).splitlines()[0].strip()
    except Exception:
        drv = "unknown"
    return (f'# provenance: env=venv:{sys.prefix} sha={sha} cmd="{cmd}" date={ts} '
            f'gpu="{torch.cuda.get_device_name(0)}" driver={drv} '
            f'cuda={torch.version.cuda} torch={torch.__version__}')


def open_csv(default_path, here, cmd, header):
    """按 BENCH_OUT 落盘(CORE 铁律 5:只写 UTC 前缀新文件,永不覆盖历史)。"""
    out_path = os.environ.get("BENCH_OUT", default_path)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w")
    f.write(provenance(here, cmd) + "\n")
    f.write(header + "\n")
    return f, out_path


def load_triton(module_attr):
    """按需 import 姊妹仓 triton-kernels 的实现(单一事实源:不复制一份)。

    找不到时静默返回 None,本 bench 仍可独立运行 —— Triton 臂是可选的
    对照,不是依赖。
    """
    src = os.environ.get("TRITON_KERNELS_SRC", "/root/projects/triton-kernels/src")
    try:
        if src not in sys.path:
            sys.path.insert(0, src)
        mod = __import__("llm_fused")
        return getattr(mod, module_attr)
    except Exception:
        return None


def rel_err(out, ref):
    """相对误差:分母用参考输出的全局 absmax,而不是逐元素。

    逐元素相对误差在近零元素上会无意义地爆炸(与 gemm/src/main.cu 同一处理)。
    """
    denom = ref.abs().max().float().clamp_min(1e-30)
    return ((out.float() - ref.float()).abs().max() / denom).item()
