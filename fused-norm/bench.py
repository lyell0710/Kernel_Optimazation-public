#!/usr/bin/env python3
"""fused_add_rmsnorm 版本梯 bench —— 七条臂,同一个 harness,同一套计时。

七条臂:v0(未融合基线) v1(融合) v2(warp 归约) v3(向量化) v4(寄存器缓存)
        pytorch_eager  torch_compile

为什么七条臂必须在同一个进程里测:本仓早期「vs 别人」的数字是自家 C++ bench
对别人 Python bench,计时口径不同,只能标注「跨 harness 推断级」。把手写
kernel 绑进 torch 后,所有臂共用下面这一段 CUDA-event 计时,差值才是算子差值。

四个形状分三个区间(取自 EXP-K04（标准库基准补齐与两区间重测）的教训:memory-bound 算子必须同时测 HBM
区间,否则数据全落在 4090 的 72MB L2 里,量到的是 L2 带宽而不是 HBM 带宽):
  decode  T=1/64   —— 引擎逐 token 解码时的真实形状,launch 主导
  prefill T=2048   —— 引擎 2K prefill,总工作集 64MB,L2 边缘
  hbm     T=32768  —— 总工作集 1GB,确定落在 HBM

用法:  BENCH_OUT=project-proof/data/<UTC>_fused-norm_r1.csv python bench.py
"""
import os, sys, json, time, datetime, subprocess, pathlib
import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
PEAK_GBPS = 1008.0          # RTX 4090 GDDR6X 理论峰值(ENV.md)

# ---------------------------------------------------------------------------
# 可选的 Triton 臂:实现在姊妹仓 triton-kernels/src/llm_fused.py(它的正主),
# 这里只 import,不复制一份 —— 单一事实源。
# 加这条臂的目的不是「比谁快」,而是补上「什么时候该用 CUDA」判断曲线的
# 第三个点:计算主导的 GEMM 手写够到 cuBLAS 86%,融合型 attention 只够到
# 自家 Triton 28%,而访存主导的融合逐元素算子(本算子)预期两边打平。
# 找不到姊妹仓时静默跳过,本 bench 仍可独立运行。
# ---------------------------------------------------------------------------
TRITON_SRC = os.environ.get("TRITON_KERNELS_SRC", "/root/projects/triton-kernels/src")
try:
    sys.path.insert(0, TRITON_SRC)
    from llm_fused import fused_add_rmsnorm as _triton_fused
    HAS_TRITON = True
except Exception as _e:
    HAS_TRITON = False
    _TRITON_WHY = repr(_e)

# ---------------------------------------------------------------------------
# 构建扩展。首编约 60-90s(要吃整套 torch 头文件),之后走 ninja 增量缓存。
# 踩坑记录:若构建过程被中断,~/.cache/torch_extensions/<py-cu>/<name>/lock
# 会残留,之后每次 load() 都会无限等锁(表现为「卡住不动、没有编译进程」)。
# 处理方式是删掉那个 lock 文件。
# ---------------------------------------------------------------------------
def build():
    src = [str(HERE / "src" / f) for f in
           ("binding.cpp", "fused_norm_v0.cu", "fused_norm_v1.cu",
            "fused_norm_v2.cu", "fused_norm_v3.cu", "fused_norm_v4.cu")]
    return load(name="fused_norm_ext", sources=src,
                extra_include_paths=[str(HERE / "include")],
                # 不开 --use_fast_math:本算子是访存主导,快速数学(影响除法/超越函数)
                # 没有性能价值,却会让 rsqrtf 的舍入偏离 PyTorch,给「与 eager 的
                # 数值差异」引入一个无法归因的来源。与本仓其他子项目的 -O3 口径一致。
                extra_cuda_cflags=["-O3", "-arch=sm_89"],
                verbose=False)


# ---------------------------------------------------------------------------
# 参考实现 = llm-engine src/layers.py rmsnorm() 与 model.py 的 `h = res + o`
# 逐行对应。舍入顺序(先把归一化结果舍到 bf16、再乘 bf16 权重)与 vLLM
# layernorm.cu 一致 —— 手写 kernel 必须复刻这个顺序,否则接进引擎后
# 逐位对拍会出现末位差异。
# ---------------------------------------------------------------------------
def eager(residual, x, w, eps):
    residual.add_(x)                                  # 就地:下一层的残差流
    xf = residual.float()                             # fp32 中间量,理由见 kernel 头注释
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(residual.dtype)


def make_inputs(T, H, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    x   = torch.randn(T, H, generator=g, device=device, dtype=torch.bfloat16)
    res = torch.randn(T, H, generator=g, device=device, dtype=torch.bfloat16)
    w   = torch.randn(H,   generator=g, device=device, dtype=torch.bfloat16)
    return x, res, w


def timeit(fn, iters, warmup=10):
    """CUDA-event 计时:一对 event 包住整段循环,再除以次数。

    为什么不逐次记 event:每对 event 有 ~1us 的记录开销,本算子最快形状只有
    几微秒,逐次计时的开销会成为被测量本身。整段包住则开销被 iters 摊薄。
    warmup 驱走冷时钟与 JIT/懒初始化(triton-kernels#EXP-T02（流水线 GEMM）的 JIT 伪影教训)。
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


def provenance(cmd):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], text=True).strip()
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


def main():
    ext = build()
    dev = "cuda"
    eps = 1e-6
    iters_env = int(os.environ.get("BENCH_ITERS", "0"))

    # (标签, T, H, iters)。iters 按单次耗时反比给:小形状多跑几轮压住抖动,
    # 1GB 形状每次 ~3ms,50 轮已是 150ms,够稳。
    shapes = [("decode",   1,     4096, 2000),
              ("decode64", 64,    4096, 2000),
              ("prefill",  2048,  4096, 300),
              ("hbm",      32768, 4096, 50)]

    compiled = torch.compile(eager, dynamic=False)

    out_path = os.environ.get("BENCH_OUT", "project-proof/data/benchmark_results.csv")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w")
    f.write(provenance(f"BENCH_ITERS={iters_env} python bench.py") + "\n")
    f.write("regime,version,T,H,latency_ms,eff_bw_GBps,pct_peak,"
            "speedup_vs_v0,max_rel_err,correctness_pass\n")

    for tag, T, H, iters in shapes:
        if iters_env:
            iters = iters_env
        x, res0, w = make_inputs(T, H, dev)

        # ---- 参考输出:在干净副本上跑一次 eager ----
        # 必须用副本:residual 是就地更新的,直接跑会污染后续所有臂的输入。
        ref_res = res0.clone()
        ref_out = eager(ref_res, x, w, eps)
        ref_absmax = ref_out.abs().max().float().item()

        # 相对误差分母用全局 absmax 而非逐元素:逐元素相对误差在近零元素上
        # 会无意义地爆炸(与 gemm/src/main.cu 同一处理)。
        def rel_err(o, r):
            return ((o.float() - r.float()).abs().max() / max(ref_absmax, 1e-30)).item()

        # 每条臂:一个「在干净副本上跑一次并返回输出」的闭包(测正确性),
        # 与一个「反复跑」的闭包(测时延)。两者分开,是因为正确性必须在
        # 未被污染的输入上判定,而时延允许 residual 随迭代漂移
        #(漂移只改变数值大小,不改变访存字节数与指令数,不影响计时口径)。
        def mk_kernel_arm(name):
            fn = getattr(ext, name)
            def once():
                r = res0.clone(); o = torch.empty_like(x)
                fn(o, r, x, w, eps)
                return o
            def loop_setup():
                r = res0.clone(); o = torch.empty_like(x)
                return lambda: fn(o, r, x, w, eps)
            return once, loop_setup

        def mk_torch_arm(f):
            def once():
                r = res0.clone()
                return f(r, x, w, eps)
            def loop_setup():
                r = res0.clone()
                return lambda: f(r, x, w, eps)
            return once, loop_setup

        plan = [(n,) + mk_kernel_arm(n) for n in ("v0", "v1", "v2", "v3", "v4")
                if not (H % 8 and n in ("v3", "v4"))]
        plan.append(("pytorch_eager",) + mk_torch_arm(eager))
        plan.append(("torch_compile",) + mk_torch_arm(compiled))
        if HAS_TRITON:
            # 形参顺序对齐 mk_torch_arm 的 (residual, x, w, eps)
            plan.append(("triton",) + mk_torch_arm(
                lambda r, x, w, eps: _triton_fused(x, r, w, eps)))

        v0_ms = None
        v3_out = None
        for name, once, loop_setup in plan:
            o = once()
            err = rel_err(o, ref_out)
            if name == "v3":
                v3_out = o.clone()
            if name == "v4" and v3_out is not None:
                # v4 缓存的是「已舍到 bf16」的值,与 v3 的重读值应当逐位相同。
                # 这条比阈值判定强:它把「优化没改变语义」变成可证伪断言。
                assert torch.equal(o, v3_out), "v4 与 v3 不逐位一致 —— 寄存器缓存改变了语义"
            ms = timeit(loop_setup(), iters)
            if name == "v0":
                v0_ms = ms
            # 有效带宽按「算法下界字节数」计:读 x + 读 residual + 写 residual
            # + 写 out = 4 次 * 2B = 8B/元素。v0 实际搬得更多(6 次),所以它的
            # 这一列偏低是应该的 —— 这一列量的是「离算法下界有多远」。
            bw = T * H * 8 / (ms * 1e-3) / 1e9
            f.write(f"{tag},{name},{T},{H},{ms:.6f},{bw:.1f},"
                    f"{bw / PEAK_GBPS * 100:.1f},{v0_ms / ms:.3f},"
                    f"{err:.3e},{'true' if err < 2e-2 else 'false'}\n")
            print(f"{tag:9s} {name:14s} {ms:9.5f} ms  {bw:7.1f} GB/s "
                  f"({bw / PEAK_GBPS * 100:5.1f}% peak)  {v0_ms / ms:6.3f}x  "
                  f"err={err:.2e}")
        print()
    f.close()
    print("written:", out_path)


if __name__ == "__main__":
    main()
