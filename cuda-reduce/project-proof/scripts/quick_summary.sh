#!/bin/bash

# 快速查看数据摘要和图表位置

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CSV_PATH="$ROOT/project-proof/data/benchmark_results.csv"
FIG_DIR="$ROOT/project-proof/docs/figures/01-benchmark"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          CUDA Reduce 性能数据摘要 (RTX 4070, N=2^24)           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$CSV_PATH" ]; then
    echo "❌ 错误: 未找到数据文件: $CSV_PATH"
    echo "请先运行: cd $ROOT/build && ./reduce_bench"
    exit 1
fi

# 用 awk 解析 CSV 和输出表格
awk -F',' '
NR==1 { next }
$1 == "baseline" { next }  # 跳过 baseline
{
    version = $1
    latency = $5
    speedup = $6

    # 按版本分类上色
    if (version == "cuBLAS") {
        marker = "🏆"
    } else if (version ~ /^v[0-2]/) {
        marker = "📌"
    } else if (version ~ /^v[3-5]/) {
        marker = "⭐"
    } else if (version ~ /^v[6-7]/) {
        marker = "📊"
    } else {
        marker = "  "
    }

    printf "%s %-10s %12.6f ms  %10.1f×\n", marker, version, latency, speedup
}
' "$CSV_PATH"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 各版本分类："
echo "  🏆 标准库参考     → cuBLAS"
echo "  ⭐ 自写最优版本   → v3-v5 (接近标准库 <3% 差距)"
echo "  📌 基础优化       → v0-v2"
echo "  📊 两级 grid-stride → v6-v7 (性能下降)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📈 生成的可视化图表:"
echo ""

for fig in "$FIG_DIR"/*.png; do
    if [ -f "$fig" ]; then
        size=$(du -h "$fig" | cut -f1)
        basename "$fig" | sed "s/^/  📄 /" | sed "s/$/ ($size)/"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 快速查看图表:"
echo "  • 与 cuBLAS 对标:    05-vs-cublas.png"
echo "  • 所有版本加速比:    04-speedup-vs-baseline.png"
echo "  • 详细延迟对比:      01-latency-overview.png"
echo ""
echo "📂 完整路径: $FIG_DIR"
echo ""
