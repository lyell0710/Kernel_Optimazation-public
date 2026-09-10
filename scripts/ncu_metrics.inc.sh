# Shared Nsight Compute knobs for Kernel_Optimazation/*/project-proof/scripts/profile_ncu.sh
# shellcheck shell=bash

NCU_PROFILE_SECTIONS=(
  # 原始八件:2026-05 那批 Laptop 报告用的就是这一组,不要删改,
  # 删了新旧报告就无法在 GUI 里逐 metric 对照。
  SpeedOfLight
  LaunchStats
  Occupancy
  WarpStateStats
  MemoryWorkloadAnalysis
  MemoryWorkloadAnalysis_Chart
  MemoryWorkloadAnalysis_Tables
  SchedulerStats
  # 追加:上面八件是为访存型算子(softmax/gemv/int8-quantize/cuda-reduce)选的,
  # 里头没有算力侧的分解。gemm 与 flash-attn 的版本梯核心是"上没上 Tensor Core"
  # (gemm v1→v2、fa2 v1→v2 都是 wmma),而 Tensor pipe 利用率只在这个 section 里,
  # 缺了它 wmma 那一跳就没有直接证据,只能拿 SOL 的 SM% 间接猜。
  # 追加不破坏兼容:旧报告只是没有这一节,原八件仍逐 metric 可比。
  ComputeWorkloadAnalysis
)

# Extended CSV metrics: throughput, occupancy 代理, 合并/ bank / L2 / 寄存器&smem 代理, warp stall 分解
# 分管线利用率:section 给不出。Compute Workload Analysis 在 CLI 导出里只有
# IPC 与 SM Busy,分管线(Tensor/FMA/ALU)那张表只在 GUI 里,CSV 拿不到。
# 而"wmma 有没有真的用上 Tensor Core"恰恰要它——SASS 的 HMMA 计数只能证明
# 指令生成了,证明不了运行时占了多少。实测 gemm v2:
#   sm__pipe_tensor_cycles_active = 25.71%,sm__inst_executed_pipe_tensor = 6.43%
# 只多一个 pass,代价很小。
NCU_TENSOR_METRICS="sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active"

NCU_CSV_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,lts__t_request_hit_rate,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,tpc__average_registers_per_thread,sm__sass_data_bytes_mem_shared,smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active,smsp__warp_issue_stalled_barrier_per_warp_active,smsp__warp_issue_stalled_membar_per_warp_active,smsp__warp_issue_stalled_short_scoreboard_per_warp_active,${NCU_TENSOR_METRICS}"
