#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_one_project() {
  local proj="$1"
  pushd "$ROOT_DIR/$proj" >/dev/null

  echo "========== [$proj] configure/build =========="
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j

  echo "========== [$proj] benchmark =========="
  case "$proj" in
    softmax)
      ./build/softmax_bench
      ;;
    gemv)
      ./build/gemv_bench
      ;;
    int8-quantize)
      ./build/int8_quantize_bench
      ;;
    *)
      echo "Unknown project: $proj"
      exit 1
      ;;
  esac

  echo "========== [$proj] plots =========="
  python project-proof/scripts/plot_latency.py
  python project-proof/scripts/plot_latency_log.py
  python project-proof/scripts/plot_speedup.py
  python project-proof/scripts/plot_correctness.py

  popd >/dev/null
}

run_one_project "softmax"
run_one_project "gemv"
run_one_project "int8-quantize"

echo "All benchmark + plot jobs completed."
