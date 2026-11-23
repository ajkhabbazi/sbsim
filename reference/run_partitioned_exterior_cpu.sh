#!/usr/bin/env bash
# Reuse the uniform-tolerance benchmark, with the two exterior applications.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity/exterior_radiation_full_cpu}"
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_DIR}/cache}"
export PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
export RUN_CPU=1 RUN_GPU=0
export APPLICATIONS=base_rad_exterior,base_rad_interior_mass_exterior
export CPU_SOLVERS="${CPU_SOLVERS:-jax_cg_resident jax_cg numpy_scipy tf_cg tf_cg_resident_fp32 tf_fdm}"
export MEASURE_ACCURACY=1
export CASE_TIMEOUT="${CASE_TIMEOUT:-36h}"

FULL_GRID="${ROOMS_PER_SIDE:-1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28}"
mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}/.worker.lock"
flock -n 9

run_phase() {
  ROOMS_PER_SIDE="$1" bash reference/run_partitioned_enclosure_all_solvers.sh
  local csv="${OUTPUT_DIR}/cpu/partitioned_enclosure_grid_time.csv"
  if [[ -s "${csv}" ]]; then
    "${PYTHON}" -m reference.plot_exterior_radiation_comparison \
      --scaled-cpu-csv "${csv}" \
      --output "${OUTPUT_DIR}/all_solvers_time_with_exterior_radiation.png"
  fi
}

# Get comparable data from every solver before the large grids. The original
# runner resumes from saved samples, so the second phase reuses these runs.
if [[ "${FULL_GRID}" == "1,2,3,"* ]]; then
  run_phase 1,2,3
fi
run_phase "${FULL_GRID}"
