#!/usr/bin/env bash
# Runs every solver of partitioned_enclosure_timeseries.py through one day with
# a daily ambient sine and staggered per-room HVAC schedules, scores each step
# against a float64 sparse direct reference trajectory, and plots the error
# over time.
#
# Uses the same environments as run_partitioned_enclosure_all_solvers.sh: the
# Poetry venv for TensorFlow, SciPy and JAX on CPU, and the .venv/jax_cuda
# overlay for JAX on GPU. CPU runs first, since the reference trajectory is
# built on the host and cached in CACHE_DIR for the GPU runs. Rerunning skips
# solvers whose output already exists, unless FORCE=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
JAX_GPU_PYTHON="${JAX_GPU_PYTHON:-${REPO_ROOT}/.venv/jax_cuda/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity/solver_agreement_timeseries}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity/cache}"
ROOMS_PER_SIDE="${ROOMS_PER_SIDE:-3}"
ROOM_SIZE_METERS="${ROOM_SIZE_METERS:-5.0}"
CV_SIZE_CM="${CV_SIZE_CM:-20.0}"
SIMULATION_HOURS="${SIMULATION_HOURS:-24}"
TIME_STEP_SECONDS="${TIME_STEP_SECONDS:-300}"
FIELD_SAVE_STRIDE="${FIELD_SAVE_STRIDE:-6}"
APPLICATIONS="${APPLICATIONS:-base,base_rad,base_rad_interior_mass}"
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-4}"
CG_TOLERANCE="${CG_TOLERANCE:-1e-8}"
ITERATION_LIMIT="${ITERATION_LIMIT:-5000}"
# FDM sweeps converge slowly once the forcing changes during the day; 5000
# sweeps are not enough by 06:00 on the base application.
TF_FDM_ITERATION_LIMIT="${TF_FDM_ITERATION_LIMIT:-50000}"
RUN_CPU="${RUN_CPU:-1}"
RUN_GPU="${RUN_GPU:-1}"
FORCE="${FORCE:-0}"
read -r -a CPU_SOLVERS <<< "${CPU_SOLVERS:-numpy_scipy tf_fdm tf_cg tf_cg_resident_fp32 jax_cg jax_cg_resident}"
read -r -a GPU_SOLVERS <<< "${GPU_SOLVERS:-tf_fdm tf_cg tf_cg_resident_fp32 jax_cg jax_cg_resident}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/benchmark.log}"
FAILURE_CSV="${OUTPUT_DIR}/benchmark_failures.csv"
export TF_CPP_MIN_LOG_LEVEL MPLCONFIGDIR

mkdir -p "${OUTPUT_DIR}/cpu" "${OUTPUT_DIR}/gpu"
[[ -f "${FAILURE_CSV}" ]] || printf 'timestamp,device,solver,exit_code\n' > "${FAILURE_CSV}"

if [[ "${BENCHMARK_LOG_ACTIVE:-0}" != "1" ]]; then
  export BENCHMARK_LOG_ACTIVE=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

run_case() {
  local device="$1"
  local solver="$2"
  local python="${PYTHON}"
  local device_env
  local force_flag=()
  local exit_code=0
  local iteration_limit="${ITERATION_LIMIT}"

  if [[ "${device}" == "cpu" ]]; then
    device_env=(CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu)
  elif [[ "${solver}" == jax* ]]; then
    python="${JAX_GPU_PYTHON}"
    device_env=(JAX_PLATFORMS=cuda)
  else
    device_env=()
  fi
  if [[ ! -x "${python}" ]]; then
    echo "[${device}] ${python} not found; skipping solver=${solver}."
    printf '%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "${device}" "${solver}" "missing_python" >> "${FAILURE_CSV}"
    return
  fi
  [[ "${FORCE}" == "1" ]] && force_flag=(--force)
  [[ "${solver}" == "tf_fdm" ]] && iteration_limit="${TF_FDM_ITERATION_LIMIT}"
  echo "[$(date --iso-8601=seconds)] [${device}] solver=${solver} iteration_limit=${iteration_limit}"
  (
    cd "${REPO_ROOT}"
    exec env -u CUDA_VISIBLE_DEVICES "${device_env[@]}" "${python}" \
      -m reference.partitioned_enclosure_timeseries \
      --rooms-per-side "${ROOMS_PER_SIDE}" \
      --room-size-meters "${ROOM_SIZE_METERS}" \
      --cv-size-cm "${CV_SIZE_CM}" \
      --simulation-hours "${SIMULATION_HOURS}" \
      --time-step-seconds "${TIME_STEP_SECONDS}" \
      --field-save-stride "${FIELD_SAVE_STRIDE}" \
      --solver "${solver}" \
      --applications "${APPLICATIONS}" \
      --convergence-threshold "${CONVERGENCE_THRESHOLD}" \
      --cg-tolerance "${CG_TOLERANCE}" \
      --iteration-limit "${iteration_limit}" \
      --cache-dir "${CACHE_DIR}" \
      --expected-device "${device}" \
      "${force_flag[@]}" \
      --output-dir "${OUTPUT_DIR}/${device}"
  ) || exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    printf '%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "${device}" "${solver}" "${exit_code}" >> "${FAILURE_CSV}"
  fi
}

echo "[$(date --iso-8601=seconds)] Starting partitioned enclosure time series run"
echo "output_dir=${OUTPUT_DIR} rooms_per_side=${ROOMS_PER_SIDE} simulation_hours=${SIMULATION_HOURS} time_step_seconds=${TIME_STEP_SECONDS} convergence_threshold=${CONVERGENCE_THRESHOLD} cg_tolerance=${CG_TOLERANCE} applications=${APPLICATIONS} cpu_solvers=${CPU_SOLVERS[*]} gpu_solvers=${GPU_SOLVERS[*]}"

if [[ "${RUN_CPU}" == "1" ]]; then
  for solver in "${CPU_SOLVERS[@]}"; do
    run_case cpu "${solver}"
  done
fi
if [[ "${RUN_GPU}" == "1" ]]; then
  for solver in "${GPU_SOLVERS[@]}"; do
    run_case gpu "${solver}"
  done
fi

(cd "${REPO_ROOT}" && "${PYTHON}" -m reference.plot_timeseries \
  --timeseries-dir "${OUTPUT_DIR}/cpu/timeseries" \
  --timeseries-dir "${OUTPUT_DIR}/gpu/timeseries" \
  --output-dir "${OUTPUT_DIR}/timeseries_plots")

echo "[$(date --iso-8601=seconds)] Partitioned enclosure time series run complete"
