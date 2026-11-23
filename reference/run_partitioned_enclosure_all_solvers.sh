#!/usr/bin/env bash
# Runs every solver of partitioned_enclosure_grid_benchmark.py on CPU and GPU
# with one Picard threshold and one CG tolerance, scores each sample against a
# float64 sparse direct reference, and plots all solvers together.
#
# TensorFlow and SciPy solvers run in the Poetry venv. JAX solvers run in the
# Poetry venv on CPU (its jaxlib is CPU only) and in the .venv/jax_cuda
# overlay on GPU, which adds jax-cuda12-plugin with its own CUDA 12.6 /
# cuDNN 9 libraries without changing the Poetry environment.
#
# Building templates and reference solutions are cached in CACHE_DIR, so the
# view factor setup and the reference solve are paid once per grid rather than
# once per solver and device. Rerunning resumes from the saved CSVs.
#
# The final air and interior mass temperatures of the reference and of every
# solver are saved as compressed .npz files under <device>/predictions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
JAX_GPU_PYTHON="${JAX_GPU_PYTHON:-${REPO_ROOT}/.venv/jax_cuda/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity/all_solvers_uniform}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity/cache}"
ROOMS_PER_SIDE="${ROOMS_PER_SIDE:-1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28}"
# GPU spsolve is a sparse QR that slows down sharply with grid size.
GPU_SPSOLVE_ROOMS_PER_SIDE="${GPU_SPSOLVE_ROOMS_PER_SIDE:-1,2,3,4,5,6,8,10,12,14}"
ROOM_SIZE_METERS="${ROOM_SIZE_METERS:-5.0}"
CV_SIZE_CM="${CV_SIZE_CM:-20.0}"
SIMULATION_STEPS="${SIMULATION_STEPS:-12}"
REPEATS="${REPEATS:-3}"
CASE_TIMEOUT="${CASE_TIMEOUT:-12h}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"
RUN_CPU="${RUN_CPU:-1}"
RUN_GPU="${RUN_GPU:-1}"
FORCE="${FORCE:-0}"
APPLICATIONS="${APPLICATIONS:-base,base_rad,base_rad_interior_mass}"
# One Picard threshold and one CG tolerance for every solver. 1e-8 is the
# tightest CG tolerance the float32 solvers reach on every application.
CONVERGENCE_THRESHOLD="${CONVERGENCE_THRESHOLD:-1e-4}"
CG_TOLERANCE="${CG_TOLERANCE:-1e-8}"
ITERATION_LIMIT="${ITERATION_LIMIT:-5000}"
MEASURE_ACCURACY="${MEASURE_ACCURACY:-1}"
# SciPy has no GPU path, so it runs on CPU only. Other registered solvers
# (jax, jax_cg_resident_fp32) can be added through these variables.
read -r -a CPU_SOLVERS <<< "${CPU_SOLVERS:-numpy_scipy tf_fdm tf_cg tf_cg_resident_fp32 jax_cg jax_cg_resident grouped_jax}"
read -r -a GPU_SOLVERS <<< "${GPU_SOLVERS:-tf_fdm tf_cg tf_cg_resident_fp32 jax_cg jax_cg_resident grouped_jax}"
TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/benchmark.log}"
FAILURE_CSV="${OUTPUT_DIR}/benchmark_failures.csv"
TIMEOUT_CSV="${OUTPUT_DIR}/benchmark_timeouts.csv"
RESULT_CSV_NAME="partitioned_enclosure_grid_time.csv"
CPU_FORCE_PENDING="${FORCE}"
GPU_FORCE_PENDING="${FORCE}"
export TF_CPP_MIN_LOG_LEVEL MPLCONFIGDIR

mkdir -p "${OUTPUT_DIR}/cpu" "${OUTPUT_DIR}/gpu"
[[ -f "${FAILURE_CSV}" ]] || printf 'timestamp,device,solver,exit_code\n' > "${FAILURE_CSV}"
[[ -f "${TIMEOUT_CSV}" ]] || printf 'timestamp,device,solver,rooms_per_side,repeat,timeout,exit_code\n' > "${TIMEOUT_CSV}"

if [[ "${BENCHMARK_LOG_ACTIVE:-0}" != "1" ]]; then
  export BENCHMARK_LOG_ACTIVE=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi

run_case() {
  local device="$1"
  local solver="$2"
  local progress_file="${OUTPUT_DIR}/.${device}_${solver}.progress"
  local force_flag=()
  local accuracy_flag=()
  local exit_code=0
  local worker_pid
  local started_at
  local next_heartbeat
  local current_size="unknown"
  local current_repeat="unknown"
  local python="${PYTHON}"
  local rooms_per_side="${ROOMS_PER_SIDE}"
  local device_env

  if [[ "${device}" == "cpu" ]]; then
    # Hides the GPU from TensorFlow and pins JAX to the host.
    device_env=(CUDA_VISIBLE_DEVICES=-1 JAX_PLATFORMS=cpu)
  elif [[ "${solver}" == jax* || "${solver}" == grouped_jax ]]; then
    python="${JAX_GPU_PYTHON}"
    # Fails instead of silently falling back to CPU if CUDA is unavailable.
    device_env=(JAX_PLATFORMS=cuda)
    [[ "${solver}" == "jax" ]] && rooms_per_side="${GPU_SPSOLVE_ROOMS_PER_SIDE}"
  else
    device_env=()
  fi
  if [[ ! -x "${python}" ]]; then
    echo "[${device}] ${python} not found; skipping solver=${solver}."
    printf '%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "${device}" "${solver}" "missing_python" >> "${FAILURE_CSV}"
    return
  fi

  rm -f "${progress_file}"
  # --force clears the device CSV, so it is applied to the first solver only.
  if [[ "${device}" == "cpu" && "${CPU_FORCE_PENDING}" == "1" ]]; then
    force_flag=(--force)
    CPU_FORCE_PENDING=0
  elif [[ "${device}" == "gpu" && "${GPU_FORCE_PENDING}" == "1" ]]; then
    force_flag=(--force)
    GPU_FORCE_PENDING=0
  fi
  [[ "${MEASURE_ACCURACY}" == "1" ]] && accuracy_flag=(--measure-accuracy)
  echo "[${device}] solver=${solver} python=${python} rooms_per_side=${rooms_per_side} applications=${APPLICATIONS}"
  started_at="$(date +%s)"
  next_heartbeat=$((started_at + HEARTBEAT_SECONDS))

  # Runs as a module so `reference` and `smart_control` resolve from the repo.
  (
    cd "${REPO_ROOT}"
    exec timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT}" \
      env -u CUDA_VISIBLE_DEVICES "${device_env[@]}" "${python}" \
      -m reference.partitioned_enclosure_grid_benchmark \
      --rooms-per-side "${rooms_per_side}" \
      --room-size-meters "${ROOM_SIZE_METERS}" \
      --cv-size-cm "${CV_SIZE_CM}" \
      --simulation-steps "${SIMULATION_STEPS}" \
      --repeats "${REPEATS}" \
      --solver "${solver}" \
      --applications "${APPLICATIONS}" \
      --convergence-threshold "${CONVERGENCE_THRESHOLD}" \
      --cg-tolerance "${CG_TOLERANCE}" \
      --iteration-limit "${ITERATION_LIMIT}" \
      --uniform-tolerances \
      "${accuracy_flag[@]}" \
      --cache-dir "${CACHE_DIR}" \
      --expected-device "${device}" \
      --progress-file "${progress_file}" \
      "${force_flag[@]}" \
      --output-dir "${OUTPUT_DIR}/${device}"
  ) &
  worker_pid=$!

  while kill -0 "${worker_pid}" 2>/dev/null; do
    sleep 1
    if (( $(date +%s) < next_heartbeat )); then
      continue
    fi
    if [[ -f "${progress_file}" ]]; then
      IFS=, read -r current_size current_repeat < "${progress_file}"
    fi
    echo "[heartbeat] device=${device} solver=${solver} elapsed_seconds=$(( $(date +%s) - started_at )) rooms_per_side=${current_size} repeat=${current_repeat}"
    next_heartbeat=$(( $(date +%s) + HEARTBEAT_SECONDS ))
  done
  wait "${worker_pid}" || exit_code=$?

  if [[ "${exit_code}" != "0" ]]; then
    printf '%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "${device}" "${solver}" "${exit_code}" >> "${FAILURE_CSV}"
    if [[ "${exit_code}" == "124" || "${exit_code}" == "137" ]]; then
      if [[ -f "${progress_file}" ]]; then
        IFS=, read -r current_size current_repeat < "${progress_file}"
      fi
      printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date --iso-8601=seconds)" "${device}" "${solver}" \
        "${current_size}" "${current_repeat}" "${CASE_TIMEOUT}" "${exit_code}" \
        >> "${TIMEOUT_CSV}"
    fi
  fi
  rm -f "${progress_file}"
}

echo "[$(date --iso-8601=seconds)] Starting partitioned enclosure benchmark (all solvers)"
echo "output_dir=${OUTPUT_DIR} cache_dir=${CACHE_DIR} rooms_per_side=${ROOMS_PER_SIDE} gpu_spsolve_rooms_per_side=${GPU_SPSOLVE_ROOMS_PER_SIDE} room_size_meters=${ROOM_SIZE_METERS} cv_size_cm=${CV_SIZE_CM} simulation_steps=${SIMULATION_STEPS} repeats=${REPEATS} convergence_threshold=${CONVERGENCE_THRESHOLD} cg_tolerance=${CG_TOLERANCE} measure_accuracy=${MEASURE_ACCURACY} cpu_solvers=${CPU_SOLVERS[*]} gpu_solvers=${GPU_SOLVERS[*]}"

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

CPU_CSV="${OUTPUT_DIR}/cpu/${RESULT_CSV_NAME}"
GPU_CSV="${OUTPUT_DIR}/gpu/${RESULT_CSV_NAME}"

plot_args=()
[[ -f "${CPU_CSV}" ]] && plot_args+=(--cpu-csv "${CPU_CSV}")
[[ -f "${GPU_CSV}" ]] && plot_args+=(--gpu-csv "${GPU_CSV}")
if (( ${#plot_args[@]} == 0 )); then
  echo "No benchmark CSV was produced; skipping plots."
else
  # One figure per solver and device.
  "${PYTHON}" "${REPO_ROOT}/reference/plot_partitioned_enclosure_benchmark.py" \
    "${plot_args[@]}" --output-dir "${OUTPUT_DIR}/per_solver"
  # Every solver and device on shared axes, with accuracy.
  "${PYTHON}" "${REPO_ROOT}/reference/plot_all_solvers.py" \
    --root "$(dirname "${OUTPUT_DIR}")" \
    --sources "$(basename "${OUTPUT_DIR}")" \
    --output-dir "${OUTPUT_DIR}/all_solvers"
  # Field-level error maps from the saved final temperatures.
  "${PYTHON}" "${REPO_ROOT}/reference/plot_prediction_fields.py" \
    --predictions-dir "${OUTPUT_DIR}/cpu/predictions" \
    --predictions-dir "${OUTPUT_DIR}/gpu/predictions" \
    --output-dir "${OUTPUT_DIR}/prediction_fields"
  # CPU against GPU for each solver.
  COMPARE_SCRIPT="${REPO_ROOT}/reference/compare_application_grid_devices.py"
  if [[ -f "${CPU_CSV}" && -f "${GPU_CSV}" && -f "${COMPARE_SCRIPT}" ]]; then
    "${PYTHON}" "${COMPARE_SCRIPT}" \
      --cpu-csv "${CPU_CSV}" --gpu-csv "${GPU_CSV}" \
      --output-dir "${OUTPUT_DIR}/cpu_vs_gpu"
  fi
fi

echo "[$(date --iso-8601=seconds)] Partitioned enclosure benchmark (all solvers) complete"
