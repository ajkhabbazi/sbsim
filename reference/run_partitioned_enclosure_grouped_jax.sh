#!/usr/bin/env bash
# Compare full-grid and grouped JAX CG with interior mass and both radiation terms.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/reference/benchmark_outputs/partitioned_enclosure_sensitivity"

export OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_ROOT}/grouped_jax_full_physics}"
export ROOMS_PER_SIDE="${ROOMS_PER_SIDE:-1,2,4,6,8,10}"
export APPLICATIONS=base_rad_interior_mass_exterior
export CPU_SOLVERS="${CPU_SOLVERS:-jax_cg grouped_jax}"
export GPU_SOLVERS="${GPU_SOLVERS:-jax_cg grouped_jax}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec bash "${REPO_ROOT}/reference/run_partitioned_enclosure_all_solvers.sh"
