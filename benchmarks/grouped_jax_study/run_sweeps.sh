#!/usr/bin/env bash
# Zone-count sweep of the grouped JAX study: grouped JAX SBSim against EnergyPlus.
#
# Each pass runs
#   - timing at every size, both engines in one process
#   - SAC at every size in two designs, one engine per process
#       7day: one 7-day episode
#       1day: 2016 transitions on a 1-day episode (7 episodes)
#
# Usage, from the repository root:
#   benchmarks/grouped_jax_study/run_sweeps.sh <epw> <energyplus-dir> <output-dir>
#
# Optional environment variables:
#   PYTHON    Python interpreter (default: python)
#   PASSES    number of passes (default: 5)
#   PLATFORM  cpu or gpu (default: cpu)
#   ZONES     space-separated zone counts
#             (default: 1 2 3 4 5 10 16 20 30 36 40 50 75 100 125 150 200 250)

set -u

if [[ $# -ne 3 ]]; then
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

EPW=$1
EPLUS_DIR=$2
OUT=$3
PYTHON=${PYTHON:-python}
PASSES=${PASSES:-5}
PLATFORM=${PLATFORM:-cpu}
read -r -a ZONES <<< "${ZONES:-1 2 3 4 5 10 16 20 30 36 40 50 75 100 125 150 200 250}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT"
LOG=$OUT/progress.txt

run() {
  "$PYTHON" -m benchmarks.grouped_jax_study.run \
    --epw "$EPW" --energyplus-dir "$EPLUS_DIR" \
    --platform "$PLATFORM" --grouped-method dense "$@"
}

{
  echo "commit $(git rev-parse HEAD)"
  echo "epw $EPW"
  echo "energyplus $EPLUS_DIR"
  echo "platform $PLATFORM"
  echo "zones ${ZONES[*]}"
  "$PYTHON" -c "import jax, numpy, platform; print('jax', jax.__version__, '| numpy', numpy.__version__, '|', platform.platform())"
} > "$OUT/run-info.txt" 2>&1

echo "start $(date '+%F %T')" >> "$LOG"
for p in $(seq 1 "$PASSES"); do
  P=$OUT/pass$p
  mkdir -p "$P/timing" "$P/sac/7day" "$P/sac/1day"
  for z in "${ZONES[@]}"; do
    run --zones "$z" --physics-zones "$z" --days 7 \
      --engines grouped_jax energyplus --section timing \
      --output "$P/timing/z$z" > "$P/timing/z$z.log" 2>&1
    echo "pass$p timing z$z exit $? $(date '+%T')" >> "$LOG"
  done
  for design in 7day 1day; do
    if [[ $design == 7day ]]; then
      opts=(--days 7)
    else
      opts=(--days 1 --transitions 2016)
    fi
    for z in "${ZONES[@]}"; do
      for e in grouped_jax energyplus; do
        run --zones "$z" --physics-zones "$z" "${opts[@]}" \
          --engines "$e" --section sac \
          --output "$P/sac/$design/z$z-$e" > "$P/sac/$design/z$z-$e.log" 2>&1
        echo "pass$p sac $design z$z $e exit $? $(date '+%T')" >> "$LOG"
      done
    done
  done
  echo "pass$p done $(date '+%F %T')" >> "$LOG"
done
echo "done $(date '+%F %T')" >> "$LOG"
