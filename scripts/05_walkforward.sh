#!/usr/bin/env bash
# Stage 05 — expanding-window walk-forward, all 16 cells, 4 workers.
# 20 refit epochs per quarterly window, warm-started. ~5 h on one H100.
# Then the OOS comparison. Produces eval_wf_surfaces/analysis/results_master.csv (16 rows).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src/fine_tuning"
export SDP_NORM=forward SDP_USE_IN2=1 SDP_Q=0.015

run_cell () {  # model arm regime rate
  SDP_USE_SENTIMENT=$2 SDP_R=$4 \
    python3 src/fine_tuning/walkforward.py --models "$1" --regimes "$3" --epochs 20 \
    >> "$HOME/wf_${1}_${3}_${2}.log" 2>&1
}

# one model family per worker, both arms, all four regimes
worker () {  # model
  for RG in 2010-2012:0.0254 2013-2015:0.0230 2016-2019:0.0236 2020-2022:0.0169; do
    DR=${RG%%:*}; RT=${RG##*:}
    run_cell "$1" 1 "$DR" "$RT"
    run_cell "$1" 0 "$DR" "$RT"
  done
  echo "worker $1 DONE"
}

worker Heston &
worker Bates &
worker Bergomi &
worker rBergomi &
wait

python3 src/fine_tuning/compare_variants_wf.py
echo "Stage 05 complete. Verify OOS results_master.csv has 16 data rows."
