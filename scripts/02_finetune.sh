#!/usr/bin/env bash
# Stage 02 — online fine-tuning of all 32 cells (4 models x 4 regimes x 2 arms).
# 4 concurrent workers, one regime each, both arms. ~4–13 h on one H100.
# Expected: 32 best_model_final_*.pth, zero collapse checkpoints.
#
# REQUIRED ENV (set below): SDP_NORM=forward disables the legacy price cap;
# omitting it silently corrupts ~20% of inversions. See README §3.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src/fine_tuning"
export SDP_USE_IN2=1 SDP_NORM=forward SDP_Q=0.015

run_cell () {  # regime rate arm
  SDP_USE_SENTIMENT=$3 SDP_R=$2 \
    python3 src/fine_tuning/train.py --date-ranges "$1" >> "$HOME/ft_${1}_${3}.log" 2>&1
}

worker () {  # regime rate  -> runs both arms sequentially
  run_cell "$1" "$2" 1
  run_cell "$1" "$2" 0
  echo "worker $1 DONE"
}

worker 2010-2012 0.0254 &
worker 2013-2015 0.0230 &
worker 2016-2019 0.0236 &
worker 2020-2022 0.0169 &
wait
echo "Stage 02 complete. Verify: find fine_tuning_output -name 'best_model_final_*.pth' | wc -l  == 32"
