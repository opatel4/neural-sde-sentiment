#!/usr/bin/env bash
# Stage 06 — delta-hedged P&L variance, all 16 cells, per-regime rate.
# 20k MC paths/day, stride 1 (every trading day). ~2 h. Produces hedging_all.log
# (16 result lines) + per-cell P&L figures.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src/fine_tuning" SDP_Q=0.015

for M in Heston Bates Bergomi rBergomi; do
  for RG in 2010-2012:0.0254 2013-2015:0.0230 2016-2019:0.0236 2020-2022:0.0169; do
    DR=${RG%%:*}; RT=${RG##*:}
    echo "=== $M $DR (r=$RT) ==="
    SDP_R=$RT python3 src/fine_tuning/run_hedging.py --model "$M" --regime "$DR"
  done
done 2>&1 | tee "$HOME/hedging_all.log"
echo "Stage 06 complete. Verify 16 'lower hedging' result lines."
