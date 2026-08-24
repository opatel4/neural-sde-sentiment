#!/usr/bin/env bash
# Stage 03 — export per-day predicted/observed surfaces for both arms.
# Produces 16 .npz per arm under results/eval_surfaces/{sentiment,nonsent}/.
# Expected: 16 + 16, no [skip] lines, price cap DISABLED, per-regime SENT_SCALE printed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/online"
export SDP_NORM=forward SDP_USE_IN2=1 SDP_Q=0.015

SDP_USE_SENTIMENT=1 python3 online/export_surfaces.py
SDP_USE_SENTIMENT=0 python3 online/export_surfaces.py
echo "Stage 03 complete. Verify 16 npz per arm."
