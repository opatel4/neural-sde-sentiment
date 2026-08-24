#!/usr/bin/env bash
# Stage 01 — offline supervised pre-training of the CNN, both arms, all families.
# Produces 8 checkpoints (4 models x {In1 baseline, In2 sentiment}).
# Expected: In2 validation loss below In1 in every family (see paper Table 2).
set -euo pipefail
cd "$(dirname "$0")/.."

SDP_DATA_DIR=${SDP_DATA_DIR:-data/offline_surfaces} \
SDP_WEIGHTS_DIR=${SDP_WEIGHTS_DIR:-results/offline_models} \
SDP_SENTIMENT=both \
  python3 src/pretrain_offline.py
echo "Stage 01 complete. Expect 8 checkpoints in results/offline_models/."
