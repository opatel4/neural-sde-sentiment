#!/usr/bin/env bash
# Stage 00 — generate synthetic offline surfaces for all four families.
# Produces 4 CSVs (~50k surfaces each) with corrected pricing + sentiment channel.
# Expected: per-family inversion-floor rate 0.22–0.70%; SS channel constant
# across strikes (std=0) within each maturity slice.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR=${SDP_OUTDIR:-data/offline_surfaces}
mkdir -p "$OUTDIR"

for M in Heston Bates Bergomi rBergomi; do
  echo "=== generating $M  $(date) ==="
  SDP_MODEL=$M SDP_OUTDIR="$OUTDIR" \
    python3 src/generate_offline_data.py
done
echo "Stage 00 complete. Verify inversion-floor rate 0.2–0.7% per family above."
