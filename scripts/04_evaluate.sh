#!/usr/bin/env bash
# Stage 04 — in-sample comparison: Diebold-Mariano, no-arbitrage, persistence,
# Giacomini-White. Reads the exported surfaces. Produces results_master.csv (16 rows).
# Do NOT interrupt — the DM/GW bootstrap looks idle under buffered stdout but is working.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/online"
python3 online/compare_variants.py
echo "Stage 04 complete. Verify results_master.csv has 16 data rows."
