#!/usr/bin/env bash
# smoke_rbergomi.sh -- prove rBergomi/2013-2015 trains at OFFLINE_WEIGHT 0.70
#
# Run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train
#
# It runs ONE cell (rBergomi, 2013-2015, sentiment arm) for a short epoch
# budget and checks the loss actually descends past the epoch-3 collapse
# point that motivated the old 0.3 override. Nothing else is trained.
#
# Env it sets (the four that must be right for a real run):
#   SDP_NORM=forward       -> disables the p97 price cap (band filter upstream)
#   SDP_USE_IN2=1          -> load the 2-channel In2 weights, call set_sent_regime
#   SDP_USE_SENTIMENT=1    -> sentiment arm; output dir = sentiment_output
#   SDP_MAX_EPOCHS=40      -> short; only need to clear the collapse window

set -euo pipefail
cd "$(dirname "$0")"

export SDP_NORM=forward
export SDP_USE_IN2=1
export SDP_USE_SENTIMENT=1
export SDP_MAX_EPOCHS="${SDP_MAX_EPOCHS:-40}"

echo "=== rBergomi / 2013-2015 smoke @ OFFLINE_WEIGHT=$(python3 -c 'import config;print(config.offline_weight_for("rBergomi"))') ==="
echo "    SDP_NORM=$SDP_NORM  SDP_USE_IN2=$SDP_USE_IN2  SDP_USE_SENTIMENT=$SDP_USE_SENTIMENT  epochs=$SDP_MAX_EPOCHS"
echo

python3 train.py --models rBergomi --date-ranges 2013-2015 2>&1 | tee ~/smoke_rbergomi.log

echo
echo "=== verdict ==="
python3 - <<'PY'
import re
lines = open("/home/opatel4/smoke_rbergomi.log").read().splitlines()
vals = []
for l in lines:
    m = re.search(r"Val Loss:\s*([0-9.]+)", l)
    if m:
        vals.append(float(m.group(1)))
if not vals:
    print("NO val-loss lines found -- inspect ~/smoke_rbergomi.log by hand")
    raise SystemExit
first = vals[0]
best  = min(vals)
last_epoch = len(vals)
print(f"epochs logged : {last_epoch}")
print(f"first val loss: {first:.6f}")
print(f"best  val loss: {best:.6f}")
print(f"improvement   : {100*(first-best)/first:+.1f}%")
if last_epoch < 5:
    print("VERDICT: STOPPED EARLY (<5 epochs) -- looks like the collapse. Keep 0.3.")
elif best < first * 0.98:
    print("VERDICT: TRAINS. Loss descends past the collapse window. 0.70 is safe -- keep the revert.")
else:
    print("VERDICT: FLAT. Not obviously collapsing but not learning either -- inspect before trusting.")
PY
