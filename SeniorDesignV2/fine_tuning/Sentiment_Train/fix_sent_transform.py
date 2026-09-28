"""
fix_sent_transform.py
=====================
make_in2_batch() currently does only:
        s = context[:, SENT_INDEX] * SENT_SCALE
which applies the SCALE but ignores SENT_SHIFT / SENT_OFFSET - so the mean
offset (the LARGER of the two mismatches) is never corrected.

Offline SS  : mean -0.4380  std 0.1629
Online sent : mean +0.2244  std 0.1418
The network pretrained on a channel centred at -0.438; feeding it raw online
sentiment centred at +0.224 makes every ordinary day look like a ~4-sigma
optimism spike.

Correct standardisation:
        s_out = (s_in - SENT_SHIFT) * SENT_SCALE + SENT_OFFSET
i.e. de-mean by the ONLINE mean, rescale to the OFFLINE spread, then re-centre
on the OFFLINE mean.

Run from fine_tuning/Sentiment_Train/:
    python3 fix_sent_transform.py
"""
import re, shutil, sys, os

p = "in2_adapter.py"
if not os.path.exists(p):
    sys.exit("run this from fine_tuning/Sentiment_Train/")

s = open(p).read()
old = "    s = context[:, SENT_INDEX] * SENT_SCALE          # (B,)"
new = ("    # standardise onto the OFFLINE distribution the filters were pretrained on:\n"
       "    #   de-mean by the online mean, rescale, re-centre on the offline mean\n"
       "    s = (context[:, SENT_INDEX] - SENT_SHIFT) * SENT_SCALE + SENT_OFFSET   # (B,)")

if "SENT_SHIFT) * SENT_SCALE + SENT_OFFSET" in s:
    print("already patched")
    sys.exit(0)
if old not in s:
    # fall back to a looser match
    m = re.search(r"^\s*s = context\[:, SENT_INDEX\] \* SENT_SCALE.*$", s, flags=re.M)
    if not m:
        sys.exit("could not find the SENT_SCALE line; inspect make_in2_batch manually")
    old = m.group(0)

shutil.copy(p, p + ".bak")
open(p, "w").write(s.replace(old, new))
print("patched make_in2_batch to apply SHIFT + SCALE + OFFSET")
