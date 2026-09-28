"""
fix_config_arm_and_scaler.py   (run in fine_tuning/Sentiment_Train/)

TWO FIXES
=========

1. OUTPUT DIRECTORY COLLISION  [required]
   config.py hardcodes
       MODEL_OUTPUT_ROOT = .../fine_tuning_output/sentiment_output/models
   so the sentiment arm (SDP_USE_SENTIMENT=1) and the baseline arm (=0) both
   write to the SAME folder. Checkpoints don't literally overwrite (the epoch
   number is in the filename) but the two arms become indistinguishable, which
   destroys the comparison the whole study rests on.
   -> output root now depends on the arm: sentiment_output / base_output.

2. rBERGOMI NORMALISATION MISMATCH  [the epoch-3 collapse]
   Observed on rBergomi / 2013-2015, BOTH arms:
       Offline mean/std : 0.3195 / 0.4612
       Real    mean/std : 0.1867 / 0.0698
       Blended          : 0.2796 / 0.3438
   OFFLINE_WEIGHT=0.70 leans the scaler on offline stats whose mean is 0.13
   too high and whose std is ~5x too large. scale_input then maps every real
   surface to about -0.26 with almost no spread, so the network sees a nearly
   constant input, settles in ~3 epochs and early-stops.

   Compare a healthy cell (Heston): offline 0.189/0.319 vs real 0.187/0.070 -
   the MEAN matches; only the std is inflated. rBergomi's mean is off as well,
   which is what breaks it.

   Root cause is the offline sampling range xi:[0.15,0.01] -> vol up to ~39%,
   far above 2013-2015's ~19%. Other regimes worked because their realised vol
   overlapped the synthetic distribution; 2013-2015 is the calmest and overlaps
   least.

   -> OFFLINE_WEIGHT becomes model-aware: rBergomi uses a lower weight so the
   scaler centres closer to the real data. Other families keep 0.70, so their
   results stay directly comparable to the existing runs.

USAGE
    python3 fix_config_arm_and_scaler.py
    python3 fix_config_arm_and_scaler.py --rbergomi-weight 0.30
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--rbergomi-weight", type=float, default=0.30,
                help="OFFLINE_WEIGHT to use for rBergomi (default 0.30)")
a = ap.parse_args()

p = "config.py"
if not os.path.exists(p):
    sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
shutil.copy(p, p + ".bak_armfix")
changed = []

# ---- fix 1: arm-dependent output root -------------------------------------
if "_ARM_DIR" not in s:
    m_out = re.search(r'^MODEL_OUTPUT_ROOT\s*=\s*os\.path\.join\([^)]*\)\s*$', s, flags=re.M)
    p_out = re.search(r'^PLOT_OUTPUT_ROOT\s*=\s*os\.path\.join\([^)]*\)\s*$', s, flags=re.M)
    if not (m_out and p_out):
        sys.exit("could not find MODEL_OUTPUT_ROOT / PLOT_OUTPUT_ROOT")
    block = (
        '# Output root depends on the arm so the sentiment run and the baseline run\n'
        '# do NOT land in the same folder (they are otherwise indistinguishable).\n'
        '_ARM_DIR = "sentiment_output" if os.environ.get("SDP_USE_SENTIMENT", "1") == "1" else "base_output"\n'
        'MODEL_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "fine_tuning_output", _ARM_DIR, "models")\n'
        'PLOT_OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "fine_tuning_output", _ARM_DIR, "plots")'
    )
    lo, hi = min(m_out.start(), p_out.start()), max(m_out.end(), p_out.end())
    s = s[:lo] + block + s[hi:]
    changed.append("arm-dependent output root")

# ---- fix 2: model-aware OFFLINE_WEIGHT ------------------------------------
if "OFFLINE_WEIGHT_BY_MODEL" not in s:
    m = re.search(r'^OFFLINE_WEIGHT\s*=\s*([0-9.]+)\s*$', s, flags=re.M)
    if not m:
        sys.exit("could not find OFFLINE_WEIGHT")
    default = m.group(1)
    block = (
        f'OFFLINE_WEIGHT = {default}\n'
        '\n'
        '# rBergomi\'s offline surfaces are far more volatile than the calm regimes\n'
        '# (offline mean/std 0.319/0.461 vs real 0.187/0.070 on 2013-2015), so a\n'
        '# 0.70 offline weight pushes the scaler off-centre and the network sees a\n'
        '# nearly constant input -> collapse at ~epoch 3. Lower weight for rBergomi\n'
        '# only; every other family keeps the original value so their numbers stay\n'
        '# comparable with earlier runs.\n'
        f'OFFLINE_WEIGHT_BY_MODEL = {{"rBergomi": {a.rbergomi_weight}}}\n'
        '\n'
        'def offline_weight_for(model_name):\n'
        '    return OFFLINE_WEIGHT_BY_MODEL.get(model_name, OFFLINE_WEIGHT)\n'
    )
    s = s[:m.start()] + block + s[m.end():]
    changed.append(f"OFFLINE_WEIGHT_BY_MODEL rBergomi={a.rbergomi_weight}")

open(p, "w").write(s)
print("config.py patched:")
for c in changed:
    print("   -", c)
if not changed:
    print("   (already patched)")

print("\nNOTE: train.py must actually USE offline_weight_for(). Check with:")
print("    grep -n 'OFFLINE_WEIGHT\\|offline_weight' train.py")
