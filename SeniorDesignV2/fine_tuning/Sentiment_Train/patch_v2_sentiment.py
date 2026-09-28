#!/usr/bin/env python3
"""
patch_v2_sentiment.py  --  V2 sentiment standardisation wiring.

Run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train

    python3 patch_v2_sentiment.py --dry-run     # show diffs, change nothing
    python3 patch_v2_sentiment.py               # apply (backs up first)
    python3 patch_v2_sentiment.py --revert      # restore the backups

Three edits:

  1. in2_adapter.py  replace the three stale SENT_* constants (derived from
                     V1's corrupted sentiment channel) with per-regime
                     dictionaries plus a set_sent_regime() resolver.

  2. train.py        call set_sent_regime(DATE_RANGE) at the top of the
                     per-regime loop, before any data or model is built.

  3. config.py       OFFLINE_WEIGHT_BY_MODEL -> {}  (rBergomi's 0.3 override
                     was justified by V1 offline moments that no longer hold).
                     Skipped unless --revert-offline-weight is passed.

Idempotent: re-running detects the patch is already applied and does nothing.
"""

import argparse
import difflib
import os
import shutil
import sys

SUFFIX = ".bak_v2sent"

# --------------------------------------------------------------------------
# measured 2026-08-16
#   offline SS, V2 surfaces, one column per maturity (constant across strikes):
#       Heston -0.4290/0.1713   Bates -0.5125/0.1864
#       Bergomi -0.4435/0.2029  rBergomi -0.3156/0.2496
#       POOLED -0.4251/0.2165
#   online FinBERT daily_sentiment, per regime, from cleanerSPXData_*_merged:
#       2010-2012 0.2970/0.1432   2013-2015 0.2703/0.1011
#       2016-2019 0.1356/0.1173   2020-2022 0.0700/0.1074
# --------------------------------------------------------------------------

NEW_CONSTANTS = '''# --- V2 sentiment standardisation (recomputed 2026-08-16) ------------------
# Offline SS distribution, measured on the V2 surfaces after the path-axis
# fix (one value per maturity, constant across strikes):
#     pooled mean -0.4251, std 0.2165
# Online FinBERT daily_sentiment drifts markedly over the sample -- yearly
# means run +0.40 (2011) down to +0.04 (2021), a spread larger than the
# pooled std -- so shift and scale are resolved per regime. Call
# set_sent_regime(DATE_RANGE) once before training each regime.
SENT_OFFLINE_MEAN = -0.4251        # offline SS mean, pooled over families
SENT_OFFLINE_STD  = 0.2165         # offline SS std,  pooled over families

SENT_ONLINE_BY_REGIME = {          # regime -> (online mean, online std)
    "2010-2012": (0.2970, 0.1432),
    "2013-2015": (0.2703, 0.1011),
    "2016-2019": (0.1356, 0.1173),
    "2020-2022": (0.0700, 0.1074),
}

# Live values read by make_in2_batch(). Default to the pooled figures so an
# unset regime still produces a sane transform rather than silently wrong one.
SENT_SHIFT  = 0.1817                       # online mean, pooled
SENT_SCALE  = SENT_OFFLINE_STD / 0.1499    # = 1.4442, pooled
SENT_OFFSET = SENT_OFFLINE_MEAN
SENT_REGIME = None


def set_sent_regime(regime):
    """Point SENT_SHIFT / SENT_SCALE at one regime's online moments.

    Must be called before make_in2_batch() for that regime. Unknown regimes
    leave the pooled fallback in place and warn loudly rather than failing,
    so a walk-forward window label that is not one of the four canonical
    regimes still trains rather than crashing mid-run.
    """
    global SENT_SHIFT, SENT_SCALE, SENT_REGIME
    if regime not in SENT_ONLINE_BY_REGIME:
        print(f"[in2_adapter] WARNING: unknown regime {regime!r}; "
              f"keeping pooled SHIFT={SENT_SHIFT:.4f} SCALE={SENT_SCALE:.4f}")
        return
    mean, std = SENT_ONLINE_BY_REGIME[regime]
    SENT_SHIFT = mean
    SENT_SCALE = SENT_OFFLINE_STD / std
    SENT_REGIME = regime
    print(f"[in2_adapter] regime {regime}: "
          f"SENT_SHIFT={SENT_SHIFT:.4f} SENT_SCALE={SENT_SCALE:.4f} "
          f"SENT_OFFSET={SENT_OFFSET:.4f}")
'''

TRAIN_ANCHOR = '''    for DATE_RANGE in CANDIDATE_DATE_RANGES:
        print(f"\\n=== Preparing data for range {DATE_RANGE} ===")
'''

TRAIN_INSERT = '''    for DATE_RANGE in CANDIDATE_DATE_RANGES:
        print(f"\\n=== Preparing data for range {DATE_RANGE} ===")
        if os.environ.get('SDP_USE_IN2', '0') == '1':
            from in2_adapter import set_sent_regime
            set_sent_regime(DATE_RANGE)
'''


def read(path):
    with open(path) as f:
        return f.read()


def show_diff(path, old, new):
    d = list(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=path, tofile=path + " (patched)", n=2))
    if not d:
        print(f"  {path}: no change")
    else:
        sys.stdout.writelines(d)
    print()


def patch_in2_adapter(src):
    if "SENT_ONLINE_BY_REGIME" in src:
        return src, "already applied"
    try:
        start = src.index("SENT_SCALE = 1.1490")
    except ValueError:
        return src, "ERROR: 'SENT_SCALE = 1.1490' not found"
    try:
        off = src.index("SENT_OFFSET = -0.4380")
    except ValueError:
        return src, "ERROR: 'SENT_OFFSET = -0.4380' not found"
    end = src.index("\n", off) + 1
    return src[:start] + NEW_CONSTANTS + src[end:], "patched"


def patch_train(src):
    if "set_sent_regime(DATE_RANGE)" in src:
        return src, "already applied"
    if TRAIN_ANCHOR not in src:
        return src, "ERROR: per-regime loop anchor not found (check line ~165)"
    if src.count(TRAIN_ANCHOR) != 1:
        return src, f"ERROR: anchor found {src.count(TRAIN_ANCHOR)}x, expected 1"
    return src.replace(TRAIN_ANCHOR, TRAIN_INSERT, 1), "patched"


def patch_config(src):
    old = 'OFFLINE_WEIGHT_BY_MODEL = {"rBergomi": 0.3}'
    new = ('# V2: the rBergomi override was justified by V1 offline moments\n'
           '# (mean/std 0.319/0.461) produced by the corrupted sentiment\n'
           '# channel. The V2 surfaces measure -0.3156/0.2496, so the scaler\n'
           '# collapse that motivated it should no longer occur. Restored to\n'
           '# a uniform weight; watch rBergomi\'s first ~20 epochs.\n'
           'OFFLINE_WEIGHT_BY_MODEL = {}')
    if "V2: the rBergomi override" in src:
        return src, "already applied"
    if old not in src:
        return src, "ERROR: OFFLINE_WEIGHT_BY_MODEL line not found"
    return src.replace(old, new, 1), "patched"


TARGETS = [
    ("in2_adapter.py", patch_in2_adapter, True),
    ("train.py",       patch_train,       True),
    ("config.py",      patch_config,      False),   # opt-in
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print diffs, write nothing")
    ap.add_argument("--revert", action="store_true",
                    help="restore .bak_v2sent backups")
    ap.add_argument("--revert-offline-weight", action="store_true",
                    help="also set OFFLINE_WEIGHT_BY_MODEL = {} in config.py")
    args = ap.parse_args()

    if args.revert:
        for name, _, _ in TARGETS:
            bak = name + SUFFIX
            if os.path.exists(bak):
                shutil.copy2(bak, name)
                print(f"reverted {name} from {bak}")
            else:
                print(f"no backup for {name}, left alone")
        return

    missing = [n for n, _, _ in TARGETS if not os.path.exists(n)]
    if missing:
        sys.exit(f"not in the right directory -- missing: {', '.join(missing)}\n"
                 f"cd ~/SeniorDesignV2/fine_tuning/Sentiment_Train first")

    results, staged, errors = [], [], []
    for name, fn, default_on in TARGETS:
        if name == "config.py" and not args.revert_offline_weight:
            results.append((name, "skipped (pass --revert-offline-weight)"))
            continue
        old = read(name)
        new, status = fn(old)
        results.append((name, status))
        if status.startswith("ERROR"):
            errors.append(name)
        elif status == "patched":
            staged.append((name, old, new))

    print("\n=== DIFFS ===\n")
    for name, old, new in staged:
        show_diff(name, old, new)

    print("=== SUMMARY ===")
    for name, status in results:
        print(f"  {name:18s} {status}")

    if errors:
        sys.exit(f"\naborted -- {len(errors)} file(s) did not match the expected "
                 f"text. Nothing written. Inspect them by hand.")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    if not staged:
        print("\nnothing to do")
        return

    for name, old, new in staged:
        shutil.copy2(name, name + SUFFIX)
        with open(name, "w") as f:
            f.write(new)
        print(f"wrote {name}  (backup: {name}{SUFFIX})")

    print("\nNext:")
    print("  python3 -c \"import in2_adapter as a; a.set_sent_regime('2013-2015')\"")
    print("  python3 in2_adapter.py        # sanity check still passes")
    print("  python3 preflight_audit.py    # transform check still passes")


if __name__ == "__main__":
    main()
