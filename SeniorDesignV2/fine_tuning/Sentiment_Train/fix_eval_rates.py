"""
fix_eval_rates.py   (run in fine_tuning/Sentiment_Train/)

Fixes two things in eval_upgrades.py.

1) RATE CONSTANTS
   S0, R, Q = 1.0, 0.045, 0.011      <- the same drift bug the retrain removed.
   These feed forwards() -> strikes() -> every IV inversion, every vega weight,
   and the benchmark table. The models were fine-tuned under per-regime market
   rates (2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 |
   2020-2022 0.0169, q=0.015), so the evaluation grid should use the same.

   Scope note: because BOTH arms are scored on the identical grid and weights,
   the rate error largely cancels in the NS-vs-S *difference*. What it distorts
   is the ABSOLUTE level of vega-weighted RMSE and the benchmark comparison.
   Expect the verdicts to be stable and the levels to move.

   put_call_parity_check() also hardcodes r=0.045, q=0.011 -> same fix.

2) lm_convention DEFAULT
   strikes() defaults to 'f_over_k', but this dataset detects 'k_over_f'.
   Any call that omits the argument inverts the strike grid across the smile.
   Default changed to 'k_over_f'; callers that pass it explicitly are unaffected.

Usage:
    python3 fix_eval_rates.py
    # then, from the repo root, with the regime's rate set:
    SDP_R=0.0230 SDP_Q=0.015 python3 fine_tuning/Sentiment_Train/compare_variants.py
"""
import os, re, sys, shutil

p = "eval_upgrades.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
if "SDP_R" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak")
done, missing = [], []

# ---- 1a. module-level constants ------------------------------------------
old = "S0, R, Q = 1.0, 0.045, 0.011"
new = ('''# PATCHED: read the same rates the pipeline was trained/exported with.
# Old hardcoded 0.045/0.011 gives drift 0.034 vs the market's ~0.008.
# Set SDP_R per regime before running (q is 0.015 in every year):
#   2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 | 2020-2022 0.0169
import os as _os
S0 = 1.0
R = float(_os.environ.get("SDP_R", 0.0231))
Q = float(_os.environ.get("SDP_Q", 0.0150))''')
if old in s: s = s.replace(old, new); done.append("module R/Q from SDP_R/SDP_Q")
else: missing.append("module R/Q")

# ---- 1b. put_call_parity_check ------------------------------------------
old2 = "    r = 0.045; q = 0.011"
new2 = "    r = R; q = Q      # PATCHED: was hardcoded 0.045/0.011"
if old2 in s: s = s.replace(old2, new2); done.append("put_call_parity_check r/q")
else: missing.append("put_call_parity_check r/q")

# ---- 2. lm_convention default -------------------------------------------
n_before = s.count("lm_convention='f_over_k'")
s = s.replace("lm_convention='f_over_k'", "lm_convention='k_over_f'")
if n_before:
    done.append(f"lm_convention default f_over_k -> k_over_f ({n_before} signatures)")
else:
    missing.append("lm_convention default")

# strikes() body: the branch comment/logic still needs to be correct either way
s = s.replace("    return F * np.exp(-LMS[None, :])                  # f_over_k (your default)",
              "    return F * np.exp(-LMS[None, :])                  # f_over_k")

open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print("eval_upgrades.py patched:")
for d in done: print("   -", d)
if missing:
    print("NOT APPLIED (inspect manually):")
    for m in missing: print("   -", m)
print("compiles OK")
print()
print("VERIFY:")
print("  SDP_R=0.0230 SDP_Q=0.015 python3 -c \"import eval_upgrades as e;"
      " print(e.R, e.Q); print(e.strikes()[0][:3])\"")
print()
print("RERUN (per regime, so each gets its own rate):")
print("  cd ~/SeniorDesignProject")
print("  for dr in 2010-2012:0.0254 2013-2015:0.0230 2016-2019:0.0236 2020-2022:0.0169; do")
print("    SDP_R=${dr##*:} SDP_Q=0.015 python3 fine_tuning/Sentiment_Train/compare_variants.py")
print("  done")
print("  python3 fine_tuning/Sentiment_Train/report_extras.py")
print()
print("NOTE: compare_variants.py loops over ALL regimes internally, so a single")
print("SDP_R cannot be right for every cell in one invocation. Check whether it")
print("accepts a regime filter; if not, the honest options are (a) accept one")
print("pooled rate and say so, or (b) add a per-cell rate lookup.")
