"""
fix_hedging_rates.py   (run in fine_tuning/Sentiment_Train/)

PROBLEM
-------
hedging.py and run_hedging.py both hardcode

    R, Q = 0.045, 0.011          # drift 0.034

which is the EXACT forward-drift bug this project spent a retrain eliminating.
The models were fine-tuned under per-regime market rates:

    2010-2012  r=0.0254     2013-2015  r=0.0230
    2016-2019  r=0.0236     2020-2022  r=0.0169     q=0.015 throughout

Those constants are used in three load-bearing places:
  * daily_spot()          - fallback when r/q columns are absent
  * _price_point()        - the r,q passed into every MC pricing
  * rolling_atm_hedge_pnl - the ATM-forward strike F_t = S_t*exp((R-Q)*T)

Pricing calibrated parameters under a drift ~4x the market's re-introduces the
error and makes the hedging numbers non-comparable with the rest of the study.

FIX
---
R and Q read SDP_R / SDP_Q, the same env vars trainer.py, export_surfaces.py and
walkforward.py use. Defaults match the pooled median (r=0.0231, q=0.0150) so an
unset environment is still sane rather than silently wrong.

Usage:  python3 fix_hedging_rates.py
Then:   SDP_R=0.0236 SDP_Q=0.015 python3 run_hedging.py --model Bergomi --regime 2016-2019
"""
import os, re, sys, shutil

BLOCK = '''# PATCHED: read the same per-regime rates the rest of the pipeline uses.
# The old hardcoded 0.045/0.011 gave drift 0.034 vs the market's ~0.008, which is
# the bug the retrain fixed. Set SDP_R per regime before running:
#   2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 | 2020-2022 0.0169
R = float(os.environ.get("SDP_R", 0.0231))
Q = float(os.environ.get("SDP_Q", 0.0150))'''

patched = []
for p in ("hedging.py", "run_hedging.py"):
    if not os.path.exists(p):
        print(f"  -- {p} not found"); continue
    s = open(p).read()
    if "SDP_R" in s:
        print(f"  .. {p} already patched"); continue
    m = re.search(r"^R,\s*Q\s*=\s*0\.045,\s*0\.011\s*$", s, flags=re.M)
    if not m:
        print(f"  -- {p}: could not find 'R, Q = 0.045, 0.011'"); continue
    shutil.copy(p, p + ".bak")
    s = s[:m.start()] + BLOCK + s[m.end():]
    if not re.search(r"^import os\b|^import os,", s, flags=re.M):
        s = "import os\n" + s
    open(p, "w").write(s)
    import py_compile; py_compile.compile(p, doraise=True)
    patched.append(p)
    print(f"  OK {p} -> R,Q from SDP_R/SDP_Q (compiles)")

print()
if patched:
    print("Verify:")
    print("  SDP_R=0.0236 SDP_Q=0.015 python3 -c \"import hedging; print(hedging.R, hedging.Q)\"")
    print("  python3 hedging.py --selftest      # FD delta must match e^-qT N(d1)")
print()
print("STILL OPEN (design, not code) - state these in the paper:")
print("  * surfaces are S-normalised, so this is a RELATIVE NS-vs-S comparison,")
print("    not an absolute tradable P&L (hedging.py's own docstring says so).")
print("  * no transaction costs; daily rebalancing.")
print("  * single point tested: ATM, T=0.25. Sentiment effects in the wings or")
print("    at long maturities are invisible to this design.")
