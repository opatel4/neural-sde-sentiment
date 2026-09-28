"""
validate_v2_norm.py -- run from ~/SeniorDesignV2

THE GATE. Run this before any V2 training.

Loads one regime under SDP_NORM=forward, then checks the one thing that must be
true if the normalisation is right: taking the stored market PRICES and inverting
them back to implied vol must recover the stored market IV - everywhere on the
grid, not just at the money.

Under the OLD (spot) convention this failed badly:
    ATM      round-trip error +0.004      (fine)
    lm=+0.5  round-trip error +0.215      (bad)
    lm=-0.5  0% inversion success         (prices below intrinsic)
    overall  80% inversion success
because the forward used to build the grid disagreed with the forward implicit
in the data, and the p97 price cap had clipped deep-ITM below intrinsic.

Under the NEW convention the normalised forward is 1 by construction and strikes
are exp(lm) exactly, so the round-trip should be near-exact everywhere.

PASS criteria:
    inversion success   > 98% at every moneyness column
    |median error|      < 0.002 vol points at every column
    no column with 100% failure
"""
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault("SDP_NORM", "forward")
sys.path.insert(0, "fine_tuning/Sentiment_Train")

from scipy.stats import norm as _norm
import data_utils as du
from config import get_spx_files_for_range

REGIME = sys.argv[1] if len(sys.argv) > 1 else "2013-2015"
LMS  = np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5])
MATS = np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0])

print(f"SDP_NORM = {os.environ['SDP_NORM']}   regime = {REGIME}")
res = du.create_iv_and_price_grids_from_raw(
    get_spx_files_for_range(REGIME), max_days=None, use_calls_only=False,
    min_points_per_day=6, min_tau_per_day=2, min_lm_per_day=3)
iv, px, ctx, dates, lm_conv = res if len(res) == 5 else (*res, "k_over_f")
iv = iv.numpy().squeeze(1); px = px.numpy().squeeze(1)
print(f"loaded {iv.shape[0]} days, convention={lm_conv}")

# In forward units: F=1, K=exp(lm), price is already C/F.
K = np.exp(LMS)[None, :]

def bs_call_fwd(sig, k, T, disc=1.0):
    sig = np.maximum(sig, 1e-9); T = np.maximum(T, 1e-9)
    srt = sig*np.sqrt(T)
    d1 = (-np.log(k) + 0.5*sig**2*T)/srt
    return disc*( _norm.cdf(d1) - k*_norm.cdf(d1-srt) )

def invert(price, k, T, disc=1.0, lo=1e-4, hi=3.0):
    if not np.isfinite(price): return np.nan
    if price <= disc*max(1-k,0)+1e-12 or price >= disc-1e-12: return np.nan
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if bs_call_fwd(mid,k,T,disc) - price > 0: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

N = min(300, iv.shape[0])
re_iv = np.full((N,8,11), np.nan)
for n in range(N):
    for i in range(8):
        for j in range(11):
            re_iv[n,i,j] = invert(px[n,i,j], float(K[0,j]), MATS[i])

print("\n" + "="*70)
print("ROUND-TRIP BY MONEYNESS COLUMN")
print("="*70)
print(f"  {'lm':>6}{'inv_ok':>9}{'med_err':>11}{'p95|err|':>11}{'mkt_iv':>9}")
worst_ok, worst_err = 1.0, 0.0
for j in range(11):
    a = iv[:N,:,j]; b = re_iv[:,:,j]
    m = np.isfinite(a) & np.isfinite(b)
    ok = float(np.isfinite(b).mean()); worst_ok = min(worst_ok, ok)
    if m.any():
        d = b[m]-a[m]
        med = float(np.median(d)); p95 = float(np.percentile(np.abs(d),95))
        worst_err = max(worst_err, abs(med))
        print(f"  {LMS[j]:>+6.1f}{ok:>9.1%}{med:>+11.5f}{p95:>11.5f}{np.nanmean(a):>9.4f}")
    else:
        print(f"  {LMS[j]:>+6.1f}{ok:>9.1%}{'--':>11}{'--':>11}{np.nanmean(a):>9.4f}")

print("\n" + "="*70)
ok_pass  = worst_ok > 0.98
err_pass = worst_err < 0.002
print(f"  worst column inversion success : {worst_ok:.1%}   {'PASS' if ok_pass else 'FAIL'}")
print(f"  worst column |median error|    : {worst_err:.5f}  {'PASS' if err_pass else 'FAIL'}")
print()
if ok_pass and err_pass:
    print("  GATE PASSED - normalisation is self-consistent across the whole grid.")
    print("  Safe to regenerate offline data and start V2 training.")
else:
    print("  GATE FAILED - do NOT start training. The normalisation still")
    print("  disagrees with the stored prices somewhere on the surface.")
print("="*70)
