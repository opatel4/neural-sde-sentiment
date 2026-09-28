"""
validate_norm_rowlevel.py -- run from ~/SeniorDesignV2

A CAREFUL consistency test, at ROW level, with the discount factor included.

WHY THE EARLIER TESTS WERE INCONCLUSIVE
---------------------------------------
Three previous attempts each measured something other than what they claimed:

  1. validate_v2_norm.py inverted GRID prices against idealised strikes
     exp(lm). Grid cells are BINNED from options at nearby moneyness, so the
     idealised strike is wrong for most cells. Error grew with |lm| - the
     signature of binning, not of bad normalisation.

  2. The re-pricing check had the same binning flaw.

  3. The V1-vs-V2 comparison set the discount to 1.0. Under V1 the vendor
     r ~ 0.025 puts a real discount in the stored prices; under V2 r ~ 0.0005
     does not. At T=2 that alone is a ~5% price difference, so the comparison
     was biased toward V1 by construction.

WHAT THIS DOES INSTEAD
----------------------
Rebuilds the row-level frame exactly as data_utils does (same rate source, same
normalisation, same put conversion), then checks Black-Scholes consistency on
INDIVIDUAL OPTIONS - no binning, correct discount, each row at its own K/F and
tau:

    C_norm  ==  e^{-r tau} [ N(d1) - (K/F) N(d2) ]      (forward units)
    C_norm  ==  N(d1) - (K/S)e^{-r tau} N(d2)           (spot units, V1)

Run it for both conventions and compare like with like.

Usage:
    python3 validate_norm_rowlevel.py 2014
    SDP_NORM=spot    python3 validate_norm_rowlevel.py 2014
    SDP_NORM=forward python3 validate_norm_rowlevel.py 2014
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm as N

sys.path.insert(0, "fine_tuning/Sentiment_Train")
YEAR = sys.argv[1] if len(sys.argv) > 1 else "2014"
MODE = os.environ.get("SDP_NORM", "spot").lower()
SPX  = "fine_tuning/Fine_Tuning_Files/SPX_Data"

d = pd.read_csv(f"{SPX}/cleanerSPXData_{YEAR}_merged.csv", low_memory=False)
price_col = "Mid" if "Mid" in d.columns else "OptionPrice"
for c in ("Strike", "Tau", "Forward", "S", "r", "q", price_col, "Implied Volatility"):
    if c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=["Strike", "Tau", "S", price_col, "Implied Volatility"])
d = d[(d.Tau > 0.02) & (d.Tau < 3)]

# vendor IV is in percent in these files
ivv = d["Implied Volatility"].values
if np.nanmedian(ivv) > 1.5:
    ivv = ivv / 100.0

if MODE == "forward":
    # same rate source data_utils uses under SDP_NORM=forward
    curve = pd.read_csv("treasury_curve.csv", index_col=0, parse_dates=True).sort_index()
    divy  = pd.read_csv("dividend_yield.csv", index_col=0, parse_dates=True)["q"].sort_index()
    tenors = np.array([float(c) for c in curve.columns])
    td = pd.to_datetime(d["Trade Date"])
    ci = np.clip(curve.index.searchsorted(td, side="right") - 1, 0, len(curve) - 1)
    rows = curve.values[ci]
    r = np.array([np.interp(t, tenors, row) for t, row in zip(d.Tau.values, rows)])
    qi = np.clip(divy.index.searchsorted(td, side="right") - 1, 0, len(divy) - 1)
    q = divy.values[qi]
    F = d.S.values * np.exp((r - q) * d.Tau.values)
    denom = F                                     # normalise by the forward
else:
    r = np.where(d.r.values > 1, d.r.values / 100, d.r.values)
    q = np.where(d.q.values > 1, d.q.values / 100, d.q.values)
    F = d.Forward.values
    denom = F * np.exp(-(r - q) * d.Tau.values)   # SpotProxy

px_norm = d[price_col].values / denom
kf = d.Strike.values / F                          # K/F is convention-free
tau = d.Tau.values
disc = np.exp(-r * tau)

# put -> call equivalent, in whichever units we are working
t = d.OptionType.astype(str).str.upper().str[0].values
is_put = t == "P"
if MODE == "forward":
    px_norm = np.where(is_put, px_norm + disc * (1.0 - kf), px_norm)
else:
    k_over_spot = d.Strike.values / denom
    px_norm = np.where(is_put,
                       px_norm + np.exp(-q * tau) - np.exp(-r * tau) * k_over_spot,
                       px_norm)

# Black-Scholes in the SAME units, using the VENDOR iv
srt = ivv * np.sqrt(tau)
d1 = (-np.log(kf) + 0.5 * ivv**2 * tau) / np.maximum(srt, 1e-12)
d2 = d1 - srt
if MODE == "forward":
    model = disc * (N.cdf(d1) - kf * N.cdf(d2))
else:
    fwd_over_spot = F / denom
    model = disc * (fwd_over_spot * N.cdf(d1) - (d.Strike.values / denom) * N.cdf(d2))

err = model - px_norm
ok = np.isfinite(err)
lm = np.log(kf)
bins = [-10, -0.4, -0.2, -0.05, 0.05, 0.2, 0.4, 10]
lab  = ["<-0.4", "-0.4..-0.2", "-0.2..-0.05", "ATM", "0.05..0.2", "0.2..0.4", ">0.4"]

print(f"\nSDP_NORM={MODE}   year={YEAR}   rows={ok.sum():,}")
print(f"  r_med={np.nanmedian(r):.5f}  q_med={np.nanmedian(q):.5f}  F/S_med={np.nanmedian(F/d.S.values):.5f}")
print(f"\n  {'moneyness':<14}{'n':>9}{'med|err|':>11}{'p95|err|':>11}{'med px':>10}{'rel err':>10}")
cat = pd.cut(lm, bins, labels=lab)
for L in lab:
    m = ok & np.asarray(cat == L)
    if m.sum() == 0:
        print(f"  {L:<14}{0:>9}"); continue
    me = float(np.median(np.abs(err[m])))
    p95 = float(np.percentile(np.abs(err[m]), 95))
    mp = float(np.median(px_norm[m]))
    print(f"  {L:<14}{m.sum():>9,}{me:>11.6f}{p95:>11.6f}{mp:>10.5f}{me/max(mp,1e-9):>10.2%}")
print(f"\n  OVERALL       {ok.sum():>9,}{float(np.median(np.abs(err[ok]))):>11.6f}"
      f"{float(np.percentile(np.abs(err[ok]),95)):>11.6f}"
      f"{float(np.median(px_norm[ok])):>10.5f}"
      f"{float(np.median(np.abs(err[ok])))/max(float(np.median(px_norm[ok])),1e-9):>10.2%}")
print("\n  Interpretation: this uses the VENDOR IV, so the residual measures how")
print("  well each convention reproduces the vendor's own quotes. Lower is better,")
print("  but note the vendor IV was itself computed under the vendor's forward -")
print("  so a small V1 advantage here is expected and is NOT evidence V1 is more")
print("  internally consistent. What matters is that neither shows a LARGE or")
print("  moneyness-skewed residual.")
