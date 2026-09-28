"""
patch_recompute_iv.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

STEP 5. Under SDP_NORM=forward, derive the market IV grid FROM the normalised
prices instead of carrying the vendor's `Implied Volatility` column.

WHY
---
The validation gate failed with a symmetric V-shaped error: +0.194 at lm=-0.5,
+0.191 at lm=+0.5, near zero around lm=+0.1. That is the signature of two
different moneyness conventions, not of a broken price.

The vendor's IV column was computed against the VENDOR's forward. V2 rebuilds
the forward from real Treasury rates and a 1.9% dividend yield, which moves F by
~1.5% in 2014 (F/S 1.0018 -> 0.9959). Strikes are therefore at different
moneyness than the vendor assumed, so the vendor IV no longer corresponds to the
V2 prices. Keeping it would mean every evaluation compares model output against
an IV surface built on a different convention.

FIX
---
Invert the V2-normalised call-equivalent prices with Black-Scholes in FORWARD
units, where F = 1 and K/F = exp(lm) exactly:

    C/F = e^{-r tau} [ N(d1) - (K/F) N(d2) ],
    d1  = ( -ln(K/F) + 0.5 sigma^2 tau ) / (sigma sqrt(tau))

Prices and IVs then agree by construction, and the gate becomes a real test of
the normalisation rather than a convention mismatch.

The vendor IV is retained as `ImpliedVolVendor` for comparison.

Usage:
    python3 patch_recompute_iv.py --dry-run
    python3 patch_recompute_iv.py
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

p = "data_utils.py"
if not os.path.exists(p):
    sys.exit("run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train")
s = open(p).read()
if "v2_iv_from_forward_price" in s:
    print("already patched"); sys.exit(0)

HELPER = '''

def v2_iv_from_forward_price(norm_price, k_over_f, tau, r, lo=1e-4, hi=3.0, iters=60):
    """
    Implied vol from a call price already normalised by the forward.

    In forward units F = 1, so
        C/F = e^{-r tau} [ N(d1) - k N(d2) ],  k = K/F,
        d1  = ( -ln k + 0.5 sig^2 tau ) / (sig sqrt(tau)),  d2 = d1 - sig sqrt(tau)

    Vectorised bisection. Returns NaN where the price is outside the
    no-arbitrage band (no positive sigma reproduces it).
    """
    from scipy.stats import norm as _nrm
    p   = np.asarray(norm_price, dtype=float)
    k   = np.asarray(k_over_f,   dtype=float)
    t   = np.maximum(np.asarray(tau, dtype=float), 1e-9)
    disc= np.exp(-np.asarray(r, dtype=float) * t)

    lower = disc * np.maximum(1.0 - k, 0.0)
    upper = disc
    bad   = ~np.isfinite(p) | (p <= lower + 1e-12) | (p >= upper - 1e-12)

    a_ = np.full_like(p, lo); b_ = np.full_like(p, hi)
    for _ in range(iters):
        mid = 0.5 * (a_ + b_)
        srt = mid * np.sqrt(t)
        d1  = (-np.log(np.maximum(k, 1e-12)) + 0.5 * mid**2 * t) / np.maximum(srt, 1e-12)
        d2  = d1 - srt
        c   = disc * (_nrm.cdf(d1) - k * _nrm.cdf(d2))
        hi_side = c > p
        b_ = np.where(hi_side, mid, b_)
        a_ = np.where(hi_side, a_, mid)
    out = 0.5 * (a_ + b_)
    out[bad] = np.nan
    return out
'''

# insert helper after the v2 band mask
anchor = "    return (p > lower * (1 + tol)) & (p < upper * (1 - tol) + tol)"
if anchor not in s:
    sys.exit("could not find v2_band_mask to anchor the helper")
s = s.replace(anchor, anchor + "\n" + HELPER, 1)

# recompute ImpliedVolUsed under forward normalisation, right after the band filter
old = """    rows_before_clean_filter = len(df)
    df = df["""
new = """    if _V2_NORM == "forward":
        # V2: derive IV from the V2-normalised price so prices and IVs share one
        # convention. The vendor IV was computed against the vendor forward and
        # is inconsistent with the rebuilt one; keep it only for comparison.
        df['ImpliedVolVendor'] = df['ImpliedVolUsed']
        _kf2 = strike_numeric.loc[df.index] / df['ForwardProxy']
        df['ImpliedVolUsed'] = v2_iv_from_forward_price(
            df['NormalizedPrice'].values, _kf2.values,
            df['Tau'].values, df['r_used'].values)
        _n_ok = np.isfinite(df['ImpliedVolUsed']).mean()
        _d = (df['ImpliedVolUsed'] - df['ImpliedVolVendor']).dropna()
        print(f"[V2] IV recomputed from forward-normalised prices: "
              f"{_n_ok:.1%} invertible, median shift vs vendor "
              f"{_d.median():+.4f} vol pts")

    rows_before_clean_filter = len(df)
    df = df["""
if old not in s:
    sys.exit("could not find the clean-filter anchor")
s = s.replace(old, new, 1)

print("patch plan:")
print("   + v2_iv_from_forward_price (vectorised forward-units inversion)")
print("   + ImpliedVolUsed recomputed from V2 prices (vendor IV kept as ImpliedVolVendor)")

if a.dry_run:
    print("\n[dry-run] nothing written."); sys.exit(0)

shutil.copy(p, p + ".bak_iv")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"\nwrote {p} (backup {p}.bak_iv); compiles OK")
print("\nRe-run the gate:")
print("  cd ~/SeniorDesignV2 && SDP_NORM=forward python3 validate_v2_norm.py 2013-2015")
print("\nNow the round-trip should be near-exact by construction. If it is NOT,")
print("the problem is in the price normalisation itself, not the convention.")
