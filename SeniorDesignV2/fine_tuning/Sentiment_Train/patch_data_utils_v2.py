"""
patch_data_utils_v2.py  -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

STEP 4 of the rebuild. Rewrites the normalisation in data_utils.py to use a
forward reconstructed from REAL rates, and replaces the damaging global price
cap with a no-arbitrage band filter.

WHAT WAS WRONG
--------------
1. RATES. The vendor `r` sits near 2-3% in every year regardless of the actual
   environment (2014: 0.0255 vs a real 3M T-bill of 0.0003). `q` is a hardcoded
   0.015. The vendor `Forward` inherits both: under real Treasury rates it
   implies a NEGATIVE dividend yield (-1.00% in 2014), which is impossible for a
   cash index. Rebuilt from FRED rates + a 1.9% dividend yield, F/S in 2014 goes
   from the vendor's 1.0018 to 0.9967 - i.e. the forward sits BELOW spot, which
   is correct when q > r. At T=2 that is a ~6% relocation of every strike.

2. PRICE CAP. filter_extreme_prices clipped normalised price at the 97th
   percentile. Price is monotonic in moneyness, so that is a moneyness filter in
   disguise: 100% of what it clipped was deep-ITM, and it clipped those BELOW
   their intrinsic value, making IV inversion impossible for lm <= -0.3.

WHAT THIS PATCH DOES
--------------------
  * loads treasury_curve.csv + dividend_yield.csv, interpolates r to each row's
    tau, and rebuilds  F = S * exp((r - q) * tau)
  * normalises prices by F (not by a reconstructed spot), so the normalised
    forward is 1 and strikes are exp(lm) exactly - no q anywhere
  * converts puts by exact forward-measure parity:  C = P + 1 - (K/F)e^{-r tau}
  * adds a row-level no-arbitrage band filter (drops ~1.2% vs the cap's 3.0%)
  * removes the global price cap from filter_extreme_prices, keeping the
    per-point winsorisation and the day-level IQR filter

Everything is gated on SDP_NORM=forward so the old path still runs unchanged.

Usage:
    python3 patch_data_utils_v2.py --dry-run
    python3 patch_data_utils_v2.py
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

p = "data_utils.py"
if not os.path.exists(p):
    sys.exit("run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train")
s = open(p).read()
if "SDP_NORM" in s:
    print("already patched"); sys.exit(0)

done, missing = [], []

# ---------------------------------------------------------------- helpers ---
HELPERS = '''

# ===========================================================================
# V2: real-rate forward reconstruction + no-arbitrage band filter
# ===========================================================================
# The vendor r/q/Forward describe a self-consistent but non-market world (see
# patch notes). Under SDP_NORM=forward we rebuild the forward from FRED
# Treasury rates and a dividend yield, normalise by F rather than by a
# reconstructed spot, and drop rows whose prices are outside the no-arbitrage
# band instead of winsorising them.
_V2_NORM   = os.environ.get("SDP_NORM", "spot").lower()        # 'forward' | 'spot'
_V2_ROOT   = os.environ.get("SDP_RATE_DIR", os.path.expanduser("~/SeniorDesignV2"))
_V2_CURVE  = None
_V2_DIVY   = None


def _v2_load_rates():
    """Lazy-load the Treasury curve and dividend yield written by the Mac scripts."""
    global _V2_CURVE, _V2_DIVY
    if _V2_CURVE is not None:
        return _V2_CURVE, _V2_DIVY
    cpath = os.path.join(_V2_ROOT, "treasury_curve.csv")
    dpath = os.path.join(_V2_ROOT, "dividend_yield.csv")
    if not (os.path.exists(cpath) and os.path.exists(dpath)):
        raise FileNotFoundError(
            f"SDP_NORM=forward needs {cpath} and {dpath}. "
            "Run build_rate_curve.py and build_dividend_yield.py on the Mac first.")
    _V2_CURVE = pd.read_csv(cpath, index_col=0, parse_dates=True).sort_index()
    _V2_DIVY  = pd.read_csv(dpath, index_col=0, parse_dates=True)["q"].sort_index()
    return _V2_CURVE, _V2_DIVY


def v2_real_rates(trade_dates, taus):
    """(r, q) per row: r interpolated across the Treasury curve at each tau."""
    curve, divy = _v2_load_rates()
    tenors = np.array([float(c) for c in curve.columns])
    td = pd.to_datetime(pd.Series(trade_dates).values)
    idx = curve.index.searchsorted(td, side="right") - 1
    idx = np.clip(idx, 0, len(curve) - 1)
    rows = curve.values[idx]                                   # (n, n_tenor)
    tau = np.asarray(taus, dtype=float)
    r = np.array([np.interp(t, tenors, row) for t, row in zip(tau, rows)])
    qi = divy.index.searchsorted(td, side="right") - 1
    qi = np.clip(qi, 0, len(divy) - 1)
    q = divy.values[qi]
    return r, q


def v2_band_mask(norm_price, k_over_f, r, tau, tol=1e-6):
    """
    Rows whose CALL-EQUIVALENT price, normalised by F, lies strictly inside the
    no-arbitrage band. In forward units the band is

        lower = e^{-r tau} * max(1 - K/F, 0)
        upper = e^{-r tau}

    Black-Scholes maps sigma in (0, inf) bijectively onto that interval, so a
    price outside it corresponds to NO implied volatility - it is a broken quote
    (stale, crossed, mis-scaled), not a mispricing to be clipped.
    """
    disc = np.exp(-np.asarray(r) * np.asarray(tau))
    lower = disc * np.maximum(1.0 - np.asarray(k_over_f), 0.0)
    upper = disc
    p = np.asarray(norm_price)
    return (p > lower * (1 + tol)) & (p < upper * (1 - tol) + tol)
'''

m = re.search(r"^(import|from)\s+.*$", s, flags=re.M)
last = 0
for mm in re.finditer(r"^(import|from)\s+.*$", s, flags=re.M):
    last = mm.end()
s = s[:last] + "\n" + HELPERS + s[last:]
done.append("v2 helpers (rate loader, band mask)")

# ------------------------------------------------- normalisation switch ------
old_norm = "    df['SpotProxy'] = df['ForwardProxy'] * np.exp(-(df['r_used'] - df['q_used']) * tau_numeric)"
new_norm = '''    if _V2_NORM == "forward":
        # V2: rebuild F from real rates; normalise by F so the normalised
        # forward is 1 and K/F = exp(lm) exactly. q disappears from the pipeline.
        _r_real, _q_real = v2_real_rates(df['Trade Date'], tau_numeric)
        df['r_used'] = _r_real
        df['q_used'] = _q_real
        _S = pd.to_numeric(df['S'], errors='coerce')
        df['ForwardProxy'] = _S * np.exp((_r_real - _q_real) * tau_numeric)
        df['SpotProxy'] = df['ForwardProxy']          # normalise by F
        print(f"[V2] forward normalisation: r_med={np.median(_r_real):.5f} "
              f"q_med={np.median(_q_real):.5f} F/S_med={(df['ForwardProxy']/_S).median():.5f}")
    else:
        df['SpotProxy'] = df['ForwardProxy'] * np.exp(-(df['r_used'] - df['q_used']) * tau_numeric)'''
if old_norm in s:
    s = s.replace(old_norm, new_norm); done.append("forward normalisation switch")
else:
    missing.append("forward normalisation switch")

# ------------------------------------------------------ put conversion -------
old_put = """        df.loc[put_mask, 'NormalizedPrice'] = (
            df.loc[put_mask, 'NormalizedPrice']
            + np.exp(-df.loc[put_mask, 'q_used'] * df.loc[put_mask, 'Tau'])
            - np.exp(-df.loc[put_mask, 'r_used'] * df.loc[put_mask, 'Tau']) * strike_over_spot.loc[put_mask]
        )"""
new_put = """        if _V2_NORM == "forward":
            # exact in forward units:  C/F = P/F + e^{-r tau}(1 - K/F)
            df.loc[put_mask, 'NormalizedPrice'] = (
                df.loc[put_mask, 'NormalizedPrice']
                + np.exp(-df.loc[put_mask, 'r_used'] * df.loc[put_mask, 'Tau'])
                * (1.0 - strike_over_spot.loc[put_mask])
            )
        else:
            df.loc[put_mask, 'NormalizedPrice'] = (
                df.loc[put_mask, 'NormalizedPrice']
                + np.exp(-df.loc[put_mask, 'q_used'] * df.loc[put_mask, 'Tau'])
                - np.exp(-df.loc[put_mask, 'r_used'] * df.loc[put_mask, 'Tau']) * strike_over_spot.loc[put_mask]
            )"""
if old_put in s:
    s = s.replace(old_put, new_put); done.append("forward-measure put conversion")
else:
    missing.append("put conversion")

# ------------------------------------------------------- band filter ---------
old_clean = """    rows_before_clean_filter = len(df)
    df = df["""
new_clean = """    if _V2_NORM == "forward":
        _kf = strike_numeric.loc[df.index] / df['ForwardProxy']
        _keep = v2_band_mask(df['NormalizedPrice'], _kf, df['r_used'], df['Tau'])
        print(f"[V2] no-arbitrage band filter: dropping {(~_keep).sum():,} of "
              f"{len(df):,} rows ({(~_keep).mean():.2%}) outside [intrinsic, forward]")
        df = df[_keep].copy()

    rows_before_clean_filter = len(df)
    df = df["""
if old_clean in s:
    s = s.replace(old_clean, new_clean, 1); done.append("no-arbitrage band filter")
else:
    missing.append("band filter")

# --------------------------------------------- disable the global cap --------
old_cap = """    global_cap = np.percentile(prices_np, global_price_cap_percentile)
    global_floor = np.percentile(prices_np, 100 - global_price_cap_percentile)
    prices_np = np.clip(prices_np, max(global_floor, 1e-7), global_cap)
    print(f"  Global price cap: {global_cap:.6f} (p{global_price_cap_percentile})")"""
new_cap = """    if os.environ.get("SDP_NORM", "spot").lower() == "forward":
        # V2: the global percentile cap is a moneyness filter in disguise -
        # price is monotonic in moneyness, so 100% of what it clipped was
        # deep-ITM, and it clipped those BELOW intrinsic. The row-level
        # no-arbitrage band filter upstream handles broken quotes properly.
        print("  Global price cap: DISABLED (band filter applied upstream)")
    else:
        global_cap = np.percentile(prices_np, global_price_cap_percentile)
        global_floor = np.percentile(prices_np, 100 - global_price_cap_percentile)
        prices_np = np.clip(prices_np, max(global_floor, 1e-7), global_cap)
        print(f"  Global price cap: {global_cap:.6f} (p{global_price_cap_percentile})")"""
if old_cap in s:
    s = s.replace(old_cap, new_cap); done.append("global price cap disabled under V2")
else:
    missing.append("global price cap")

print("patch plan:")
for d in done: print("   +", d)
for m_ in missing: print("   ! NOT MATCHED:", m_)

if a.dry_run:
    print("\n[dry-run] nothing written.")
    sys.exit(0)
if missing:
    sys.exit("\nrefusing to write with unmatched patterns - inspect manually.")

shutil.copy(p, p + ".bak_v2")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"\nwrote {p} (backup {p}.bak_v2); compiles OK")
print("\nNEXT - validation gate before ANY training:")
print("  SDP_NORM=forward python3 -c \\\"import data_utils\\\"   # imports cleanly?")
print("  then re-run the round-trip check: invert market_px back to market_iv")
print("  and confirm recovery within basis points ACROSS THE WHOLE GRID,")
print("  including lm <= -0.3 which currently fails 80% of the time.")
