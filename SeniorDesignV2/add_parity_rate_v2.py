"""
add_parity_rate.py -- run from ~/SeniorDesignV2

Adds an `r_parity` column to every SPX CSV in V2, WITHOUT touching the vendor's
`r`. Both columns then coexist so the two can be compared and either used.

WHY
---
The vendor `r` column is flat across tenors and does not track observable rates:
2014 shows 0.0255 at 3 months and 0.0252 at 5 years, when 3-month T-bills paid
about 0.03%. Put-call parity recovers the discount rate the options market is
actually using, from traded prices:

        C - P = S*e^{-q*tau} - K*e^{-r*tau}
    =>  r = -ln( (S*e^{-q*tau} - (C-P)) / K ) / tau

Measured across all 13 years this tracks the real rate cycle (2012: 0.0032,
2014: 0.0074, 2018: 0.0252, 2022: 0.0319) unlike the vendor column.

CAVEAT worth stating in the paper: parity-implied r absorbs any error in q.
Since q is a hardcoded 0.015 here, r_parity is an EFFECTIVE cost-of-carry rate
rather than a clean risk-free rate. It is internally consistent with the option
prices, which is what matters for calibration.

Method: for each (trade date, expiry) use near-the-money matched call/put pairs
(|K/S - 1| <= 0.10), take the median implied r, then fall back to the daily
median and finally the yearly median where no pair exists.

Usage:
    python3 add_parity_rate.py --dry-run     # report only
    python3 add_parity_rate.py               # write r_parity in place
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd

SPX = os.path.expanduser("~/SeniorDesignV2/fine_tuning/Fine_Tuning_Files/SPX_Data")
R_MIN, R_MAX = -0.005, 0.10          # plausible band for an effective rate
NTM = 0.10                            # |K/S - 1| window for the pairs

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--spx", default=SPX)
a = ap.parse_args()

files = sorted(glob.glob(f"{a.spx}/cleanerSPXData_*_merged.csv"))
if not files:
    sys.exit(f"no CSVs under {a.spx}")

print(f"{'year':>6}{'rows':>10}{'pairs':>9}{'r_vendor':>11}{'r_parity':>11}"
      f"{'p05':>9}{'p95':>9}{'covered':>9}")
print("-" * 76)

for f in files:
    year = os.path.basename(f).split("_")[1]
    d = pd.read_csv(f, low_memory=False)
    price_col = "Mid" if "Mid" in d.columns else "OptionPrice"
    need = {"Trade Date", "Expiry", "Strike", "Tau", price_col, "OptionType"}
    if not need.issubset(d.columns):
        print(f"{year:>6}  missing columns, skipped"); continue

    for c in ("Strike", "Tau", price_col, "Forward", "r", "q"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # FIXED: use the CSV's own S column. Deriving spot as
    #     Forward * exp(-(r_vendor - q_vendor)*tau)
    # makes the parity inversion CIRCULAR - r_parity then just restates the
    # vendor drift we are trying to replace. S is one clean value per trade
    # date (verified: nunique per day = 1, F/S = 1.0005 at short tau), so use
    # it directly and let parity measure the carry independently.
    r_v = np.where(d.r > 1, d.r / 100, d.r) if "r" in d.columns else np.full(len(d), np.nan)
    q_v = np.where(d.q > 1, d.q / 100, d.q) if "q" in d.columns else np.full(len(d), np.nan)
    if "S" not in d.columns:
        print(f"{year:>6}  no S column, skipped"); continue
    spot = pd.to_numeric(d["S"], errors="coerce")

    t = d.OptionType.astype(str).str.upper().str[0]
    calls = d[t == "C"][["Trade Date", "Expiry", "Strike", "Tau", price_col]]
    puts  = d[t == "P"][["Trade Date", "Expiry", "Strike", "Tau", price_col]]
    m = calls.merge(puts, on=["Trade Date", "Expiry", "Strike", "Tau"],
                    suffixes=("_c", "_p"))
    if len(m) == 0:
        print(f"{year:>6}{len(d):>10}{0:>9}   no matched pairs"); continue

    # attach the spot for those rows
    sp = pd.DataFrame({"Trade Date": d["Trade Date"], "spot": spot}).dropna()
    sp = sp.groupby("Trade Date")["spot"].median()
    m = m.join(sp, on="Trade Date")
    m = m[np.abs(m.Strike / m.spot - 1) <= NTM]

    qh = float(np.nanmedian(q_v)) if np.isfinite(q_v).any() else 0.015
    if not np.isfinite(qh):
        qh = 0.015
    arg = (m.spot * np.exp(-qh * m.Tau) - (m[f"{price_col}_c"] - m[f"{price_col}_p"])) / m.Strike
    ok = arg > 0
    m = m[ok].copy()
    m["r_imp"] = -np.log(arg[ok]) / m.Tau[ok]
    m = m[(m.r_imp > R_MIN) & (m.r_imp < R_MAX)]
    if len(m) == 0:
        print(f"{year:>6}{len(d):>10}{0:>9}   no plausible rates"); continue

    by_de = m.groupby(["Trade Date", "Expiry"])["r_imp"].median()
    by_d  = m.groupby("Trade Date")["r_imp"].median()
    yr_med = float(m.r_imp.median())

    key = list(zip(d["Trade Date"], d["Expiry"]))
    r_new = pd.Series([by_de.get(k, np.nan) for k in key], index=d.index)
    r_new = r_new.fillna(d["Trade Date"].map(by_d)).fillna(yr_med)
    covered = float(r_new.notna().mean())

    print(f"{year:>6}{len(d):>10}{len(m):>9}{np.nanmedian(r_v):>11.4f}"
          f"{np.nanmedian(r_new):>11.4f}{np.nanpercentile(r_new,5):>9.4f}"
          f"{np.nanpercentile(r_new,95):>9.4f}{covered:>9.1%}")

    if not a.dry_run:
        d["r_parity"] = r_new.values
        d.to_csv(f, index=False)

print()
if a.dry_run:
    print("[dry-run] nothing written. Re-run without --dry-run to add r_parity.")
else:
    print("r_parity written to every file. The vendor `r` column is untouched.")
print("\nNOTE: r_parity is an EFFECTIVE cost-of-carry rate (it absorbs the")
print("hardcoded q=0.015). Under forward normalisation it is needed only for")
print("the discount factor, not for the drift.")
