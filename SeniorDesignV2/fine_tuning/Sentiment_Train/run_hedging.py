"""
run_hedging.py   (Phase 2 economic test — delta-hedged P&L)
==========================================================
DROP INTO fine_tuning/Sentiment_Train/  (needs config, data_utils, pricers,
utils, hedging alongside it) and run from inside that folder:

    cd fine_tuning/Sentiment_Train
    python run_hedging.py --model Heston --regime 2013-2015
    cd ../..

It reads the calibrated parameters the exporter already saved (for BOTH
variants), reconstructs the daily SPX spot from the same CSVs, then delta-hedges
a rolling constant-maturity ATM option with each variant's MODEL delta and
compares hedging-error variance. Lower variance = better hedger.

COST: each hedged day runs several Monte-Carlo pricings. Defaults are trimmed
(fewer paths, and --stride to hedge every k-th day) so a regime finishes in
minutes on CPU. Raise --paths / drop --stride for the final paper numbers.

OUTPUT:
    full_project_output/eval_surfaces/figures/hedge_<Model>_<Range>.png (+ .pdf)
    prints the NS vs S hedging-error variance.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROJECT_ROOT, get_spx_files_for_range
import hedging

REPO_ROOT = os.path.dirname(PROJECT_ROOT)
EVAL = os.path.join(REPO_ROOT, "full_project_output", "eval_surfaces")
FIG  = os.path.join(EVAL, "figures")
# PATCHED: read the same per-regime rates the rest of the pipeline uses.
# The old hardcoded 0.045/0.011 gave drift 0.034 vs the market's ~0.008, which is
# the bug the retrain fixed. Set SDP_R per regime before running:
#   2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 | 2020-2022 0.0169
R = float(os.environ.get("SDP_R", 0.0231))
Q = float(os.environ.get("SDP_Q", 0.0150))
def daily_spot(date_range):
    """Per-trade-date spot from the raw CSVs: SpotProxy = Forward * e^{-(r-q)tau}."""
    frames = []
    for f in get_spx_files_for_range(date_range):
        df = pd.read_csv(f, low_memory=False)
        if not {"Trade Date", "Forward", "Tau"}.issubset(df.columns):
            continue
        df["Trade Date"] = pd.to_datetime(df["Trade Date"])
        r = pd.to_numeric(df["r"], errors="coerce") if "r" in df.columns else R
        q = pd.to_numeric(df["q"], errors="coerce") if "q" in df.columns else Q
        fwd = pd.to_numeric(df["Forward"], errors="coerce")
        tau = pd.to_numeric(df["Tau"], errors="coerce")
        df["_spot"] = fwd * np.exp(-(r - q) * tau)
        frames.append(df[["Trade Date", "_spot"]].dropna())
    if not frames:
        return {}
    alld = pd.concat(frames)
    s = alld.groupby("Trade Date")["_spot"].median()
    return {str(k.date()): float(v) for k, v in s.items()}


def load_params(variant, model, date_range):
    p = os.path.join(EVAL, variant, f"{model}_{date_range}.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["params"], np.array([str(x) for x in d["dates"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Heston")
    ap.add_argument("--regime", default="2013-2015")
    ap.add_argument("--target_T", type=float, default=0.25)
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=1, help="hedge every k-th day")
    args = ap.parse_args()

    par_ns, dts_ns = load_params("nonsent",   args.model, args.regime)
    par_s,  dts_s  = load_params("sentiment", args.model, args.regime)
    if par_s is None or par_ns is None:
        print("ERROR: run export_surfaces.py in BOTH packages first "
              f"(missing npz for {args.model} {args.regime}).")
        return

    # common dates, in order
    common = np.intersect1d(dts_ns, dts_s)
    ia = np.array([np.where(dts_ns == c)[0][0] for c in common])
    ib = np.array([np.where(dts_s == c)[0][0] for c in common])
    par_ns, par_s, dates = par_ns[ia], par_s[ib], common

    spot_map = daily_spot(args.regime)
    spot = np.array([spot_map.get(d, np.nan) for d in dates])
    # PATCHED: parameters were calibrated on S-NORMALISED surfaces (S=1,
    # option values ~0.12). Hedging against the raw index level (~2600)
    # mixes scales and the P&L just tracks index moves. Rebase to S_0=1 so
    # spot moves are relative and commensurate with the option values.
    _s0 = np.nanmedian(spot)
    spot = spot / _s0
    ok = np.isfinite(spot)
    par_ns, par_s, dates, spot = par_ns[ok], par_s[ok], dates[ok], spot[ok]

    # optional subsample to keep MC cost sane
    if args.stride > 1:
        par_ns, par_s, dates, spot = (par_ns[::args.stride], par_s[::args.stride],
                                      dates[::args.stride], spot[::args.stride])
    print(f"{args.model} {args.regime}: hedging {len(dates)} days "
          f"(paths={args.paths}, stride={args.stride}) ...")

    pnl_ns = hedging.rolling_atm_hedge_pnl(args.model, par_ns, spot,
                                           target_T=args.target_T, N_paths=args.paths)
    pnl_s  = hedging.rolling_atm_hedge_pnl(args.model, par_s, spot,
                                           target_T=args.target_T, N_paths=args.paths)
    sum_ns, sum_s = hedging.hedge_error_summary(pnl_ns), hedging.hedge_error_summary(pnl_s)

    print("\n=== Delta-hedged P&L (lower variance is better) ===")
    print(f"  no sentiment : var={sum_ns['var']:.3e}  std={sum_ns['std']:.3e}  n={sum_ns['n']}")
    print(f"  sentiment    : var={sum_s['var']:.3e}  std={sum_s['std']:.3e}  n={sum_s['n']}")
    better = "sentiment" if sum_s['var'] < sum_ns['var'] else "no sentiment"
    print(f"  -> lower hedging-error variance: {better}")

    os.makedirs(FIG, exist_ok=True)
    d = np.array([np.datetime64(x) for x in dates[1:]])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d, np.nancumsum(pnl_ns), color="#4C72B0", lw=1.2, label="no sentiment")
    ax.plot(d, np.nancumsum(pnl_s),  color="#DD8452", lw=1.2, label="sentiment")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_ylabel("cumulative hedged P&L"); ax.set_xlabel("trade date")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Delta-hedged P&L — {args.model} {args.regime}\n"
                 f"variance  NS {sum_ns['var']:.2e}  |  S {sum_s['var']:.2e}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"hedge_{args.model}_{args.regime}.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"hedge_{args.model}_{args.regime}.pdf"), bbox_inches="tight")
    print(f"\n  figure -> {FIG}/hedge_{args.model}_{args.regime}.png")


if __name__ == "__main__":
    main()
