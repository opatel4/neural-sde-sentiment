#!/usr/bin/env python3
"""
make_fig5_aor.py -- build Fig5 (delta-hedged P&L) to Annals of Operations
Research specification, as a two-panel figure:

    (a) Heston   2020-2022
    (b) rBergomi 2020-2022

READ-ONLY with respect to every original. Imports run_hedging and hedging
rather than editing them, and writes only into figures/aor_fig5/.
"""
import os
import re
import sys
import glob
import argparse

import numpy as np

import run_hedging as rh
import hedging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MM = 1.0 / 25.4
W = 119 * MM
DPI = 600
FS = 8.0

_SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _SANS,
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.35,
    "lines.linewidth": 0.9,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "savefig.dpi": DPI,
    "figure.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "aor_fig5")

CELLS = [("Heston", "2020-2022", "a"), ("rBergomi", "2020-2022", "b")]

EXPECT = {
    ("Heston", "2020-2022"):   (1.751e-05, 1.533e-05),
    ("rBergomi", "2020-2022"): (1.107e-05, 9.322e-06),
}

C_NS, C_S = "#1f4e79", "#b45309"


def resolved_font():
    from matplotlib.font_manager import findfont, FontProperties
    try:
        return os.path.basename(findfont(FontProperties(family=["sans-serif"])))
    except Exception:
        return "unknown"


def detect_run_settings(model, regime):
    pats = [os.path.expanduser(p) for p in (
        "~/SeniorDesignV2/**/hedging_all.log",
        "~/**/hedging_all.log",
    )]
    for pat in pats:
        for f in glob.glob(pat, recursive=True)[:8]:
            try:
                txt = open(f, errors="ignore").read()
            except OSError:
                continue
            m = re.search(
                rf"{re.escape(model)}\s+{re.escape(regime)}:\s*hedging\s+(\d+)\s+days\s*"
                rf"\(paths=(\d+),\s*stride=(\d+)\)", txt)
            if m:
                return int(m.group(2)), int(m.group(3)), f
    return None, None, None


def prep(model, regime, stride):
    par_ns, dts_ns = rh.load_params("nonsent", model, regime)
    par_s, dts_s = rh.load_params("sentiment", model, regime)
    if par_s is None or par_ns is None:
        raise SystemExit(f"missing npz for {model} {regime}")

    common = np.intersect1d(dts_ns, dts_s)
    ia = np.array([np.where(dts_ns == c)[0][0] for c in common])
    ib = np.array([np.where(dts_s == c)[0][0] for c in common])
    par_ns, par_s, dates = par_ns[ia], par_s[ib], common

    spot_map = rh.daily_spot(regime)
    spot = np.array([spot_map.get(d, np.nan) for d in dates])
    spot = spot / np.nanmedian(spot)
    ok = np.isfinite(spot)
    par_ns, par_s, dates, spot = par_ns[ok], par_s[ok], dates[ok], spot[ok]

    if stride > 1:
        par_ns, par_s, dates, spot = (par_ns[::stride], par_s[::stride],
                                      dates[::stride], spot[::stride])
    return par_ns, par_s, dates, spot


def compute(model, regime, target_T, paths, stride):
    par_ns, par_s, dates, spot = prep(model, regime, stride)
    print(f"  {model} {regime}: {len(dates)} days, paths={paths}, stride={stride}")
    pnl_ns = hedging.rolling_atm_hedge_pnl(model, par_ns, spot,
                                           target_T=target_T, N_paths=paths)
    pnl_s = hedging.rolling_atm_hedge_pnl(model, par_s, spot,
                                          target_T=target_T, N_paths=paths)
    s_ns = hedging.hedge_error_summary(pnl_ns)
    s_s = hedging.hedge_error_summary(pnl_s)
    d = np.array([np.datetime64(x) for x in dates[1:]])
    return dict(dates=d, pnl_ns=pnl_ns, pnl_s=pnl_s, sum_ns=s_ns, sum_s=s_s,
                spot=spot, model=model, regime=regime)


def report(res):
    key = (res["model"], res["regime"])
    v_ns, v_s = res["sum_ns"]["var"], res["sum_s"]["var"]
    chg = 100.0 * (v_s - v_ns) / v_ns
    print(f"    reproduced : baseline {v_ns:.4e}  sentiment {v_s:.4e}  change {chg:+.1f}%")
    if key in EXPECT:
        e_ns, e_s = EXPECT[key]
        e_chg = 100.0 * (e_s - e_ns) / e_ns
        d_ns = 100.0 * abs(v_ns - e_ns) / e_ns
        d_s = 100.0 * abs(v_s - e_s) / e_s
        print(f"    Table 8    : baseline {e_ns:.4e}  sentiment {e_s:.4e}  change {e_chg:+.1f}%")
        tag = "MATCH" if max(d_ns, d_s) < 2.0 else "DRIFT"
        print(f"    [{tag}] baseline off by {d_ns:.2f}%, sentiment off by {d_s:.2f}%")
        if tag == "DRIFT":
            print("    -> different Monte Carlo draw than Table 8; seed the MC "
                  "or refresh Table 8 from this run")


def panel_label(ax, text):
    ax.text(-0.20, 1.03, text, transform=ax.transAxes, fontsize=FS,
            fontweight="bold", va="bottom", ha="left")


def draw(results):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15))
    for ax, res in zip(axes, results):
        d = res["dates"]
        ax.plot(d, np.nancumsum(res["pnl_ns"]), color=C_NS, ls="-",
                lw=0.9, label="Baseline")
        ax.plot(d, np.nancumsum(res["pnl_s"]), color=C_S, ls="--",
                lw=0.9, dashes=(3.2, 1.6), label="Sentiment")
        ax.axhline(0, color="0.45", lw=0.4, ls=":")
        ax.set_xlabel("Trade date")
        ax.grid(alpha=0.28)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", pad=2)
        panel_label(ax, res["_label"])
    axes[0].set_ylabel("Cumulative hedged P&L")
    axes[0].legend(frameon=False, loc="best", handlelength=2.4,
                   borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    return fig


def save(fig):
    os.makedirs(OUT, exist_ok=True)
    over = os.environ.get("AOR_OVERWRITE") == "1"
    wrote = []
    for name in ("Fig5.pdf", "Fig5.png"):
        path = os.path.join(OUT, name)
        if os.path.exists(path) and not over:
            print(f"  {name}: already exists, left alone (AOR_OVERWRITE=1 to replace)")
            continue
        fig.savefig(path, dpi=DPI, facecolor="white")
        wrote.append(path)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  size {w*25.4:.1f} x {h*25.4:.1f} mm  (limit 119 x 195)  raster {DPI} dpi")
    for p in wrote:
        print(f"  wrote {p}  [{os.path.getsize(p)/1024:.0f} kB]")


def dump(results):
    os.makedirs(OUT, exist_ok=True)
    for res in results:
        f = os.path.join(OUT, f"fig5_pnl_{res['model']}_{res['regime']}.npz")
        if os.path.exists(f) and os.environ.get("AOR_OVERWRITE") != "1":
            print(f"  {os.path.basename(f)}: exists, left alone")
            continue
        np.savez_compressed(
            f, dates=res["dates"], pnl_ns=res["pnl_ns"], pnl_s=res["pnl_s"],
            spot=res["spot"],
            var_ns=res["sum_ns"]["var"], var_s=res["sum_s"]["var"],
            std_ns=res["sum_ns"]["std"], std_s=res["sum_s"]["std"],
            n_ns=res["sum_ns"]["n"], n_s=res["sum_s"]["n"],
            model=res["model"], regime=res["regime"])
        print(f"  wrote {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_T", type=float, default=0.25)
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    args = ap.parse_args()

    print(f"AOR Fig5: 119 mm wide, {DPI} dpi raster plus vector PDF, {FS:g} pt lettering")
    print(f"  font resolved to {resolved_font()}")
    print(f"  output -> {OUT}\n")

    results = []
    for model, regime, label in CELLS:
        p, s, src = detect_run_settings(model, regime)
        paths = args.paths if args.paths is not None else (p if p else 20000)
        stride = args.stride if args.stride is not None else (s if s else 1)
        if src and args.paths is None and args.stride is None:
            print(f"  settings from {src}")
        res = compute(model, regime, args.target_T, paths, stride)
        res["_label"] = label
        report(res)
        results.append(res)
        print()

    dump(results)
    save(draw(results))

    print("\nCaption for the manuscript (no period at the end, AOR rule):\n")
    print("Delta-hedged profit and loss in the 2020 to 2022 regime for "
          "(a) the Heston family and (b) the rough Bergomi family, the two "
          "cells with the largest reduction in hedging error variance under "
          "the sentiment channel. Solid lines give the baseline arm and "
          "dashed lines the sentiment arm")


if __name__ == "__main__":
    main()
