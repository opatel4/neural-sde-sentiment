#!/usr/bin/env python3
"""
make_param_fig_aor.py -- calibrated-parameter comparison figure for the Heston
family, baseline arm against sentiment arm, to AOR specification.

Reads calibrated parameters through run_hedging.load_params, i.e. the same
arrays behind the reported results. No simulation. Writes only into
figures/aor_params/. Originals untouched.
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import run_hedging as rh

MM, W, DPI, FS = 1 / 25.4, 119 / 25.4, 600, 8.0
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                        "Nimbus Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.5, "grid.linewidth": 0.35, "lines.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "savefig.dpi": DPI, "figure.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_NS, C_S = "#1f4e79", "#b45309"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "aor_params")

HESTON = [
    (r"$\kappa$",  "mean reversion",     0.10,  3.00),
    (r"$\theta$",  "long-run variance",  0.01,  0.15),
    (r"$\sigma$",  "vol of vol",         0.10,  0.80),
    (r"$v_0$",     "initial variance",   0.01,  0.15),
    (r"$\rho$",    "correlation",       -0.90, -0.10),
]
LABELS = "abcdef"


def as_dates(dts):
    return np.array([np.datetime64(str(d)[:10]) for d in dts])


def load(regime):
    par_ns, dts_ns = rh.load_params("nonsent", "Heston", regime)
    par_s, dts_s = rh.load_params("sentiment", "Heston", regime)
    if par_ns is None or par_s is None:
        sys.exit(f"missing npz for Heston {regime}")
    common = np.intersect1d(dts_ns, dts_s)
    ia = np.array([np.where(dts_ns == c)[0][0] for c in common])
    ib = np.array([np.where(dts_s == c)[0][0] for c in common])
    return par_ns[ia], par_s[ib], as_dates(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="2016-2019")
    args = ap.parse_args()

    par_ns, par_s, dates = load(args.regime)
    n, d = par_ns.shape
    print(f"Heston {args.regime}: {n} common dates, {d} parameters per day")
    print("  assumed order: " + ", ".join(p[1] for p in HESTON))
    if d != len(HESTON):
        print(f"  WARNING: npz has {d} parameters but {len(HESTON)} labels are "
              f"defined. Check the generator's parameter order.")
    for j in range(min(d, len(HESTON))):
        lo, hi = HESTON[j][2], HESTON[j][3]
        a, b = par_ns[:, j], par_s[:, j]
        frac = np.mean(np.abs(a - b)) / (hi - lo)
        print(f"  {HESTON[j][1]:>18}: baseline [{a.min():.3f}, {a.max():.3f}] "
              f"sentiment [{b.min():.3f}, {b.max():.3f}] "
              f"mean |diff| = {frac*100:.2f}% of range")

    fig, axes = plt.subplots(2, 3, figsize=(W, 2.9), sharex=True)
    flat = axes.ravel()
    for j, (sym, name, lo, hi) in enumerate(HESTON):
        if j >= d:
            break
        ax = flat[j]
        ax.plot(dates, par_ns[:, j], color=C_NS, ls="-", lw=0.6,
                label="Baseline")
        ax.plot(dates, par_s[:, j], color=C_S, ls="--", lw=0.6,
                dashes=(3.0, 1.5), label="Sentiment")
        ax.set_ylim(lo, hi)
        ax.set_ylabel(sym)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", pad=2)
        ax.text(-0.30, 1.05, LABELS[j], transform=ax.transAxes, fontsize=FS,
                fontweight="bold", va="bottom", ha="left")

    legend_ax = flat[len(HESTON)]
    legend_ax.axis("off")
    h, lab = flat[0].get_legend_handles_labels()
    legend_ax.legend(h, lab, frameon=False, loc="center", handlelength=2.6)

    for ax in axes[-1]:
        ax.set_xlabel("Trade date")

    fig.tight_layout(pad=0.4)

    os.makedirs(OUT, exist_ok=True)
    over = os.environ.get("AOR_OVERWRITE") == "1"
    stem = f"FigParams_Heston_{args.regime}"
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"{stem}.{ext}")
        if os.path.exists(p) and not over:
            print(f"  {stem}.{ext}: exists, left alone (AOR_OVERWRITE=1)")
            continue
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  wrote {p}  [{os.path.getsize(p)/1024:.0f} kB]")
    w, hgt = fig.get_size_inches()
    plt.close(fig)
    print(f"  size {w*25.4:.1f} x {hgt*25.4:.1f} mm (limit 119 x 195), {DPI} dpi")


if __name__ == "__main__":
    main()
