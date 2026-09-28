#!/usr/bin/env python3
"""Two separate AOR hedging figures from the existing PDFs, reusing
extract_fig5_aor's extraction helpers. Heston -> Fig5, rBergomi -> Fig6.
Writes only into figures/aor_fig5/. No simulation, no originals touched."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import extract_fig5_aor as E   # reuse extract_curves, calibrate, to_data, etc.

NAMES = {"Heston": "Fig5", "rBergomi": "Fig6"}
FAMILY = {"Heston": "Heston", "rBergomi": "rough Bergomi"}


def one(model, regime, pdf, stem):
    if not os.path.exists(pdf):
        print(f"  MISSING {pdf}"); return
    curves, page = E.extract_curves(pdf)
    fy, fx = E.calibrate(page, stem)
    if fy is None or fx is None or not curves["baseline"] or not curves["sentiment"]:
        print(f"  {stem}: extraction incomplete, skipping"); return

    fig, ax = plt.subplots(figsize=(E.W, 2.4))
    for key, color, ls, dashes, label in [
        ("baseline",  E.C_NS, "-",  None,       "Baseline"),
        ("sentiment", E.C_S,  "--", (3.2, 1.6), "Sentiment"),
    ]:
        yr, val = E.to_data(curves[key], fy, fx)
        order = np.argsort(yr)
        kw = dict(color=color, ls=ls, lw=0.9, label=label)
        if dashes:
            kw["dashes"] = dashes
        ax.plot(E.year_to_dt(yr[order]), val[order], **kw)
        print(f"  {stem} {key}: {len(val)} pts, "
              f"y-range [{val.min():.3e}, {val.max():.3e}]")
    ax.axhline(0, color="0.45", lw=0.4, ls=":")
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Cumulative hedged P&L")
    ax.grid(alpha=0.28)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="best", handlelength=2.4)
    fig.tight_layout(pad=0.4)

    os.makedirs(E.OUT, exist_ok=True)
    over = os.environ.get("AOR_OVERWRITE") == "1"
    for ext in ("pdf", "png"):
        p = os.path.join(E.OUT, f"{stem}.{ext}")
        if os.path.exists(p) and not over:
            print(f"  {stem}.{ext}: exists, left alone (AOR_OVERWRITE=1 to replace)")
            continue
        fig.savefig(p, dpi=E.DPI, facecolor="white")
        print(f"  wrote {p}  [{os.path.getsize(p)/1024:.0f} kB]")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {stem} size {w*25.4:.1f} x {h*25.4:.1f} mm, {E.DPI} dpi\n")


if __name__ == "__main__":
    print(f"Two separate AOR hedging figures -> {E.OUT}\n")
    for model, regime, lab, pdf in E.CELLS:
        one(model, regime, pdf, NAMES[model])
    print("Captions (no period at end, AOR rule):\n")
    print("Fig5: Delta-hedged profit and loss for the Heston family in the "
          "2020 to 2022 regime. Solid lines show the baseline arm and dashed "
          "lines the sentiment arm")
    print("Fig6: Delta-hedged profit and loss for the rough Bergomi family in "
          "the 2020 to 2022 regime. Solid lines show the baseline arm and "
          "dashed lines the sentiment arm")
