#!/usr/bin/env python3
"""
extract_fig5_aor.py -- rebuild Fig5 to AOR spec WITHOUT re-running the hedge
simulation, by extracting the plotted P&L curves from the vector PDFs that
run_hedging.py already saved for the Table 8 run.

No Monte Carlo. Reads the two existing PDFs, redraws them as one two-panel
AOR figure. Writes only into figures/aor_fig5/. Originals are never modified.

Requires PyMuPDF (import fitz):   pip install --user pymupdf

Run:
    cd ~/SeniorDesignV2/fine_tuning/Sentiment_Train
    python3 extract_fig5_aor.py
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import fitz  # PyMuPDF
except Exception as e:
    sys.exit("PyMuPDF not available: %s\n  pip install --user pymupdf" % e)

# ------------------------------------------------------------------ inputs
SRC = os.path.expanduser(
    "~/SeniorDesignV2/full_project_output/eval_surfaces/figures")
CELLS = [
    ("Heston",   "2020-2022", "a", os.path.join(SRC, "hedge_Heston_2020-2022.pdf")),
    ("rBergomi", "2020-2022", "b", os.path.join(SRC, "hedge_rBergomi_2020-2022.pdf")),
]

# original run_hedging.py stroke colours
BLUE   = (0x4C/255, 0x72/255, 0xB0/255)   # "no sentiment" -> baseline
ORANGE = (0xDD/255, 0x84/255, 0x52/255)   # "sentiment"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "aor_fig5")

# ------------------------------------------------------------------ AOR format
MM, W, DPI, FS = 1 / 25.4, 119 / 25.4, 600, 8.0
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                        "Nimbus Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.5, "grid.linewidth": 0.35, "lines.linewidth": 0.9,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "savefig.dpi": DPI, "figure.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_NS, C_S = "#1f4e79", "#b45309"


def cdist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def path_points(items):
    """Ordered vertices of a stroked matplotlib path (list of 'l'/'c' items)."""
    pts = []
    for it in items:
        op = it[0]
        if op == "l":
            _, p1, p2 = it
            if not pts:
                pts.append((p1.x, p1.y))
            pts.append((p2.x, p2.y))
        elif op == "c":
            _, p1, p2, p3, p4 = it
            if not pts:
                pts.append((p1.x, p1.y))
            pts.append((p4.x, p4.y))
    return pts


def extract_curves(pdf):
    """Return {'baseline':[(x,y)...], 'sentiment':[...]} in PDF page coords."""
    page = fitz.open(pdf)[0]
    cands = {"baseline": [], "sentiment": []}
    for d in page.get_drawings():
        col = d.get("color")
        if col is None:
            continue
        pts = path_points(d["items"])
        if len(pts) < 20:            # skip legend swatches, axhline, ticks
            continue
        if cdist(col, BLUE) < 0.18:
            cands["baseline"].append(pts)
        elif cdist(col, ORANGE) < 0.18:
            cands["sentiment"].append(pts)
    out = {}
    for k, v in cands.items():
        out[k] = max(v, key=len) if v else []
    return out, page


def read_ticks(page):
    """Numeric axis tick labels: return (yticks, xticks) as [(value, cx, cy)]."""
    yt, xt = [], []
    words = page.get_text("dict")["blocks"]
    spans = []
    for b in words:
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp["text"].strip()
                x0, y0, x1, y1 = sp["bbox"]
                spans.append((t, (x0 + x1) / 2, (y0 + y1) / 2, x0, y1))
    # y ticks: numeric, left side of page; x ticks: year-like, bottom
    xs = [s[1] for s in spans]
    ys = [s[2] for s in spans]
    if not xs:
        return yt, xt, spans
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    for t, cx, cy, x0, y1 in spans:
        num = _parsenum(t)
        yr = _parseyear(t)
        if yr is not None and cy > ymin + 0.6 * (ymax - ymin):
            xt.append((float(yr), cx, cy))
        elif num is not None and cx < xmin + 0.4 * (xmax - xmin):
            yt.append((num, cx, cy))
    return yt, xt, spans


def _parsenum(t):
    t = t.replace("−", "-").replace("−", "-").replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", t):
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _parseyear(t):
    t = t.strip()
    m = re.fullmatch(r"(20[0-2]\d)-(\d{2})(?:-\d{2})?", t)
    if m:
        return int(m.group(1)) + (int(m.group(2)) - 1) / 12.0
    return float(t) if re.fullmatch(r"20[0-2]\d", t) else None


def linfit(pairs):
    """pairs=[(value, pagecoord)]; return (m,b) so value = m*pagecoord + b."""
    if len(pairs) < 2:
        return None
    v = np.array([p[0] for p in pairs], float)
    c = np.array([p[1] for p in pairs], float)
    A = np.vstack([c, np.ones_like(c)]).T
    m, b = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(m), float(b)


def robust_linfit(pairs):
    fit = linfit(pairs)
    if fit is None or len(pairs) <= 2:
        return fit
    m, b = fit
    span = max(v for v, _ in pairs) - min(v for v, _ in pairs)
    tol = max(abs(span) * 0.05, 1e-9)
    keep = [p for p in pairs if abs(p[0] - (m * p[1] + b)) <= tol]
    return linfit(keep) if 2 <= len(keep) < len(pairs) else fit


def calibrate(page, cell):
    yt, xt, spans = read_ticks(page)
    fy = robust_linfit([(v, cy) for v, cx, cy in yt])       # value = fy(page_y)
    fx = robust_linfit([(v, cx) for v, cx, cy in xt])       # year  = fx(page_x)
    print(f"  [{cell}] y-ticks read: "
          + ", ".join(f"{v:g}" for v, _, _ in sorted(yt)))
    print(f"  [{cell}] x-ticks read: "
          + ", ".join(f"{int(v)}" for v, _, _ in sorted(xt)))
    if fy is None or fx is None:
        print(f"  [{cell}] WARN: could not calibrate an axis; dumping all text:")
        for t, cx, cy, *_ in spans:
            print(f"        {t!r:>12}  x={cx:.1f} y={cy:.1f}")
    return fy, fx


def to_data(pts, fy, fx):
    my, by = fy
    mx, bx = fx
    year = np.array([mx * x + bx for x, y in pts])
    val = np.array([my * y + by for x, y in pts])
    return year, val


def year_to_dt(years):
    return np.array([np.datetime64("1970") +
                     np.timedelta64(int((yr - 1970) * 365.25), "D")
                     for yr in years])


def build():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15))
    ok = True
    for ax, (model, regime, lab, pdf) in zip(axes, CELLS):
        if not os.path.exists(pdf):
            print(f"  MISSING {pdf}"); ok = False; continue
        curves, page = extract_curves(pdf)
        fy, fx = calibrate(page, lab)
        if fy is None or fx is None or not curves["baseline"] or not curves["sentiment"]:
            print(f"  [{lab}] extraction incomplete "
                  f"(baseline pts={len(curves['baseline'])}, "
                  f"sentiment pts={len(curves['sentiment'])})")
            ok = False; continue
        for key, color, ls, dashes, label in [
            ("baseline",  C_NS, "-",  None,        "Baseline"),
            ("sentiment", C_S,  "--", (3.2, 1.6),  "Sentiment"),
        ]:
            yr, val = to_data(curves[key], fy, fx)
            order = np.argsort(yr)
            dt = year_to_dt(yr[order])
            kw = dict(color=color, ls=ls, lw=0.9, label=label)
            if dashes:
                kw["dashes"] = dashes
            ax.plot(dt, val[order], **kw)
            print(f"  [{lab}] {key}: {len(val)} pts, "
                  f"y-range [{val.min():.3e}, {val.max():.3e}]")
        ax.axhline(0, color="0.45", lw=0.4, ls=":")
        ax.set_xlabel("Trade date"); ax.grid(alpha=0.28)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.text(-0.20, 1.03, lab, transform=ax.transAxes, fontsize=FS,
                fontweight="bold", va="bottom", ha="left")
    axes[0].set_ylabel("Cumulative hedged P&L")
    axes[0].legend(frameon=False, loc="best", handlelength=2.4)
    fig.tight_layout(pad=0.4)
    if not ok:
        print("\n  extraction had problems; figure written for inspection but "
              "check the diagnostics above before using it")
    over = os.environ.get("AOR_OVERWRITE") == "1"
    for name in ("Fig5.pdf", "Fig5.png"):
        p = os.path.join(OUT, name)
        if os.path.exists(p) and not over:
            print(f"  {name}: exists, left alone (AOR_OVERWRITE=1 to replace)")
            continue
        fig.savefig(p, dpi=DPI, facecolor="white")
        print(f"  wrote {p}  [{os.path.getsize(p)/1024:.0f} kB]")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  size {w*25.4:.1f} x {h*25.4:.1f} mm (limit 119 x 195), {DPI} dpi")


if __name__ == "__main__":
    print(f"Extracting Fig5 from existing PDFs in\n  {SRC}\n  -> {OUT}\n")
    build()
    print("\nCaption (no period at end, AOR rule):\n"
          "Delta-hedged profit and loss in the 2020 to 2022 regime for "
          "(a) the Heston family and (b) the rough Bergomi family. Solid lines "
          "give the baseline arm and dashed lines the sentiment arm")
