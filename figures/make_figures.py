#!/usr/bin/env python3
"""
make_figures.py  --  regenerate the paper's figures from the backup npz surfaces.

Run from inside SDP_FINAL_20260823/:
    python3 make_figures.py

Reads:  in_sample/{sentiment,nonsent}/<Model>_<Regime>.npz
        logs/fix3_bergomi_b.log      (for the convergence figure)
Writes: figures/paper/*.png

No VM, no GPU, no torch. Only numpy + matplotlib.
"""
import os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
INS  = os.path.join(ROOT, "in_sample")
OUT  = os.path.join(ROOT, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

MODELS  = ["Heston", "Bates", "Bergomi", "rBergomi"]
REGIMES = ["2010-2012", "2013-2015", "2016-2019", "2020-2022"]
# 11 log-moneyness nodes on [-0.5, 0.5]; 8 maturity slices.
LM = np.linspace(-0.5, 0.5, 11)
TAU = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])  # adjust if your grid differs

def load(arm, model, regime):
    p = os.path.join(INS, arm, f"{model}_{regime}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)

# ---------------------------------------------------------------- 1. convergence
def fig_convergence():
    """Huber loss + relative error vs epoch, from a representative training log."""
    log = os.path.join(ROOT, "logs", "fix3_bergomi_b.log")
    if not os.path.exists(log):
        cand = glob.glob(os.path.join(ROOT, "logs", "*.log"))
        log = max(cand, key=os.path.getsize) if cand else None
    if not log:
        print("  [convergence] no log found, skipping"); return
    hub, rel = [], []
    for line in open(log, errors="ignore"):
        mh = re.search(r"huber=([0-9.]+)", line)
        mr = re.search(r"rel=([0-9.]+)", line)
        if mh and mr:
            hub.append(float(mh.group(1))); rel.append(float(mr.group(1)))
    if not hub:
        print("  [convergence] no loss lines parsed, skipping"); return
    ep = np.arange(1, len(hub)+1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hub, lw=0.8); ax[0].set_title("Huber loss"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Huber"); ax[0].grid(alpha=.3)
    ax[1].plot(ep, rel, lw=0.8, color="tab:orange"); ax[1].set_title("Relative error")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("rel"); ax[1].grid(alpha=.3)
    fig.suptitle(f"Training convergence ({os.path.basename(log)})")
    fig.tight_layout(); f = os.path.join(OUT, "fig_convergence.png")
    fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f}")

# ---------------------------------------------------------------- 2. smile + term
def fig_smile_term(model="Heston", regime="2016-2019"):
    """Predicted vs observed price smile and ATM term structure, both arms, one day."""
    s = load("sentiment", model, regime); n = load("nonsent", model, regime)
    if s is None or n is None:
        print(f"  [smile] missing {model} {regime}, skipping"); return
    day = len(s["dates"]) // 2                       # a representative mid-sample day
    mkt = s["market_px"][day, 0]                     # 8x11
    ps  = s["model_px"][day, 0]
    pn  = n["model_px"][day, 0]
    atm = 5                                          # centre log-moneyness column (11//2)
    mid_tau = 4                                       # a mid maturity slice for the smile

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(LM, mkt[mid_tau], "k-o", lw=2, ms=4, label="Observed")
    ax[0].plot(LM, pn[mid_tau], "s--", ms=3, label="Baseline")
    ax[0].plot(LM, ps[mid_tau], "^--", ms=3, label="Sentiment")
    ax[0].set_title(f"{model} {regime}: price smile (tau={TAU[mid_tau]})")
    ax[0].set_xlabel("log-moneyness"); ax[0].set_ylabel("normalised price")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(TAU, mkt[:, atm], "k-o", lw=2, ms=4, label="Observed")
    ax[1].plot(TAU, pn[:, atm], "s--", ms=3, label="Baseline")
    ax[1].plot(TAU, ps[:, atm], "^--", ms=3, label="Sentiment")
    ax[1].set_title(f"{model} {regime}: ATM term structure")
    ax[1].set_xlabel("maturity"); ax[1].set_ylabel("normalised price")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); f = os.path.join(OUT, f"fig_smile_term_{model}_{regime}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f}")

# ---------------------------------------------------------------- 3. parity
def fig_parity(model="Heston", regime="2016-2019"):
    """Predicted vs observed price scatter, both arms, all days/points."""
    s = load("sentiment", model, regime); n = load("nonsent", model, regime)
    if s is None or n is None:
        print(f"  [parity] missing {model} {regime}, skipping"); return
    mkt = s["market_px"][:, 0].ravel()
    ps  = s["model_px"][:, 0].ravel()
    pn  = n["model_px"][:, 0].ravel()
    m = np.isfinite(mkt) & np.isfinite(ps) & np.isfinite(pn) & (mkt > 0)
    mkt, ps, pn = mkt[m], ps[m], pn[m]
    lim = np.percentile(mkt, 99.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, pred, name in [(ax[0], pn, "Baseline"), (ax[1], ps, "Sentiment")]:
        a.scatter(mkt, pred, s=1, alpha=.15)
        a.plot([0, lim], [0, lim], "r-", lw=1)
        a.set_xlim(0, lim); a.set_ylim(0, lim)
        a.set_title(f"{model} {regime}: {name}")
        a.set_xlabel("observed price"); a.set_ylabel("predicted price")
        a.grid(alpha=.3)
    fig.tight_layout(); f = os.path.join(OUT, f"fig_parity_{model}_{regime}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f}")

# ---------------------------------------------------------------- 4. error heatmap
def fig_error_heatmap(model="Heston", regime="2016-2019"):
    """Per-node RMSE across the 8x11 grid, both arms side by side."""
    s = load("sentiment", model, regime); n = load("nonsent", model, regime)
    if s is None or n is None:
        print(f"  [heatmap] missing {model} {regime}, skipping"); return
    def rmse(d):
        err = d["model_px"][:, 0] - d["market_px"][:, 0]      # (N,8,11)
        return np.sqrt(np.nanmean(err**2, axis=0))            # (8,11)
    rn, rs = rmse(n), rmse(s)
    vmax = max(np.nanmax(rn), np.nanmax(rs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, r, name in [(ax[0], rn, "Baseline"), (ax[1], rs, "Sentiment")]:
        im = a.imshow(r, aspect="auto", origin="lower", vmin=0, vmax=vmax,
                      extent=[LM[0], LM[-1], 0, len(TAU)])
        a.set_title(f"{model} {regime}: RMSE, {name}")
        a.set_xlabel("log-moneyness"); a.set_ylabel("maturity index")
        fig.colorbar(im, ax=a, fraction=.046)
    fig.tight_layout(); f = os.path.join(OUT, f"fig_error_heatmap_{model}_{regime}.png")
    fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f}")

if __name__ == "__main__":
    print("Generating figures ->", OUT)
    fig_convergence()
    # representative cells: Heston 2016-2019 is a robust-both-protocols sentiment win
    for mdl, rg in [("Heston", "2016-2019"), ("Heston", "2020-2022"), ("Bates", "2020-2022")]:
        fig_smile_term(mdl, rg)
        fig_parity(mdl, rg)
        fig_error_heatmap(mdl, rg)
    print("done.")
