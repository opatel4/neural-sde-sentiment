"""
hedging.py   (PHASE 2 — validate in your environment before trusting numbers)
============================================================================
DROP INTO A PACKAGE FOLDER (needs pricers.py + utils.py alongside it):
    fine_tuning/Sentiment_Train/  (or Non_Sentiment_Train/)

Economic test: delta-hedged P&L. A better pricing model should hedge with lower
error variance. The delta is MODEL-CONSISTENT (finite-difference through your own
MC pricer), not a Black-Scholes delta — that's the point of the test.

Two things to validate first (a `__main__` self-test does #1 for you):
  1. FD machinery: run `python hedging.py --selftest` — a Black-Scholes analog
     must recover N(d1). This confirms the central-difference logic.
  2. Spot series + units: rolling_atm_hedge_pnl assumes you can supply a real
     SPX close series aligned to the same trade dates as your surfaces. With
     S-normalized surfaces you are hedging a synthetic constant-maturity ATM
     contract, so report this as a RELATIVE comparison (NS vs S hedging-error
     variance), not an absolute tradable P&L. Say so in the paper.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

# PATCHED: pricers.py builds its random draws on CUDA when available, so
# CPU-built inputs raised 'expected all tensors on the same device'.
_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATCHED: read the same per-regime rates the rest of the pipeline uses.
# The old hardcoded 0.045/0.011 gave drift 0.034 vs the market's ~0.008, which is
# the bug the retrain fixed. Set SDP_R per regime before running:
#   2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 | 2020-2022 0.0169
R = float(os.environ.get("SDP_R", 0.0231))
Q = float(os.environ.get("SDP_Q", 0.0150))
# ---------------------------------------------------------------------------
# Model-consistent delta via central finite difference through the MC pricer
# ---------------------------------------------------------------------------
def _price_point(model_type, params_row, K, T, S, x_nodes, w_nodes,
                 N_paths=20000, N_steps=100, seed=0):
    """One option price at (K,T,S) for the given calibrated params. Uses your pricers."""
    from pricers import (Hestonprice_Batched, Batesprice_Batched,
                         Bergomiprice_Batched, rBergomiprice_Batched)
    torch.manual_seed(seed)                                  # common random numbers across bumps
    p = torch.as_tensor(params_row, dtype=torch.float32, device=_DEV).view(1, -1)
    Kv = torch.as_tensor([K], dtype=torch.float32, device=_DEV)
    S_t = torch.as_tensor(float(S), dtype=torch.float32, device=_DEV)
    r = torch.as_tensor(R, device=_DEV); q = torch.as_tensor(Q, device=_DEV); Tt = torch.as_tensor(float(T), device=_DEV)
    if model_type == "Heston":
        out = Hestonprice_Batched(p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],
                                  Tt,Kv,S_t,r,q,x_nodes,w_nodes,N_paths=N_paths,N_steps=N_steps)
    elif model_type == "Bates":
        out = Batesprice_Batched(p[:,0],p[:,1],p[:,2],p[:,3],p[:,4],p[:,5],p[:,6],p[:,7],
                                 Tt,Kv,S_t,r,q,x_nodes,w_nodes,N_paths=N_paths,N_steps=N_steps)
    elif model_type == "Bergomi":
        out = Bergomiprice_Batched(p[:,0],p[:,1],p[:,2],p[:,3],
                                   Tt,Kv,S_t,r,q,N_paths=N_paths,N_steps=N_steps)
    elif model_type == "rBergomi":
        out = rBergomiprice_Batched(p[:,0],p[:,1],p[:,2],p[:,3],
                                    Tt,Kv,S_t,r,q,N_paths=N_paths,N_steps=N_steps)
    else:
        raise ValueError(model_type)
    return float(out.detach().reshape(-1)[0])


def model_delta(model_type, params_row, K, T, S=1.0, h=1e-3,
                x_nodes=None, w_nodes=None, N_paths=40000, N_steps=100):
    """d Price / d Spot via central difference (common random numbers => low MC noise)."""
    if x_nodes is None:
        from utils import GenerateGaussLaguerre
        x_nodes, w_nodes = GenerateGaussLaguerre(32, device=_DEV)
    up = _price_point(model_type, params_row, K, T, S + h, x_nodes, w_nodes, N_paths, N_steps, seed=12345)
    dn = _price_point(model_type, params_row, K, T, S - h, x_nodes, w_nodes, N_paths, N_steps, seed=12345)
    return (up - dn) / (2 * h)


# ---------------------------------------------------------------------------
# Rolling constant-maturity ATM delta-hedged P&L
# ---------------------------------------------------------------------------
def rolling_atm_hedge_pnl(model_type, params_by_day, spot_series, target_T=0.25,
                          h=1e-3, N_paths=40000, N_steps=100):
    """
    params_by_day : (N, d) calibrated params, one row per trade date (chronological).
    spot_series   : (N,)   real underlying close aligned to the same dates.
    Holds a constant-maturity ATM call, re-struck/re-hedged daily.
    Returns per-day hedged P&L: V_{t+1} - V_t - delta_t*(S_{t+1} - S_t).
    Compare .var() between NS and S variants — lower is better.
    """
    from utils import GenerateGaussLaguerre
    xn, wn = GenerateGaussLaguerre(32, device=_DEV)
    S = np.asarray(spot_series, float)
    N = len(S); pnl = np.full(N - 1, np.nan)
    for t in range(N - 1):
        F_t = S[t] * np.exp((R - Q) * target_T)
        K_atm = F_t                                          # ATM-forward
        V_t   = _price_point(model_type, params_by_day[t],   K_atm, target_T, S[t],   xn, wn, N_paths, N_steps, seed=7)
        d_t   = model_delta(model_type, params_by_day[t], K_atm, target_T, S[t], h, xn, wn, N_paths, N_steps)
        V_t1  = _price_point(model_type, params_by_day[t+1], K_atm, target_T, S[t+1], xn, wn, N_paths, N_steps, seed=7)
        pnl[t] = (V_t1 - V_t) - d_t * (S[t+1] - S[t])
    return pnl


def hedge_error_summary(pnl):
    p = pnl[np.isfinite(pnl)]
    return {"n": int(len(p)), "mean": float(p.mean()),
            "std": float(p.std(ddof=1)), "var": float(p.var(ddof=1))}


# ---------------------------------------------------------------------------
# Self-test: FD delta on a Black-Scholes analog must equal N(d1)
# ---------------------------------------------------------------------------
def _selftest():
    from scipy.stats import norm
    K, T, sig = 1.0, 0.25, 0.2
    def bs(S):
        F = S * np.exp((R - Q) * T)
        d1 = (np.log(F / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T)); d2 = d1 - sig*np.sqrt(T)
        return np.exp(-R*T) * (F*norm.cdf(d1) - K*norm.cdf(d2))
    h = 1e-4
    fd = (bs(1.0 + h) - bs(1.0 - h)) / (2*h)
    F = np.exp((R - Q) * T)
    d1 = (np.log(F/K) + 0.5*sig**2*T) / (sig*np.sqrt(T))
    analytic = np.exp(-Q*T) * norm.cdf(d1)
    print(f"FD delta   = {fd:.6f}")
    print(f"analytic   = {analytic:.6f}   (e^-qT N(d1))")
    print("FD machinery OK" if abs(fd - analytic) < 1e-4 else "FD MISMATCH — check h / discounting")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("Import model_delta / rolling_atm_hedge_pnl from your driver, or run --selftest.")
