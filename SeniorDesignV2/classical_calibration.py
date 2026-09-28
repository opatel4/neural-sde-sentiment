#!/usr/bin/env python3
"""
classical_calibration.py — calibrate one cell by classical least squares and
compare the resulting surface RMSE against the neural calibration.

WHY: the paper reports that a previous-day persistence forecast beats every
calibrated model. A referee will ask whether that reflects the model class or
an underfitting neural calibration. This answers it: same pricer, same
surfaces, same metric — only the optimiser differs.

RUN (on the VM, from the repo root or the online/ dir):
    export PYTHONPATH=$PWD/online
    export SDP_NORM=forward SDP_Q=0.015 SDP_R=0.0236
    python3 classical_calibration.py --model Heston --regime 2016-2019 --days 40

Compare the printed classical RMSE against results/in_sample/analysis/results_master.csv
for the same cell.
"""
import os, sys, time, argparse
import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "online"))

from data_utils import create_iv_and_price_grids_from_raw, filter_extreme_prices
from config import get_spx_files_for_range
import pricers

# Table-1 bounds, as (lo, hi) — must match generate_offline_data.py
BOUNDS = {
    "Heston":   [("kappa",0.1,2.9), ("theta",0.01,0.14), ("sigma",0.1,0.7),
                 ("v0",0.01,0.14), ("rho",-0.9,0.8)],
    "Bates":    [("kappa",0.1,2.9), ("theta",0.01,0.14), ("sigma",0.1,0.7),
                 ("v0",0.01,0.14), ("rho",-0.9,0.8),
                 ("lambdJ",0.0,2.0), ("muJ",-0.2,0.25), ("sigmaJ",0.01,0.19)],
    "Bergomi":  [("xi",0.01,0.15), ("nu",0.5,3.5), ("rho",-0.95,0.85), ("beta",0.0,10.0)],
    "rBergomi": [("xi",0.01,0.15), ("nu",0.5,3.5), ("rho",-0.95,0.85), ("H",0.025,0.475)],
}

PRICER = {
    "Heston":   pricers.Hestonprice_Batched,
    "Bates":    pricers.Batesprice_Batched,
    "Bergomi":  pricers.Bergomiprice_Batched,
    "rBergomi": pricers.rBergomiprice_Batched,
}

TAU = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def price_surface(model, params, S, r, q, lm_grid, device=None):
    device = device or DEV
    """Price the full 8x11 grid for one parameter vector. Returns (8,11) numpy."""
    fn = PRICER[model]
    rows = []
    for T in TAU:
        K = torch.tensor(S * np.exp(np.asarray(lm_grid)), dtype=torch.float32, device=device)
        args = [torch.tensor([float(p)], dtype=torch.float32, device=device) for p in params]
        out = fn(*args, T, K, S, r, q)
        rows.append(np.asarray(out.detach().cpu(), dtype=float).ravel())
    return np.vstack(rows)


def calibrate_day(model, market_px, S, r, q, lm_grid, x0=None, maxiter=60):
    """Least-squares fit of the parameter vector to one day's surface."""
    spec = BOUNDS[model]
    lo = np.array([b[1] for b in spec]); hi = np.array([b[2] for b in spec])
    if x0 is None:
        x0 = lo + 0.5 * (hi - lo)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    finite = np.isfinite(market_px)

    def resid(x):
        try:
            pred = price_surface(model, x, S, r, q, lm_grid)
        except Exception:
            return np.full(finite.sum(), 1e3)
        d = (pred - market_px)[finite]
        return np.nan_to_num(d, nan=1e3, posinf=1e3, neginf=1e3)

    sol = least_squares(resid, x0, bounds=(lo, hi), max_nfev=maxiter,
                        xtol=1e-8, ftol=1e-8, diff_step=1e-3)
    return sol.x, sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Heston")
    ap.add_argument("--regime", default="2016-2019")
    ap.add_argument("--days", type=int, default=40,
                    help="number of days to calibrate (subsampled evenly)")
    ap.add_argument("--maxiter", type=int, default=60)
    a = ap.parse_args()

    R = float(os.environ.get("SDP_R", 0.0236))
    Q = float(os.environ.get("SDP_Q", 0.015))

    print(f"[classical] {a.model} {a.regime}  r={R} q={Q}  SDP_NORM={os.environ.get('SDP_NORM')}")

    iv, px, ctx, dates, lm = create_iv_and_price_grids_from_raw(
        get_spx_files_for_range(a.regime), max_days=None, use_calls_only=False)
    iv, px, ctx, dates = filter_extreme_prices(iv, px, ctx, dates, 97, 2, 98, 2.0)

    px = px.squeeze(1).numpy()          # (N,8,11)
    N = px.shape[0]
    lm_grid = np.linspace(-0.5, 0.5, px.shape[2])
    idx = np.linspace(0, N - 1, min(a.days, N)).astype(int)

    print(f"[classical] {N} days available; calibrating {len(idx)} of them")

    S = 1.0                              # forward-normalised: F = 1
    errs, params_hist, x_warm = [], [], None
    t0 = time.time()

    for j, i in enumerate(idx):
        mk = px[i]
        x, sol = calibrate_day(a.model, mk, S, R, Q, lm_grid,
                               x0=x_warm, maxiter=a.maxiter)
        x_warm = x                       # warm-start next day
        pred = price_surface(a.model, x, S, R, Q, lm_grid)
        f = np.isfinite(mk) & np.isfinite(pred)
        rmse = float(np.sqrt(np.mean((pred[f] - mk[f]) ** 2)))
        errs.append(rmse); params_hist.append(x)
        if j % 10 == 0:
            print(f"   day {j+1}/{len(idx)} ({dates[i]}): px-RMSE={rmse:.6f}  "
                  f"nfev={sol.nfev}  {time.time()-t0:.0f}s")

    errs = np.array(errs); P = np.array(params_hist)
    print("\n" + "=" * 66)
    print(f"CLASSICAL LEAST-SQUARES CALIBRATION — {a.model} {a.regime}")
    print("=" * 66)
    print(f"  days calibrated      : {len(errs)}")
    print(f"  mean  price RMSE     : {errs.mean():.6f}")
    print(f"  median price RMSE    : {np.median(errs):.6f}")
    print(f"  pooled price RMSE    : {np.sqrt((errs**2).mean()):.6f}")
    print(f"  wall clock           : {time.time()-t0:.0f}s")
    print("\n  parameter stability (std of daily changes / uniform-draw std):")
    spec = BOUNDS[a.model]
    for k, (nm, lo, hi) in enumerate(spec):
        unif = (hi - lo) / np.sqrt(6.0)          # std of difference of 2 uniforms
        obs = np.std(np.diff(P[:, k]))
        print(f"    {nm:8s} IR = {obs/unif:.3f}   mean={P[:,k].mean():.4f}")
    print("\n  COMPARE against the neural calibration for this cell in")
    print("  results/in_sample/analysis/results_master.csv")
    print("  (note: that file reports vega-weighted IV RMSE; this reports price")
    print("   RMSE. For a like-for-like number, run the same price-RMSE")
    print("   computation on the exported npz — see README.)")


if __name__ == "__main__":
    main()
