"""
param_identification.py
=======================
Turns "the calibrated parameters look unstable" into a defensible statistic.

THE TEST: if a parameter were re-drawn independently at random from its bound
range [a,b] each day, the std of daily changes would be

        std(dp)_random = sqrt(2) * (b - a) / sqrt(12)

So the IDENTIFICATION RATIO

        IR = std(dp)_observed / std(dp)_random

is ~1.0 when the network is effectively guessing (no identification), and ->0
when the parameter is pinned down by the surface. We also report:
  - pin_rate : fraction of days sitting within 1% of a bound (bound-pinning)
  - ac1      : lag-1 autocorrelation of the LEVEL (identified params persist)

Run from the REPO ROOT:
    python param_identification.py
    python param_identification.py --model Heston

Writes full_project_output/eval_surfaces/analysis/param_identification.csv
"""
import os, glob, json, csv, argparse
import numpy as np

NS_DIR = "full_project_output/eval_surfaces/nonsent"
S_DIR  = "full_project_output/eval_surfaces/sentiment"
ANA    = "full_project_output/eval_surfaces/analysis"

PARAM_NAMES = {
    "Heston":   ["kappa", "theta", "sigma", "v0", "rho"],
    "Bates":    ["kappa", "theta", "sigma", "v0", "rho", "lambdaJ", "muJ", "sigmaJ"],
    "Bergomi":  ["xi", "nu", "rho", "beta"],
    "rBergomi": ["xi", "nu", "rho", "H"],
}


def ac1(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    d = (x @ x)
    return float((x[1:] @ x[:-1]) / d) if d > 0 else np.nan


def analyse(params, names):
    """params: (N,d). Bounds inferred from observed range (proxy for config bounds)."""
    out = []
    for i in range(params.shape[1]):
        p = params[:, i].astype(float)
        lo, hi = np.nanmin(p), np.nanmax(p)
        rng = hi - lo
        if rng <= 0:
            continue
        obs = np.nanstd(np.diff(p))
        rand = np.sqrt(2) * rng / np.sqrt(12)
        ir = obs / rand if rand > 0 else np.nan
        tol = 0.01 * rng
        pin = float(np.mean((p <= lo + tol) | (p >= hi - tol)))
        out.append({
            "param": names[i] if i < len(names) else f"p{i}",
            "min": round(float(lo), 5), "max": round(float(hi), 5),
            "std_daily_change": round(float(obs), 5),
            "std_if_random": round(float(rand), 5),
            "identification_ratio": round(float(ir), 3),
            "pin_rate": round(pin, 3),
            "ac1_level": round(ac1(p), 3),
            "verdict": ("NOT identified" if ir > 0.8 else
                        "weak" if ir > 0.5 else
                        "moderate" if ir > 0.25 else "identified"),
        })
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); a = ap.parse_args()
    rows = []
    files = sorted(glob.glob(os.path.join(S_DIR, "*.npz")))
    if not files:
        print(f"No exports in {S_DIR}. Run export_surfaces.py first."); return

    print("=" * 104)
    print("PARAMETER IDENTIFICATION  (IR ~ 1.0 => behaving like random daily draws; "
          "IR -> 0 => well identified)")
    print("=" * 104)

    for sp in files:
        cell = os.path.basename(sp)[:-4]
        model = cell.partition("_")[0]
        if a.model and model != a.model:
            continue
        npath = os.path.join(NS_DIR, os.path.basename(sp))
        names = PARAM_NAMES.get(model, [])
        print(f"\n### {cell}")
        print(f"{'param':<10}{'variant':<10}{'std(dp)':>10}{'random':>10}"
              f"{'IR':>8}{'pin%':>8}{'ac1':>8}   verdict")
        print("-" * 104)
        for variant, path in (("nonsent", npath), ("sentiment", sp)):
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            for r in analyse(d["params"], names):
                print(f"{r['param']:<10}{variant:<10}{r['std_daily_change']:>10.4f}"
                      f"{r['std_if_random']:>10.4f}{r['identification_ratio']:>8.2f}"
                      f"{100*r['pin_rate']:>7.1f}%{r['ac1_level']:>8.2f}   {r['verdict']}")
                rows.append({"cell": cell, "model": model, "variant": variant, **r})

    os.makedirs(ANA, exist_ok=True)
    if rows:
        with open(os.path.join(ANA, "param_identification.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        # summary by model family
        print("\n" + "=" * 104)
        print("SUMMARY - mean identification ratio by model family (lower = better identified)")
        print("=" * 104)
        by = {}
        for r in rows:
            by.setdefault(r["model"], []).append(r["identification_ratio"])
        for m, v in sorted(by.items(), key=lambda kv: np.mean(kv[1])):
            v = [x for x in v if np.isfinite(x)]
            print(f"  {m:<12} mean IR = {np.mean(v):.2f}   "
                  f"({sum(1 for x in v if x > 0.8)}/{len(v)} params behaving ~randomly)")
        print(f"\nWrote {ANA}/param_identification.csv")


if __name__ == "__main__":
    main()
