"""
calendar_diagnostic.py
======================
The reported calendar-arbitrage violation rate (8-18%) is suspicious: Heston,
Bates and Bergomi are martingale models whose TRUE surfaces cannot violate
calendar arbitrage. So a high rate means either (a) the fitted surface really is
pathological, or (b) Monte-Carlo noise in the pricer is flipping the sign of
small total-variance increments.

This script distinguishes them three ways:

  1. TOLERANCE SWEEP  - real violations survive a generous tolerance; noise
                        vanishes as tolerance grows past the MC error scale.
  2. MATURITY PROFILE - MC noise bites hardest where the true increment in total
                        variance between adjacent maturities is smallest
                        (the long end). Real pathology is usually short-dated.
  3. MAGNITUDE        - distribution of violation sizes vs the typical increment.
                        If |violation| << typical increment, it's noise.

Run from the REPO ROOT:
    python calendar_diagnostic.py
    python calendar_diagnostic.py --cell Heston_2013-2015
"""
import os, glob, argparse
import numpy as np
import eval_upgrades as eu

NS_DIR = "full_project_output/eval_surfaces/nonsent"
S_DIR  = "full_project_output/eval_surfaces/sentiment"


def total_variance(iv):                      # (N,8,11) -> w = sigma^2 * T
    return (iv ** 2) * eu.MATS[:, None]


def sweep(iv_stack, tols):
    w = total_variance(iv_stack)
    dv = np.diff(w, axis=1)                  # (N,7,11) increments across maturity
    fin = np.isfinite(dv)
    return [float(np.mean(dv[fin] < -t)) for t in tols], dv, fin


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cell"); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(S_DIR, "*.npz")))
    if a.cell:
        files = [f for f in files if os.path.basename(f)[:-4] == a.cell]
    if not files:
        print("No matching exports found."); return

    tols = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    print("=" * 100)
    print("CALENDAR VIOLATION DIAGNOSTIC")
    print("=" * 100)

    for sp in files[:6]:
        cell = os.path.basename(sp)[:-4]
        d = np.load(sp, allow_pickle=True)
        lm = str(d["lm_convention"])
        iv = eu.surfaces_to_iv(d["model_px"], lm)

        rates, dv, fin = sweep(iv, tols)
        print(f"\n### {cell}")
        print("  tolerance sweep (violation rate at increasing tolerance):")
        for t, r in zip(tols, rates):
            bar = "#" * int(60 * r)
            print(f"    tol={t:<8.0e} rate={r:6.3f}  {bar}")

        # typical positive increment, for scale
        pos = dv[fin & (dv > 0)]
        typ = float(np.median(pos)) if pos.size else float("nan")
        neg = -dv[fin & (dv < 0)]
        med_viol = float(np.median(neg)) if neg.size else float("nan")
        print(f"  median POSITIVE increment (true signal scale) : {typ:.5f}")
        print(f"  median violation magnitude                    : {med_viol:.5f}")
        if np.isfinite(typ) and np.isfinite(med_viol):
            print(f"  ratio violation/signal                        : {med_viol/typ:.3f}"
                  f"   {'<-- noise-scale' if med_viol < 0.5*typ else '<-- material'}")

        # maturity profile at a strict tolerance
        prof = []
        for i in range(dv.shape[1]):
            sl = dv[:, i, :]; f = np.isfinite(sl)
            prof.append(float(np.mean(sl[f] < -1e-6)) if f.any() else np.nan)
        print("  violation rate by maturity gap "
              "(T_i -> T_{i+1}), strict tol:")
        for i, r in enumerate(prof):
            print(f"    {eu.MATS[i]:.1f}->{eu.MATS[i+1]:.1f}:  {r:6.3f}  {'#'*int(50*r)}")
        short = np.nanmean(prof[:3]); long_ = np.nanmean(prof[4:])
        print(f"  short-end mean {short:.3f}  |  long-end mean {long_:.3f}"
              f"   -> {'LONG-end concentrated (noise signature)' if long_ > short else 'SHORT-end concentrated (likely real)'}")

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("  * If rates collapse toward 0 as tolerance passes ~1e-3, and violations")
    print("    concentrate at the LONG end with magnitude << typical increment,")
    print("    the violations are Monte-Carlo noise -> do NOT report as a model defect.")
    print("  * If rates persist at generous tolerance and sit at the SHORT end,")
    print("    they are real -> report them.")
    print("=" * 100)


if __name__ == "__main__":
    main()
