"""
compare_variants.py
===================
RUN FROM THE REPO ROOT after export_surfaces.py has been run in BOTH packages.

    python compare_variants.py

For every (model, regime) present in both variant surface folders it:
  1. aligns non-sentiment and sentiment exports on common dates,
  2. computes the primary metric (vega-weighted IV RMSE) for each variant,
  3. runs the Clark-West nested test (NS nested in S),
  4. runs no-arbitrage checks (calendar + butterfly) per variant,
  5. scores cheap benchmarks (previous-day, ATM-flat),
  6. reports parameter turnover per variant,
  7. runs conditional buckets + Giacomini-White on sentiment / news-volume.

Outputs:
    fine_tuning_output/analysis/results_master.csv   (one row per model,regime)
    fine_tuning_output/analysis/<Model>_<Range>.json (full detail per cell)

Point NS_DIR / S_DIR at wherever export_surfaces.py wrote its .npz files.
"""

import os, glob, json
import numpy as np
import eval_upgrades as eu       # keep eval_upgrades.py on the path / repo root

# ---- Defaults assume you run this from the REPO ROOT (parent of fine_tuning/).
# For walk-forward results, swap "eval_surfaces" -> "eval_wf_surfaces".
NS_DIR = "full_project_output/eval_surfaces/nonsent"
S_DIR  = "full_project_output/eval_surfaces/sentiment"
OUT    = "full_project_output/eval_surfaces/analysis"
# Optional: dict {date_str -> vix_decimal} to enable the VIX-flat benchmark
VIX_BY_DATE = None
# ---------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)
PARAM_NAMES = {
    "Heston":  ["kappa", "theta", "sigma", "v0", "rho"],
    "Bates":   ["kappa", "theta", "sigma", "v0", "rho", "lambdaJ", "muJ", "sigmaJ"],
    "Bergomi": ["xi", "nu", "rho", "beta"],
    "rBergomi":["xi", "nu", "rho", "H"],
}


def _cells():
    ns = {os.path.basename(f): f for f in glob.glob(os.path.join(NS_DIR, "*.npz"))}
    s  = {os.path.basename(f): f for f in glob.glob(os.path.join(S_DIR,  "*.npz"))}
    for name in sorted(set(ns) & set(s)):
        yield name[:-4], ns[name], s[name]


def analyze(ns_path, s_path):
    ns = np.load(ns_path, allow_pickle=True)
    s  = np.load(s_path,  allow_pickle=True)
    lm = str(ns["lm_convention"])
    model = str(s["model"]); dr = str(s["date_range"])

    ia, ib, dates = eu.align_on_dates(ns["dates"], s["dates"])
    if len(dates) < 20:
        return None

    market_iv = s["market_iv"][ib]                       # identical to NS on common dates
    model_px_ns, model_px_s = ns["model_px"][ia], s["model_px"][ib]
    model_iv_ns = eu.surfaces_to_iv(model_px_ns, lm)
    model_iv_s  = eu.surfaces_to_iv(model_px_s,  lm)
    w = eu.vega_weights(market_iv, lm)
    # COMMON-VALID MASK: score both variants on IDENTICAL grid points, so a
    # model with more IV-inversion failures cannot look better by having its
    # hardest points silently dropped.
    mask = eu.common_valid_mask(market_iv, model_iv_ns, model_iv_s, weights=w)
    cov = eu.coverage_report(market_iv, model_iv_ns, model_iv_s, w)

    # (1) primary + supporting
    prim = {"vw_iv_rmse_ns": eu.vw_iv_rmse(model_iv_ns, market_iv, w, mask),
            "vw_iv_rmse_s":  eu.vw_iv_rmse(model_iv_s,  market_iv, w, mask)}
    sup_ns = eu.supporting_metrics(model_iv_ns, market_iv, model_px_ns, s["market_px"][ib])
    sup_s  = eu.supporting_metrics(model_iv_s,  market_iv, model_px_s,  s["market_px"][ib])

    # (2) Clark-West (nested) AND Diebold-Mariano (separately-estimated models)
    cw = eu.clark_west(market_iv, model_iv_ns, model_iv_s, w, mask=mask)
    loss_ns_d = eu.daily_vw_iv_sqerr(model_iv_ns, market_iv, w, mask)
    loss_s_d  = eu.daily_vw_iv_sqerr(model_iv_s,  market_iv, w, mask)
    dm = eu.diebold_mariano(loss_ns_d, loss_s_d)

    # (3) no-arbitrage per variant
    arb_ns = eu.arbitrage_report(model_iv_ns, model_px_ns, lm)
    arb_s  = eu.arbitrage_report(model_iv_s,  model_px_s,  lm)

    # (4) benchmarks
    vix = None
    if VIX_BY_DATE is not None:
        vix = np.array([VIX_BY_DATE.get(d, np.nan) for d in dates])
    bench = eu.benchmark_table(market_iv, model_iv_ns, model_iv_s, w, vix)
    # NOTE: atm_flat (and vix_flat) use SAME-DAY information -> they are shape
    # diagnostics, not forecasts. previous_day is the only true forecast baseline.

    # (5) parameter turnover
    names = PARAM_NAMES.get(model)
    turn_ns = eu.param_turnover(ns["params"][ia], names)
    turn_s  = eu.param_turnover(s["params"][ib],  names)

    # (6) conditional analysis on sentiment magnitude + news volume (lag by 1 day)
    ctx = s["context"][ib]                               # (N,2) [sentiment, articles_scaled]
    sent_mag = np.abs(ctx[:, 0]); articles = ctx[:, 1]
    z_sent = np.roll(sent_mag, 1); z_sent[0] = np.nan
    z_art  = np.roll(articles, 1); z_art[0]  = np.nan
    # Use the RAW daily loss differential (the DM statistic), NOT the CW-adjusted
    # one: the two variants are separately estimated, so CW's correction is not
    # valid here and would flip bucket signs.
    dm_diff = loss_ns_d - loss_s_d                       # >0 favours sentiment
    buckets = {
        "by_sentiment_magnitude": eu.conditional_buckets(dm_diff, z_sent),
        "by_news_volume":         eu.conditional_buckets(dm_diff, z_art),
    }
    loss_ns = loss_ns_d
    loss_s  = loss_s_d
    H = np.column_stack([np.ones(len(dates)), z_sent, z_art])
    gw = eu.giacomini_white(loss_ns, loss_s, H)

    return {
        "model": model, "date_range": dr, "n_days": int(len(dates)),
        "primary": prim, "supporting_ns": sup_ns, "supporting_s": sup_s,
        "coverage": cov,
        "clark_west": {k: v for k, v in cw.items() if k != "f_day"},
        "diebold_mariano": dm,
        "arbitrage_ns": arb_ns, "arbitrage_s": arb_s,
        "benchmarks": bench,
        "param_turnover_ns": turn_ns, "param_turnover_s": turn_s,
        "conditional_buckets": buckets, "giacomini_white": gw,
    }


def main():
    rows = []
    for cell, ns_path, s_path in _cells():
        res = analyze(ns_path, s_path)
        if res is None:
            print(f"[skip] {cell}: too few common days"); continue
        with open(os.path.join(OUT, f"{cell}.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        p, cw = res["primary"], res["clark_west"]
        better = "S" if p["vw_iv_rmse_s"] < p["vw_iv_rmse_ns"] else "NS"
        d = res["diebold_mariano"]
        sig = "***" if d["dm_p_two_sided"] < 0.01 else \
              "**" if d["dm_p_two_sided"] < 0.05 else \
              "*" if d["dm_p_two_sided"] < 0.10 else ""
        rows.append({
            "model": res["model"], "regime": res["date_range"], "n": res["n_days"],
            "vwRMSE_NS": round(p["vw_iv_rmse_ns"], 6),
            "vwRMSE_S":  round(p["vw_iv_rmse_s"], 6),
            "better": better,
            "DM_t": round(d["dm_t"], 2), "DM_p": round(d["dm_p_two_sided"], 4),
            "sig_DM": sig,
            "CW_t": round(cw["cw_t"], 2), "CW_p": round(cw["cw_p_one_sided"], 4),
            "cal_viol_S": round(res["arbitrage_s"]["calendar_viol_rate"], 4),
            "bfly_viol_S": round(res["arbitrage_s"]["butterfly_viol_rate"], 4),
            "GW_p": round(res["giacomini_white"]["gw_p"], 4),
            "coverage": round(res["coverage"]["common_coverage"], 4),
            "inv_ok_NS": round(res["coverage"]["inversion_success_ns"], 4),
            "inv_ok_S": round(res["coverage"]["inversion_success_s"], 4),
        })
        print(f"[ok] {cell}: NS={p['vw_iv_rmse_ns']:.5f} S={p['vw_iv_rmse_s']:.5f} "
              f"DM_t={d['dm_t']:.2f}{sig} (favours {d['favours']})")

    if rows:
        import csv
        keys = list(rows[0].keys())
        with open(os.path.join(OUT, "results_master.csv"), "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader(); wtr.writerows(rows)
        print(f"\nWrote {os.path.join(OUT, 'results_master.csv')} ({len(rows)} cells)")
        print("sig_DM: *** p<0.01, ** p<0.05, * p<0.10 (two-sided Diebold-Mariano).")
        print("DM is the primary test here: the two variants are separately estimated,")
        print("which breaks the estimation-nesting assumption Clark-West requires.")
        print("CW columns retained for reference only - interpret with caution.")


if __name__ == "__main__":
    main()
