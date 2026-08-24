"""
eval_upgrades.py
================
Package-agnostic evaluation methodologies for the sentiment / non-sentiment
neural-SDE option-pricing comparison. NO torch dependency — everything here
operates on plain numpy arrays produced by `export_surfaces.py`.

Grid + market conventions are copied verbatim from RealWorldFineTuner so the
Black-Scholes math matches how your surfaces were built:
    maturities      = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    log-moneyness   = [-0.5 ... 0.5] (11 points)
    S = 1.0, r = 0.045, q = 0.011   (normalized, discounted call prices)

If your data-prep used different r/q for the *market* IV column, set R/Q below
to match. Note: for the NS-vs-S *comparison* (Clark-West), any market-side
convention offset cancels, so CW is robust to a small r/q mismatch; only the
absolute vega-weighted RMSE *level* is sensitive to it.
"""

import numpy as np
from scipy.stats import norm, chi2

# ---------------------------------------------------------------------------
# Grid + market constants (must match RealWorldFineTuner)
# ---------------------------------------------------------------------------
MATS = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
LMS  = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ATM_COL = 5                       # index of log-moneyness 0.0
# PATCHED: read the same rates the pipeline was trained/exported with.
# Old hardcoded 0.045/0.011 gives drift 0.034 vs the market's ~0.008.
# Set SDP_R per regime before running (q is 0.015 in every year):
#   2010-2012 0.0254 | 2013-2015 0.0230 | 2016-2019 0.0236 | 2020-2022 0.0169
import os as _os
S0 = 1.0
R = float(_os.environ.get("SDP_R", 0.0231))
Q = float(_os.environ.get("SDP_Q", 0.0150))


def forwards():
    return S0 * np.exp((R - Q) * MATS)                # (8,)


def strikes(lm_convention='k_over_f'):
    """(8,11) strike grid, matching trainer.price_one_maturity."""
    F = forwards()[:, None]                           # (8,1)
    if lm_convention == 'k_over_f':
        return F * np.exp(LMS[None, :])
    return F * np.exp(-LMS[None, :])                  # f_over_k


# ---------------------------------------------------------------------------
# Black-Scholes: price, vega, implied-vol inversion
# ---------------------------------------------------------------------------
def bs_call(sig, K, T, F):
    sig = np.maximum(sig, 1e-8); T = np.maximum(T, 1e-8)
    srt = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / srt
    d2 = d1 - srt
    return np.exp(-R * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_vega(sig, K, T, F):
    sig = np.maximum(sig, 1e-8); T = np.maximum(T, 1e-8)
    d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    return np.exp(-R * T) * F * norm.pdf(d1) * np.sqrt(T)


def _implied_vol_scalar(price, K, T, F, lo=1e-4, hi=3.0):
    intrinsic = max(np.exp(-R * T) * (F - K), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-9:
        return np.nan
    flo, fhi = bs_call(lo, K, T, F) - price, bs_call(hi, K, T, F) - price
    if flo * fhi > 0:                                 # no sign change -> unbracketable
        return np.nan
    for _ in range(80):                               # bisection (no scipy dependency)
        mid = 0.5 * (lo + hi); fm = bs_call(mid, K, T, F) - price
        if abs(fm) < 1e-10:
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def surfaces_to_iv(price_surface, lm_convention='k_over_f'):
    """(N,1,8,11) or (N,8,11) model prices -> (N,8,11) implied vols (NaN where unbracketable)."""
    P = price_surface.squeeze(1) if price_surface.ndim == 4 else price_surface
    K = strikes(lm_convention); F = forwards()
    out = np.full(P.shape, np.nan)
    for n in range(P.shape[0]):
        for i in range(8):
            for j in range(11):
                out[n, i, j] = _implied_vol_scalar(P[n, i, j], K[i, j], MATS[i], F[i])
    return out


def vega_weights(market_iv, lm_convention='k_over_f'):
    """(N,8,11) BS vega at MARKET iv -> weights, identical for both variants."""
    K = strikes(lm_convention); F = forwards()[None, :, None]
    return bs_vega(market_iv, K[None, :, :], MATS[None, :, None], F)


# ---------------------------------------------------------------------------
# Metric 1: vega-weighted IV RMSE (primary) + supporting metrics
# ---------------------------------------------------------------------------
def common_valid_mask(market_iv, *model_ivs, weights=None):
    """
    Shared finite mask so BOTH variants are scored on IDENTICAL grid points.

    Without this each model is masked independently, and a model with MORE
    IV-inversion failures can look better because its hardest points silently
    disappear. Pass every model surface that will be compared.
    """
    m = np.isfinite(market_iv)
    for iv in model_ivs:
        m = m & np.isfinite(iv)
    if weights is not None:
        m = m & np.isfinite(weights) & (weights > 0)
    return m


def coverage_report(market_iv, iv_ns, iv_s, w):
    """IV-inversion success rates + common coverage, so masking is auditable."""
    common = common_valid_mask(market_iv, iv_ns, iv_s, weights=w)
    per_day = common.reshape(common.shape[0], -1).sum(axis=1)
    return {
        "inversion_success_ns": float(np.isfinite(iv_ns).mean()),
        "inversion_success_s":  float(np.isfinite(iv_s).mean()),
        "common_coverage":      float(common.mean()),
        "n_points_total":       int(market_iv.size),
        "n_points_common":      int(common.sum()),
        "valid_points_per_day_mean": float(per_day.mean()),
        "valid_points_per_day_min":  int(per_day.min()),
        "days_with_zero_valid":      int((per_day == 0).sum()),
    }


def vw_iv_rmse(model_iv, market_iv, w, mask=None):
    """Pass `mask` (from common_valid_mask) so both variants use the same points."""
    e2 = (model_iv - market_iv) ** 2
    m = (np.isfinite(e2) & np.isfinite(w)) if mask is None else mask
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.sum(w[m] * e2[m]) / np.sum(w[m])))


def daily_vw_iv_sqerr(model_iv, market_iv, w, mask=None):
    """
    Per-day vega-weighted mean squared IV error -> (N,).

    Numerator AND denominator use the SAME mask: an excluded point removes its
    vega from the denominator too. The previous version skipped NaN errors in
    the numerator but kept their vegas in the denominator, which mechanically
    understated the loss of whichever model had more inversion failures.
    """
    if mask is None:
        mask = np.isfinite(model_iv) & np.isfinite(market_iv) & np.isfinite(w)
    e2 = np.where(mask, (model_iv - market_iv) ** 2, 0.0)
    wm = np.where(mask, w, 0.0)
    num = np.sum(wm * e2, axis=(1, 2))
    den = np.sum(wm, axis=(1, 2))
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)


def supporting_metrics(model_iv, market_iv, model_px, market_px):
    d_iv = model_iv - market_iv
    d_px = (model_px.squeeze(1) if model_px.ndim == 4 else model_px) - \
           (market_px.squeeze(1) if market_px.ndim == 4 else market_px)
    return {
        "iv_mae":   float(np.nanmean(np.abs(d_iv))),
        "iv_rmse":  float(np.sqrt(np.nanmean(d_iv ** 2))),
        "px_mae":   float(np.nanmean(np.abs(d_px))),
        "px_rmse":  float(np.sqrt(np.nanmean(d_px ** 2))),
    }


# ---------------------------------------------------------------------------
# Metric 2: Clark-West nested test (+ Newey-West)
# ---------------------------------------------------------------------------
def newey_west_t(f, lags=None):
    f = np.asarray(f, float); f = f[np.isfinite(f)]; T = len(f)
    if T < 8:
        return np.nan, np.nan, 0
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    e = f - f.mean(); var = (e @ e) / T
    for L in range(1, lags + 1):
        var += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / T
    se = np.sqrt(max(var, 1e-30) / T)
    return f.mean(), f.mean() / se, lags


def diebold_mariano(loss_ns, loss_s, lags=None):
    """
    DM test on daily loss differential d_t = loss_ns - loss_s  (>0 favours sentiment).
    Use this when the two models are SEPARATELY ESTIMATED (different init/training),
    which breaks the estimation-nesting assumption Clark-West relies on.
    Two-sided p-value: H1 = the two models differ in predictive accuracy.
    """
    d = np.asarray(loss_ns, float) - np.asarray(loss_s, float)
    mean, t, L = newey_west_t(d, lags)
    p2 = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    return {"dm_mean": float(mean), "dm_t": float(t), "dm_p_two_sided": float(p2),
            "nw_lags": L, "favours": ("sentiment" if mean > 0 else "no_sentiment")}


def clark_west(market_iv, iv_ns, iv_s, w, lags=None, mask=None):
    """
    Nested CW test. NS (smaller) nested in S (larger).
    f_pt > 0 favours the sentiment model. One-sided p: H1 = sentiment better.
    Returns the CW-adjusted per-day differential too (for conditional analysis).
    """
    if mask is None:
        mask = common_valid_mask(market_iv, iv_ns, iv_s, weights=w)
    e1  = (market_iv - iv_ns) ** 2
    e2  = (market_iv - iv_s) ** 2
    adj = (iv_ns - iv_s) ** 2
    f_pt = np.where(mask, e1 - (e2 - adj), 0.0)
    wm   = np.where(mask, w, 0.0)
    num = np.sum(wm * f_pt, axis=(1, 2))
    den = np.sum(wm, axis=(1, 2))
    f_day = np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0)
    mean, t, L = newey_west_t(f_day, lags)
    return {"cw_mean": float(mean), "cw_t": float(t),
            "cw_p_one_sided": float(1 - norm.cdf(t)) if np.isfinite(t) else np.nan,
            "nw_lags": L, "f_day": f_day}


# ---------------------------------------------------------------------------
# Metric 3: no-arbitrage checks
# ---------------------------------------------------------------------------
def calendar_violation_rate(iv_surface, tol=1e-6):
    """Total implied variance non-decreasing in maturity (per moneyness column)."""
    w = (iv_surface ** 2) * MATS[:, None]
    dv = np.diff(w, axis=0)
    fin = np.isfinite(dv)
    return float(np.mean(dv[fin] < -tol)) if fin.any() else np.nan


def butterfly_violation_rate(price_surface, lm_convention='k_over_f', tol=1e-8):
    """Convexity of call price in strike (per maturity row)."""
    P = price_surface.squeeze() if price_surface.ndim > 2 else price_surface
    K = strikes(lm_convention)
    viol = cnt = 0
    for i in range(8):
        order = np.argsort(K[i]); k = K[i][order]; c = P[i][order]
        for j in range(1, 10):
            if not np.all(np.isfinite([c[j-1], c[j], c[j+1]])):
                continue
            left  = (c[j]   - c[j-1]) / (k[j]   - k[j-1])
            right = (c[j+1] - c[j])   / (k[j+1] - k[j])
            cnt += 1; viol += int(right - left < -tol)
    return viol / cnt if cnt else np.nan


def arbitrage_report(model_iv_surfaces, model_px_surfaces, lm_convention='k_over_f'):
    """Average calendar + butterfly violation rate across all days."""
    cal = [calendar_violation_rate(model_iv_surfaces[n]) for n in range(len(model_iv_surfaces))]
    but = [butterfly_violation_rate(model_px_surfaces[n], lm_convention)
           for n in range(len(model_px_surfaces))]
    return {"calendar_viol_rate": float(np.nanmean(cal)),
            "butterfly_viol_rate": float(np.nanmean(but))}


def put_call_parity_check(raw_csv_path, tol=0.02):
    """
    DATA QC on the raw merged CSV (not model output): C - P vs S e^{-qT} - K e^{-rT}
    for matched (date, strike, tau). Returns violation fraction. Column names are
    best-effort; adjust to your schema. Returns None if columns are absent.
    """
    import pandas as pd
    df = pd.read_csv(raw_csv_path, low_memory=False)
    need = {'Trade Date', 'Strike', 'Tau', 'OptionType'}
    price_col = 'Mid' if 'Mid' in df.columns else ('OptionPrice' if 'OptionPrice' in df.columns else None)
    if not need.issubset(df.columns) or price_col is None or 'Forward' not in df.columns:
        return None
    r = R; q = Q      # PATCHED: was hardcoded 0.045/0.011
    df = df.dropna(subset=['Strike', 'Tau', 'Forward', price_col])
    calls = df[df['OptionType'].astype(str).str.upper() == 'C']
    puts  = df[df['OptionType'].astype(str).str.upper() == 'P']
    m = calls.merge(puts, on=['Trade Date', 'Strike', 'Tau'], suffixes=('_c', '_p'))
    if len(m) == 0:
        return None
    S = m['Forward_c'] * np.exp(-(r - q) * m['Tau'])
    lhs = m[f'{price_col}_c'] - m[f'{price_col}_p']
    rhs = S * np.exp(-q * m['Tau']) - m['Strike'] * np.exp(-r * m['Tau'])
    return float(np.mean(np.abs(lhs - rhs) > tol))


# ---------------------------------------------------------------------------
# Metric 4: cheap benchmarks (score with the same vega-weighted IV RMSE)
# ---------------------------------------------------------------------------
def persistence_iv(market_iv):
    """Previous-day surface. Row 0 is NaN (dropped in scoring)."""
    pred = market_iv.copy(); pred[1:] = market_iv[:-1]; pred[0] = np.nan
    return pred


def atm_flat_iv(market_iv):
    """Flat surface at that day's ATM IV (no external data needed)."""
    atm = market_iv[:, :, ATM_COL:ATM_COL + 1]
    return np.broadcast_to(atm, market_iv.shape).copy()


def vix_flat_iv(vix_series):
    """Flat surface at daily VIX (as decimal). vix_series: (N,) in decimals."""
    v = np.asarray(vix_series, float)[:, None, None]
    return np.broadcast_to(v, (len(v), 8, 11)).copy()


def benchmark_table(market_iv, model_iv_ns, model_iv_s, w, vix_series=None):
    rows = {
        "previous_day":  vw_iv_rmse(persistence_iv(market_iv), market_iv, w),
        "atm_flat":      vw_iv_rmse(atm_flat_iv(market_iv),    market_iv, w),
        "cnn_no_sent":   vw_iv_rmse(model_iv_ns, market_iv, w),
        "cnn_sentiment": vw_iv_rmse(model_iv_s,  market_iv, w),
    }
    if vix_series is not None:
        rows["vix_flat"] = vw_iv_rmse(vix_flat_iv(vix_series), market_iv, w)
    return rows


# ---------------------------------------------------------------------------
# Metric 5: parameter stability
# ---------------------------------------------------------------------------
def param_turnover(params, names=None):
    d = np.diff(params, axis=0)
    out = {"std_of_daily_change": np.nanstd(d, axis=0).tolist(),
           "mean_abs_change":     np.nanmean(np.abs(d), axis=0).tolist()}
    if names:
        out["param_names"] = list(names)
    return out


# ---------------------------------------------------------------------------
# Metric 6: conditional analysis + Giacomini-White
# ---------------------------------------------------------------------------
def conditional_buckets(f_day, z, n_buckets=3, labels=("low", "mid", "high")):
    """
    f_day: CW-adjusted daily differential (>0 favours sentiment).
    z:     conditioning variable at t (align/lag before calling if needed).
    Returns per-bucket mean differential + Newey-West t.
    """
    f = np.asarray(f_day, float); z = np.asarray(z, float)
    m = np.isfinite(f) & np.isfinite(z); f, z = f[m], z[m]
    edges = np.quantile(z, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    lab = np.digitize(z, edges[1:-1])
    res = {}
    for b in range(n_buckets):
        fb = f[lab == b]
        mean, t, _ = newey_west_t(fb)
        res[labels[b] if b < len(labels) else b] = {
            "n": int(len(fb)), "mean_diff": float(mean), "t": float(t)}
    return res


def giacomini_white(loss_ns, loss_s, instruments):
    """
    GW conditional predictive ability test.
    loss_ns, loss_s : (N,) per-day loss series (e.g. daily vega-weighted IV MSE).
    instruments     : (N, q) known at t-1 (INCLUDE a constant column of ones).
                      Lag your conditioners by one period before passing them in.
    H0: equal conditional predictive ability. stat ~ chi2(q).
    """
    d = np.asarray(loss_ns, float) - np.asarray(loss_s, float)      # >0 favours sentiment
    H = np.asarray(instruments, float)
    m = np.isfinite(d) & np.all(np.isfinite(H), axis=1)
    d, H = d[m], H[m]
    T, qd = H.shape
    mt = H * d[:, None]                        # moment (T,q)
    mbar = mt.mean(axis=0)
    # Newey-West HAC of the moment vector (raw, since H0 mean is 0)
    lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    S = (mt.T @ mt) / T
    for L in range(1, lags + 1):
        G = (mt[L:].T @ mt[:-L]) / T
        S += (1 - L / (lags + 1)) * (G + G.T)
    try:
        stat = T * mbar @ np.linalg.solve(S, mbar)
    except np.linalg.LinAlgError:
        stat = T * mbar @ np.linalg.pinv(S) @ mbar
    return {"gw_stat": float(stat), "gw_dof": int(qd),
            "gw_p": float(1 - chi2.cdf(stat, qd)),
            "mean_diff_favouring_sentiment": float(d.mean())}


# ---------------------------------------------------------------------------
# Alignment helper (intersect two exports on common dates)
# ---------------------------------------------------------------------------
def align_on_dates(dates_a, dates_b):
    """Return (idx_a, idx_b) that select the common dates in sorted order."""
    da = np.asarray(dates_a); db = np.asarray(dates_b)
    common = np.intersect1d(da, db)
    ia = np.array([np.where(da == c)[0][0] for c in common])
    ib = np.array([np.where(db == c)[0][0] for c in common])
    return ia, ib, common
