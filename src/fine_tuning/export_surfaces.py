import re
"""
export_surfaces.py
==================
DROP THIS FILE INTO BOTH:
    fine_tuning/Non_Sentiment_Train/
    fine_tuning/Sentiment_Train/

Run from inside each folder AFTER training/fine-tuning has produced checkpoints:
    cd fine_tuning/Non_Sentiment_Train && python export_surfaces.py
    cd fine_tuning/Sentiment_Train     && python export_surfaces.py

For every (model, regime) with a checkpoint it loads the best epoch, rebuilds
the exact tuner used in evaluate.py, runs a NON-shuffled loader (so row n = day n),
and saves per-day arrays to:
    fine_tuning_output/<variant>_surfaces/<Model>_<Range>.npz

The companion driver compare_variants.py (run from the repo root) reads these.
This script reuses your config / data_utils / trainer verbatim and works
unchanged in both packages — it auto-detects whether the model takes a sentiment
context argument.
"""

import sys, os, glob, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import (
    PROJECT_ROOT, MODEL_OUTPUT_ROOT, CANDIDATE_MODELS, CANDIDATE_DATE_RANGES,
    get_spx_files_for_range,
)
from data_utils import (
    get_model_registry, build_model_from_config,
    create_iv_and_price_grids_from_raw, filter_extreme_prices,
)
from trainer import RealWorldFineTuner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Robust: "Non_Sentiment_Train" also contains "Sentiment_Train", so key off the basename.
_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
# PATCHED: both arms now run from Sentiment_Train/, so the package name no
# longer identifies the arm. Use the same flag training used.
_USE_SENT = os.environ.get("SDP_USE_SENTIMENT", "1") == "1"
_USE_IN2  = os.environ.get("SDP_USE_IN2", "0") == "1"
VARIANT = "sentiment" if _USE_SENT else "nonsent"

# This repo's trained checkpoints live under <repo>/full_project_output/... , NOT
# under the fine_tuning/fine_tuning_output path config.py points at. Search both.
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                       # .../Senior Design Research Project
_OUT_NAME = os.path.basename(os.path.dirname(MODEL_OUTPUT_ROOT))  # sentiment_output / base_output
# PATCHED: search the CURRENT run's output first. full_project_output may hold
# stale checkpoints from the previous study.
CKPT_ROOTS = [
    MODEL_OUTPUT_ROOT,
    os.path.join(REPO_ROOT, "full_project_output", _OUT_NAME, "models"),
]
OUT_DIR = os.path.join(REPO_ROOT, "full_project_output", "eval_surfaces", VARIANT)
os.makedirs(OUT_DIR, exist_ok=True)


def _find_checkpoint(model_name, date_range):
    """PATCHED: pick by NUMERIC epoch (lexical sort put epoch_99 after epoch_700)
    and prefer the *final* checkpoint, which is what training last wrote."""
    def _ep(f):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    for root in CKPT_ROOTS:
        d = os.path.join(root, model_name, date_range)
        finals = glob.glob(os.path.join(d, "best_model_final_epoch_*.pth"))
        if finals:
            return max(finals, key=_ep)
        cands = glob.glob(os.path.join(d, "best_model_epoch_*.pth"))
        if cands:
            return max(cands, key=_ep)
        if os.path.exists(os.path.join(d, "best_model.pth")):
            return os.path.join(d, "best_model.pth")
    return None


def _model_takes_context(model):
    return len(inspect.signature(model.forward).parameters) >= 2


def _extract(tuner, iv_t, px_t, ctx_t):
    """Chronological per-day extraction. Handles context vs no-context models."""
    loader = DataLoader(TensorDataset(iv_t, px_t, ctx_t), batch_size=32, shuffle=False)
    takes_ctx = _model_takes_context(tuner.model)
    tuner.model.eval()
    m_iv, m_px, p_px, prm = [], [], [], []
    with torch.no_grad():
        for miv, mpx, ctx in loader:
            miv = miv.to(device).float(); ctx = ctx.to(device).float()
            scaled = tuner.scale_input(miv)
            # PATCHED: In2 needs the (B,2,8,11) stacked input, built exactly as
            # training built it - standardised sentiment for the sentiment arm,
            # constant fill for the baseline arm. Without this the model never
            # sees sentiment and BOTH arms would export identical surfaces.
            if _USE_IN2:
                from in2_adapter import make_in2_batch
                x = make_in2_batch(scaled, ctx, use_sentiment=_USE_SENT)
                _, params = tuner.model(x)
            elif takes_ctx:
                _, params = tuner.model(scaled, ctx)
            else:
                _, params = tuner.model(scaled)
            px = tuner.calculate_model_price_surface(params, for_eval=True)
            m_iv.append(miv.cpu().numpy()); m_px.append(mpx.numpy())
            p_px.append(px.cpu().numpy()); prm.append(params.cpu().numpy())
    return (np.concatenate(m_iv).squeeze(1), np.concatenate(m_px),
            np.concatenate(p_px), np.concatenate(prm))


def _np(x):
    """data_utils returns torch tensors; normalize to numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    print(f"[{VARIANT}] device={device}  ->  writing to {OUT_DIR}")
    registry = get_model_registry()

    # cache market data per regime (same loading path as evaluate.py __main__)
    cache = {}
    for dr in CANDIDATE_DATE_RANGES:
        res = create_iv_and_price_grids_from_raw(
            get_spx_files_for_range(dr), max_days=None, use_calls_only=False,
            min_points_per_day=6, min_tau_per_day=2, min_lm_per_day=3)
        # Sentiment package returns 5 (with context); non-sentiment returns 4.
        if len(res) == 5:
            iv, px, ctx, dates, lm = res
            iv, px, ctx, dates = filter_extreme_prices(
                iv, px, ctx, dates, global_price_cap_percentile=97,
                per_point_low_pct=2, per_point_high_pct=98, day_outlier_iqr_factor=2.0)
            ctx = _np(ctx)
        else:
            iv, px, dates, lm = res
            iv, px, dates = filter_extreme_prices(
                iv, px, dates, global_price_cap_percentile=97,
                per_point_low_pct=2, per_point_high_pct=98, day_outlier_iqr_factor=2.0)
            # no sentiment input in this variant; keep a zero context for shape parity
            ctx = np.zeros((len(dates), 2), dtype=np.float32)
        # sort chronologically so row n == day n (time-series tests depend on this)
        order = np.argsort(pd.to_datetime(dates).values)
        iv, px = _np(iv), _np(px)
        cache[dr] = (iv[order], px[order], ctx[order],
                     np.array([str(pd.Timestamp(dates[i]).date()) for i in order]), lm)

    for model_name in CANDIDATE_MODELS:
        for dr in CANDIDATE_DATE_RANGES:
            ckpt_path = _find_checkpoint(model_name, dr)
            if ckpt_path is None:
                print(f"  [skip] no checkpoint: {model_name} {dr}")
                continue
            if _USE_IN2:
                from in2_adapter import set_sent_regime
                set_sent_regime(dr)
            iv_t_np, px_t_np, ctx_t_np, dates, lm = cache[dr]
            iv_t = torch.as_tensor(iv_t_np, dtype=torch.float32)
            px_t = torch.as_tensor(px_t_np, dtype=torch.float32)
            ctx_t = torch.as_tensor(ctx_t_np, dtype=torch.float32)

            ckpt = torch.load(ckpt_path, map_location=device)
            cfg = registry[model_name]
            # PATCHED: the checkpoints are 2-channel ParameterCNN_In2.
            if _USE_IN2:
                from in2_adapter import build_model_from_config_in2
                model = build_model_from_config_in2(cfg, device, strict=False)
            else:
                model = build_model_from_config(cfg, device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            tuner = RealWorldFineTuner(
                model=model, x_mean=ckpt["x_mean"].to(device),
                x_std=ckpt["x_std"].to(device), device=device,
                model_config=cfg, lm_convention=lm,
                train_mc_settings=ckpt.get("train_mc_settings"),
                eval_mc_settings=ckpt.get("eval_mc_settings"))

            market_iv, market_px, model_px, params = _extract(tuner, iv_t, px_t, ctx_t)

            ctx_np = (ctx_t_np.detach().cpu().numpy()
                      if hasattr(ctx_t_np, "detach") else np.asarray(ctx_t_np))
            out = os.path.join(OUT_DIR, f"{model_name}_{dr}.npz")
            np.savez_compressed(
                out, variant=VARIANT, model=model_name, date_range=dr,
                lm_convention=lm, dates=dates,
                market_iv=market_iv.astype(np.float32),       # (N,8,11)
                market_px=market_px.astype(np.float32),       # (N,1,8,11)
                model_px=model_px.astype(np.float32),         # (N,1,8,11)
                params=params.astype(np.float32),             # (N,d)
                context=ctx_np.astype(np.float32))            # (N,2) [sentiment, articles_scaled]
            print(f"  [ok] {model_name} {dr}: {len(dates)} days -> {out}")

    print(f"[{VARIANT}] done.")


if __name__ == "__main__":
    main()
