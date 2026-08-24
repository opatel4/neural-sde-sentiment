"""
walkforward.py   (PHASE 2 — compute-heavy; read the cost note before running)
============================================================================
DROP INTO A PACKAGE FOLDER and run from inside it:
    cd fine_tuning/Sentiment_Train && python walkforward.py

Replaces the single shuffled train/test split with a chronological expanding
window: train on all quarters up to Q, test on Q+1, roll forward. Emits OOS
surfaces per regime into
    fine_tuning_output/<variant>_wf_surfaces/<Model>_<Range>.npz
in the SAME schema as export_surfaces.py, so compare_variants.py (with NS_DIR /
S_DIR pointed at the *_wf_surfaces folders) runs Clark-West on genuinely
out-of-sample forecasts — which is what the reviewer feedback is really asking.

COST: your refit is 750-epoch MC-in-the-loop. A monthly roll over 13 years is
infeasible. This uses QUARTERLY test blocks and WARM-STARTS each window from the
previous window's weights with REFIT_EPOCHS (default 75) instead of restarting at
750. If still too expensive for all 8 variants, run only Heston + Bates and keep
the fixed-regime tables for the rest (label them clearly).
"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import (PROJECT_ROOT, MODEL_OUTPUT_ROOT, CANDIDATE_MODELS, CANDIDATE_DATE_RANGES,
                    get_spx_files_for_range, EPOCHS, LR, BATCH_SIZE, OFFLINE_WEIGHT,
                    offline_weight_for)
from data_utils import (get_model_registry, build_model_from_config,
                        recover_mean_std_from_offline_csv, make_blended_scaler,
                        create_iv_and_price_grids_from_raw, filter_extreme_prices)
from trainer import RealWorldFineTuner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- knobs (overridable via env vars, e.g. WF_REFIT_EPOCHS=5 WF_MODELS=Heston) ----
REFIT_EPOCHS   = int(os.environ.get("WF_REFIT_EPOCHS", 75))
MIN_TRAIN_QTRS = int(os.environ.get("WF_MIN_TRAIN_QTRS", 4))
MODELS_TO_RUN  = (os.environ["WF_MODELS"].split(",")
                  if os.environ.get("WF_MODELS") else CANDIDATE_MODELS)
RANGES_TO_RUN  = (os.environ["WF_RANGES"].split(",")
                  if os.environ.get("WF_RANGES") else CANDIDATE_DATE_RANGES)
# ----------------------------------------------------------------------------

_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
# PATCHED: both arms run from Sentiment_Train/ now, so the folder name no longer
# identifies the arm. Use the same flags training and export used.
_USE_SENT = os.environ.get("SDP_USE_SENTIMENT", "1") == "1"
_USE_IN2  = os.environ.get("SDP_USE_IN2", "0") == "1"
VARIANT = "sentiment" if _USE_SENT else "nonsent"
# This repo uses full_project_output/ (not fine_tuning/fine_tuning_output/).
_REPO_ROOT = os.path.dirname(PROJECT_ROOT)
OUT_DIR = os.path.join(_REPO_ROOT, "full_project_output", "eval_wf_surfaces", VARIANT)
os.makedirs(OUT_DIR, exist_ok=True)


def _takes_context(model):
    return len(inspect.signature(model.forward).parameters) >= 2


def _loaders(iv, px, ctx, mask, batch, shuffle):
    ds = TensorDataset(torch.tensor(iv[mask]), torch.tensor(px[mask]), torch.tensor(ctx[mask]))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _extract(tuner, iv, px, ctx, mask):
    loader = _loaders(iv, px, ctx, mask, 32, False)
    takes = _takes_context(tuner.model); tuner.model.eval()
    m_iv, m_px, p_px, prm = [], [], [], []
    with torch.no_grad():
        for a, b, c in loader:
            a = a.to(device).float(); c = c.to(device).float()
            s = tuner.scale_input(a)
            _, pr = tuner.model(s, c) if takes else tuner.model(s)
            pxs = tuner.calculate_model_price_surface(pr, for_eval=True)
            m_iv.append(a.cpu().numpy()); m_px.append(b.numpy())
            p_px.append(pxs.cpu().numpy()); prm.append(pr.cpu().numpy())
    return (np.concatenate(m_iv).squeeze(1), np.concatenate(m_px),
            np.concatenate(p_px), np.concatenate(prm))


def walk_forward_regime(model_name, date_range, registry):
    if _USE_IN2:
        from in2_adapter import set_sent_regime
        set_sent_regime(date_range)
    iv, px, ctx, dates, lm = create_iv_and_price_grids_from_raw(
        get_spx_files_for_range(date_range), max_days=None, use_calls_only=False,
        min_points_per_day=6, min_tau_per_day=2, min_lm_per_day=3)
    iv, px, ctx, dates = filter_extreme_prices(
        iv, px, ctx, dates, 97, 2, 98, 2.0)
    iv, px, ctx = iv.numpy(), px.numpy(), ctx.numpy()
    order = np.argsort(pd.to_datetime(dates).values)
    iv, px, ctx = iv[order], px[order], ctx[order]
    d = pd.to_datetime(np.array(dates)[order])
    qs = pd.PeriodIndex(d, freq='Q'); blocks = list(pd.unique(qs))

    cfg = registry[model_name]
    off_mean, off_std = recover_mean_std_from_offline_csv(cfg["offline_csv"])

    carried_state = None                         # warm-start weights across windows
    oos = {"iv": [], "mpx": [], "ppx": [], "prm": [], "dates": [], "ctx": []}

    for i in range(MIN_TRAIN_QTRS, len(blocks)):
        tr = np.isin(qs, blocks[:i]); te = qs == blocks[i]
        if tr.sum() < 30 or te.sum() < 3:
            continue

        # PATCHED: In2 checkpoints are 2-channel; build the matching model.
        if _USE_IN2:
            from in2_adapter import build_model_from_config_in2
            model = build_model_from_config_in2(cfg, device, strict=True)
        else:
            model = build_model_from_config(cfg, device)   # loads offline checkpoint
        if carried_state is not None:
            model.load_state_dict(carried_state)             # warm-start
        X_mean, X_std = make_blended_scaler(
            off_mean, off_std, torch.tensor(iv[tr]),
            offline_weight=offline_weight_for(model_name))
        tuner = RealWorldFineTuner(model=model, x_mean=X_mean.to(device),
                                   x_std=X_std.to(device), device=device,
                                   model_config=cfg, lm_convention=lm)

        train_loader = _loaders(iv, px, ctx, tr, BATCH_SIZE, True)
        tuner.fine_tune(train_loader, val_loader=None, epochs=REFIT_EPOCHS, lr=LR)
        carried_state = {k: v.detach().clone() for k, v in tuner.model.state_dict().items()}

        m_iv, m_px, p_px, prm = _extract(tuner, iv, px, ctx, te)
        oos["iv"].append(m_iv); oos["mpx"].append(m_px); oos["ppx"].append(p_px)
        oos["prm"].append(prm); oos["ctx"].append(ctx[te])
        oos["dates"].append(np.array([str(x.date()) for x in d[te]]))
        print(f"    {model_name} {date_range} test {blocks[i]}: {te.sum()} days OOS")

    if not oos["dates"]:
        return
    np.savez_compressed(
        os.path.join(OUT_DIR, f"{model_name}_{date_range}.npz"),
        variant=VARIANT, model=model_name, date_range=date_range, lm_convention=lm,
        dates=np.concatenate(oos["dates"]),
        market_iv=np.concatenate(oos["iv"]).astype(np.float32),
        market_px=np.concatenate(oos["mpx"]).astype(np.float32),
        model_px=np.concatenate(oos["ppx"]).astype(np.float32),
        params=np.concatenate(oos["prm"]).astype(np.float32),
        context=np.concatenate(oos["ctx"]).astype(np.float32))


def main():
    import argparse, traceback
    global REFIT_EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",  default=None, help="comma-separated, e.g. Heston,Bates")
    ap.add_argument("--regimes", default=None, help="comma-separated, e.g. 2013-2015")
    ap.add_argument("--epochs",  type=int, default=REFIT_EPOCHS,
                    help="refit epochs per window (use 5 for a smoke test)")
    a = ap.parse_args()

    REFIT_EPOCHS = a.epochs
    models  = a.models.split(",")  if a.models  else MODELS_TO_RUN
    regimes = a.regimes.split(",") if a.regimes else RANGES_TO_RUN

    print(f"[{VARIANT}] walk-forward -> {OUT_DIR} (refit_epochs={REFIT_EPOCHS})")
    print(f"  models={models}  regimes={regimes}")
    registry = get_model_registry()
    for m in models:
        for dr in regimes:
            print(f"  == {m} {dr} ==")
            try:
                walk_forward_regime(m, dr, registry)
            except Exception as e:
                traceback.print_exc()
                print(f"    [error] {m} {dr}: {e}")
    print(f"[{VARIANT}] walk-forward done.")


if __name__ == "__main__":
    main()
