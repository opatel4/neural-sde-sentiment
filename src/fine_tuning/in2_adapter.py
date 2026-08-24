"""
in2_adapter.py  —  wires the In2 (two-channel) architecture into the ONLINE
fine-tuning package WITHOUT editing data_utils.py / model.py in place.

Drop into fine_tuning/Sentiment_Train/ next to model_in2.py.

WHY AN ADAPTER
--------------
Your existing files must keep working: export_surfaces.py still needs the old
model.py + In1 weights to read your current 32 checkpoints. So instead of
rewriting data_utils.py, this module provides In2 versions of the three things
that change:

    get_model_registry_in2()      -> points at CNN_*_In2_OutBase.pth
    build_model_from_config_in2() -> builds ParameterCNN_In2, strict load
    make_in2_batch()              -> builds the (B,2,8,11) input from a batch

WHAT CHANGES vs In1
-------------------
  In1: model(scaled_iv, context)   context = [sentiment, articles] scalars
  In2: model(x)                    x[:,0]=IV surface, x[:,1]=sentiment heatmap
       (offline dense layer is (256, 2816) = 32*8*11 -> NO scalar path at all)

TRAINER EDITS REQUIRED (3 call sites, from your grep):
    line ~172:  _, predicted_params = self.model(scaled_input, context)
    line ~216:  _, predicted_params = self.model(scaled_input, context)
    line ~329:  _, predicted_params = self.model(scaled_input, context)
  each becomes:
    x = make_in2_batch(scaled_input, context, use_sentiment=USE_SENTIMENT)
    _, predicted_params = self.model(x)

!! OPEN QUESTION — CONFIRM WITH WHOEVER BUILT THE OFFLINE In2 DATA !!
The offline CSVs carry PER-CELL sentiment as SS_T{T}_K{K} columns on a strike
axis of K0.5..K1.5 (moneyness K/S). The online grid is LogMoneyness on a
'k_over_f' convention spanning -0.5..+0.5. If the offline channel was built
differently from what make_in2_batch() produces here, the pretrained conv
filters will not mean the same thing on real data — a SILENT degradation, not
an error. Ask: (a) how is SS scaled/normalized offline? (b) does K1.0 map to
LogMoneyness 0? Then set SENT_MODE / SENT_SCALE below to match.
"""

import os
import torch

from model_in2 import ParameterCNN_In2, build_in2_input   # noqa: F401

# --- match these to the offline construction once confirmed ----------------
SENT_MODE  = "broadcast"   # "broadcast" = daily scalar over whole grid
                           # "per_maturity" = (B,8) broadcast across strikes
# --- V2 sentiment standardisation (recomputed 2026-08-16) ------------------
# Offline SS distribution, measured on the V2 surfaces after the path-axis
# fix (one value per maturity, constant across strikes):
#     pooled mean -0.4251, std 0.2165
# Online FinBERT daily_sentiment drifts markedly over the sample -- yearly
# means run +0.40 (2011) down to +0.04 (2021), a spread larger than the
# pooled std -- so shift and scale are resolved per regime. Call
# set_sent_regime(DATE_RANGE) once before training each regime.
SENT_OFFLINE_MEAN = -0.0296        # offline SS mean, pooled (new coefficients 2026-08-16)
SENT_OFFLINE_STD  = 0.0927         # offline SS std,  pooled over families

SENT_ONLINE_BY_REGIME = {          # regime -> (online mean, online std)
    "2010-2012": (0.2970, 0.1432),
    "2013-2015": (0.2703, 0.1011),
    "2016-2019": (0.1356, 0.1173),
    "2020-2022": (0.0700, 0.1074),
}

# Live values read by make_in2_batch(). Default to the pooled figures so an
# unset regime still produces a sane transform rather than silently wrong one.
SENT_SHIFT  = 0.1817                       # online mean, pooled
SENT_SCALE  = SENT_OFFLINE_STD / 0.1499    # = 1.4442, pooled
SENT_OFFSET = SENT_OFFLINE_MEAN
SENT_REGIME = None


def set_sent_regime(regime):
    """Point SENT_SHIFT / SENT_SCALE at one regime's online moments.

    Must be called before make_in2_batch() for that regime. Unknown regimes
    leave the pooled fallback in place and warn loudly rather than failing,
    so a walk-forward window label that is not one of the four canonical
    regimes still trains rather than crashing mid-run.
    """
    global SENT_SHIFT, SENT_SCALE, SENT_REGIME
    if regime not in SENT_ONLINE_BY_REGIME:
        print(f"[in2_adapter] WARNING: unknown regime {regime!r}; "
              f"keeping pooled SHIFT={SENT_SHIFT:.4f} SCALE={SENT_SCALE:.4f}")
        return
    mean, std = SENT_ONLINE_BY_REGIME[regime]
    SENT_SHIFT = mean
    SENT_SCALE = SENT_OFFLINE_STD / std
    SENT_REGIME = regime
    print(f"[in2_adapter] regime {regime}: "
          f"SENT_SHIFT={SENT_SHIFT:.4f} SENT_SCALE={SENT_SCALE:.4f} "
          f"SENT_OFFSET={SENT_OFFSET:.4f}")
SENT_INDEX = 0             # which context column holds daily sentiment
BASELINE_FILL = 0.0        # constant value for the no-sentiment variant
# ---------------------------------------------------------------------------


def get_model_registry_in2(fine_tuning_files_dir=None):
    """Same shape as get_model_registry(), but pointing at the In2 weights."""
    from data_utils import load_bounds_from_state_dict
    if fine_tuning_files_dir is None:
        from config import PROJECT_ROOT
        fine_tuning_files_dir = os.path.join(PROJECT_ROOT, "Fine_Tuning_Files")
    ft = fine_tuning_files_dir

    registry = {}
    for name in ("Heston", "Bates", "Bergomi", "rBergomi"):
        w = os.path.join(ft, f"CNN_{name}_In2_OutBase.pth")
        # offline CSVs in the new bundle dropped the "Sentiment_" prefix
        c_new = os.path.join(ft, f"{name}_IV_Surface_Data_Final.csv")
        c_old = os.path.join(ft, f"{name}_Sentiment_IV_Surface_Data_Final.csv")
        registry[name] = {
            "weights_path": w,
            "offline_csv": c_new if os.path.exists(c_new) else c_old,
        }
    for name in registry:
        if not os.path.exists(registry[name]["weights_path"]):
            raise FileNotFoundError(
                f"In2 weights missing: {registry[name]['weights_path']}\n"
                "Copy CNN_*_In2_OutBase.pth into Fine_Tuning_Files/ first."
            )
        registry[name]["param_bounds"] = load_bounds_from_state_dict(
            registry[name]["weights_path"]
        )
    return registry


def build_model_from_config_in2(model_config, device, strict=True):
    """
    Build ParameterCNN_In2 and load the In2 offline weights.

    Unlike the In1 builder (which silently left mismatched weights randomly
    initialised), this uses strict=True: an In2 checkpoint MUST load cleanly.
    A shape error here means the offline and online architectures disagree,
    which you want to fail loudly rather than train through.
    """
    model = ParameterCNN_In2(param_bounds=model_config["param_bounds"]).to(device)
    ck = torch.load(model_config["weights_path"], map_location=device)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd, strict=strict)
    return model


def make_in2_batch(scaled_iv, context=None, use_sentiment=True):
    """
    Build the (B, 2, 8, 11) network input.

    scaled_iv : (B,1,8,11) or (B,8,11) — ALREADY normalised by scale_input()
    context   : (B,2) [daily_sentiment, articles_scaled] from the loader,
                or None for the baseline variant.
    use_sentiment : False -> constant BASELINE_FILL channel, so the baseline
                and sentiment arms share an IDENTICAL architecture and differ
                only in channel content. This removes the architecture/
                batch-size confound the In1 setup had.
    """
    iv = scaled_iv if scaled_iv.dim() == 4 else scaled_iv.unsqueeze(1)
    if not use_sentiment or context is None:
        return build_in2_input(iv, None, use_sentiment=False,
                               fill_value=BASELINE_FILL)

    # standardise onto the OFFLINE distribution the filters were pretrained on:
    #   de-mean by the online mean, rescale, re-centre on the offline mean
    s = (context[:, SENT_INDEX] - SENT_SHIFT) * SENT_SCALE + SENT_OFFSET   # (B,)
    if SENT_MODE == "per_maturity":
        s = s.unsqueeze(1).expand(s.size(0), 8)      # (B,8)
    return build_in2_input(iv, s, use_sentiment=True)


def sanity_check(model_config, device="cpu"):
    """Load the In2 weights and push one fake batch through. Fail loudly."""
    m = build_model_from_config_in2(model_config, device, strict=True)
    m.eval()
    iv = torch.rand(3, 1, 8, 11, device=device)
    ctx = torch.randn(3, 2, device=device) * 0.1
    with torch.no_grad():
        for label, use in (("sentiment", True), ("baseline", False)):
            x = make_in2_batch(iv, ctx, use_sentiment=use)
            _, p = m(x)
            print(f"  {label:10} input {tuple(x.shape)} -> params {tuple(p.shape)}")
    print("  In2 sanity check PASSED")
    return True


if __name__ == "__main__":
    reg = get_model_registry_in2()
    for name, cfg in reg.items():
        print(f"[{name}] {os.path.basename(cfg['weights_path'])} "
              f"bounds={len(cfg['param_bounds'])} params")
        sanity_check(cfg)
