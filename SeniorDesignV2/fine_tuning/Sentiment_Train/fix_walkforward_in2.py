"""
fix_walkforward_in2.py   (run in fine_tuning/Sentiment_Train/)

walkforward.py's METHODOLOGY is already sound:
  * scaler is rebuilt per window from iv[tr] only  -> no normalisation leakage
  * build_model_from_config loads the OFFLINE checkpoint, not a fine-tuned one
    -> window 1 has never seen SPX data
  * tr = prior quarters, te = next quarter        -> no train/test overlap
  * warm-start carries only weights fitted on PRIOR quarters

What is broken is In2 compatibility (same three issues export_surfaces.py had),
plus the rBergomi offline-weight override:

1. build_model_from_config builds the 1-channel VolatilitySurfaceCNN. The
   checkpoints and the offline weights are 2-channel ParameterCNN_In2.
2. _extract() (and anything calling the model) passes (scaled, ctx) or (scaled);
   In2 needs the stacked (B,2,8,11) input from make_in2_batch.
3. VARIANT is inferred from the package folder name, which is now always
   "Sentiment_Train" -> both arms would write to eval_wf_surfaces/sentiment/.
4. OFFLINE_WEIGHT is the flat constant, so rBergomi would not get the 0.3
   override that stopped it collapsing.

Usage:  python3 fix_walkforward_in2.py
"""
import os, re, sys, shutil

p = "walkforward.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
if "make_in2_batch" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak")
done = []

# ---- 3. VARIANT from the env flag ----------------------------------------
old = '''_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
VARIANT = "nonsent" if _PKG.startswith("Non") else "sentiment"'''
new = '''_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
# PATCHED: both arms run from Sentiment_Train/ now, so the folder name no longer
# identifies the arm. Use the same flags training and export used.
_USE_SENT = os.environ.get("SDP_USE_SENTIMENT", "1") == "1"
_USE_IN2  = os.environ.get("SDP_USE_IN2", "0") == "1"
VARIANT = "sentiment" if _USE_SENT else "nonsent"'''
if old in s: s = s.replace(old, new); done.append("VARIANT from SDP_USE_SENTIMENT")

# ---- 4. model-aware offline weight ---------------------------------------
s2 = s.replace("BATCH_SIZE, OFFLINE_WEIGHT)", "BATCH_SIZE, OFFLINE_WEIGHT,\n                    offline_weight_for)")
if s2 != s:
    s = s2; done.append("import offline_weight_for")
s2 = s.replace("off_mean, off_std, torch.tensor(iv[tr]), offline_weight=OFFLINE_WEIGHT)",
               "off_mean, off_std, torch.tensor(iv[tr]),\n            offline_weight=offline_weight_for(model_name))")
if s2 != s:
    s = s2; done.append("per-model offline weight in the window scaler")

# ---- 1. In2 model construction -------------------------------------------
old = "        model = build_model_from_config(cfg, device)         # loads offline checkpoint"
new = '''        # PATCHED: In2 checkpoints are 2-channel; build the matching model.
        if _USE_IN2:
            from in2_adapter import build_model_from_config_in2
            model = build_model_from_config_in2(cfg, device, strict=True)
        else:
            model = build_model_from_config(cfg, device)   # loads offline checkpoint'''
if old in s: s = s.replace(old, new); done.append("In2 model construction")

# ---- 2. In2 forward pass in _extract -------------------------------------
m = re.search(r"def _extract\(tuner, iv, px, ctx, mask\):.*?\n\n", s, flags=re.S)
if m:
    body = m.group(0)
    nb = body
    nb = nb.replace("""            if takes_ctx:
                _, params = tuner.model(scaled, ctx)
            else:
                _, params = tuner.model(scaled)""",
"""            # PATCHED: In2 needs the stacked 2-channel input, built exactly as
            # training built it (standardised sentiment vs constant fill).
            if _USE_IN2:
                from in2_adapter import make_in2_batch
                _, params = tuner.model(make_in2_batch(scaled, ctx, use_sentiment=_USE_SENT))
            elif takes_ctx:
                _, params = tuner.model(scaled, ctx)
            else:
                _, params = tuner.model(scaled)""")
    if nb != body:
        s = s.replace(body, nb); done.append("In2 forward pass in _extract")

open(p, "w").write(s)
print("walkforward.py patched:")
for d in done: print("   -", d)
expected = ["VARIANT from SDP_USE_SENTIMENT", "import offline_weight_for",
            "per-model offline weight in the window scaler",
            "In2 model construction", "In2 forward pass in _extract"]
miss = [e for e in expected if e not in done]
if miss:
    print("NOT APPLIED (inspect manually):")
    for m_ in miss: print("   -", m_)
import py_compile; py_compile.compile(p, doraise=True); print("compiles OK")
print()
print("NOTE: the trainer's own call sites are already In2-aware (SDP_USE_IN2),")
print("so tuner.fine_tune() inside each window needs no change.")
print()
print("SMOKE TEST (one cell, 5 epochs, ~2 min):")
print("  cd ~/SeniorDesignProject")
print("  SDP_USE_IN2=1 SDP_USE_SENTIMENT=1 SDP_R=0.0230 SDP_Q=0.015 \\")
print("    python3 -u fine_tuning/Sentiment_Train/walkforward.py \\")
print("    --models Heston --regimes 2013-2015 --epochs 5")
