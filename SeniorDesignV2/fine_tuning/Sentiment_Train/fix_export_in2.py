"""
fix_export_in2.py   (run in fine_tuning/Sentiment_Train/)

export_surfaces.py was written for the OLD two-package In1 setup. Four problems
under the new single-package In2 design:

1. VARIANT is inferred from the package DIRECTORY NAME:
       VARIANT = "nonsent" if _PKG.startswith("Non") else "sentiment"
   Both arms now run from Sentiment_Train/, so BOTH would be labelled
   "sentiment" and the second would overwrite the first.
   -> derive VARIANT from SDP_USE_SENTIMENT, exactly as training did.

2. Model is built with build_model_from_config (1-channel VolatilitySurfaceCNN).
   The checkpoints are 2-channel ParameterCNN_In2. load_state_dict would raise,
   or (worse, with the permissive loader) silently leave layers random.
   -> build_model_from_config_in2(..., strict=True) when SDP_USE_IN2=1.

3. _find_checkpoint prefers best_model_epoch_*.pth over
   best_model_final_epoch_*.pth and takes cands[-1] by LEXICAL sort, so
   "epoch_99" sorts after "epoch_700". Wrong checkpoint, silently.
   -> pick by NUMERIC epoch, preferring the final checkpoint.

4. CKPT_ROOTS looks in <repo>/full_project_output/... first. Your run wrote to
   fine_tuning/fine_tuning_output/... . The old path may hold STALE checkpoints
   from the previous study, which would be picked up in preference.
   -> search MODEL_OUTPUT_ROOT first.

Usage:  python3 fix_export_in2.py           (writes .bak first)
"""
import os, re, sys, shutil

p = "export_surfaces.py"
if not os.path.exists(p):
    sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
if "build_model_from_config_in2" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak")
changed = []

# --- 1. VARIANT from the env flag, not the folder name ---------------------
old_variant = '''_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
VARIANT = "nonsent" if _PKG.startswith("Non") else "sentiment"'''
new_variant = '''_PKG = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
# PATCHED: both arms now run from Sentiment_Train/, so the package name no
# longer identifies the arm. Use the same flag training used.
_USE_SENT = os.environ.get("SDP_USE_SENTIMENT", "1") == "1"
_USE_IN2  = os.environ.get("SDP_USE_IN2", "0") == "1"
VARIANT = "sentiment" if _USE_SENT else "nonsent"'''
if old_variant in s:
    s = s.replace(old_variant, new_variant); changed.append("VARIANT from SDP_USE_SENTIMENT")

# --- 4. prefer the path this run actually wrote to --------------------------
old_roots = '''CKPT_ROOTS = [
    os.path.join(REPO_ROOT, "full_project_output", _OUT_NAME, "models"),
    MODEL_OUTPUT_ROOT,
]'''
new_roots = '''# PATCHED: search the CURRENT run's output first. full_project_output may hold
# stale checkpoints from the previous study.
CKPT_ROOTS = [
    MODEL_OUTPUT_ROOT,
    os.path.join(REPO_ROOT, "full_project_output", _OUT_NAME, "models"),
]'''
if old_roots in s:
    s = s.replace(old_roots, new_roots); changed.append("checkpoint search order")

# --- 3. numeric epoch selection --------------------------------------------
old_find = re.search(r"def _find_checkpoint.*?\n    return None\n", s, flags=re.S)
if old_find:
    new_find = '''def _find_checkpoint(model_name, date_range):
    """PATCHED: pick by NUMERIC epoch (lexical sort put epoch_99 after epoch_700)
    and prefer the *final* checkpoint, which is what training last wrote."""
    def _ep(f):
        m = re.search(r"epoch_(\\d+)\\.pth$", os.path.basename(f))
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
'''
    s = s.replace(old_find.group(0), new_find); changed.append("numeric epoch selection")
    if not re.search(r"^import re\b", s, flags=re.M):
        s = "import re\n" + s

# --- 2. build the In2 model -------------------------------------------------
old_build = '''            model = build_model_from_config(cfg, device)
            model.load_state_dict(ckpt["model_state_dict"])'''
new_build = '''            # PATCHED: the checkpoints are 2-channel ParameterCNN_In2.
            if _USE_IN2:
                from in2_adapter import build_model_from_config_in2
                model = build_model_from_config_in2(cfg, device, strict=False)
            else:
                model = build_model_from_config(cfg, device)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)'''
if old_build in s:
    s = s.replace(old_build, new_build); changed.append("In2 model construction")

open(p, "w").write(s)
print("export_surfaces.py patched:")
for c in changed: print("   -", c)
missing = [x for x in ["VARIANT from SDP_USE_SENTIMENT","checkpoint search order",
                       "numeric epoch selection","In2 model construction"] if x not in changed]
if missing:
    print("NOT APPLIED (inspect manually):")
    for m in missing: print("   -", m)
import py_compile; py_compile.compile(p, doraise=True); print("compiles OK")
print()
print("STILL TO VERIFY: _extract() must feed the model a 2-channel input")
print("  grep -n 'def _extract' -A 25 export_surfaces.py")
