"""
fix_evaluate_in2.py   (run in fine_tuning/Sentiment_Train/)

evaluate.py produces the paper's figures (training history, parity, surface
heatmaps, smile + term structure, aggregate error heatmaps) but predates In2.

FIXES
1. line ~424 / ~462: build_model_from_config builds the 1-CHANNEL
   VolatilitySurfaceCNN. The checkpoints are 2-channel ParameterCNN_In2.
2. line ~166: tuner_obj.model(scaled_input, context) - In2 takes ONE stacked
   (B,2,8,11) argument built by make_in2_batch, with the arm's channel content.
3. checkpoint choice: candidates[-1] on a LEXICAL sort puts epoch_99 after
   epoch_700, and it ignores best_model_final_epoch_*.pth entirely. Pick by
   NUMERIC epoch, preferring the final checkpoint.
4. PLOT_OUTPUT_ROOT already follows config's _ARM_DIR, so sentiment and
   baseline figures land in separate trees - nothing to change, but the run
   MUST set SDP_USE_SENTIMENT or both arms overwrite each other.

Usage:
    python3 fix_evaluate_in2.py
    # then, from the repo root:
    SDP_USE_IN2=1 SDP_USE_SENTIMENT=1 SDP_R=0.0230 SDP_Q=0.015 \
      python3 fine_tuning/Sentiment_Train/evaluate.py
"""
import os, re, sys, shutil

p = "evaluate.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
if "build_model_from_config_in2" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p, p + ".bak")
done, missing = [], []

# ---- flags near the top -------------------------------------------------
anchor = re.search(r"^device\s*=.*$", s, flags=re.M)
if anchor:
    s = s[:anchor.end()] + (
        "\n\n# PATCHED: In2 wiring. Both arms run from this package now, so the arm\n"
        "# must come from the env flag, exactly as training and export did.\n"
        "_USE_IN2  = os.environ.get('SDP_USE_IN2', '0') == '1'\n"
        "_USE_SENT = os.environ.get('SDP_USE_SENTIMENT', '1') == '1'\n"
    ) + s[anchor.end():]
    done.append("In2/arm flags")
else:
    missing.append("In2/arm flags (no 'device =' line found)")

# ---- 1. model construction (indentation-preserving) ---------------------
# plot_model sits at function level (4 spaces); zero_model is nested inside
# `if RUN_ZERO_SHOT_COMPARISON:` (8 spaces). Assuming one indent breaks the file,
# so match whatever leading whitespace each site actually has.
BUILD_RE = re.compile(
    r"^(?P<ind>[ \t]+)(?P<var>\w+) = build_model_from_config\(selected_config, device\)[ \t]*$",
    flags=re.M)

def _build_repl(m):
    ind, var = m.group("ind"), m.group("var")
    return (f"{ind}if _USE_IN2:\n"
            f"{ind}    from in2_adapter import build_model_from_config_in2\n"
            f"{ind}    {var} = build_model_from_config_in2(selected_config, device, strict=False)\n"
            f"{ind}else:\n"
            f"{ind}    {var} = build_model_from_config(selected_config, device)")

s, n = BUILD_RE.subn(_build_repl, s)
if n: done.append(f"In2 model construction ({n} site(s), indentation preserved)")
else: missing.append("In2 model construction")

# ---- 2. forward pass ----------------------------------------------------
FWD_RE = re.compile(
    r"^(?P<ind>[ \t]+)_, predicted_params = tuner_obj\.model\(scaled_input, context\)[ \t]*$",
    flags=re.M)

def _fwd_repl(m):
    ind = m.group("ind")
    return (f"{ind}# PATCHED: In2 takes the stacked 2-channel input, built the same\n"
            f"{ind}# way training built it (standardised sentiment vs constant fill).\n"
            f"{ind}if _USE_IN2:\n"
            f"{ind}    from in2_adapter import make_in2_batch\n"
            f"{ind}    _x = make_in2_batch(scaled_input, context, use_sentiment=_USE_SENT)\n"
            f"{ind}    _, predicted_params = tuner_obj.model(_x)\n"
            f"{ind}else:\n"
            f"{ind}    _, predicted_params = tuner_obj.model(scaled_input, context)")

s, nf = FWD_RE.subn(_fwd_repl, s)
if nf: done.append(f"In2 forward pass ({nf} site(s))")
else: missing.append("In2 forward pass")

# ---- 3. numeric checkpoint selection ------------------------------------
old = '''    candidates = sorted(glob.glob(os.path.join(model_dir, "best_model_epoch_*.pth")))
    if not candidates:
        # Fall back to plain best_model.pth for backwards compatibility
        fallback = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(fallback):
            candidates = [fallback]
    if not candidates:
        print(f"  [SKIP] No checkpoint found in: {model_dir}")
        return None
    checkpoint_path = candidates[-1]  # highest epoch number'''
new = '''    # PATCHED: lexical sort put epoch_99 AFTER epoch_700, and the *_final_*
    # checkpoints were ignored entirely. Choose by numeric epoch, final first.
    def _ep(f):
        m = re.search(r"epoch_(\\d+)\\.pth$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    finals = glob.glob(os.path.join(model_dir, "best_model_final_epoch_*.pth"))
    periodic = glob.glob(os.path.join(model_dir, "best_model_epoch_*.pth"))
    if finals:
        checkpoint_path = max(finals, key=_ep)
    elif periodic:
        checkpoint_path = max(periodic, key=_ep)
    elif os.path.exists(os.path.join(model_dir, "best_model.pth")):
        checkpoint_path = os.path.join(model_dir, "best_model.pth")
    else:
        print(f"  [SKIP] No checkpoint found in: {model_dir}")
        return None'''
if old in s:
    s = s.replace(old, new); done.append("numeric checkpoint selection")
    if not re.search(r"^import re\b", s, flags=re.M):
        s = "import re\n" + s
else:
    missing.append("numeric checkpoint selection")

open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print("evaluate.py patched:")
for d in done: print("   -", d)
if missing:
    print("NOT APPLIED (inspect manually):")
    for m in missing: print("   -", m)
print("compiles OK")
print()
print("RUN BOTH ARMS (figures go to separate trees via config._ARM_DIR):")
print("  cd ~/SeniorDesignProject")
print("  for arm in 1 0; do")
print("    SDP_USE_IN2=1 SDP_USE_SENTIMENT=$arm SDP_R=0.0230 SDP_Q=0.015 \\")
print("      python3 fine_tuning/Sentiment_Train/evaluate.py")
print("  done")
print()
print("CAUTION - two of the five figures span the full moneyness range:")
print("  plot_smile_and_term_structure  and  plot_aggregate_error_heatmaps")
print("The evaluation grid inverts IV reliably only near ATM (round-trip error")
print("0.004 at lm=0, but +0.215 at lm=+0.5 with 80% success). Treat those two")
print("figures as unreliable in the wings, or restrict them to |lm| <= 0.2.")
