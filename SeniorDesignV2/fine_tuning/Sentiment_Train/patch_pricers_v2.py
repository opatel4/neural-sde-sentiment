"""
patch_pricers_v2.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

STEP 6. Puts the Monte-Carlo pricers in FORWARD measure so they match the V2
normalisation.

WHY
---
Prices are now normalised by F, so a normalised price is C/F and the normalised
forward is 1. A simulator that starts at S and drifts at (r - q) is producing
S_T/S, not F_T/F. Under the forward measure the underlying is a martingale:

    d log X = -0.5 V dt + sqrt(V) dW      X_0 = 1

so the drift term (r - q) disappears and log_S starts at 0. Discounting is
already applied separately at the payoff (`* exp(-r*T)`), so it is untouched.

Bates keeps its jump compensator: that term exists to keep the JUMP part a
martingale and is independent of the (r - q) drift. Only (r - q) is removed.

All four models have the same shape:
    log_S = log_S + (r_val - q_val - 0.5*V_t [- jump_comp]) * dt + vol * dW
so this is four one-line edits plus four initialisations.

Gated on SDP_NORM=forward, so the old behaviour is unchanged by default.

Usage:
    python3 patch_pricers_v2.py --dry-run
    python3 patch_pricers_v2.py
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

p = "pricers.py"
if not os.path.exists(p):
    sys.exit("run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train")
s = open(p).read()
if "_V2_FWD" in s:
    print("already patched"); sys.exit(0)

# ---- module flag ---------------------------------------------------------
m = None
for mm in re.finditer(r"^(import|from)\s+.*$", s, flags=re.M):
    m = mm
flag = ('\n\n# V2: under forward normalisation the simulated underlying IS the\n'
        '# normalised forward, a martingale starting at 1 with no (r-q) drift.\n'
        '# Discounting is applied separately at the payoff and is unaffected.\n'
        'import os as _os\n'
        '_V2_FWD = _os.environ.get("SDP_NORM", "spot").lower() == "forward"\n')
s = s[:m.end()] + flag + s[m.end():]

# ---- drift lines ---------------------------------------------------------
DRIFT_RE = re.compile(
    r"^(?P<ind>[ \t]+)\+ \(r_val - q_val(?P<rest>[^)]*)\) \* dt[ \t]*$", flags=re.M)
def drift_repl(mo):
    ind, rest = mo.group("ind"), mo.group("rest")
    return (f"{ind}+ ((0.0 if _V2_FWD else (r_val - q_val)){rest}) * dt")
s, n_drift = DRIFT_RE.subn(drift_repl, s)

# ---- log_S initialisations ----------------------------------------------
INIT1_RE = re.compile(
    r"^(?P<ind>[ \t]+)log_S = torch\.full\(\((?P<shape>[^)]*)\), math\.log\(S_val\),(?P<rest>.*)$",
    flags=re.M)
def init1_repl(mo):
    ind, shape, rest = mo.group("ind"), mo.group("shape"), mo.group("rest")
    return (f"{ind}log_S = torch.full(({shape}), "
            f"(0.0 if _V2_FWD else math.log(S_val)),{rest}")
s, n_i1 = INIT1_RE.subn(init1_repl, s)

# multi-line variant: math.log(_to_float_scalar(S)) on its own line
INIT2_RE = re.compile(
    r"^(?P<ind>[ \t]+)math\.log\(_to_float_scalar\(S\)\),[ \t]*$", flags=re.M)
def init2_repl(mo):
    ind = mo.group("ind")
    return f"{ind}(0.0 if _V2_FWD else math.log(_to_float_scalar(S))),"
s, n_i2 = INIT2_RE.subn(init2_repl, s)

print("patch plan:")
print(f"   + drift lines neutralised under SDP_NORM=forward : {n_drift}")
print(f"   + log_S initialisations (single-line)            : {n_i1}")
print(f"   + log_S initialisations (multi-line)             : {n_i2}")
print(f"   = models covered: {n_i1 + n_i2} (expect 4)")

if n_drift != 4 or (n_i1 + n_i2) != 4:
    print("\n!! expected 4 of each - inspect pricers.py before writing")
    if not a.dry_run:
        sys.exit("refusing to write")

if a.dry_run:
    print("\n[dry-run] nothing written."); sys.exit(0)

shutil.copy(p, p + ".bak_v2")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"\nwrote {p} (backup {p}.bak_v2); compiles OK")
print("\nVALIDATE before regenerating anything:")
print("  SDP_NORM=forward python3 -c \"")
print("  import torch, pricers as P, utils")
print("  xn,wn = utils.GenerateGaussLaguerre(32, device=torch.device('cpu'))")
print("  t=lambda v: torch.tensor([[v]],dtype=torch.float32)")
print("  K=torch.tensor([0.8,1.0,1.25])")
print("  c=P.Hestonprice_Batched(t(2.0),t(0.04),t(0.4),t(0.04),t(-0.7),")
print("      1.0,K,torch.tensor(1.0),torch.tensor(0.0),torch.tensor(0.0),xn,wn,N_paths=200000)")
print("  print('E[X_T] should be ~1.0 ; prices:', c)\"")
print("\nThe martingale check: with r=q=0 and S=1 the simulated mean must be 1.")
