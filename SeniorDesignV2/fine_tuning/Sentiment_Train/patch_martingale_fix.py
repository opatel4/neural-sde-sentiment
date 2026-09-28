"""
patch_martingale_fix.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

Applies an empirical martingale correction to the Monte-Carlo pricers.

THE PROBLEM (measured, not assumed)
-----------------------------------
Under forward normalisation the simulated underlying must satisfy E[X_T] = 1.
Probing with a near-zero strike (whose call price IS E[X_T]) over 24 parameter
draws from the generator's own ranges:

    model      T=0.1     T=1.0      T=2.0     worst draw @T=2
    Heston    -1.2%     -10%       -19.1%        -39%
    Bates     -0.9%      -8%       -15.2%        -39%
    Bergomi   +0.004%   +0.005%    +0.004%       -0.22%
    rBergomi  +0.001%   -0.006%    -0.005%       -0.20%

Heston and Bates lose mass; Bergomi and rBergomi do not. The mechanism is the
variance scheme: Heston/Bates use Euler with

    V_t = clamp(V_next, min=1e-8)

and only 58% / 62% of sampled parameters satisfy Feller (2*kappa*theta >=
sigma^2), so variance hits the clamp often. The Ito correction -0.5*V_t then no
longer matches the realised quadratic variation and the bias compounds with T.
Bergomi/rBergomi build variance as xi*exp(nu*X - ...), lognormal and strictly
positive, so there is nothing to clamp and no bias.

NOTE: this bias was present in V1 as well. There the (r - q) drift - inflated by
the vendor's r ~ 2.5% - pushed prices back up and partially masked it.

THE FIX
-------
Empirical martingale correction: rescale the terminal values so the sample mean
is exactly 1 (or exactly S_val under spot normalisation) before computing
payoffs:

    S_T <- S_T * (target / mean(S_T))

This is standard practice for Heston Euler schemes. It is exact by construction,
costs one reduction per pricing call, and leaves the shape of the distribution
untouched - it removes only the discretisation drift.

Applied to ALL FOUR models for consistency; for Bergomi/rBergomi the correction
is ~1e-4 and therefore harmless.

Usage:
    python3 patch_martingale_fix.py --dry-run
    python3 patch_martingale_fix.py
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

p = "pricers.py"
if not os.path.exists(p):
    sys.exit("run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train")
s = open(p).read()
if "_martingale_correct" in s:
    print("already patched"); sys.exit(0)

HELPER = '''

def _martingale_correct(S_T, target):
    """
    Empirical martingale correction.

    Euler + variance clamping makes E[S_T] drift away from its theoretical value
    (measured: -19% for Heston at T=2). Rescaling by the realised sample mean
    restores the martingale property exactly without altering the shape of the
    terminal distribution.

    S_T    : (num_samples, N_paths) terminal underlying
    target : scalar, the value E[S_T] must take (1.0 in forward units, S_val in
             spot units)
    """
    m = S_T.mean(dim=1, keepdim=True)
    return S_T * (target / torch.clamp(m, min=1e-12))
'''

# insert the helper before the first pricer definition
mdef = re.search(r"^def \w*MC_Batched", s, flags=re.M)
if not mdef:
    sys.exit("could not find a *MC_Batched definition to anchor the helper")
s = s[:mdef.start()] + HELPER.strip("\n") + "\n\n\n" + s[mdef.start():]

# apply it wherever S_T is formed
RE = re.compile(r"^(?P<ind>[ \t]*)S_T = torch\.exp\(log_S\)[ \t]*$", flags=re.M)
def repl(mo):
    ind = mo.group("ind")
    return (f"{ind}S_T = torch.exp(log_S)\n"
            f"{ind}# martingale correction: Euler + variance clamping loses mass\n"
            f"{ind}# (measured -19% for Heston at T=2). Rescale so E[S_T] is exact.\n"
            f"{ind}S_T = _martingale_correct(S_T, 1.0 if _V2_FWD else S_val)")
s, n = RE.subn(repl, s)

print(f"patch plan:\n   + _martingale_correct helper\n   + applied at {n} S_T sites (expect 4)")
if n != 4:
    print("\n!! expected 4 - inspect pricers.py")
    if not a.dry_run:
        sys.exit("refusing to write")
if a.dry_run:
    print("\n[dry-run] nothing written."); sys.exit(0)

# Bergomi/rBergomi initialise log_S from _to_float_scalar(S), not S_val;
# make sure a spot-mode target exists in those scopes.
if "S_val" not in s.split("def Bergomi_1F_MC_Batched")[1][:1200]:
    s = s.replace("def Bergomi_1F_MC_Batched(xi, nu, rho, beta, T, K_strikes, S, r, q, N_paths=3000, N_steps=100):",
                  "def Bergomi_1F_MC_Batched(xi, nu, rho, beta, T, K_strikes, S, r, q, N_paths=3000, N_steps=100):\n"
                  "    S_val = _to_float_scalar(S)")
if "S_val" not in s.split("def rBergomi_MC_Batched")[1][:1200]:
    s = s.replace("def rBergomi_MC_Batched(xi, nu, rho, H, T, K_strikes, S, r, q, N_paths=4000, N_steps=150):",
                  "def rBergomi_MC_Batched(xi, nu, rho, H, T, K_strikes, S, r, q, N_paths=4000, N_steps=150):\n"
                  "    S_val = _to_float_scalar(S)")

shutil.copy(p, p + ".bak_mart")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"\nwrote {p} (backup {p}.bak_mart); compiles OK")
print("\nRE-RUN THE DIAGNOSTIC - every defect should now be ~1e-5:")
print("  SDP_NORM=forward python3 diagnose_martingale.py")
print("\nThen re-check the intrinsic bound:")
print("  Heston/Bates at K=0.7 must price >= 0.30, not 0.18.")
