"""
patch_full_truncation.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

Switches the Heston/Bates variance scheme from CLAMPED Euler to FULL TRUNCATION,
behind SDP_VSCHEME=trunc so it can be A/B tested against the current scheme.

WHY, GIVEN THE MARTINGALE FIX IS ALREADY IN
-------------------------------------------
The martingale correction forces E[S_T] = 1 exactly. That fixes the MEAN. It does
not fix the SHAPE of the terminal distribution: clamping variance at 1e-8 still
distorts the left tail, because paths that would have gone negative are pinned at
a floor and then diffuse from there. Option prices away from the money can still
be biased even when the mean is right.

Current scheme (clamped / "absorption"):
    V_next = V + kappa(theta - V)dt + sigma*sqrt(max(V,0))*sqrt(dt)*Z
    V      = clamp(V_next, min=1e-8)          <- the state itself is floored

Full truncation (Lord, Koekkoek & van Dijk 2010):
    keep the state V possibly negative; use max(V,0) only where a variance is
    needed - in the drift's mean reversion, in the diffusion coefficient, and in
    the log-spot Ito term.

Lord et al. find full truncation has the smallest bias of the simple Euler
variants, and the gap is widest exactly where this project sits: only 58% (Heston)
and 62% (Bates) of the generator's sampled parameters satisfy Feller
(2*kappa*theta >= sigma^2), so the variance boundary is hit often.

Bergomi/rBergomi are untouched - their variance is xi*exp(nu*X - ...), strictly
positive by construction, so there is nothing to truncate.

Usage:
    python3 patch_full_truncation.py --dry-run
    python3 patch_full_truncation.py
    # then A/B:
    SDP_NORM=forward                      python3 compare_vscheme.py
    SDP_NORM=forward SDP_VSCHEME=trunc    python3 compare_vscheme.py
"""
import os, re, sys, shutil, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

p = "pricers.py"
if not os.path.exists(p):
    sys.exit("run from ~/SeniorDesignV2/fine_tuning/Sentiment_Train")
s = open(p).read()
if "_V2_TRUNC" in s:
    print("already patched"); sys.exit(0)

# flag next to the existing _V2_FWD flag
if "_V2_FWD" not in s:
    sys.exit("expected _V2_FWD (run patch_pricers_v2.py first)")
s = s.replace(
    '_V2_FWD = _os.environ.get("SDP_NORM", "spot").lower() == "forward"',
    '_V2_FWD = _os.environ.get("SDP_NORM", "spot").lower() == "forward"\n'
    '# Variance scheme: "clamp" (current) or "trunc" (full truncation).\n'
    '# Full truncation keeps V possibly negative and applies max(V,0) only where\n'
    '# a variance is needed - smallest bias among simple Euler variants when the\n'
    '# Feller condition is often violated, which it is here (58%/62% satisfied).\n'
    '_V2_TRUNC = _os.environ.get("SDP_VSCHEME", "clamp").lower() == "trunc"')

# --- the Heston/Bates variance update -------------------------------------
OLD_V = """        V_next = (
            V_t
            + kappa * (theta - V_t) * dt
            + sigma * torch.sqrt(torch.clamp(V_t, min=0.0)) * sqrt_dt * ZV
        )
        V_t = torch.clamp(V_next, min=1e-8)"""
NEW_V = """        _Vpos = torch.clamp(V_t, min=0.0)
        V_next = (
            V_t
            + kappa * (theta - _Vpos) * dt
            + sigma * torch.sqrt(_Vpos) * sqrt_dt * ZV
        )
        if _V2_TRUNC:
            # full truncation: the STATE may go negative; only its positive part
            # is ever used as a variance.
            V_t = V_next
        else:
            V_t = torch.clamp(V_next, min=1e-8)"""
n_v = s.count(OLD_V)
s = s.replace(OLD_V, NEW_V)

# --- the log-spot update must use max(V,0) --------------------------------
OLD_S1 = """            + ((0.0 if _V2_FWD else (r_val - q_val)) - 0.5 * V_t) * dt
            + torch.sqrt(V_t) * sqrt_dt * ZS"""
NEW_S1 = """            + ((0.0 if _V2_FWD else (r_val - q_val)) - 0.5 * torch.clamp(V_t, min=0.0)) * dt
            + torch.sqrt(torch.clamp(V_t, min=0.0)) * sqrt_dt * ZS"""
n_s1 = s.count(OLD_S1)
s = s.replace(OLD_S1, NEW_S1)

OLD_S2 = """            + ((0.0 if _V2_FWD else (r_val - q_val)) - 0.5 * V_t - jump_comp) * dt
            + torch.sqrt(V_t) * sqrt_dt * ZS"""
NEW_S2 = """            + ((0.0 if _V2_FWD else (r_val - q_val)) - 0.5 * torch.clamp(V_t, min=0.0) - jump_comp) * dt
            + torch.sqrt(torch.clamp(V_t, min=0.0)) * sqrt_dt * ZS"""
n_s2 = s.count(OLD_S2)
s = s.replace(OLD_S2, NEW_S2)

print("patch plan:")
print(f"   + SDP_VSCHEME flag")
print(f"   + variance updates rewritten   : {n_v} (expect 2: Heston, Bates)")
print(f"   + log-spot uses max(V,0)       : {n_s1} Heston-style + {n_s2} Bates-style")
if n_v != 2 or n_s1 < 1 or n_s2 < 1:
    print("\n!! unexpected counts - inspect pricers.py")
    if not a.dry_run:
        sys.exit("refusing to write")
if a.dry_run:
    print("\n[dry-run] nothing written."); sys.exit(0)

shutil.copy(p, p + ".bak_trunc")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print(f"\nwrote {p} (backup {p}.bak_trunc); compiles OK")
print("\nNOTE: with SDP_VSCHEME unset the behaviour is IDENTICAL to before -")
print("the clamp path is preserved, so this is a safe A/B.")
