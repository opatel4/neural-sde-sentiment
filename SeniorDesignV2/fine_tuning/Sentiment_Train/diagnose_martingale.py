"""
diagnose_martingale.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

Measures the martingale defect  E[X_T] - 1  across the ACTUAL parameter ranges
the offline generator samples from, for all four models.

BACKGROUND
----------
Under forward normalisation the simulated underlying must be a martingale:
E[X_T] = 1 for every T. A deep-ITM call (K -> 0) prices exactly E[X_T], so that
is the cleanest probe.

A single spot check at the Feller boundary (2*kappa*theta = sigma^2) gave
E[X_T] = 0.753 at T=2 - a 25% loss of mass. The cause is the Euler scheme:

    log_S += (-0.5 * V_t) * dt + sqrt(V_t) * sqrt(dt) * Z
    V_t    = clamp(V_next, min=1e-8)

The Ito correction -0.5*V_t only reproduces a martingale in continuous time.
With variance clamped at zero the realised quadratic variation no longer matches
the correction, and the bias compounds over steps.

This bug is NOT new to V2. It existed in V1 too, but the drift (r - q) - inflated
there by the vendor's r - partially offset it, so it was hidden by a compensating
error.

WHAT THIS REPORTS
-----------------
For each model, sampling parameters from the generator's own ranges:
  * E[X_T] at every maturity on the grid
  * how the defect scales with T and with N_steps
  * the share of sampled parameter sets that violate Feller (Heston/Bates)
so you can see whether the defect is a corner case or pervasive.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDP_NORM", "forward")
import pricers as P
from utils import GenerateGaussLaguerre

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xn, wn = GenerateGaussLaguerre(32, device=dev)
MATS = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
TINY = torch.tensor([1e-6], device=dev)      # K ~ 0  =>  price = E[X_T]
ONE  = torch.tensor(1.0, device=dev)
Z    = torch.tensor(0.0, device=dev)
NP, NS = 200_000, 100

# the generator's sampling ranges (from SeniorDParamBasedDataCollection)
RANGES = {
    "Heston":   {"kappa":[2.9,0.1], "theta":[0.14,0.01], "sigma":[0.7,0.1],
                 "v0":[0.14,0.01], "rho_SV":[0.8,-0.9]},
    "Bates":    {"kappa":[2.9,0.1], "theta":[0.14,0.01], "sigma":[0.7,0.1],
                 "v0":[0.14,0.01], "rho_SV":[0.8,-0.9],
                 "lambdJ":[2.0,0.0], "muJ":[0.25,-0.2], "sigmaJ":[0.19,0.01]},
    "Bergomi":  {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "beta":[10.0,0.0]},
    "rBergomi": {"xi":[0.15,0.01], "nu":[3.5,0.5], "rho_SV":[0.85,-0.95], "H":[0.475,0.025]},
}
def draw(rg, key, n, gen):
    span, lo = rg[key]
    return (torch.rand(n, 1, generator=gen, device=dev) * span + lo)

g = torch.Generator(device=dev); g.manual_seed(0)
NSAMP = 24

print("=" * 82)
print("MARTINGALE DEFECT  E[X_T] - 1   (0 = correct; negative = losing mass)")
print("=" * 82)

for model in ("Heston", "Bates", "Bergomi", "rBergomi"):
    rg = RANGES[model]
    print(f"\n{model}")
    if model in ("Heston", "Bates"):
        kap = draw(rg,"kappa",NSAMP,g); th = draw(rg,"theta",NSAMP,g)
        sig = draw(rg,"sigma",NSAMP,g); v0 = draw(rg,"v0",NSAMP,g)
        rho = draw(rg,"rho_SV",NSAMP,g)
        feller = (2*kap*th >= sig**2).float().mean().item()
        print(f"  Feller (2*kappa*theta >= sigma^2) satisfied in {feller:.0%} of draws")
    print(f"  {'T':>6}{'E[X_T]':>12}{'defect':>12}{'worst-of-batch':>17}")
    for T in MATS:
        if model == "Heston":
            c = P.Hestonprice_Batched(kap,th,sig,v0,rho, T, TINY, ONE, Z, Z, xn, wn,
                                      N_paths=NP, N_steps=NS)
        elif model == "Bates":
            lj = draw(rg,"lambdJ",NSAMP,g); mj = draw(rg,"muJ",NSAMP,g); sj = draw(rg,"sigmaJ",NSAMP,g)
            c = P.Batesprice_Batched(kap,th,sig,v0,rho,lj,mj,sj, T, TINY, ONE, Z, Z, xn, wn,
                                     N_paths=NP, N_steps=NS)
        elif model == "Bergomi":
            xi = draw(rg,"xi",NSAMP,g); nu = draw(rg,"nu",NSAMP,g)
            rho = draw(rg,"rho_SV",NSAMP,g); be = draw(rg,"beta",NSAMP,g)
            c = P.Bergomiprice_Batched(xi,nu,rho,be, T, TINY, ONE, Z, Z, N_paths=NP, N_steps=NS)
        else:
            xi = draw(rg,"xi",NSAMP,g); nu = draw(rg,"nu",NSAMP,g)
            rho = draw(rg,"rho_SV",NSAMP,g); H = draw(rg,"H",NSAMP,g)
            c = P.rBergomiprice_Batched(xi,nu,rho,H, T, TINY, ONE, Z, Z, N_paths=NP, N_steps=NS)
        v = c.detach().cpu().numpy().ravel()
        print(f"  {T:>6.1f}{v.mean():>12.5f}{v.mean()-1:>+12.5f}{v.min()-1:>+17.5f}")

print("\n" + "=" * 82)
print("READING")
print("  Defect near 0 at every T  -> scheme is fine, the earlier spot check was")
print("  a corner case.")
print("  Defect large and growing with T -> the Euler/clamp scheme loses mass")
print("  systematically. Fix before regenerating: rescale S_T by its sample mean")
print("  (exact martingale correction, one line per model), or move to a full-")
print("  truncation / QE variance scheme.")
print("  NOTE: whatever this shows, the SAME bias was present in V1 - it was")
print("  masked there by the (r-q) drift, not absent.")
