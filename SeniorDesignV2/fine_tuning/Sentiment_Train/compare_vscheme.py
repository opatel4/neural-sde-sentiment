"""
compare_vscheme.py -- run in ~/SeniorDesignV2/fine_tuning/Sentiment_Train

A/B test: clamped Euler vs full truncation for the Heston/Bates variance, with
the martingale correction active in both.

The martingale correction already forces E[S_T] = 1, so the MEAN is right either
way. What it cannot fix is the SHAPE of the terminal distribution. This measures
whether the shape difference is large enough to matter for option prices - i.e.
whether it exceeds Monte-Carlo noise.

Method: price the same parameter draws and the same strike/maturity grid under
both schemes with a common seed, then compare in IMPLIED VOL points (the unit
the models are actually judged in) rather than price.

    diff > ~0.002 vol pts   -> the scheme choice matters, use full truncation
    diff ~ MC noise         -> clamping is adequate once the mean is corrected

Usage:
    SDP_NORM=forward python3 compare_vscheme.py
"""
import os, sys, subprocess, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MATS = [0.1, 0.6, 1.2, 2.0]
LMS  = [-0.4, -0.2, 0.0, 0.2, 0.4]

WORKER = r'''
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pricers as P
from utils import GenerateGaussLaguerre
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xn, wn = GenerateGaussLaguerre(32, device=dev)
MATS = %s; LMS = %s
g = torch.Generator(device=dev); g.manual_seed(12345)
N = 16
def d(span, lo): return torch.rand(N,1,generator=g,device=dev)*span + lo
kap=d(2.9,0.1); th=d(0.14,0.01); sig=d(0.7,0.1); v0=d(0.14,0.01); rho=d(0.8,-0.9)
lj=d(2.0,0.0);  mj=d(0.25,-0.2); sj=d(0.19,0.01)
one=torch.tensor(1.0,device=dev); z=torch.tensor(0.0,device=dev)
out={"Heston":[], "Bates":[]}
for T in MATS:
    K = torch.tensor([float(np.exp(l)) for l in LMS], device=dev)
    torch.manual_seed(999)
    c = P.Hestonprice_Batched(kap,th,sig,v0,rho,T,K,one,z,z,xn,wn,N_paths=300000,N_steps=100)
    out["Heston"].append(c.detach().cpu().numpy().tolist())
    torch.manual_seed(999)
    c = P.Batesprice_Batched(kap,th,sig,v0,rho,lj,mj,sj,T,K,one,z,z,xn,wn,N_paths=300000,N_steps=100)
    out["Bates"].append(c.detach().cpu().numpy().tolist())
print("@@JSON@@" + json.dumps(out))
''' % (MATS, LMS)

wpath = os.path.join(HERE, "_vscheme_worker.py")
open(wpath, "w").write(WORKER)

def run(scheme):
    env = dict(os.environ); env["SDP_VSCHEME"] = scheme
    env.setdefault("SDP_NORM", "forward")
    r = subprocess.run([sys.executable, wpath], capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[8:])
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    raise SystemExit(f"worker failed for scheme={scheme}")

print("running clamped Euler ...");   A = run("clamp")
print("running full truncation ..."); B = run("trunc")

from scipy.stats import norm as N
def iv(price, k, T, lo=1e-4, hi=3.0):
    if not np.isfinite(price) or price <= max(1-k,0)+1e-9 or price >= 1-1e-9: return np.nan
    for _ in range(80):
        m=0.5*(lo+hi); srt=m*np.sqrt(T)
        d1=(-np.log(k)+0.5*m*m*T)/srt
        if N.cdf(d1)-k*N.cdf(d1-srt) > price: hi=m
        else: lo=m
    return 0.5*(lo+hi)

print("\n" + "="*76)
print("IMPLIED-VOL DIFFERENCE:  full truncation  -  clamped Euler   (vol points)")
print("="*76)
for model in ("Heston","Bates"):
    print(f"\n{model}")
    print(f"  {'T':>6}" + "".join(f"{l:>10.1f}" for l in LMS))
    big = 0.0
    for ti,T in enumerate(MATS):
        a = np.array(A[model][ti]); b = np.array(B[model][ti])   # (N, nK)
        row=[]
        for j,l in enumerate(LMS):
            k=float(np.exp(l))
            va=np.array([iv(x,k,T) for x in a[:,j]])
            vb=np.array([iv(x,k,T) for x in b[:,j]])
            d=np.nanmedian(vb-va); row.append(d)
            big=max(big, abs(d))
        print(f"  {T:>6.1f}" + "".join(f"{x:>+10.4f}" for x in row))
    print(f"  largest |difference| : {big:.4f} vol points")

print("\n" + "="*76)
print("READING")
print("  < 0.002 vol pts : clamping is adequate once the martingale correction")
print("                    is applied - keep the current scheme.")
print("  > 0.005 vol pts : the scheme materially changes the surface - switch to")
print("                    full truncation before regenerating (set SDP_VSCHEME=trunc).")
print("  Both runs share seeds, so the difference is scheme, not MC noise.")
os.remove(wpath)
