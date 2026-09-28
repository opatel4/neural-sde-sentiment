"""
eval_preflight.py -- run in fine_tuning/Sentiment_Train/ BEFORE any analysis.
Audits the exported surfaces and every evaluation script's inputs/outputs.
"""
import os, sys, glob, re, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # ~/SeniorDesignProject
SURF = f"{REPO}/full_project_output/eval_surfaces"
MODELS  = ["Heston","Bates","Bergomi","rBergomi"]
REGIMES = ["2010-2012","2013-2015","2016-2019","2020-2022"]
fails, warns = [], []
def chk(c,l,w=False):
    print(f"  {'OK  ' if c else ('WARN' if w else 'FAIL')} {l}")
    if not c: (warns if w else fails).append(l)

print("="*74); print("1. EXPORTED SURFACES - completeness"); print("="*74)
for arm in ("sentiment","nonsent"):
    n=len(glob.glob(f"{SURF}/{arm}/*.npz"))
    chk(n==16, f"{arm:9} {n}/16 npz files")
missing=[(a,m,d) for a in ("sentiment","nonsent") for m in MODELS for d in REGIMES
         if not os.path.exists(f"{SURF}/{a}/{m}_{d}.npz")]
chk(not missing, f"all 32 (arm,model,regime) files present ({len(missing)} missing)")
for x in missing: print("       missing:", x)

print("\n"+"="*74); print("2. SURFACE CONTENT - shapes, finiteness, alignment"); print("="*74)
KEYS=["market_iv","market_px","model_px","params","context","dates"]
shape_prob=[]; nan_prob=[]; align_prob=[]
for m in MODELS:
    for d in REGIMES:
        try:
            a=np.load(f"{SURF}/sentiment/{m}_{d}.npz", allow_pickle=True)
            b=np.load(f"{SURF}/nonsent/{m}_{d}.npz", allow_pickle=True)
        except Exception as e:
            shape_prob.append((m,d,str(e))); continue
        miss=[k for k in KEYS if k not in a.files]
        if miss: shape_prob.append((m,d,f"missing keys {miss}")); continue
        if a["market_iv"].shape != b["market_iv"].shape:
            align_prob.append((m,d,"arm shapes differ")); continue
        if len(a["dates"]) != len(b["dates"]) or not np.array_equal(a["dates"],b["dates"]):
            align_prob.append((m,d,"date vectors differ between arms"))
        for nm,arr in (("sent",a),("base",b)):
            for k in ("market_iv","model_px","market_px"):
                v=arr[k].astype(float)
                if not np.isfinite(v).all():
                    nan_prob.append((m,d,nm,k,float(np.mean(~np.isfinite(v)))))
chk(not shape_prob, f"all files load with expected keys ({len(shape_prob)} problems)")
for x in shape_prob: print("       ",x)
chk(not align_prob, f"arms aligned on dates & shapes ({len(align_prob)} problems)")
for x in align_prob: print("       ",x)
chk(not nan_prob, f"no NaN/Inf in iv/px arrays ({len(nan_prob)} arrays affected)", w=True)
for x in nan_prob[:6]: print(f"        {x[0]} {x[1]} {x[2]} {x[3]}: {x[4]:.2%} non-finite")

print("\n"+"="*74); print("3. SURFACES REFLECT THE RIGHT ARM"); print("="*74)
# variant tag stored inside each npz must match its folder
tagbad=[]
for arm in ("sentiment","nonsent"):
    for m in MODELS:
        for d in REGIMES:
            f=f"{SURF}/{arm}/{m}_{d}.npz"
            if not os.path.exists(f): continue
            z=np.load(f, allow_pickle=True)
            v=str(z["variant"]) if "variant" in z.files else "?"
            if v!=arm: tagbad.append((arm,m,d,v))
chk(not tagbad, f"internal 'variant' tag matches folder ({len(tagbad)} mismatches)")
for x in tagbad[:8]: print("       ",x)
# and the two arms must not be identical
ident=[(m,d) for m in MODELS for d in REGIMES
       if os.path.exists(f"{SURF}/sentiment/{m}_{d}.npz")
       and np.allclose(np.load(f"{SURF}/sentiment/{m}_{d}.npz")["model_px"],
                       np.load(f"{SURF}/nonsent/{m}_{d}.npz")["model_px"])]
chk(not ident, f"arms produce different prices ({len(ident)} identical pairs)")

print("\n"+"="*74); print("4. CONTEXT / SENTIMENT SANITY"); print("="*74)
for d in REGIMES:
    z=np.load(f"{SURF}/sentiment/Heston_{d}.npz", allow_pickle=True)
    c=z["context"].astype(float)
    s=c[:,0]
    chk(np.isfinite(s).all() and s.std()>1e-6,
        f"{d} sentiment col: mean={s.mean():+.4f} std={s.std():.4f} n={len(s)}")

print("\n"+"="*74); print("5. EVALUATION SCRIPTS - present & importable"); print("="*74)
need = {"eval_upgrades.py":"metrics + DM test",
        "compare_variants.py":"arm comparison",
        "param_identification.py":"parameter identifiability",
        "report_extras.py":"tables",
        "calendar_diagnostic.py":"no-arbitrage checks"}
for f,desc in need.items():
    ok=os.path.exists(f)
    chk(ok, f"{f:26} {desc}")
    if ok:
        try:
            src=open(f).read(); compile(src,f,"exec")
        except SyntaxError as e:
            chk(False, f"{f} syntax error line {e.lineno}")

print("\n"+"="*74); print("6. WHAT THE SCRIPTS EXPECT (paths they will read/write)"); print("="*74)
for f in [x for x in need if os.path.exists(x)]:
    src=open(f).read()
    reads=set(re.findall(r'eval_surfaces[^\'"\)]*', src))
    writes=set(re.findall(r'(?:analysis|figures|results_master)[^\'"\)]*', src))
    print(f"  {f}")
    for r in sorted(reads)[:3]:  print(f"        reads : {r}")
    for w in sorted(writes)[:3]: print(f"        writes: {w}")
    if not reads and not writes: print("        (no explicit eval_surfaces/analysis paths found)")

print("\n"+"="*74); print("7. OUTPUT DIRS WRITABLE"); print("="*74)
for d in [f"{REPO}/full_project_output/analysis", f"{REPO}/full_project_output/figures"]:
    try:
        os.makedirs(d, exist_ok=True); t=os.path.join(d,".w"); open(t,"w").close(); os.remove(t)
        chk(True, f"{d.replace(REPO,'~')} writable")
    except Exception as e:
        chk(False, f"{d}: {e}")
n_stale=len(glob.glob(f"{REPO}/full_project_output/analysis/*.csv"))
chk(n_stale==0, f"analysis dir has no stale CSVs (found {n_stale})", w=True)

print("\n"+"="*74)
print("READY TO EVALUATE" if not fails else "NOT READY -- unresolved:")
for f in fails: print("   -",f)
for w in warns: print("   warn:",w)
print("="*74)
