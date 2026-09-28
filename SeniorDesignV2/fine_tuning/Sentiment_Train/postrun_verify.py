"""
postrun_verify.py -- run in fine_tuning/Sentiment_Train/ AFTER the 32-cell run,
BEFORE evaluation. Confirms what is actually on disk is complete and coherent.

Checks:
 1. all 32 (arm, model, regime) cells present, one usable checkpoint each
 2. every checkpoint loads, has sane best_epoch / best_loss, no NaN weights
 3. checkpoint metadata (model_type, x_mean/x_std) matches expectations
 4. the two arms are genuinely different models (not accidental duplicates)
 5. the rBergomi/2016-2019 rerun replaced the collapsed cell
 6. inputs still intact: surfaces, offline weights, SPX data, code fixes
"""
import os, sys, glob, itertools, importlib
import numpy as np, pandas as pd, torch

ROOT = os.path.expanduser("~/SeniorDesignProject/fine_tuning")
OUT  = f"{ROOT}/fine_tuning_output"
FT   = f"{ROOT}/Fine_Tuning_Files"
MODELS  = ["Heston","Bates","Bergomi","rBergomi"]
REGIMES = ["2010-2012","2013-2015","2016-2019","2020-2022"]
ARMS    = {"sentiment_output":"sentiment", "base_output":"baseline"}
fails, warns = [], []
def chk(c,l,warn=False):
    print(f"  {'OK  ' if c else ('WARN' if warn else 'FAIL')} {l}")
    if not c: (warns if warn else fails).append(l)

def pick(d):
    """the checkpoint a downstream script would use: highest-epoch 'final', else highest epoch"""
    fs = glob.glob(f"{d}/*.pth")
    if not fs: return None
    def ep(f):
        b=os.path.basename(f)
        try: return int(b.split("epoch_")[1].split(".")[0])
        except: return -1
    fin=[f for f in fs if "final" in os.path.basename(f)]
    return max(fin or fs, key=ep)

print("="*74); print("1. CELL COVERAGE  (32 = 2 arms x 4 models x 4 regimes)"); print("="*74)
found={}
for arm_dir,arm in ARMS.items():
    for m,r in itertools.product(MODELS,REGIMES):
        d=f"{OUT}/{arm_dir}/models/{m}/{r}"
        ck=pick(d) if os.path.isdir(d) else None
        found[(arm,m,r)]=ck
n=sum(1 for v in found.values() if v)
chk(n==32, f"cells with a checkpoint: {n}/32")
for k,v in sorted(found.items()):
    if not v: print(f"       MISSING: {k[0]:9} {k[1]:9} {k[2]}")

print("\n"+"="*74); print("2. CHECKPOINT HEALTH"); print("="*74)
meta={}
bad_ep=[]
for k,f in sorted(found.items()):
    if not f: continue
    try:
        ck=torch.load(f,map_location="cpu")
    except Exception as e:
        chk(False,f"{k} load error {e}"); continue
    be=ck.get("best_epoch",-1); bl=ck.get("best_loss",float('nan'))
    sd=ck.get("model_state_dict",{})
    nan=any(torch.isnan(v).any().item() for v in sd.values() if torch.is_floating_point(v))
    meta[k]=dict(best_epoch=be,best_loss=float(bl),model_type=ck.get("model_type"),
                 x_mean=float(ck["x_mean"].mean()) if "x_mean" in ck else None,
                 x_std=float(ck["x_std"].mean()) if "x_std" in ck else None, nan=nan, file=f)
    if be<50 or nan: bad_ep.append((k,be,nan))
chk(not bad_ep, f"all checkpoints converged & finite (problem cells: {len(bad_ep)})")
for k,be,nan in bad_ep: print(f"       {k} best_epoch={be} nan_weights={nan}  <-- {os.path.basename(meta[k]['file'])}")

if meta:
    eps=[v['best_epoch'] for v in meta.values()]; los=[v['best_loss'] for v in meta.values()]
    print(f"       best_epoch: min={min(eps)} med={int(np.median(eps))} max={max(eps)}")
    print(f"       best_loss : min={min(los):.6f} med={np.median(los):.6f} max={max(los):.6f}")

print("\n"+"="*74); print("3. MODEL TYPE MATCHES DIRECTORY"); print("="*74)
mism=[(k,v['model_type']) for k,v in meta.items() if v['model_type'] and v['model_type']!=k[1]]
chk(not mism, f"model_type matches folder for all cells ({len(mism)} mismatches)")
for k,mt in mism: print(f"       {k} says model_type={mt}")

print("\n"+"="*74); print("4. ARMS ARE GENUINELY DIFFERENT"); print("="*74)
same=0; cmpd=0
for m,r in itertools.product(MODELS,REGIMES):
    a=meta.get(("sentiment",m,r)); b=meta.get(("baseline",m,r))
    if not(a and b): continue
    cmpd+=1
    wa=torch.load(a['file'],map_location='cpu')['model_state_dict']
    wb=torch.load(b['file'],map_location='cpu')['model_state_dict']
    k0=[k for k in wa if wa[k].is_floating_point()][0]
    if torch.allclose(wa[k0],wb[k0]): same+=1; print(f"       IDENTICAL weights: {m} {r}")
chk(same==0 and cmpd==16, f"{cmpd}/16 pairs compared, {same} identical (want 0)")

print("\n"+"="*74); print("5. rBERGOMI 2016-2019 BASELINE (the rerun cell)"); print("="*74)
k=("baseline","rBergomi","2016-2019")
if k in meta:
    v=meta[k]
    chk(v['best_epoch']>=500, f"best_epoch={v['best_epoch']} loss={v['best_loss']:.6f} file={os.path.basename(v['file'])}")
else: chk(False,"rBergomi/2016-2019 baseline missing")

print("\n"+"="*74); print("6. INPUTS STILL INTACT"); print("="*74)
Ts=['0.1','0.3','0.6','0.9','1.2','1.5','1.8','2.0']
for m in MODELS:
    f=f"{FT}/{m}_IV_Surface_Data_Final.csv"
    d=pd.read_csv(f,nrows=1500); ss=[c for c in d.columns if c.startswith("SS_")]
    u=d[[c for c in ss if "_T0.1_" in c]].iloc[0].nunique()
    prof=np.array([[d[[c for c in ss if f"_T{T}_" in c]].iloc[i].mean() for T in Ts] for i in range(len(d))])
    mm=prof.mean(axis=0); corr=float(np.corrcoef(mm[:-1],mm[1:])[0,1])
    chk(u==1 and corr>0.3, f"{m:9} surface SS-unique={u} T-corr={corr:+.3f}")
for m in MODELS:
    w=f"{FT}/CNN_{m}_In2_OutBase.pth"
    sd=torch.load(w,map_location="cpu"); sd=sd.get("model_state_dict",sd) if isinstance(sd,dict) else sd
    src=os.path.expanduser(f"~/offline_models/CNN_{m}_In2_OutBase.pth")
    same_f = os.path.exists(src) and open(src,'rb').read(8192)==open(w,'rb').read(8192)
    chk(sd["conv_block.0.weight"].shape[1]==2 and same_f, f"{m:9} In2 weights 2-channel & retrained")
for dr in REGIMES:
    y0,y1=dr.split("-"); yrs=range(int(y0),int(y1)+1)
    r=pd.concat([pd.read_csv(f"{FT}/SPX_Data/cleanerSPXData_{y}_merged.csv",usecols=["r","S","Forward"],low_memory=False) for y in yrs])
    chk(0<r.r.median()<0.10 and abs((r.Forward/r.S).median()-1)<0.02,
        f"{dr} r_med={r.r.median():.4f} F/S={(r.Forward/r.S).median():.4f}")

print("\n"+"="*74)
print("READY FOR EVALUATION" if not fails else "NOT READY -- unresolved:")
for f in fails: print("   -",f)
for w in warns: print("   warn:",w)
print("="*74)
