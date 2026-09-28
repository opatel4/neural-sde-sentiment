"""
audit_all_results.py -- run from ~/SeniorDesignProject
Re-verifies every evaluation already produced, independently of the scripts that
produced them. Recomputes headline numbers from the raw .npz files so a bug in a
comparison script cannot hide behind its own output.
"""
import os, sys, glob, json, csv
import numpy as np

REPO = os.path.expanduser("~/SeniorDesignProject")
EVAL = f"{REPO}/full_project_output/eval_surfaces"
WF   = f"{REPO}/full_project_output/eval_wf_surfaces"
AN   = f"{EVAL}/analysis"
MODELS=["Heston","Bates","Bergomi","rBergomi"]
REGIMES=["2010-2012","2013-2015","2016-2019","2020-2022"]
fails, warns = [], []
def chk(c,l,w=False):
    print(f"  {'OK  ' if c else ('WARN' if w else 'FAIL')} {l}")
    if not c: (warns if w else fails).append(l)

print("="*76); print("1. IN-SAMPLE SURFACES - integrity"); print("="*76)
for arm in ("sentiment","nonsent"):
    n=len(glob.glob(f"{EVAL}/{arm}/*.npz")); chk(n==16, f"{arm:9} {n}/16 files")
bad=[]
for m in MODELS:
    for d in REGIMES:
        try:
            a=np.load(f"{EVAL}/sentiment/{m}_{d}.npz",allow_pickle=True)
            b=np.load(f"{EVAL}/nonsent/{m}_{d}.npz",allow_pickle=True)
        except Exception as e: bad.append((m,d,str(e))); continue
        if not np.array_equal(a["dates"],b["dates"]): bad.append((m,d,"date mismatch"))
        if np.allclose(a["model_px"],b["model_px"]): bad.append((m,d,"arms identical"))
        for k in ("market_iv","model_px"):
            if not np.isfinite(a[k]).all(): bad.append((m,d,f"{k} non-finite (S)"))
chk(not bad, f"paired, aligned, distinct, finite ({len(bad)} problems)")
for x in bad[:6]: print("       ",x)

print("\n"+"="*76); print("2. RECOMPUTE headline metric INDEPENDENTLY"); print("="*76)
print("   (unweighted RMSE on model_px vs market_px, common finite cells)")
print(f"   {'model':10}{'regime':12}{'RMSE_NS':>10}{'RMSE_S':>10}{'better':>9}{'ndays':>7}")
recomp={}
for m in MODELS:
    for d in REGIMES:
        a=np.load(f"{EVAL}/sentiment/{m}_{d}.npz"); b=np.load(f"{EVAL}/nonsent/{m}_{d}.npz")
        mk=a["market_px"].astype(float); ps=a["model_px"].astype(float); pn=b["model_px"].astype(float)
        msk=np.isfinite(mk)&np.isfinite(ps)&np.isfinite(pn)
        rs=float(np.sqrt(np.mean((ps[msk]-mk[msk])**2)))
        rn=float(np.sqrt(np.mean((pn[msk]-mk[msk])**2)))
        recomp[(m,d)]=(rn,rs)
        print(f"   {m:10}{d:12}{rn:10.5f}{rs:10.5f}{('S' if rs<rn else 'NS'):>9}{len(a['dates']):7}")

print("\n"+"="*76); print("3. CROSS-CHECK vs results_master.csv"); print("="*76)
f=f"{AN}/results_master.csv"
if os.path.exists(f):
    rows=list(csv.DictReader(open(f)))
    chk(len(rows)==16, f"results_master.csv has {len(rows)}/16 rows")
    agree=dis=0
    for r in rows:
        k=(r["model"],r["regime"])
        if k not in recomp: continue
        rn,rs=recomp[k]
        indep = "S" if rs<rn else "NS"
        (agree:=agree+1) if indep==r["better"] else (dis:=dis+1)
    print(f"       price-RMSE agrees with vega-weighted-IV verdict in {agree}/16 cells")
    print(f"       (disagreement is EXPECTED where the two metrics rank differently;")
    print(f"        large disagreement means the metric choice drives the result)")
else:
    chk(False,"results_master.csv present")

print("\n"+"="*76); print("4. SIGNIFICANCE TALLY from results_master.csv"); print("="*76)
if os.path.exists(f):
    rows=list(csv.DictReader(open(f)))
    S_sig=sum(1 for r in rows if r["better"]=="S" and r["sig_DM"])
    S_ns =sum(1 for r in rows if r["better"]=="S" and not r["sig_DM"])
    N_sig=sum(1 for r in rows if r["better"]=="NS" and r["sig_DM"])
    N_ns =sum(1 for r in rows if r["better"]=="NS" and not r["sig_DM"])
    print(f"       sentiment better: {S_sig} significant, {S_ns} not")
    print(f"       baseline  better: {N_sig} significant, {N_ns} not")
    chk(S_sig+S_ns+N_sig+N_ns==16, "tally covers all 16 cells")
    cov=[float(r["coverage"]) for r in rows]
    chk(min(cov)>0.5, f"min common coverage {min(cov):.3f} (below 0.5 = most of grid discarded)", w=True)
    byfam={}
    for r in rows: byfam.setdefault(r["model"],[]).append(r["better"])
    print("       by family:", {k:f"{v.count('S')}S/{v.count('NS')}NS" for k,v in byfam.items()})

print("\n"+"="*76); print("5. PARAMETER IDENTIFICATION"); print("="*76)
f2=f"{AN}/param_identification.csv"
if os.path.exists(f2):
    rows=list(csv.DictReader(open(f2)))
    chk(len(rows)>0, f"param_identification.csv has {len(rows)} rows")
    fam={}
    for r in rows:
        cell=r.get("cell") or r.get("model") or ""
        famname=cell.split("_")[0] if "_" in cell else cell
        try: fam.setdefault(famname,[]).append(float(r.get("IR") or r.get("ir")))
        except: pass
    for k,v in sorted(fam.items(), key=lambda x: np.mean(x[1]) if x[1] else 9):
        if v: print(f"       {k:10} mean IR={np.mean(v):.3f}  n={len(v)}")
    print("       (lower IR = better identified; IR~1.0 = indistinguishable from random)")
else: chk(False,"param_identification.csv present", w=True)

print("\n"+"="*76); print("6. BENCHMARKS - do models beat persistence?"); print("="*76)
f3=f"{AN}/benchmarks_table.csv"
if os.path.exists(f3):
    rows=list(csv.DictReader(open(f3)))
    beat=sum(1 for r in rows if str(r.get("beats_persistence","")).strip().upper() in ("YES","TRUE","1"))
    print(f"       cells where a model beats previous-day persistence: {beat}/{len(rows)}")
    chk(beat>0, f"at least one cell beats persistence ({beat}/{len(rows)})", w=True)
    if beat==0:
        print("       -> persistence dominates everywhere. This bounds every other")
        print("          claim in the study and must be stated prominently.")
else: chk(False,"benchmarks_table.csv present", w=True)

print("\n"+"="*76); print("7. WALK-FORWARD (out-of-sample) - progress"); print("="*76)
for arm in ("sentiment","nonsent"):
    n=len(glob.glob(f"{WF}/{arm}/*.npz")); chk(n==16, f"{arm:9} {n}/16 files", w=(n<16))
wf=glob.glob(f"{WF}/*/*.npz")
if wf:
    z=np.load(wf[0],allow_pickle=True)
    print(f"       sample: {os.path.basename(wf[0])} {len(z['dates'])} OOS days "
          f"({z['dates'][0]} .. {z['dates'][-1]})")
    print("       NOTE: OOS refits used a capped epoch budget and were still")
    print("       improving at the cap -> OOS numbers are a conservative lower bound.")

print("\n"+"="*76)
print("ALL CHECKS PASSED" if not fails else "PROBLEMS FOUND:")
for x in fails: print("   -",x)
for x in warns: print("   warn:",x)
print("="*76)
