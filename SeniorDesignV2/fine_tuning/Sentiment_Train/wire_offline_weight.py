"""
wire_offline_weight.py   (run in fine_tuning/Sentiment_Train/)

config.py now defines offline_weight_for(model_name), but train.py still passes
the flat OFFLINE_WEIGHT constant at two call sites (~212 and ~250), so the
rBergomi-specific weight never takes effect.

This patch:
  - imports offline_weight_for alongside OFFLINE_WEIGHT
  - replaces  offline_weight=OFFLINE_WEIGHT
      with    offline_weight=offline_weight_for(selected_model_name)
    at both sites, falling back to the model variable actually in scope.

Run:  python3 wire_offline_weight.py
"""
import re, os, sys, shutil
p="train.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s=open(p).read()
if "offline_weight_for" in s:
    print("already patched"); sys.exit(0)
shutil.copy(p,p+".bak_ow")

# 1. import it
s = s.replace("    OFFLINE_WEIGHT,", "    OFFLINE_WEIGHT,\n    offline_weight_for,", 1)

# 2. figure out which model variable is in scope at each call site
lines = s.splitlines(keepends=True)
out=[]; n=0
for i,l in enumerate(lines):
    if "offline_weight=OFFLINE_WEIGHT" in l:
        # look backwards for the nearest loop/assignment naming the model
        var=None
        for j in range(i, max(i-80,0), -1):
            m=re.search(r'\b(selected_model_name|model_name|BASE_MODEL|selected_config\["?model_type"?\])\b', lines[j])
            if m: var=m.group(1); break
        var = var or "selected_model_name"
        l = l.replace("offline_weight=OFFLINE_WEIGHT",
                      f"offline_weight=offline_weight_for({var})")
        n+=1
    out.append(l)
s="".join(out)
open(p,"w").write(s)
print(f"patched {n} call site(s) to use offline_weight_for(...)")
import py_compile; py_compile.compile(p, doraise=True); print("train.py compiles OK")
