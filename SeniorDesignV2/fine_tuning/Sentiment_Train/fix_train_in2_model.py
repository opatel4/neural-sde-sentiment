"""
fix_train_in2_model.py  (run in fine_tuning/Sentiment_Train/)
Makes train.py build the In2 model when SDP_USE_IN2=1, else the old In1 model.
The trainer call sites are already In2-aware; only model CONSTRUCTION was still
hardcoded to the 1-channel VolatilitySurfaceCNN via build_model_from_config.
"""
import re, shutil, sys, os
p="train.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s=open(p).read()
if "build_model_from_config_in2" in s:
    print("already patched"); sys.exit(0)

old="            model = build_model_from_config(selected_config, device)"
if old not in s:
    m=re.search(r"^\s*model = build_model_from_config\(selected_config, device\)\s*$", s, flags=re.M)
    if not m: sys.exit("could not find the build_model_from_config call at ~line 235")
    old=m.group(0)
indent=old[:len(old)-len(old.lstrip())]
new=(f"{indent}if os.environ.get('SDP_USE_IN2','0') == '1':\n"
     f"{indent}    from in2_adapter import build_model_from_config_in2\n"
     f"{indent}    model = build_model_from_config_in2(selected_config, device, strict=True)\n"
     f"{indent}else:\n"
     f"{indent}    model = build_model_from_config(selected_config, device)")
shutil.copy(p,p+".bak")
open(p,"w").write(s.replace(old,new))
# make sure os is imported
t=open(p).read()
if not re.search(r"^import os\b", t, flags=re.M):
    t="import os\n"+t; open(p,"w").write(t)
print("patched train.py: In2 model built when SDP_USE_IN2=1")
