"""
fix_compare_per_regime_rate.py   (run in fine_tuning/Sentiment_Train/)

compare_variants.py has no CLI args and loops over all 16 cells in one pass, so
a single SDP_R env var cannot be correct for every cell. eval_upgrades reads its
R/Q once at IMPORT time, which means the grid/vega weights would use one rate
for all four regimes.

This patch makes analyze() set eval_upgrades.R / .Q per cell, from the regime
name, before it computes anything. Rates are the pooled medians measured from
the SPX CSVs:

    2010-2012  0.0254     2013-2015  0.0230
    2016-2019  0.0236     2020-2022  0.0169     q = 0.015 everywhere

Scope note: both arms are always scored on the SAME grid within a cell, so this
mostly moves the ABSOLUTE RMSE level, not the NS-vs-S verdict. It matters for
the benchmark table and for reporting levels honestly.

Usage:  python3 fix_compare_per_regime_rate.py
"""
import os, re, sys, shutil

p = "compare_variants.py"
if not os.path.exists(p): sys.exit("run from fine_tuning/Sentiment_Train/")
s = open(p).read()
if "_REGIME_RATES" in s:
    print("already patched"); sys.exit(0)

block = '''
# --- PATCHED: per-regime rates -------------------------------------------
# eval_upgrades reads R/Q at import time, but this script processes all four
# regimes in one pass. Set them per cell so the strike grid, IV inversion and
# vega weights use the rate that regime was actually trained under.
_REGIME_RATES = {"2010-2012": 0.0254, "2013-2015": 0.0230,
                 "2016-2019": 0.0236, "2020-2022": 0.0169}
_Q_ALL = 0.0150

def _set_rate_for(date_range):
    import eval_upgrades as _eu
    _eu.R = _REGIME_RATES.get(date_range, 0.0231)
    _eu.Q = _Q_ALL
    return _eu.R, _eu.Q
# --------------------------------------------------------------------------
'''

# insert the block after the imports
m = re.search(r"^(import|from)\s+.*$", s, flags=re.M)
last_import_end = 0
for mm in re.finditer(r"^(import|from)\s+.*$", s, flags=re.M):
    last_import_end = mm.end()
s = s[:last_import_end] + "\n" + block + s[last_import_end:]

# call it at the top of analyze(), which knows the date_range
m2 = re.search(r"def analyze\([^)]*\):\n", s)
if not m2:
    sys.exit("could not find analyze(); inspect manually")
# find how date_range is derived inside analyze
after = s[m2.end():m2.end()+1200]
if "date_range" in after:
    # inject right after the first line that binds date_range
    m3 = re.search(r"^(\s*)(\w*date_range\w*\s*=\s*.*)$", after, flags=re.M)
    if m3:
        indent = m3.group(1)
        ins = f"\n{indent}_set_rate_for(date_range)   # PATCHED: per-regime rate\n"
        pos = m2.end() + m3.end()
        s = s[:pos] + ins + s[pos:]
        note = "injected after date_range assignment inside analyze()"
    else:
        ins = f"    _set_rate_for(date_range)   # PATCHED: per-regime rate\n"
        s = s[:m2.end()] + ins + s[m2.end():]
        note = "injected at top of analyze() (verify date_range is a parameter)"
else:
    ins = "    _set_rate_for(date_range)   # PATCHED: per-regime rate\n"
    s = s[:m2.end()] + ins + s[m2.end():]
    note = "injected at top of analyze() - VERIFY date_range is in scope"

shutil.copy(p, p + ".bak")
open(p, "w").write(s)
import py_compile; py_compile.compile(p, doraise=True)
print("compare_variants.py patched:")
print("   - _REGIME_RATES table + _set_rate_for()")
print(f"   - {note}")
print("compiles OK")
print()
print("VERIFY the injection landed somewhere sensible:")
print("  grep -n '_set_rate_for' compare_variants.py")
print("  sed -n '/def analyze/,/+12p/p' compare_variants.py | head -18")
