#!/usr/bin/env bash
# audit_repo.sh — verify the repo is complete AND not stale.
# Run:  cd ~/Downloads/neural-sde-sentiment && bash audit_repo.sh
# Checks content, not just presence. PASS/FAIL per item.

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
pass=0; fail=0; warn=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }
wrn()  { echo "  WARN  $1"; warn=$((warn+1)); }
has()  { [ -f "$1" ]; }

echo "=================================================================="
echo " AUDIT: $REPO"
echo "=================================================================="

echo
echo "--- 1. STRUCTURE -------------------------------------------------"
for d in src src/fine_tuning src/sentiment data scripts results figures paper; do
  [ -d "$d" ] && ok "dir $d" || bad "dir $d MISSING"
done
for f in README.md requirements.txt .gitignore data/README.md paper/neural_sde_sentiment_paper.md; do
  has "$f" && ok "file $f" || bad "file $f MISSING"
done

echo
echo "--- 2. CODE COMPLETENESS -----------------------------------------"
n=$(find src -name "*.py" | wc -l | tr -d ' ')
[ "$n" -ge 16 ] && ok "src has $n .py files (>=16)" || bad "src has only $n .py files"
for f in src/generate_offline_data.py src/pretrain_offline.py \
         src/sentiment/regression_sentiment.py src/fine_tuning/config.py \
         src/fine_tuning/train.py src/fine_tuning/trainer.py \
         src/fine_tuning/data_utils.py src/fine_tuning/in2_adapter.py \
         src/fine_tuning/walkforward.py src/fine_tuning/evaluate.py \
         src/fine_tuning/export_surfaces.py src/fine_tuning/compare_variants.py \
         src/fine_tuning/compare_variants_wf.py src/fine_tuning/run_hedging.py; do
  has "$f" && ok "$(basename $f)" || bad "$f MISSING"
done
n=$(ls scripts/0[0-6]_*.sh 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -eq 7 ] || [ "$n" -eq 6 ] && ok "scripts: $n stage scripts" || bad "scripts: expected 6-7, found $n"

echo
echo "--- 3. NOT STALE: corrected constants ----------------------------"
# in2_adapter: corrected offline sentiment moments
if grep -q "SENT_OFFLINE_MEAN *= *-0.0296" src/fine_tuning/in2_adapter.py 2>/dev/null; then
  ok "in2_adapter SENT_OFFLINE_MEAN = -0.0296 (corrected)"
else
  bad "in2_adapter SENT_OFFLINE_MEAN is NOT -0.0296 -> STALE"; grep -n "SENT_OFFLINE_MEAN" src/fine_tuning/in2_adapter.py 2>/dev/null | head -2
fi
if grep -q "SENT_OFFLINE_STD *= *0.0927" src/fine_tuning/in2_adapter.py 2>/dev/null; then
  ok "in2_adapter SENT_OFFLINE_STD = 0.0927 (corrected)"
else
  bad "in2_adapter SENT_OFFLINE_STD is NOT 0.0927 -> STALE"; grep -n "SENT_OFFLINE_STD" src/fine_tuning/in2_adapter.py 2>/dev/null | head -2
fi
# trainer: relaxed early stopping
grep -q "max_increase_streak *= *30" src/fine_tuning/trainer.py 2>/dev/null \
  && ok "trainer max_increase_streak = 30 (fixed)" \
  || bad "trainer max_increase_streak != 30 -> STALE early-stop rule"
grep -q "early_stop_arm_epoch *= *20" src/fine_tuning/trainer.py 2>/dev/null \
  && ok "trainer early_stop_arm_epoch = 20 (fixed)" \
  || bad "trainer early_stop_arm_epoch != 20 -> STALE"
# config: no per-model offline weight override
if grep -qE "OFFLINE_WEIGHT_BY_MODEL *= *\{\s*\}" src/fine_tuning/config.py 2>/dev/null; then
  ok "config OFFLINE_WEIGHT_BY_MODEL = {} (override removed)"
else
  wrn "config OFFLINE_WEIGHT_BY_MODEL not empty — check:"; grep -n "OFFLINE_WEIGHT_BY_MODEL" src/fine_tuning/config.py 2>/dev/null | head -2
fi
# data_utils: forward-mode price-cap gate reads the env var
grep -q 'SDP_NORM' src/fine_tuning/data_utils.py 2>/dev/null \
  && ok "data_utils gates the price cap on SDP_NORM" \
  || bad "data_utils has NO SDP_NORM gate -> price cap always on"
# walkforward: per-regime sentiment standardisation
grep -q "set_sent_regime" src/fine_tuning/walkforward.py 2>/dev/null \
  && ok "walkforward calls set_sent_regime (patched)" \
  || bad "walkforward MISSING set_sent_regime -> wrong sentiment scaling OOS"
for f in evaluate export_surfaces train; do
  grep -q "set_sent_regime" src/fine_tuning/$f.py 2>/dev/null \
    && ok "$f.py calls set_sent_regime" \
    || wrn "$f.py has no set_sent_regime — verify by hand"
done

echo
echo "--- 4. NOT STALE: generator coefficients -------------------------"
G=src/generate_offline_data.py
if grep -qE "0\.6855|0\.68555|0\.6856" $G 2>/dev/null; then
  ok "generator has corrected y_lag1 coefficient (~0.6856)"
else
  bad "generator does NOT contain the corrected coefficients -> STALE"
  echo "        expected intercept 0.06556, y_lag1 0.68555, ret -3.22856"
fi
grep -qiE "full.?trunc" $G 2>/dev/null && ok "generator: full-truncation scheme present" \
  || wrn "generator: no 'full truncation' string found — verify by hand"
grep -qiE "martingale|rescal" $G 2>/dev/null && ok "generator: martingale correction present" \
  || wrn "generator: no martingale/rescale string — verify by hand"
grep -qE "laguerre" $G 2>/dev/null && ok "generator: Gauss-Laguerre quadrature present" \
  || wrn "generator: no laguerre reference — verify by hand"

echo
echo "--- 5. PORTABILITY: no machine-specific absolute paths -----------"
hits=$(grep -rlE "/home/opatel4|/Users/ompatel|fscresearchvm" src config scripts 2>/dev/null)
if [ -z "$hits" ]; then
  ok "no absolute VM/Mac paths in src, config, scripts"
else
  bad "ABSOLUTE PATHS FOUND — a reviewer cannot run this as-is:"
  echo "$hits" | sed 's/^/        /'
  echo "        offending lines:"
  grep -rnE "/home/opatel4|/Users/ompatel|fscresearchvm" src config scripts 2>/dev/null | head -12 | sed 's/^/        /'
fi

echo
echo "--- 6. RESULTS: complete and correct shape -----------------------"
for f in results/in_sample/analysis/results_master.csv results/walkforward/analysis/results_master.csv; do
  if has "$f"; then
    r=$(( $(wc -l < "$f") - 1 ))
    [ "$r" -eq 16 ] && ok "$(basename $(dirname $(dirname $f)))/results_master.csv has 16 rows" \
                    || bad "$f has $r rows, expected 16"
  else bad "$f MISSING"; fi
done
n=$(ls results/in_sample/analysis/*.json 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -ge 16 ] && ok "in-sample per-cell JSONs: $n" || bad "in-sample JSONs: $n (expected >=16)"
n=$(ls results/walkforward/analysis/*.json 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -ge 16 ] && ok "walkforward per-cell JSONs: $n" || bad "walkforward JSONs: $n (expected >=16)"
has results/hedging_all.log && { n=$(grep -c "lower hedging" results/hedging_all.log); \
  [ "$n" -eq 16 ] && ok "hedging_all.log has 16 results" || bad "hedging_all.log has $n results"; } \
  || bad "results/hedging_all.log MISSING"
has results/in_sample/analysis/param_identification.csv && ok "param_identification.csv present" \
  || wrn "param_identification.csv missing"
# spot-check a headline number the paper reports
if has results/walkforward/analysis/results_master.csv; then
  grep -q "Heston,2016-2019" results/walkforward/analysis/results_master.csv \
    && ok "OOS CSV contains Heston 2016-2019 (paper's key surviving cell)" \
    || wrn "could not find Heston,2016-2019 row — check delimiter"
fi

echo
echo "--- 7. FIGURES ---------------------------------------------------"
n=$(find figures -name "*.png" | wc -l | tr -d ' ')
[ "$n" -ge 26 ] && ok "figures: $n png (16 hedging + 10 paper)" || bad "figures: only $n png"
[ -d figures/paper ] && ok "figures/paper present" || bad "figures/paper MISSING"
has figures/make_figures.py && ok "make_figures.py present" || bad "make_figures.py MISSING"
if has figures/make_figures.py; then
  grep -q "0.6, 0.9, 1.2, 1.5, 1.8" figures/make_figures.py \
    && ok "make_figures TAU grid corrected [0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0]" \
    || bad "make_figures TAU grid is STALE (wrong maturity labels)"
fi

echo
echo "--- 8. DATA ------------------------------------------------------"
has data/spx_returns_sentiment_merged.csv && {
  c=$(head -1 data/spx_returns_sentiment_merged.csv)
  echo "$c" | grep -q "daily_sentiment" && ok "merged CSV has daily_sentiment column" \
    || bad "merged CSV missing daily_sentiment"
  r=$(( $(wc -l < data/spx_returns_sentiment_merged.csv) - 1 ))
  echo "        rows: $r"
} || bad "data/spx_returns_sentiment_merged.csv MISSING"
has data/treasury_curve.csv && ok "treasury_curve.csv present" \
  || wrn "treasury_curve.csv missing — needed for SDP_NORM=forward"
has data/dividend_yield.csv && ok "dividend_yield.csv present" \
  || wrn "dividend_yield.csv missing (constant 1.9% may be hardcoded)"

echo
echo "--- 9. HYGIENE: nothing large or licensed committed ---------------"
n=$(find . -name "*.pth" -not -path './.git/*' | wc -l | tr -d ' ')
[ "$n" -eq 0 ] && ok "no .pth committed (gitignored, use a Release)" || wrn "$n .pth files present — will bloat/reject on push"
n=$(find . -name "*.npz" -not -path './.git/*' | wc -l | tr -d ' ')
[ "$n" -eq 0 ] && ok "no .npz committed" || wrn "$n .npz files present"
big=$(find . -type f -size +50M -not -path './.git/*' 2>/dev/null)
[ -z "$big" ] && ok "no file over 50MB" || { wrn "large files (GitHub caps at 100MB):"; echo "$big" | sed 's/^/        /'; }
[ -d data/SPX_Data ] && bad "data/SPX_Data present — LICENSED, must not be committed" || ok "no licensed SPX data in tree"
echo "        repo size: $(du -sh . 2>/dev/null | cut -f1)"

echo
echo "--- 10. KNOWN OPEN ITEMS -----------------------------------------"
has src/sentiment/finbert_scoring.py && ok "finbert_scoring.py present" \
  || wrn "finbert_scoring.py MISSING — README references it (ask Sam / add stub)"

echo
echo "=================================================================="
printf " PASS %d   WARN %d   FAIL %d\n" $pass $warn $fail
echo "=================================================================="
[ "$fail" -eq 0 ] && echo " No blocking failures." || echo " Fix the FAIL items before pushing."

echo
echo "--- 11. IMPORTS RESOLVE ------------------------------------------"
( cd src/fine_tuning && python3 -c "
import ast, os, sys
local = {f[:-3] for f in os.listdir('.') if f.endswith('.py')}
std = getattr(sys, 'stdlib_module_names', set())
third = {'numpy','pandas','torch','scipy','matplotlib','tqdm','sklearn'}
missing = set()
for f in sorted(os.listdir('.')):
    if not f.endswith('.py'): continue
    try: tree = ast.parse(open(f, errors='ignore').read())
    except SyntaxError: continue
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods = [n.module.split('.')[0]]
        elif isinstance(n, ast.Import):
            mods = [a.name.split('.')[0] for a in n.names]
        for m in mods:
            if m not in local and m not in std and m not in third:
                missing.add((f, m))
print('  FAIL  unresolved:', sorted(missing)) if missing else print('  PASS  all local imports resolve')
" )
