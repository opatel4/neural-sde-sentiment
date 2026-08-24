#!/usr/bin/env bash
# restructure_A.sh — move to the offline/ + online/ layout.
# RUN ON A BRANCH:  git checkout -b restructure && bash restructure_A.sh
# Verify with audit + import check BEFORE committing. Abandon with:
#   git checkout main && git branch -D restructure
set -euo pipefail
cd "$(dirname "$0")"

[ -d src ] || { echo "ERROR: no src/ — are you in the repo root?"; exit 1; }

echo "== moving offline stage =="
mkdir -p offline
git mv src/generate_offline_data.py offline/generate_offline_data.py
git mv src/pretrain_offline.py      offline/pretrain_offline.py

echo "== moving online stage (all 20 modules stay siblings) =="
mkdir -p online
for f in src/fine_tuning/*.py; do
  git mv "$f" "online/$(basename "$f")"
done

echo "== moving sentiment =="
mkdir -p sentiment
for f in src/sentiment/*.py; do
  git mv "$f" "sentiment/$(basename "$f")"
done

rmdir src/fine_tuning src/sentiment src 2>/dev/null || true

echo "== patching scripts: src/fine_tuning -> online, src/ -> offline/ =="
sed -i '' \
  -e 's|\$PWD/src/fine_tuning|$PWD/online|g' \
  -e 's|src/fine_tuning/|online/|g' \
  -e 's|src/generate_offline_data.py|offline/generate_offline_data.py|g' \
  -e 's|src/pretrain_offline.py|offline/pretrain_offline.py|g' \
  scripts/*.sh

echo "== patching README paths =="
sed -i '' \
  -e 's|src/fine_tuning/|online/|g' \
  -e 's|src/generate_offline_data.py|offline/generate_offline_data.py|g' \
  -e 's|src/pretrain_offline.py|offline/pretrain_offline.py|g' \
  -e 's|src/sentiment/|sentiment/|g' \
  -e 's|\$PWD/src/fine_tuning|$PWD/online|g' \
  README.md

echo "== patching audit =="
sed -i '' \
  -e 's|src/fine_tuning|online|g' \
  -e 's|src/generate_offline_data.py|offline/generate_offline_data.py|g' \
  -e 's|src/pretrain_offline.py|offline/pretrain_offline.py|g' \
  -e 's|src/sentiment/|sentiment/|g' \
  -e 's|find src -name|find offline online sentiment -name|g' \
  -e 's|^for d in src|for d in offline online sentiment|' \
  audit_repo.sh

echo
echo "== new structure =="
find . -maxdepth 2 -type d -not -path './.git*' | sort
echo
echo "== NOW VERIFY (do not commit until both are clean) =="
echo "   bash audit_repo.sh 2>&1 | tail -8"
echo
echo "== if green =="
echo "   git add -A && git commit -m 'Restructure to offline/online layout'"
echo "   git checkout main && git merge restructure && git push"
echo
echo "== if broken =="
echo "   git checkout main && git branch -D restructure"
