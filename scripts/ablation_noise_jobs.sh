#!/usr/bin/env bash
# Appendix E sanity checks from the reproduction plan:
# - jobs=1 vs jobs>1 should yield identical aggregates (same seeds, deterministic workers).
# - reward-noise-mode shared vs independent shifts cross-algorithm coupling only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0
COMMON=(--preset quick --runs 8 --seed 4242 --T 4000 --record-every 0 --aeags-arm-schedule fixed)

filter_summary() {
  grep -E '^\[|cumulative market unstability' || true
}

echo "=== A. jobs=1 vs jobs=8 (four 'unstability' lines should match pairwise) ==="
echo "--- jobs=1 ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 1 2>&1 | filter_summary
echo "--- jobs=8 ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 8 2>&1 | filter_summary

echo "=== B. reward-noise-mode shared vs independent ==="
echo "--- shared ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 1 --reward-noise-mode shared 2>&1 | filter_summary
echo "--- independent ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 1 --reward-noise-mode independent 2>&1 | filter_summary

echo "=== C. aeags-arm-schedule fixed vs random (AE-AGS instability line should change) ==="
echo "--- fixed ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 1 --aeags-arm-schedule fixed 2>&1 | grep -A3 '^\[AE-AGS\]' || true
echo "--- random ---"
python -m ae_ags.run_experiment "${COMMON[@]}" --jobs 1 --aeags-arm-schedule random 2>&1 | grep -A3 '^\[AE-AGS\]' || true
