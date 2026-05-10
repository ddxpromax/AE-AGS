#!/usr/bin/env bash
# Appendix Fig.1 funnel: medium-T grid over confidence factor, algo2 outer loop,
# arm-rank jitter, and pull tie-break. Writes results/paper_run/fig1_funnel_scan_t15000.txt
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[[ -f scripts/parallel_defaults.sh ]] && source scripts/parallel_defaults.sh || true

python -m ae_ags.scan_fig1_knobs \
  --T 15000 --runs 10 --jobs 4 \
  --confidence-factors 5,5.5,6,6.5 \
  --algo2-outer-loops pick_one,round_sweep \
  --arm-schedules fixed \
  --pull-tiebreaks random,smallest_arm \
  --ucb-time-scales horizon \
  --arm-rank-jitter-scales 0,1e-9 \
  | tee results/paper_run/fig1_funnel_scan_t15000.txt
