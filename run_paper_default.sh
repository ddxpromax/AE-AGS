#!/usr/bin/env bash
# Default paper single-setting run: Appendix Fig.1 **knee ~15k** preset (paper_fig1_knee15k).
# For C-ETC appendix (f) scale (c_etc_log_coeff≈8.35) use: --preset paper_default --config configs/paper_default.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/parallel_defaults.sh
source "$SCRIPT_DIR/scripts/parallel_defaults.sh"

python -m ae_ags.run_experiment --preset paper_fig1_knee15k --config configs/paper_fig1_knee15k.json "$@"
