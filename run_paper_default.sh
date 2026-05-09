#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/parallel_defaults.sh
source "$SCRIPT_DIR/scripts/parallel_defaults.sh"

python -m ae_ags.run_experiment --preset paper_default --config configs/paper_default.json "$@"
