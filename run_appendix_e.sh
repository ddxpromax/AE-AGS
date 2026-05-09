#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck source=scripts/parallel_defaults.sh
source "$SCRIPT_DIR/scripts/parallel_defaults.sh"

python -m ae_ags.sweep_appendix_e --T 100000 --runs 20 --jobs 8 --clip-rewards 0 --rectify-regret 0 "$@"
