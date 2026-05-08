#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python sweep_appendix_e.py --T 100000 --runs 20 --jobs 8 --clip-rewards 0 --rectify-regret 0 "$@"
