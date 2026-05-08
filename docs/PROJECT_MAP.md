# Project Map

This document explains how the repository is organized and where to edit when extending functionality.

## 1) Execution Entry Points

- Package-first execution:
  - `python -m ae_ags.run_experiment`
  - `python -m ae_ags.sweep_appendix_e`
  - `python -m ae_ags.plot_from_run_json`

- `run_paper_default.sh`
  - One-command wrapper for paper default run.

- `run_appendix_e.sh`
  - One-command wrapper for paper-scale Appendix E sweeps.

## 2) Core Algorithm & Environment

- `ae_ags/aeags_centralized.py`
  - Centralized AE-AGS policy.
  - Includes UCB/LCB updates, `Better` relation updates, and arm-guided matching subroutine.

- `ae_ags/baselines.py`
  - Baseline implementations:
    - `CETCKnownDelta`
    - `PhasedETC`
    - `RandomMatchingPolicy`
  - Shared GS helper for player-proposing matching.

- `ae_ags/market.py`
  - Matching market simulator:
    - proposal/rejection flow
    - reward generation (Gaussian noise)
    - weak stability checking
    - stable baseline reward approximation for larger `N`
  - Market generation modes:
    - `paper_rank` (default)
    - `level_uniform` (legacy)

## 3) Config and Shell Scripts

- `configs/paper_default.json`
  - Default paper-scale parameters.

- `run_paper_default.sh`
  - Wrapper over `python -m ae_ags.run_experiment` with paper default config.

- `run_appendix_e.sh`
  - Wrapper over `python -m ae_ags.sweep_appendix_e` with paper-scale sweep parameters.

## 4) Outputs

- `results/paper_run/`
  - One-setting run outputs (JSON + plots + text summary).

- `results/appendix_e_full/`
  - Full sweep outputs (JSON + 4 figures).

## 5) Non-source/Reference Material

- `2409_Bandit_Learning_in_Matchi.pdf`
- `2409_Bandit_Learning_in_Matchi_extracted.txt`

## 6) Ignore Rules

- `.gitignore` excludes Python cache artifacts and temporary smoke result folders.
- Final `results/` is intentionally tracked (high-cost reproducibility artifacts).
