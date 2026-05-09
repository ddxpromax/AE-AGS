# AE-AGS Reproduction Project

Reproduction workspace for:

> *Bandit Learning in Matching Markets with Indifference* (ICLR 2025)

This repo focuses on:
- centralized AE-AGS implementation,
- paper-scale experiment execution,
- Appendix E sweep automation,
- result export (JSON + figures).

---

## Project Layout

### Core code
- `ae_ags/`: main Python package (all core implementation now lives here).
- `ae_ags/market.py`: matching market simulator, stability checking, synthetic data generation.
- `ae_ags/aeags_centralized.py`: centralized AE-AGS (Algorithm 1/2/3 style).
- `ae_ags/baselines.py`: baselines (`CETCKnownDelta`, `PhasedETC`, `RandomMatchingPolicy`).
- `ae_ags/run_experiment.py`: single-setting experiment runner (supports presets, parallel runs, JSON export, trajectories).
- `ae_ags/sweep_appendix_e.py`: full Appendix E sweep runner.
- `ae_ags/plot_from_run_json.py`: plot curves from `run_experiment.py --save-json` output.

### Config / scripts
- `configs/paper_default.json`: default paper-scale config.
- `run_paper_default.sh`: one-command paper default run.
- `run_appendix_e.sh`: one-command Appendix E sweep run.

### Outputs
- `results/paper_run/`: paper default run outputs.
- `results/appendix_e_full/`: Appendix E sweep outputs.

### Reference material
- `2409_Bandit_Learning_in_Matchi.pdf`
- `2409_Bandit_Learning_in_Matchi_extracted.txt`

---

## Quick Start

```bash
cd /root/AE-AGS
python -m ae_ags.run_experiment --preset quick
```

---

## Recommended Workflows

### 1) Paper default (raw, paper-aligned)
```bash
./run_paper_default.sh
```

With explicit parallelism:
```bash
./run_paper_default.sh --runs 20 --jobs 8
```

### 2) Save trajectory JSON for plotting
```bash
python -m ae_ags.run_experiment \
  --preset paper_default \
  --runs 20 --jobs 8 \
  --record-every 1000 \
  --save-json results/paper_run/one_run_curve.json
```

### 3) Plot from saved JSON
```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/one_run_curve.json \
  --output-dir results/paper_run/plots
```

### 4) Full Appendix E sweep
```bash
./run_appendix_e.sh
```

---

## Faster runs (wall-clock)

Cost is dominated by **\(T \times\)** per-round simulation (matching + stability check × 4 policies). Appendix E sweep multiplies that by several \((\Delta, N)\) points.

Practical knobs:

| Idea | Effect |
|------|--------|
| **`--runs 20 --jobs 8`** (or `jobs ≤` CPU cores) | Best lever: parallel independent repeats (ProcessPoolExecutor). |
| **BLAS single-thread defaults** | `run_paper_default.sh` / `run_appendix_e.sh` source `scripts/parallel_defaults.sh` unless you already exported `OMP_NUM_THREADS` etc. **`make paper-j8`** / **`make paper-json`** prefix the same vars (override via `PARALLEL_BLAS=...`). |
| **`--record-every 5000`** (or larger) vs `1000` | Fewer trajectory snapshots ⇒ slightly less aggregation / JSON churn; main loop cost barely changes. |
| Appendix E **`--runs`** | Paper uses 20; lowering (e.g. 10) for dry runs scales linearly but is no longer apples-to-apples with Figure 2. |

Implementation-side (already in code): **`is_stable_matching` reuses cached player inverse ranks**, **`Better` updates are vectorized**, **`log(T)` precomputed**, and **AE-AGS reuses a fixed arm→player propose order** for the whole run instead of sorting `arm_rank` every round.

---

## Notes on Metrics

- **Stable regret (paper Eq. (1))** uses a per-player *regret reference* (not an algorithm baseline like C-ETC):
  \(\mu_{i,m_i}=\min_{\text{stable }m'}\mu_{i,m'(i)}\), computed in
  `MatchingMarket.stable_regret_reference_per_player`. Step regret is
  \(\mu_{i,m_i}-X_{i,A_i(t)}(t)\) on the matched arm outcome each round.

- AE-AGS empirical means \(T_{i,j}\) update only when the player was **actually matched**
  to the arm the platform assigned (Algorithm 3, lines 7–9).

- `paper_default` uses raw paper-style settings:
  - `clip_rewards=0`
  - `rectify_regret=0`
- Negative sampled rewards / cumulative regrets can therefore appear and are expected.
- For an engineering-style non-negative view:
```bash
python -m ae_ags.run_experiment --preset paper_clean
```

---

## Documentation

- `docs/PROJECT_MAP.md`: architecture and file responsibilities.
- `docs/COMMANDS.md`: copy-paste command handbook.
- `Makefile`: short aliases (`make quick`, `make paper-j8`, `make sweep`).

---

## Current Scope

- Baselines are practical reproductions designed to be closer to paper behavior.
- This is not yet a line-by-line full reimplementation of every paper baseline detail.
- Future work:
  - decentralized AE-AGS (Algorithm 4/5),
  - deeper protocol-level alignment for all baselines,
  - additional reproducibility tooling (seeded manifests / run registry).
