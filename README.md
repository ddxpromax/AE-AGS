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
- `ae_ags/scan_fig1_knobs.py`: grid scan of AE-AGS knobs vs C-ETC unstability (Fig. 1(f) alignment).
- `ae_ags/test_algorithm2_ai.py`: regression check that Algorithm 2 compares incumbent + newcomer on \(T_{i,j}\) (`python -m ae_ags.test_algorithm2_ai`).

### Config / scripts
- `configs/paper_default.json`: default paper-scale config.
- `run_paper_default.sh`: one-command paper default run.
- `run_appendix_e.sh`: one-command Appendix E sweep run.

### Outputs
- `results/paper_run/`: paper default run outputs.
- `results/appendix_e_full/`: Appendix E sweep outputs.

**Algorithm 2 `A_i` fix (incumbent tentative match included when comparing proposals):** after this correction, rerun Fig. 1 curves from `one_run_curve_appendix_e_fixed.json` / `figure1_appendix_e_fixed.png` (commands in plan). With `preset paper_default`, `runs=20`, `seed=0`, `fixed` arm schedule, a representative mean **AE-AGS cumulative market unstability \(\approx 5.0\times 10^4\)** vs **C-ETC \(\approx 4.3\times 10^4\)**—closer than the pre-fix \(\sim 6.0\times 10^4\) plateau. Older JSON (`one_run_curve_appendix_e*.json` from before the `A_i` change) should not be cited as matching current code.

Other archived knobs/sweeps: `one_run_curve_appendix_e_round_robin.json`, `one_run_curve_appendix_e_random.json` (~\(6.9\times 10^4\) AE-AGS, avoid for repro), etc., under `results/paper_run/`; prefer **`fixed`** arm scheduling over **`random`**.

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

Appendix **Figure 1** (six panels: regret for \(p_1,\ldots,p_5\) plus cumulative market unstability; three algorithms):

```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/one_run_curve.json \
  --output-dir results/paper_run/plots \
  --paper-figure1
# `make paper-figure1` if JSON already exists under results/paper_run/
```
(Re-save JSON after upgrading the repo so `curve` contains `per_player_stable_regret_mean` / `..._se`.)

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

### Appendix Fig. 1 reproduction toggles (`run_experiment`)

- **`--aeags-arm-schedule`**: Algorithm 2 chooses which **unmatched arm** proposes when several are eligible. Options: **`fixed`** (lowest arm index; default presets—closest to deterministic pseudo-code implementations), **`round_robin`** (cyclic scan), **`random`** (legacy uniform choice). Compare schedules if AE-AGS vs C-ETC **unstability ordering** disagrees with the PDF.
- **`--reward-noise-mode`**: **`shared`** (default): deterministic Gaussian keyed by `(seed, t, i, a)`, identical across algorithms when two policies realize the same `(t,i,a)`; **`independent`**: separate salt per algorithm for ablations against the appendix wording “drawn independently” (does not change fairness of *relative* rankings unless you rely on coupling).
- **`--aeags-player-pull-tiebreak`**: when several arms tie on **minimum** pull count in Algorithm 2 line 6, **`random`** (default) breaks ties uniformly; **`smallest_arm`** picks the lowest arm index (deterministic, closer to some reference scripts).
- **`--aeags-ucb-time-scale`**: **`horizon`** (default) uses \(\ln(T)\) in the UCB/LCB radius as in the paper; **`elapsed`** uses \(\ln(t)\) at the current round—**empirical only**, not the theorem statement.

Knob scan (same seed, prints `AE_unst`, `CETC_unst`, `AE−CETC`):

```bash
python -m ae_ags.scan_fig1_knobs --T 8000 --runs 8 --jobs 4 \
  --arm-schedules fixed,random,round_robin \
  --pull-tiebreaks random,smallest_arm \
  --ucb-time-scales horizon,elapsed
```

Reproducibility: parallel **`--jobs`** only affects wall-clock; aggregates should match **`--jobs 1`** for the same **`--runs`** / **`--seed`**. Sanity script: [`scripts/ablation_noise_jobs.sh`](scripts/ablation_noise_jobs.sh).

### C-ETC `log_coeff` (theory vs “pixel” alignment)

| `--c-etc-log-coeff` | Typical use |
|---------------------|-------------|
| **4** | Theorem-style order \(\propto \ln(T)/\Delta^2\) (conservative exploration). |
| **8.35** | Default in `paper_default` / `configs/paper_default.json`: empirically sets C-ETC mean cumulative unstability near **~43%** of rounds at \(T=10^5\), \(N=K=5\), matching Appendix Fig.1(f) scale; **not** an author-published constant. |
| **2** | `quick` preset: faster, shallower exploration. |

To scan coefficients, rerun with different `--c-etc-log-coeff` (each run writes its own `--save-json` if desired).

---

## Notes on Metrics

- **Market stability** mirrors the blocking-pair definition in the paper §3: a pair `(p_i,a_j)` blocks only when **`µ_{i,j} > µ_{i,Ā_i}` strictly** **and** the arm **`π_{j,i} ≺ π_{j, partner}` strictly**. Tied `µ` or tied `π` must not invent a bogus strict order (`argsort` tie-breaking was removed for stability checks).

- Arms break ties among equally preferred proposers **uniformly at random**, per the paper §3 (`resolve_round`; each algorithm run passes its **own** `rng`), instead of deterministic lowest player index only.

- **Appendix Fig. 1 knobs** (see `configs/paper_default.json`): `aeags_confidence_factor` stays at the theoretical `6` in the AE-AGS radius (with `ln(T)` or `ln(t)` per `aeags_ucb_time_scale`). `c_etc_log_coeff` scales C-ETC pulls per directed pair as `coeff · ln(T)/Δ²` (see table above: **~4** theory-scale vs **~8.35** Fig.1(f) scale). `p_etc_explore_coef` scales phased exploration length. Offline GS commits apply tiny Gaussian perturbations to `μ̂` before player-proposing GS (Appendix B style ties). JSON may set `aeags_arm_schedule`, `reward_noise_mode`, `aeags_player_pull_tiebreak`, `aeags_ucb_time_scale`.

- **Stable regret (paper Eq. (1))** uses a per-player *regret reference* (not an algorithm baseline like C-ETC):
  \(\mu_{i,m_i}=\min_{\text{stable }m'}\mu_{i,m'(i)}\), computed in
  `MatchingMarket.stable_regret_reference_per_player`. Step regret is
  \(\mu_{i,m_i}-X_{i,A_i(t)}(t)\) on the matched arm outcome each round.
  Because \(\mu_{i,m_i}\) is the **worst** stable payoff for player \(i\),
  typical realized rewards \(X\) can sit **above** that benchmark for long stretches, so the
  **cumulative sum can decrease** and go **negative** — this is compatible with Fig. 1 in the paper
  (see the extracted axis ranges in the appendix) and does **not**, by itself, mean the run is invalid.
  Use `--rectify-regret 1` (`paper_clean`) for a nonnegative per-round “gap” view.

- **Reward noise (`--reward-noise-mode`).** Default **`shared`**: within one repeat, all algorithms share the same \(\mu\) matrix; matched rewards use a deterministic stream keyed by `(experiment_seed, t, i, a)` **plus** optional per-policy salt (`independent`). Two policies that realize the same `(t,i,a)` see the **same** draw iff `shared`; `independent` breaks cross-algorithm coupling for ablations. `resolve_round` tie-breaking still uses **each policy’s** RNG.

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
