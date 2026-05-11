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
- `ae_ags/test_aeags_assign_order.py`: Algorithm 3 observe semantics + optional arm-rank jitter tie-break smoke (`python -m ae_ags.test_aeags_assign_order`).

### Config / scripts
- `configs/paper_fig1_knee15k.json`: **default** single-setting paper run (`run_paper_default.sh`, `make paper-j8`): C-ETC `log_coeff` smaller so the regret knee is near \(\sim 10^4\)–\(2\times10^4\) rounds (Appendix Fig. 1 (a)–(e) visual default).
- `configs/paper_default.json`: **appendix (f) scale** preset (`c_etc_log_coeff≈8.35`); use when you want C-ETC cumulative unstability aligned to the PDF’s panel (f) rather than the knee-15k regret layout.
- `run_paper_default.sh`: one-command run using **`paper_fig1_knee15k`** (override with `--preset` / `--config` if needed).
- `run_appendix_e.sh`: one-command Appendix E sweep run.
- [`scripts/fig1_funnel_scan.sh`](scripts/fig1_funnel_scan.sh): medium-\(T\) Cartesian scan for Fig. 1(f) knobs (writes [`results/paper_run/fig1_funnel_scan_t15000.txt`](results/paper_run/fig1_funnel_scan_t15000.txt)).

### Outputs
- `results/paper_run/`: paper default run outputs (older Fig. 1 PNGs also live under `results/paper_run/plots/`).
- [`results/paper_run/fig1_knee15k/plots/`](results/paper_run/fig1_knee15k/plots/): **default** location for Appendix Fig. 1 six-panels: **paper-style signed regret** (`stable_regret_reference=worst`, `rectify_regret=0`) from `run_paper_default.sh` / `make paper-json` → `one_run_curve.json` + `figure1_sixpanels.png`; **rectified display** → `one_run_curve_rectified.json` + `figure1_sixpanels_rectified.png` (`make paper-json-rectified` / `make paper-figure1-rectified`); **best-stable benchmark ablation** → `one_run_curve_best_stable_ref.json` + `figure1_sixpanels_best_stable_ref.png` (`make paper-json-bestref` / `make paper-figure1-bestref`). See [`docs/COMMANDS.md`](docs/COMMANDS.md).
- `results/appendix_e_full/`: Appendix E sweep outputs.

**Algorithm 2 `A_i` fix (incumbent tentative match included when comparing proposals):** after this correction, Fig. 1 curves at **`paper_default`** with **theorem-scale** `aeags_confidence_factor=6`, `pick_one`, `pull_tiebreak=random` still show **AE-AGS cumulative unstability \(\approx 5.0\times 10^4\)** vs **C-ETC \(\approx 4.3\times 10^4\)** (`appendix_e_fig1_pick_one.json` / `figure1_appendix_e_pick_one.png`). A **two-stage funnel** on medium \(T\) (see `results/paper_run/fig1_funnel_scan_t15000.txt`) picked **`aeags_confidence_factor=5`**, **`round_sweep`**, **`smallest_arm`**, **`aeags_arm_rank_jitter_scale=0`**: at full **`paper_default`** length this yields **AE-AGS \(\approx 4.22\times 10^4\)** vs **C-ETC \(\approx 4.33\times 10^4\)** (`appendix_e_fig1_funnel_best_cf5_rs_sm.json`, `figure1_appendix_e_funnel_best_cf5_rs_sm.png`). That **\(5\neq 6\)** combination is **empirical Fig. 1 alignment**, not the theorem’s confidence constant. For **theorem \(6\)** with the same outer/tie-break but full \(T\), see `appendix_e_fig1_cf6_round_sweep_smallest_arm.json`. Older JSON from before the `A_i` change should not be cited as current code.

Other archived knobs/sweeps: `one_run_curve_appendix_e_round_robin.json`, `one_run_curve_appendix_e_random.json` (~\(6.9\times 10^4\) AE-AGS, avoid for repro), etc., under `results/paper_run/`; prefer **`fixed`** arm scheduling over **`random`**. Per-seed spread for the funnel-best knobs (short \(T\)): `results/paper_run/fig1_seed_table_funnel_best_t8k.txt`.

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

### 1) Paper default (Fig. 1 knee ~15k: `paper_fig1_knee15k`)
```bash
./run_paper_default.sh
```

With explicit parallelism:
```bash
./run_paper_default.sh --runs 20 --jobs 8
```

For **appendix (f) C-ETC scale** (`c_etc_log_coeff≈8.35`) instead:
```bash
python -m ae_ags.run_experiment --preset paper_default --config configs/paper_default.json --runs 20 --jobs 8
```

### 2) Save trajectory JSON for plotting
```bash
mkdir -p results/paper_run/fig1_knee15k
python -m ae_ags.run_experiment \
  --preset paper_fig1_knee15k --config configs/paper_fig1_knee15k.json \
  --runs 20 --jobs 8 \
  --record-every 1000 \
  --save-json results/paper_run/fig1_knee15k/one_run_curve.json
```

### 3) Plot from saved JSON
```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/fig1_knee15k/one_run_curve.json \
  --output-dir results/paper_run/fig1_knee15k/plots
```

Appendix **Figure 1** (six panels: regret for \(p_1,\ldots,p_5\) plus cumulative market unstability; three algorithms):

```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/fig1_knee15k/one_run_curve.json \
  --output-dir results/paper_run/fig1_knee15k/plots \
  --paper-figure1
# `make paper-figure1` if JSON already exists under results/paper_run/fig1_knee15k/
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

Implementation-side (already in code): **`is_stable_matching` reuses cached player inverse ranks**, **`Better` updates are vectorized**, **`log(T)` precomputed**, and **AE-AGS uses a fixed arm→player propose order** from the market when **`--aeags-arm-rank-jitter-scale 0`** (default); a positive scale **re-sorts `arm_rank` with tiny Gaussian jitter each round** (Appendix B–style arm-side tie-break).

### Appendix Fig. 1 reproduction toggles (`run_experiment`)

- **`--aeags-arm-schedule`**: Algorithm 2 chooses which **unmatched arm** proposes when several are eligible. Options: **`fixed`** (lowest arm index; default presets—closest to deterministic pseudo-code implementations), **`round_robin`** (cyclic scan), **`random`** (legacy uniform choice). Compare schedules if AE-AGS vs C-ETC **unstability ordering** disagrees with the PDF.
- **`--reward-noise-mode`**: **`shared`** (default): deterministic Gaussian keyed by `(seed, t, i, a)`, identical across algorithms when two policies realize the same `(t,i,a)`; **`independent`**: separate salt per algorithm for ablations against the appendix wording “drawn independently” (does not change fairness of *relative* rankings unless you rely on coupling).
- **`--aeags-player-pull-tiebreak`**: when several arms tie on **minimum** pull count in Algorithm 2 line 6, **`random`** (default) breaks ties uniformly; **`smallest_arm`** picks the lowest arm index (deterministic, closer to some reference scripts).
- **`--aeags-ucb-time-scale`**: **`horizon`** (default) uses \(\ln(T)\) in the UCB/LCB radius as in the paper; **`elapsed`** uses \(\ln(t)\) at the current round—**empirical only**, not the theorem statement.
- **`--aeags-algo2-outer-loop`**: **`pick_one`** (default): when several unmatched arms could propose in one outer iteration, pick one via **`--aeags-arm-schedule`**. **`round_sweep`**: repeat full sweeps over arm indices \(j=0..K-1\); each unmatched arm takes at most one propose step per sweep (order fixed by index; **`arm_schedule` does not reorder simultaneous eligibles** in this mode).
- **`--aeags-arm-rank-jitter-scale`**: **`0`** (default) uses the cached deterministic propose order. **`>0`** adds i.i.d. Gaussian noise (scaled by this factor) to integer **`arm_rank`** entries before sorting **each round**—closer in spirit to the offline GS tie-break in [`gs_commit_matching_from_mu_hat`](ae_ags/baselines.py), at the cost of deviating from a fixed proposal list.

Knob scan (same seed unless `--seed-list` is set; prints `AE_unst`, `CETC_unst`, `AE−CETC`; comma lists for **`--confidence-factors`**, **`--algo2-outer-loops`**, **`--arm-rank-jitter-scales`**; **`--seed-list`** uses the **first** entry of each list for the per-seed table):

```bash
python -m ae_ags.scan_fig1_knobs --T 8000 --runs 8 --jobs 4 \
  --confidence-factors 6 \
  --algo2-outer-loops pick_one,round_sweep \
  --arm-schedules fixed,random,round_robin \
  --pull-tiebreaks random,smallest_arm \
  --ucb-time-scales horizon \
  --arm-rank-jitter-scales 0,1e-9 \
  --seed-list 0,1,2,3,4,5,6,7,8,9
```

Omit **`--seed-list`** if you only need the Cartesian grid over the other knobs. Reproduce the recorded funnel grid via **`./scripts/fig1_funnel_scan.sh`**.

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

- **Appendix Fig. 1 knobs** (defaults: `configs/paper_fig1_knee15k.json`; for C-ETC (f) scale see `configs/paper_default.json`): the **theorem** AE-AGS radius uses **`aeags_confidence_factor = 6`** with **`ln(T)`** when `aeags_ucb_time_scale=horizon`. For **figures / scans**, [`scan_fig1_knobs.py`](ae_ags/scan_fig1_knobs.py) accepts a **`confidence-factors` list**: values \(\neq 6\) are explicitly **empirical knob search**, not the stated bound constants. **`c_etc_log_coeff`** scales C-ETC pulls per directed pair as `coeff · ln(T)/Δ²` (see table above: **~4** theory-scale vs **~8.35** Fig.1(f) scale). **`p_etc_explore_coef`** scales phased exploration length. Offline GS commits apply tiny Gaussian perturbations to `μ̂` before player-proposing GS (Appendix B style ties). JSON may set `aeags_arm_schedule`, `reward_noise_mode`, `aeags_player_pull_tiebreak`, `aeags_ucb_time_scale`, **`aeags_algo2_outer_loop`**, **`aeags_arm_rank_jitter_scale`**.

- **Stable regret (paper Eq. (1))** uses a per-player *regret reference* (not an algorithm baseline like C-ETC):
  \(\mu_{i,m_i}=\min_{\text{stable }m'}\mu_{i,m'(i)}\), computed in
  `MatchingMarket.stable_regret_reference_per_player`. Step regret is
  \(\mu_{i,m_i}-X_{i,A_i(t)}(t)\) on the matched arm outcome each round.
  Because \(\mu_{i,m_i}\) is the **worst** stable payoff for player \(i\),
  typical realized rewards \(X\) can sit **above** that benchmark for long stretches, so the
  **cumulative sum can decrease** and go **negative** — this is compatible with Fig. 1 in the paper
  (see the extracted axis ranges in the appendix) and does **not**, by itself, mean the run is invalid.
  Use `--rectify-regret 1` for a nonnegative per-round “gap” view. Preset **`paper_clean`** also enables **`clip_rewards`** (stronger deviation from raw sampling). For Appendix Fig. 1–style curves with the **same** hyperparameters as `paper_default` or `paper_fig1_knee15k` but **only** nonnegative cumulative regret in panels (a)–(e), use **`paper_default_rectified`** / **`paper_fig1_knee15k_rectified`**, or `--config configs/paper_default_rectified.json` (equivalently `--rectify-regret 1` on top of those presets).

- **Canonical paper-style runs** keep **`rectify_regret=0`** and **`stable_regret_reference=worst`** (default): cumulative regret follows the signed sum in Eq. (1) without discarding negative per-step contributions.

- **Ablation — best stable benchmark** (not in the paper’s regret definition): `stable_regret_reference=best` uses \(\max_{\text{stable }m'}\mu_{i,m'(i)}\) per player. Preset **`paper_fig1_knee15k_best_stable_ref`**, config [`configs/paper_fig1_knee15k_best_stable_ref.json`](configs/paper_fig1_knee15k_best_stable_ref.json), or CLI **`--stable-regret-reference best`**. Use only to test whether Appendix Fig. 1 (a)–(e) shapes line up better; do not conflate with theorem statements.

- **Reward noise (`--reward-noise-mode`).** Default **`shared`**: within one repeat, all algorithms share the same \(\mu\) matrix; matched rewards use a deterministic stream keyed by `(experiment_seed, t, i, a)` **plus** optional per-policy salt (`independent`). Two policies that realize the same `(t,i,a)` see the **same** draw iff `shared`; `independent` breaks cross-algorithm coupling for ablations. `resolve_round` tie-breaking still uses **each policy’s** RNG.

- AE-AGS empirical means \(T_{i,j}\) update only when the player was **actually matched**
  to the arm the platform assigned (Algorithm 3, lines 7–9).

- `paper_default` uses raw paper-style settings:
  - `clip_rewards=0`
  - `rectify_regret=0`
- Negative sampled rewards / cumulative regrets can therefore appear and are expected.
- For an engineering-style nonnegative cumulative regret (panels (a)–(e)) without reward clipping (default knee15k layout; artifacts under `fig1_knee15k/plots/`):
```bash
make paper-json-rectified
make paper-figure1-rectified
# -> results/paper_run/fig1_knee15k/one_run_curve_rectified.json
# -> results/paper_run/fig1_knee15k/plots/figure1_sixpanels_rectified.png
```
- `paper_clean` additionally clips rewards:
```bash
python -m ae_ags.run_experiment --preset paper_clean
```

---

## Documentation

- `docs/PROJECT_MAP.md`: architecture and file responsibilities.
- `docs/COMMANDS.md`: copy-paste command handbook.
- `Makefile`: short aliases (`make quick`, `make paper-j8`, `make paper-json-rectified`, `make paper-json-bestref`, `make sweep`).

---

## Current Scope

- Baselines are practical reproductions designed to be closer to paper behavior.
- This is not yet a line-by-line full reimplementation of every paper baseline detail.
- Future work:
  - decentralized AE-AGS (Algorithm 4/5),
  - deeper protocol-level alignment for all baselines,
  - additional reproducibility tooling (seeded manifests / run registry).
