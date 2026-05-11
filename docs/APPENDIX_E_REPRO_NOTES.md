# Appendix E six-panel reproduction — goals and commands

This note implements the triage from the internal plan: negative stable regret (panels a–e) versus panel (f) alignment, without editing the plan file.

## Goal decision (clarify-goal)

Three objectives **trade off**; pick one as primary when tuning:

| Primary goal | Preset / knobs | What you get | What you give up |
|--------------|----------------|--------------|------------------|
| C-ETC regret knee near ~15k in (a)–(e) | `paper_fig1_knee15k` (`c_etc_log_coeff≈2.5`) | Visual match for C-ETC regret shape | Panel (f) three-way layout vs PDF is **not** tuned; AE-AGS / P-ETC (f) can drift |
| Appendix-scale (f) for C-ETC | `paper_default` (`c_etc_log_coeff≈8.35`, AE theorem-style \(cf{=}6\)) | C-ETC unstability scale + paper Algorithm 3 coefficient in UCB/LCB | C-ETC regret knee not at ~15k; AE on (f) may be worse than funnel-tuned AE |
| AE-AGS tuned for Appendix E Fig. 1 interplay | **`paper_appendix_e_fig1`** (`c_etc_log_coeff≈8.35`, **`cf=5`**, **`round_sweep`**, **`smallest_arm`**, worst-stable regret) | Uses README **full‑horizon funnel** knobs in one preset | **`cf≠6`** — empirical alignment, not the theorem line in Algorithm 3 |

**Defaults in this repo:** `run_paper_default.sh` / `make paper-json` keep **`paper_fig1_knee15k`** (regret knee ~15k). For Appendix E \(\theta\) at \(\approx 8.35\): use **`paper_appendix_e_fig1`** (funnel AE) or **`paper_default`** (theorem AE); both set `c_etc_log_coeff` consistently for C‑ETC.

## Q1 — Why cumulative stable regret can be negative

- Reference per player \(i\): \(\mu_{i,m_i}=\min_{m'\text{ stable}}\mu_{i,m'(i)}\) — worst stable payoff (see `MatchingMarket.stable_regret_reference_per_player` in `ae_ags/market.py`).
- Per-round term: `reference - realized_reward`. High realized draws or good arms ⇒ negative steps ⇒ negative cumulative sums are **expected** under `rectify_regret=false`.
- Plots average over Monte Carlo **runs** only; there is **no** demeaning across algorithms in `paper_figure1.py`.

### Nonnegative panels (a)–(e) like many paper figures

If you want cumulative stable regret trajectories to stay **nonnegative** while keeping the same hyperparameters as the main presets (no reward clipping), use the built-in per-step rectifier `max(μ_ref - X, 0)`:

- **`--preset paper_default_rectified`** — same as `paper_default` except `rectify_regret=true`.
- **`--preset paper_fig1_knee15k_rectified`** — same as `paper_fig1_knee15k` except `rectify_regret=true`.
- Or **`--config configs/paper_default_rectified.json`** / **`configs/paper_fig1_knee15k_rectified.json`**.

This is **not** identical to the strict paper Eq. (1) sum; it is a reproducibility knob for figure shape. **`paper_clean`** turns on rectification **and** `clip_rewards`, which changes the reward process as well.

**Canonical paths (knee15k, rectified):** `results/paper_run/fig1_knee15k/one_run_curve_rectified.json` and `results/paper_run/fig1_knee15k/plots/figure1_sixpanels_rectified.png` — use `make paper-json-rectified` and `make paper-figure1-rectified`.

**Best-stable regret ablation (same folder):** `one_run_curve_best_stable_ref.json`, `figure1_sixpanels_best_stable_ref.png` — `make paper-json-bestref` / `make paper-figure1-bestref`. Paper Eq. (1) remains **worst** stable + **no** rectify; this is for exploratory figure fit only.

**Verify locally:**

```bash
cd /root/AE-AGS
python -m ae_ags.diagnostics_stable_regret --runs 2 --T 2000
```

Example output (illustrates negative **final** cumulative regret for some players while `rectify_regret` is off): `run_index=0` showed `stable_regret_reference_per_player` `[0.7, 1.0, 0.9, 0.8, 0.7]` and final AE-AGS cumulative stable regret approximately `[-522, 76, 35, -183, -247]` — consistent with `reference - reward` steps averaging below zero when realized payoffs sit above the worst-stable benchmark.

## Q2 — Scan (paper_default scale) and artifacts

Outputs from this implementation live under `results/paper_run/plan_appendix_e/`:

- `fig1_knob_scan_paper_default.txt` — table from `scan_fig1_knobs` (AE vs C-ETC cumulative unstability).
- `sixpanel_paper_default_best.json` — full `run_experiment` at `T=100000`, `runs=8` (same content as `sixpanel_paper_default_funnel_cf5.json`).
- `sixpanel_paper_default_funnel_cf5.json` — duplicate of the above (kept for an explicit filename).
- `plots/figure1_plan_best.png` — six-panel figure from `paper_figure1`.

### Scan outcome (short horizon caveat)

The saved scan uses `T=8000`, `runs=6`, `c_etc_log_coeff=8.35`. The row with the lowest AE unstability was **`aeags_confidence_factor=6`**, `pull_tiebreak=smallest_arm`, with both `pick_one` and `round_sweep` giving `AE_unst≈6395` vs `CETC_unst≈6933` (`AE-CETC≈-538`). **That ranking does not carry to `T=100000` with the same knobs:** a full run with `cf=6`, `round_sweep`, `smallest_arm` gave `AE_unst=51039` vs `CETC_unst=43263` (AE worse on panel (f)).

**Full-horizon choice (AE-AGS better than C-ETC on cumulative unstability):** README funnel knobs at paper scale: **`aeags_confidence_factor=5`**, **`aeags_algo2_outer_loop=round_sweep`**, **`aeags_player_pull_tiebreak=smallest_arm`**, **`aeags_arm_rank_jitter_scale=0`**. One replicate batch (`T=100000`, `runs=8`, `seed=0`) reported `cumulative_market_unstability`: **AE-AGS 43061**, **C-ETC 43263**, **P-ETC 89885**. This is **empirical knob tuning**, not the theorem confidence constant 6.

Regenerate scan (medium cost; **raise `--T`** before trusting row selection for long-horizon Fig.1(f)):

```bash
cd /root/AE-AGS
source scripts/parallel_defaults.sh 2>/dev/null || true
mkdir -p results/paper_run/plan_appendix_e/plots
python -m ae_ags.scan_fig1_knobs \
  --T 8000 --runs 6 --jobs 4 \
  --c-etc-log-coeff 8.35 \
  --confidence-factors 5,5.5,6 \
  --algo2-outer-loops pick_one,round_sweep \
  --pull-tiebreaks random,smallest_arm \
  --arm-rank-jitter-scales 0 \
  | tee results/paper_run/plan_appendix_e/fig1_knob_scan_paper_default.txt
```

**Reproduce the committed six-panel JSON + PNG** (funnel `cf=5` at full length):

```bash
python -m ae_ags.run_experiment --preset paper_default --jobs 4 --record-every 1000 \
  --T 100000 --runs 8 \
  --aeags-confidence-factor 5 --aeags-algo2-outer-loop round_sweep \
  --aeags-player-pull-tiebreak smallest_arm --aeags-arm-rank-jitter-scale 0 \
  --save-json results/paper_run/plan_appendix_e/sixpanel_paper_default_best.json
python -m ae_ags.paper_figure1 \
  --input-json results/paper_run/plan_appendix_e/sixpanel_paper_default_best.json \
  --output results/paper_run/plan_appendix_e/plots/figure1_plan_best.png
```

**Knee15k preset** (regret visual only; do not expect (f) to match appendix PDF):

```bash
python -m ae_ags.run_experiment --preset paper_fig1_knee15k --jobs 4 --record-every 1000 \
  --save-json results/paper_run/fig1_knee15k/one_run_curve.json
python -m ae_ags.paper_figure1 \
  --input-json results/paper_run/fig1_knee15k/one_run_curve.json \
  --output results/paper_run/fig1_knee15k/plots/figure1_sixpanels.png
```

## Full funnel (optional, longer)

```bash
./scripts/fig1_funnel_scan.sh
```

Writes `results/paper_run/fig1_funnel_scan_t15000.txt` (see script for grid).
