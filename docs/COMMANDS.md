# Command Handbook

All commands assume:

```bash
cd /root/AE-AGS
```

## A. Fast sanity check

```bash
python -m ae_ags.run_experiment --preset quick
```

## B. Paper default run

```bash
./run_paper_default.sh
```

Parallel:

```bash
./run_paper_default.sh --runs 20 --jobs 8
```

## C. Save trajectory JSON

```bash
python -m ae_ags.run_experiment \
  --preset paper_default \
  --runs 20 --jobs 8 \
  --record-every 1000 \
  --save-json results/paper_run/one_run_curve.json
```

## D. Plot from run JSON

```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/one_run_curve.json \
  --output-dir results/paper_run/plots
```

Appendix Figure 1 layout (panels (a)–(e) cumulative stable regret per player, (f) market unstability; AE-AGS / C-ETC / P-ETC only) requires a JSON produced by **current** `run_experiment` (includes `per_player_stable_regret_*` in each algorithm’s `curve`):

```bash
python -m ae_ags.plot_from_run_json \
  --input-json results/paper_run/one_run_curve.json \
  --output-dir results/paper_run/plots \
  --paper-figure1
# -> results/paper_run/plots/figure1_paper_sixpanels.png
```

Or: `python -m ae_ags.paper_figure1 --input-json results/paper_run/one_run_curve.json`

## E. Full Appendix E sweep

```bash
./run_appendix_e.sh
```

Custom output directory:

```bash
python -m ae_ags.sweep_appendix_e --T 100000 --runs 20 --jobs 8 --output-dir results/appendix_e_full
```

## F. Paper-clean (non-negative display)

```bash
python -m ae_ags.run_experiment --preset paper_clean --jobs 8
```

## H. Faster wall-clock runs

Parallel repeats (recommended):

```bash
python -m ae_ags.run_experiment --preset paper_default --config configs/paper_default.json --runs 20 --jobs 8
```

When using several workers, pin BLAS to one thread each to reduce oversubscription:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m ae_ags.run_experiment --preset paper_default --runs 20 --jobs 8
```

Optional alignment with Appendix E baselines (defaults are in `configs/paper_default.json`):

```bash
python -m ae_ags.run_experiment --preset paper_default \
  --c-etc-log-coeff 8.35 --p-etc-explore-coef 0.52 --aeags-confidence-factor 6
```

## G. Git workflow

```bash
git status --short --branch
git add .
git commit -m "update experiments and docs"
git push
```
