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

## G. Git workflow

```bash
git status --short --branch
git add .
git commit -m "update experiments and docs"
git push
```
