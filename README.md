# AE-AGS Reproduction (Minimal)

This folder contains a minimal, runnable reproduction of the core AE-AGS idea from:

> *Bandit Learning in Matching Markets with Indifference* (ICLR 2025)

## Files

- `market.py`: matching market simulator with indifference and stability checker.
- `aeags_centralized.py`: centralized AE-AGS core logic (Algorithm 1/2/3 style).
- `baselines.py`: simple baselines (`ExploreThenCommit`, `RandomMatchingPolicy`).
- `run_experiment.py`: run toy experiments and print cumulative stable regret / unstability.

## Quick Start

```bash
cd /root/AE-AGS
python run_experiment.py --preset quick
```

## Terminal Command Cheat Sheet

Copy-paste commands (no need to memorize parameters):

```bash
# 1) Quick sanity run (small/fast)
cd /root/AE-AGS
python run_experiment.py --preset quick

# 2) Paper default (raw, strict setting)
./run_paper_default.sh

# 3) Paper default + parallel workers (faster)
./run_paper_default.sh --runs 20 --jobs 8

# 4) Non-negative display mode (engineering view)
python run_experiment.py --preset paper_clean --jobs 8

# 5) Run Appendix-E sweeps (delta + market-size)
./run_appendix_e.sh

# 6) Save one run to JSON file
python run_experiment.py --preset paper_default --jobs 8 --save-json results/one_run.json

# 7) Git push current branch
git add .
git commit -m "update experiments"
git push
```

## One-Command Paper Default (Raw, Paper-Aligned)

Use the paper-style default scale (N=5, K=5, T=100000, runs=20):

```bash
cd /root/AE-AGS
./run_paper_default.sh
```

You can still override fields, for example:

```bash
./run_paper_default.sh --runs 5
```

Run in parallel across independent repeats:

```bash
./run_paper_default.sh --runs 20 --jobs 8
```

## Config-Driven Run

```bash
python run_experiment.py --preset paper_default --config configs/paper_default.json
```

## Appendix E Sweeps (One Command)

Run both sweeps from Appendix E:

```bash
cd /root/AE-AGS
./run_appendix_e.sh
```

Artifacts are saved under `results/`:
- `appendix_e_sweeps.json`
- `delta_sweep_max_stable_regret.png`
- `delta_sweep_unstability.png`
- `size_sweep_max_stable_regret.png`
- `size_sweep_unstability.png`

## About Negative Rewards / Regrets

By default (`paper_default`), this repo now uses paper-aligned raw settings:

- `clip_rewards=0`
- `rectify_regret=0`

So negative sampled rewards / cumulative regrets can appear and are expected.

If you want a cleaned-up non-negative reporting style, use:

```bash
python run_experiment.py --preset paper_clean
```

## Notes

- This is a **minimal reproduction scaffold** focused on correctness and executability.
- The baseline `C-ETC(simple)` is a simplified ETC-style implementation for comparison; it is not a line-by-line copy of the paper's exact baseline.
- Next step for strict paper-level reproduction is to add:
  - decentralized AE-AGS (Algorithm 4/5),
  - exact phase/communication protocol,
  - same plotting and statistical protocol as Appendix E.
