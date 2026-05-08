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

## One-Command Paper Default

Use the paper-style default scale (N=5, K=5, T=100000, runs=20):

```bash
cd /root/AE-AGS
./run_paper_default.sh
```

You can still override fields, for example:

```bash
./run_paper_default.sh --runs 5
```

## Config-Driven Run

```bash
python run_experiment.py --preset paper_default --config configs/paper_default.json
```

## Notes

- This is a **minimal reproduction scaffold** focused on correctness and executability.
- The baseline `C-ETC(simple)` is a simplified ETC-style implementation for comparison; it is not a line-by-line copy of the paper's exact baseline.
- Next step for strict paper-level reproduction is to add:
  - decentralized AE-AGS (Algorithm 4/5),
  - exact phase/communication protocol,
  - same plotting and statistical protocol as Appendix E.
