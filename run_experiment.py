from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from aeags_centralized import AEAGSCentralized
from baselines import ExploreThenCommit, RandomMatchingPolicy
from market import MatchingMarket, make_random_market


@dataclass
class RunResult:
    stable_regret: np.ndarray  # [N]
    unstable_count: int


def run_policy(
    market: MatchingMarket,
    policy,
    horizon: int,
    baseline_reward: np.ndarray,
) -> RunResult:
    stable_regret = np.zeros(market.N, dtype=float)
    unstable_count = 0
    for _ in range(horizon):
        actions = policy.assign_actions(market.arm_rank)
        matched_arm, rewards = market.resolve_round(actions)
        policy.observe(matched_arm, rewards)

        # Stable regret definition used in paper.
        stable_regret += baseline_reward - rewards
        if not market.is_stable_matching(matched_arm):
            unstable_count += 1
    return RunResult(stable_regret=stable_regret, unstable_count=unstable_count)


def summarize(name: str, result: RunResult) -> None:
    sr = result.stable_regret
    print(f"[{name}]")
    print(f"  max cumulative stable regret: {sr.max():.2f}")
    print(f"  mean cumulative stable regret: {sr.mean():.2f}")
    print(f"  cumulative market unstability: {result.unstable_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=20000)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    alg_names = ["AE-AGS", "C-ETC(simple)", "Random"]
    agg: Dict[str, List[RunResult]] = {k: [] for k in alg_names}

    for r in range(args.runs):
        market = make_random_market(args.N, args.K, delta=args.delta, seed=args.seed + 1000 * r)
        baseline_reward = market.stable_baseline_reward()

        aeags = AEAGSCentralized(args.N, args.K, args.T, seed=args.seed + r)
        etc = ExploreThenCommit(args.N, args.K, explore_rounds=30, seed=args.seed + r)
        rnd = RandomMatchingPolicy(args.N, args.K, seed=args.seed + r)

        agg["AE-AGS"].append(run_policy(market, aeags, args.T, baseline_reward))
        agg["C-ETC(simple)"].append(run_policy(market, etc, args.T, baseline_reward))
        agg["Random"].append(run_policy(market, rnd, args.T, baseline_reward))

    print(f"Experiment: N={args.N}, K={args.K}, T={args.T}, delta={args.delta}, runs={args.runs}")
    for name in alg_names:
        mean_sr = np.mean([x.stable_regret for x in agg[name]], axis=0)
        mean_unstable = float(np.mean([x.unstable_count for x in agg[name]]))
        summarize(name, RunResult(stable_regret=mean_sr, unstable_count=int(mean_unstable)))


if __name__ == "__main__":
    main()
