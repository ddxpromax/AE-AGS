"""
Print stable-regret reference vs realized rewards for a few rounds (Appendix E sanity).

Demonstrates that cumulative stable regret can go negative: reference is the worst
stable payoff per player; realized Gaussian rewards can exceed it often.
"""

from __future__ import annotations

import argparse

import numpy as np

from .aeags_centralized import AEAGSCentralized
from .market import make_random_market
from .run_experiment import run_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Stable regret reference vs rewards (diagnostic).")
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--T", type=int, default=500)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=int, default=1, help="How many market draws (run_index) to print.")
    p.add_argument("--rectify-regret", action="store_true")
    p.add_argument(
        "--stable-regret-reference",
        type=str,
        choices=["worst", "best"],
        default="worst",
        help='Benchmark: "worst" = paper Eq. (1); "best" = max stable payoff (ablation).',
    )
    args = p.parse_args()

    for run_index in range(int(args.runs)):
        market = make_random_market(
            args.N,
            args.K,
            delta=float(args.delta),
            sigma=float(args.sigma),
            clip_rewards=False,
            model="paper_rank",
            seed=int(args.seed) + 1000 * run_index,
        )
        rng_ref = np.random.default_rng(int(args.seed) + 424242 + run_index)
        ref = market.stable_regret_reference_per_player(
            rng=rng_ref, reference=str(args.stable_regret_reference).lower()
        )
        print(f"\n=== run_index={run_index} market seed={int(args.seed) + 1000 * run_index} ===")
        print("mu row min / max:", float(market.mu.min()), float(market.mu.max()))
        print(f"stable_regret_reference_per_player ({args.stable_regret_reference} stable μ per player):", ref)

        policy = AEAGSCentralized(
            args.N,
            args.K,
            int(args.T),
            seed=int(args.seed) + run_index,
            market=market,
            confidence_factor=6.0,
            arm_schedule="fixed",
            player_pull_tiebreak="random",
            ucb_time_scale="horizon",
            algo2_outer_loop="pick_one",
            arm_rank_jitter_scale=0.0,
        )
        res = run_policy(
            market,
            policy,
            int(args.T),
            ref,
            bool(args.rectify_regret),
            record_every=max(1, int(args.T) // 5),
            seed=int(args.seed) + 11,
            reward_experiment_seed=int(args.seed) + 1_001_311 * int(run_index),
        )
        print("final cumulative stable regret per player:", res.stable_regret)
        print("rectify_regret:", bool(args.rectify_regret))


if __name__ == "__main__":
    main()
