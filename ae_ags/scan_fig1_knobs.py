"""
Grid scan for Appendix Fig.1 knobs (AE-AGS vs C-ETC unstability).

Example (fast screen): --T 8000 --runs 8 --record-every 0
Paper-scale check: inherit from paper_default and pass --T 100000 --runs 20 ...
"""

from __future__ import annotations

import argparse
from itertools import product
from typing import Any, Dict, List

from .run_experiment import _aggregate_results, run_one_repeat

ALG_NAMES = ["AE-AGS", "C-ETC", "P-ETC", "Random"]


def _run_aggregate(
    n_players: int,
    n_arms: int,
    horizon: int,
    delta: float,
    sigma: float,
    clip_rewards: bool,
    rectify_regret: bool,
    market_model: str,
    record_every: int,
    seed: int,
    runs: int,
    *,
    aeags_confidence_factor: float,
    c_etc_log_coeff: float,
    p_etc_explore_coef: float,
    aeags_arm_schedule: str,
    reward_noise_mode: str,
    aeags_player_pull_tiebreak: str,
    aeags_ucb_time_scale: str,
    jobs: int,
) -> Dict[str, Dict[str, Any]]:
    aeags_arm_schedule = str(aeags_arm_schedule).lower().replace("-", "_")
    reward_noise_mode = str(reward_noise_mode).lower()
    aeags_player_pull_tiebreak = str(aeags_player_pull_tiebreak).lower().replace("-", "_")
    aeags_ucb_time_scale = str(aeags_ucb_time_scale).lower().replace("-", "_")

    agg: Dict[str, List[Any]] = {k: [] for k in ALG_NAMES}
    if jobs <= 1:
        for r in range(runs):
            one = run_one_repeat(
                n_players,
                n_arms,
                horizon,
                delta,
                sigma,
                clip_rewards,
                rectify_regret,
                market_model,
                record_every,
                seed,
                r,
                float(aeags_confidence_factor),
                float(c_etc_log_coeff),
                float(p_etc_explore_coef),
                aeags_arm_schedule,
                reward_noise_mode,
                aeags_player_pull_tiebreak,
                aeags_ucb_time_scale,
            )
            for name in ALG_NAMES:
                agg[name].append(one[name])
    else:
        import concurrent.futures

        max_workers = min(int(jobs), int(runs))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    run_one_repeat,
                    n_players,
                    n_arms,
                    horizon,
                    delta,
                    sigma,
                    clip_rewards,
                    rectify_regret,
                    market_model,
                    record_every,
                    seed,
                    r,
                    float(aeags_confidence_factor),
                    float(c_etc_log_coeff),
                    float(p_etc_explore_coef),
                    aeags_arm_schedule,
                    reward_noise_mode,
                    aeags_player_pull_tiebreak,
                    aeags_ucb_time_scale,
                )
                for r in range(runs)
            ]
            for f in concurrent.futures.as_completed(futures):
                one = f.result()
                for name in ALG_NAMES:
                    agg[name].append(one[name])

    return _aggregate_results(agg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan AE-AGS knobs vs C-ETC cumulative unstability.")
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=8000)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--market-model", type=str, default="paper_rank")
    parser.add_argument("--aeags-confidence-factor", type=float, default=6.0)
    parser.add_argument("--c-etc-log-coeff", type=float, default=8.35)
    parser.add_argument("--p-etc-explore-coef", type=float, default=0.52)
    parser.add_argument("--reward-noise-mode", type=str, default="shared")
    parser.add_argument(
        "--arm-schedules",
        type=str,
        default="fixed,random,round_robin",
        help="Comma-separated: fixed,random,round_robin",
    )
    parser.add_argument(
        "--pull-tiebreaks",
        type=str,
        default="random,smallest_arm",
        help="Comma-separated: random,smallest_arm",
    )
    parser.add_argument(
        "--ucb-time-scales",
        type=str,
        default="horizon",
        help="Comma-separated: horizon,elapsed",
    )
    args = parser.parse_args()

    schedules = [s.strip() for s in args.arm_schedules.split(",") if s.strip()]
    tiebreaks = [s.strip() for s in args.pull_tiebreaks.split(",") if s.strip()]
    ucbs = [s.strip() for s in args.ucb_time_scales.split(",") if s.strip()]

    print(
        f"# scan N={args.N} K={args.K} T={args.T} runs={args.runs} seed={args.seed} "
        f"jobs={args.jobs} market={args.market_model} c_etc={args.c_etc_log_coeff}"
    )
    print("schedule\tpull_tb\tucb_ts\tAE_unst\tCETC_unst\tAE-CETC")

    for sched, ptb, uts in product(schedules, tiebreaks, ucbs):
        summary = _run_aggregate(
            args.N,
            args.K,
            args.T,
            args.delta,
            args.sigma,
            False,
            False,
            args.market_model,
            int(args.record_every),
            int(args.seed),
            int(args.runs),
            aeags_confidence_factor=float(args.aeags_confidence_factor),
            c_etc_log_coeff=float(args.c_etc_log_coeff),
            p_etc_explore_coef=float(args.p_etc_explore_coef),
            aeags_arm_schedule=sched,
            reward_noise_mode=str(args.reward_noise_mode),
            aeags_player_pull_tiebreak=ptb,
            aeags_ucb_time_scale=uts,
            jobs=int(args.jobs),
        )
        ae_u = float(summary["AE-AGS"]["cumulative_market_unstability"])
        ce_u = float(summary["C-ETC"]["cumulative_market_unstability"])
        print(f"{sched}\t{ptb}\t{uts}\t{ae_u:.1f}\t{ce_u:.1f}\t{ae_u - ce_u:.1f}")


if __name__ == "__main__":
    main()
