"""
Grid scan for Appendix Fig.1 knobs (AE-AGS vs C-ETC unstability).

Example (fast screen): ``--T 8000 --runs 8 --record-every 0``.

Default ``--c-etc-log-coeff`` is **2.5** (same as ``paper_fig1_knee15k``). For appendix Fig.1(f) C-ETC scale use ``--c-etc-log-coeff 8.35``. For heavier paper-scale grids add e.g. ``--T 100000 --runs 20``.

Also supports:
- Comma lists for --confidence-factors and --algo2-outer-loops (product with other axes).
- Optional --seed-list for a per-seed AE−CETC table (fixed: arm_schedule=fixed, pull_tiebreak=random,
  ucb=horizon; uses the first confidence factor and first outer-loop from the respective lists).
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
    aeags_algo2_outer_loop: str,
    aeags_arm_rank_jitter_scale: float,
    jobs: int,
) -> Dict[str, Dict[str, Any]]:
    aeags_arm_schedule = str(aeags_arm_schedule).lower().replace("-", "_")
    reward_noise_mode = str(reward_noise_mode).lower()
    aeags_player_pull_tiebreak = str(aeags_player_pull_tiebreak).lower().replace("-", "_")
    aeags_ucb_time_scale = str(aeags_ucb_time_scale).lower().replace("-", "_")
    aeags_algo2_outer_loop = str(aeags_algo2_outer_loop).lower().replace("-", "_")

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
                aeags_algo2_outer_loop,
                float(aeags_arm_rank_jitter_scale),
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
                    aeags_algo2_outer_loop,
                    float(aeags_arm_rank_jitter_scale),
                )
                for r in range(runs)
            ]
            for f in concurrent.futures.as_completed(futures):
                one = f.result()
                for name in ALG_NAMES:
                    agg[name].append(one[name])

    return _aggregate_results(agg)


def _split_csv_nums(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


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
    parser.add_argument(
        "--confidence-factors",
        type=str,
        default="6",
        help="Comma-separated aeags_confidence_factor values (theorem uses 6; other values are empirical knob search).",
    )
    parser.add_argument(
        "--c-etc-log-coeff",
        type=float,
        default=2.5,
        help="C-ETC log coeff (default 2.5 = paper_fig1_knee15k; use 8.35 for appendix Fig.1(f) scale).",
    )
    parser.add_argument("--p-etc-explore-coef", type=float, default=0.52)
    parser.add_argument("--reward-noise-mode", type=str, default="shared")
    parser.add_argument(
        "--arm-schedules",
        type=str,
        default="fixed",
        help="Comma-separated: fixed,random,round_robin (`round_sweep` mode ignores scheduling among simultaneous eligibles).",
    )
    parser.add_argument(
        "--algo2-outer-loops",
        type=str,
        default="pick_one,round_sweep",
        help="Comma-separated: pick_one, round_sweep (hyphens allowed).",
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
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help=(
            "Optional comma-separated seeds; prints a second table (AE_unst, CETC_unst, AE-CETC) using "
            "first entries of arm-schedules, pull-tiebreaks, ucb-time-scales, confidence-factors, "
            "algo2-outer-loops, and arm-rank-jitter-scales."
        ),
    )
    parser.add_argument(
        "--arm-rank-jitter-scales",
        type=str,
        default="0",
        help="Comma-separated aeags_arm_rank_jitter_scale (0 = deterministic arm propose order).",
    )
    args = parser.parse_args()

    schedules = _split_csv_nums(args.arm_schedules)
    tiebreaks = _split_csv_nums(args.pull_tiebreaks)
    ucbs = _split_csv_nums(args.ucb_time_scales)
    cfs_raw = _split_csv_nums(args.confidence_factors)
    confidence_factors = [float(x) for x in cfs_raw]
    outer_loops = [s.lower().replace("-", "_") for s in _split_csv_nums(args.algo2_outer_loops)]
    jitter_scales = [float(x) for x in _split_csv_nums(args.arm_rank_jitter_scales)]

    print(
        f"# scan N={args.N} K={args.K} T={args.T} runs={args.runs} seed={args.seed} "
        f"jobs={args.jobs} market={args.market_model} c_etc={args.c_etc_log_coeff}"
    )
    print("cf\touter\tjitter\tschedule\tpull_tb\tucb_ts\tAE_unst\tCETC_unst\tAE-CETC")

    for cf, outer, jitter, sched, ptb, uts in product(
        confidence_factors, outer_loops, jitter_scales, schedules, tiebreaks, ucbs
    ):
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
            aeags_confidence_factor=float(cf),
            c_etc_log_coeff=float(args.c_etc_log_coeff),
            p_etc_explore_coef=float(args.p_etc_explore_coef),
            aeags_arm_schedule=sched,
            reward_noise_mode=str(args.reward_noise_mode),
            aeags_player_pull_tiebreak=ptb,
            aeags_ucb_time_scale=uts,
            aeags_algo2_outer_loop=outer,
            aeags_arm_rank_jitter_scale=float(jitter),
            jobs=int(args.jobs),
        )
        ae_u = float(summary["AE-AGS"]["cumulative_market_unstability"])
        ce_u = float(summary["C-ETC"]["cumulative_market_unstability"])
        print(f"{cf}\t{outer}\t{jitter}\t{sched}\t{ptb}\t{uts}\t{ae_u:.1f}\t{ce_u:.1f}\t{ae_u - ce_u:.1f}")

    seed_list = _split_csv_nums(args.seed_list)
    if seed_list:
        cf0 = float(confidence_factors[0])
        outer0 = str(outer_loops[0])
        jitter0 = float(jitter_scales[0])
        sched0 = str(schedules[0])
        ptb0 = str(tiebreaks[0])
        uts0 = str(ucbs[0])
        print("")
        print(
            f"# seed sensitivity: sched={sched0} pull_tb={ptb0} ucb={uts0} cf={cf0} outer={outer0} "
            f"arm_rank_jitter={jitter0} (first list entries)"
        )
        print("seed\tAE_unst\tCETC_unst\tAE-CETC")
        for s in seed_list:
            sd = int(s)
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
                sd,
                int(args.runs),
                aeags_confidence_factor=cf0,
                c_etc_log_coeff=float(args.c_etc_log_coeff),
                p_etc_explore_coef=float(args.p_etc_explore_coef),
                aeags_arm_schedule=sched0,
                reward_noise_mode=str(args.reward_noise_mode),
                aeags_player_pull_tiebreak=ptb0,
                aeags_ucb_time_scale=uts0,
                aeags_algo2_outer_loop=outer0,
                aeags_arm_rank_jitter_scale=jitter0,
                jobs=int(args.jobs),
            )
            ae_u = float(summary["AE-AGS"]["cumulative_market_unstability"])
            ce_u = float(summary["C-ETC"]["cumulative_market_unstability"])
            print(f"{sd}\t{ae_u:.1f}\t{ce_u:.1f}\t{ae_u - ce_u:.1f}")


if __name__ == "__main__":
    main()
