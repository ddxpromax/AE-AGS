from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from aeags_centralized import AEAGSCentralized
from baselines import ExploreThenCommit, RandomMatchingPolicy
from market import MatchingMarket, make_random_market


@dataclass
class RunResult:
    stable_regret: np.ndarray  # [N]
    unstable_count: int


PRESETS = {
    "quick": {
        "N": 5,
        "K": 5,
        "T": 5000,
        "delta": 0.1,
        "sigma": 1.0,
        "clip_rewards": False,
        "rectify_regret": False,
        "seed": 0,
        "runs": 3,
    },
    "paper_default": {
        "N": 5,
        "K": 5,
        "T": 100000,
        "delta": 0.1,
        "sigma": 1.0,
        "clip_rewards": False,
        "rectify_regret": False,
        "seed": 0,
        "runs": 20,
    },
    "paper_clean": {
        "N": 5,
        "K": 5,
        "T": 100000,
        "delta": 0.1,
        "sigma": 1.0,
        "clip_rewards": True,
        "rectify_regret": True,
        "seed": 0,
        "runs": 20,
    },
}


def _load_json_config(path: str | None) -> Dict[str, float | int]:
    if not path:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")
    return data


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    # Precedence: CLI explicit value > config > preset > parser defaults.
    parser_defaults = {
        "N": 5,
        "K": 5,
        "T": 20000,
        "delta": 0.1,
        "sigma": 1.0,
        "clip_rewards": False,
        "rectify_regret": False,
        "seed": 0,
        "runs": 3,
        "jobs": 1,
    }

    preset_vals = PRESETS.get(args.preset, {})
    cfg_vals = _load_json_config(args.config)

    resolved = {}
    for key in parser_defaults:
        cli_value = getattr(args, key)
        if cli_value is not None:
            resolved[key] = cli_value
        elif key in cfg_vals:
            resolved[key] = cfg_vals[key]
        elif key in preset_vals:
            resolved[key] = preset_vals[key]
        else:
            resolved[key] = parser_defaults[key]

    for k, v in resolved.items():
        setattr(args, k, v)
    return args


def run_policy(
    market: MatchingMarket,
    policy,
    horizon: int,
    baseline_reward: np.ndarray,
    rectify_regret: bool,
) -> RunResult:
    stable_regret = np.zeros(market.N, dtype=float)
    unstable_count = 0
    for _ in range(horizon):
        actions = policy.assign_actions(market.arm_rank)
        matched_arm, rewards = market.resolve_round(actions)
        policy.observe(matched_arm, rewards)

        # Paper-aligned raw metric: baseline - sampled reward.
        # Optional rectify mode is retained for engineering-style reporting.
        step_regret = baseline_reward - rewards
        if rectify_regret:
            step_regret = np.maximum(step_regret, 0.0)
        stable_regret += step_regret
        if not market.is_stable_matching(matched_arm):
            unstable_count += 1
    return RunResult(stable_regret=stable_regret, unstable_count=unstable_count)


def summarize(name: str, result: RunResult) -> None:
    sr = result.stable_regret
    print(f"[{name}]")
    print(f"  max cumulative stable regret: {sr.max():.2f}")
    print(f"  mean cumulative stable regret: {sr.mean():.2f}")
    print(f"  cumulative market unstability: {result.unstable_count}")


def run_one_repeat(
    n_players: int,
    n_arms: int,
    horizon: int,
    delta: float,
    sigma: float,
    clip_rewards: bool,
    rectify_regret: bool,
    seed: int,
    run_index: int,
) -> Dict[str, RunResult]:
    market = make_random_market(
        n_players,
        n_arms,
        delta=delta,
        sigma=sigma,
        clip_rewards=clip_rewards,
        seed=seed + 1000 * run_index,
    )
    baseline_reward = market.stable_baseline_reward()

    aeags = AEAGSCentralized(n_players, n_arms, horizon, seed=seed + run_index)
    etc = ExploreThenCommit(n_players, n_arms, explore_rounds=30, seed=seed + run_index)
    rnd = RandomMatchingPolicy(n_players, n_arms, seed=seed + run_index)

    return {
        "AE-AGS": run_policy(market, aeags, horizon, baseline_reward, rectify_regret),
        "C-ETC(simple)": run_policy(market, etc, horizon, baseline_reward, rectify_regret),
        "Random": run_policy(market, rnd, horizon, baseline_reward, rectify_regret),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, choices=sorted(PRESETS.keys()), default="quick")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian noise std.")
    parser.add_argument(
        "--clip-rewards",
        type=int,
        choices=[0, 1],
        default=None,
        help="Clip observed rewards to [0,1] (1=yes, 0=no).",
    )
    parser.add_argument(
        "--rectify-regret",
        type=int,
        choices=[0, 1],
        default=None,
        help="Clamp per-step regret to non-negative (1=yes, 0=no).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None, help="Parallel worker processes for runs.")
    args = _resolve_args(parser.parse_args())

    alg_names = ["AE-AGS", "C-ETC(simple)", "Random"]
    agg: Dict[str, List[RunResult]] = {k: [] for k in alg_names}

    jobs = max(1, int(args.jobs))
    clip_rewards = bool(int(args.clip_rewards))
    rectify_regret = bool(int(args.rectify_regret))
    if jobs == 1:
        for r in range(args.runs):
            one = run_one_repeat(
                args.N,
                args.K,
                args.T,
                args.delta,
                args.sigma,
                clip_rewards,
                rectify_regret,
                args.seed,
                r,
            )
            for name in alg_names:
                agg[name].append(one[name])
    else:
        max_workers = min(jobs, args.runs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    run_one_repeat,
                    args.N,
                    args.K,
                    args.T,
                    args.delta,
                    args.sigma,
                    clip_rewards,
                    rectify_regret,
                    args.seed,
                    r,
                )
                for r in range(args.runs)
            ]
            for f in concurrent.futures.as_completed(futures):
                one = f.result()
                for name in alg_names:
                    agg[name].append(one[name])

    print(
        f"Experiment: N={args.N}, K={args.K}, T={args.T}, delta={args.delta}, sigma={args.sigma}, "
        f"clip_rewards={int(clip_rewards)}, rectify_regret={int(rectify_regret)}, runs={args.runs}, jobs={jobs}"
    )
    for name in alg_names:
        mean_sr = np.mean([x.stable_regret for x in agg[name]], axis=0)
        mean_unstable = float(np.mean([x.unstable_count for x in agg[name]]))
        summarize(name, RunResult(stable_regret=mean_sr, unstable_count=int(mean_unstable)))


if __name__ == "__main__":
    main()
