from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .aeags_centralized import AEAGSCentralized
from .baselines import CETCKnownDelta, PhasedETC, RandomMatchingPolicy
from .market import MatchingMarket, make_random_market


@dataclass
class RunResult:
    stable_regret: np.ndarray
    unstable_count: int
    curve_steps: np.ndarray | None = None
    curve_player_regret: np.ndarray | None = None
    curve_unstable: np.ndarray | None = None


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
        "record_every": 500,
        "market_model": "paper_rank",
        "aeags_confidence_factor": 6.0,
        "c_etc_log_coeff": 2.0,
        "p_etc_explore_coef": 0.5,
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
        "record_every": 1000,
        "market_model": "paper_rank",
        # Appendix E-style baselines (Gaussian tie-break before offline GS; resolve_round tie-break is random).
        # C-ETC exploratory length ~ theta*ln(T)/Delta^2 per pair. Theorem-scale theta~4 is conservative;
        # theta~8.35 aligns 8-run means near Appendix Fig.1(f) cumulative unstability (~43% unstable rounds for C-ETC at T=100k).
        "aeags_confidence_factor": 6.0,
        "c_etc_log_coeff": 8.35,
        "p_etc_explore_coef": 0.52,
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
        "record_every": 1000,
        "market_model": "paper_rank",
        "aeags_confidence_factor": 6.0,
        "c_etc_log_coeff": 8.35,
        "p_etc_explore_coef": 0.52,
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
        "record_every": 0,
        "market_model": "paper_rank",
        "aeags_confidence_factor": 6.0,
        "c_etc_log_coeff": 4.0,
        "p_etc_explore_coef": 0.52,
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
    regret_reference_mu: np.ndarray,
    rectify_regret: bool,
    record_every: int = 0,
    seed: int = 0,
) -> RunResult:
    rng = np.random.default_rng(seed)
    stable_regret = np.zeros(market.mu.shape[0], dtype=float)
    unstable_count = 0
    steps: List[int] = []
    curve_regret: List[np.ndarray] = []
    curve_unstable: List[int] = []

    for t in range(1, horizon + 1):
        actions = policy.assign_actions(market.arm_rank)
        matched_arm, rewards = market.resolve_round(actions, rng)
        policy.observe(actions, matched_arm, rewards)

        step_regret = regret_reference_mu - rewards
        if rectify_regret:
            step_regret = np.maximum(step_regret, 0.0)
        stable_regret += step_regret
        if not market.is_stable_matching(matched_arm):
            unstable_count += 1
        if record_every > 0 and (t % record_every == 0 or t == horizon):
            steps.append(t)
            curve_regret.append(stable_regret.copy())
            curve_unstable.append(unstable_count)

    return RunResult(
        stable_regret=stable_regret,
        unstable_count=unstable_count,
        curve_steps=np.array(steps, dtype=int) if steps else None,
        curve_player_regret=np.stack(curve_regret, axis=0) if curve_regret else None,
        curve_unstable=np.array(curve_unstable, dtype=float) if curve_unstable else None,
    )


def summarize(name: str, result: RunResult) -> None:
    sr = result.stable_regret
    print(f"[{name}]")
    print(f"  max cumulative stable regret: {sr.max():.2f}")
    print(f"  mean cumulative stable regret: {sr.mean():.2f}")
    print(f"  cumulative market unstability: {result.unstable_count}")


def _aggregate_results(agg: Dict[str, List[RunResult]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, runs in agg.items():
        mean_sr = np.mean([x.stable_regret for x in runs], axis=0)
        mean_unstable = float(np.mean([x.unstable_count for x in runs]))
        payload: Dict[str, Any] = {
            "stable_regret_per_player": [float(v) for v in mean_sr.tolist()],
            "max_cumulative_stable_regret": float(np.max(mean_sr)),
            "mean_cumulative_stable_regret": float(np.mean(mean_sr)),
            "cumulative_market_unstability": float(mean_unstable),
        }
        if runs and runs[0].curve_steps is not None:
            steps = runs[0].curve_steps
            curve_sr = np.stack([r.curve_player_regret for r in runs], axis=0)
            curve_unst = np.stack([r.curve_unstable for r in runs], axis=0)
            max_curve = np.max(curve_sr, axis=2)
            mean_curve = np.mean(curve_sr, axis=2)
            n_std = np.sqrt(max(1, len(runs)))
            # Per-player cumulative stable regret trajectories (paper Figure 1 (a)-(e)).
            ppm = np.mean(curve_sr, axis=0)
            ppe = np.std(curve_sr, axis=0, ddof=0) / n_std
            payload["curve"] = {
                "steps": [int(s) for s in steps.tolist()],
                "max_stable_regret_mean": [float(v) for v in np.mean(max_curve, axis=0).tolist()],
                "max_stable_regret_se": [
                    float(v) for v in (np.std(max_curve, axis=0, ddof=0) / n_std).tolist()
                ],
                "mean_stable_regret_mean": [float(v) for v in np.mean(mean_curve, axis=0).tolist()],
                "mean_stable_regret_se": [
                    float(v) for v in (np.std(mean_curve, axis=0, ddof=0) / n_std).tolist()
                ],
                "per_player_stable_regret_mean": [
                    [float(v) for v in ppm[:, pi].tolist()] for pi in range(ppm.shape[1])
                ],
                "per_player_stable_regret_se": [
                    [float(v) for v in ppe[:, pi].tolist()] for pi in range(ppe.shape[1])
                ],
                "unstability_mean": [float(v) for v in np.mean(curve_unst, axis=0).tolist()],
                "unstability_se": [
                    float(v) for v in (np.std(curve_unst, axis=0, ddof=0) / n_std).tolist()
                ],
            }
        out[name] = payload
    return out


def run_one_repeat(
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
    run_index: int,
    aeags_confidence_factor: float = 6.0,
    c_etc_log_coeff: float = 4.0,
    p_etc_explore_coef: float = 0.52,
) -> Dict[str, RunResult]:
    market = make_random_market(
        n_players,
        n_arms,
        delta=delta,
        sigma=sigma,
        clip_rewards=clip_rewards,
        model=market_model,
        seed=seed + 1000 * run_index,
    )
    regret_ref_rng = np.random.default_rng(seed + 424242 + run_index)
    regret_reference_mu = market.stable_regret_reference_per_player(rng=regret_ref_rng)

    aeags = AEAGSCentralized(
        n_players,
        n_arms,
        horizon,
        seed=seed + run_index,
        market=market,
        confidence_factor=float(aeags_confidence_factor),
    )
    c_etc = CETCKnownDelta(
        n_players,
        n_arms,
        horizon,
        delta=delta,
        seed=seed + run_index,
        log_coeff=float(c_etc_log_coeff),
    )
    p_etc = PhasedETC(
        n_players,
        n_arms,
        horizon,
        delta=delta,
        seed=seed + run_index,
        explore_coef=float(p_etc_explore_coef),
    )
    rnd = RandomMatchingPolicy(n_players, n_arms, seed=seed + run_index)

    return {
        "AE-AGS": run_policy(market, aeags, horizon, regret_reference_mu, rectify_regret, record_every, seed + 11),
        "C-ETC": run_policy(market, c_etc, horizon, regret_reference_mu, rectify_regret, record_every, seed + 22),
        "P-ETC": run_policy(market, p_etc, horizon, regret_reference_mu, rectify_regret, record_every, seed + 33),
        "Random": run_policy(market, rnd, horizon, regret_reference_mu, rectify_regret, record_every, seed + 44),
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
    parser.add_argument("--clip-rewards", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rectify-regret", type=int, choices=[0, 1], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None, help="Parallel worker processes for runs.")
    parser.add_argument("--record-every", type=int, default=None, help="Record curve every N steps. 0 disables.")
    parser.add_argument(
        "--market-model",
        type=str,
        choices=["paper_rank", "paper_strict_perm", "level_uniform"],
        default=None,
    )
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save summary JSON.")
    parser.add_argument(
        "--aeags-confidence-factor",
        type=float,
        default=None,
        help="AE-AGS UCB/LCB rad uses sqrt(this * log T / counts); paper theory uses 6 (default preset).",
    )
    parser.add_argument(
        "--c-etc-log-coeff",
        type=float,
        default=None,
        help="Centralized ETC pulls per directed pair ≃ coeff * ln(T)/Δ² (Appendix Eq. order match).",
    )
    parser.add_argument(
        "--p-etc-explore-coef",
        type=float,
        default=None,
        help="P-ETC explore length multiplier on (s+1)*ln(T)/Δ² per phased round block.",
    )
    args = _resolve_args(parser.parse_args())

    alg_names = ["AE-AGS", "C-ETC", "P-ETC", "Random"]
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
                args.market_model,
                int(args.record_every),
                args.seed,
                r,
                float(args.aeags_confidence_factor),
                float(args.c_etc_log_coeff),
                float(args.p_etc_explore_coef),
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
                    args.market_model,
                    int(args.record_every),
                    args.seed,
                    r,
                    float(args.aeags_confidence_factor),
                    float(args.c_etc_log_coeff),
                    float(args.p_etc_explore_coef),
                )
                for r in range(args.runs)
            ]
            for f in concurrent.futures.as_completed(futures):
                one = f.result()
                for name in alg_names:
                    agg[name].append(one[name])

    print(
        f"Experiment: N={args.N}, K={args.K}, T={args.T}, delta={args.delta}, sigma={args.sigma}, "
        f"clip_rewards={int(clip_rewards)}, rectify_regret={int(rectify_regret)}, runs={args.runs}, jobs={jobs}, "
        f"market_model={args.market_model}, record_every={int(args.record_every)}, "
        f"aeags_CF={float(args.aeags_confidence_factor):.4g}, c_etc_log={float(args.c_etc_log_coeff):.4g}, "
        f"p_etc_explore={float(args.p_etc_explore_coef):.4g}"
    )

    summary = _aggregate_results(agg)
    for name in alg_names:
        summarize(
            name,
            RunResult(
                stable_regret=np.array(summary[name]["stable_regret_per_player"], dtype=float),
                unstable_count=int(summary[name]["cumulative_market_unstability"]),
            ),
        )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "N": int(args.N),
                "K": int(args.K),
                "T": int(args.T),
                "delta": float(args.delta),
                "sigma": float(args.sigma),
                "clip_rewards": int(clip_rewards),
                "rectify_regret": int(rectify_regret),
                "runs": int(args.runs),
                "jobs": int(jobs),
                "seed": int(args.seed),
                "market_model": str(args.market_model),
                "record_every": int(args.record_every),
                "aeags_confidence_factor": float(args.aeags_confidence_factor),
                "c_etc_log_coeff": float(args.c_etc_log_coeff),
                "p_etc_explore_coef": float(args.p_etc_explore_coef),
            },
            "summary": summary,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved summary JSON to: {out_path}")


if __name__ == "__main__":
    main()

