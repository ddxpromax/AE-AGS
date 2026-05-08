from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .run_experiment import run_one_repeat

ALG_NAMES = ["AE-AGS", "C-ETC", "P-ETC", "Random"]


def run_setting(
    n_players: int,
    n_arms: int,
    horizon: int,
    delta: float,
    sigma: float,
    clip_rewards: bool,
    rectify_regret: bool,
    runs: int,
    jobs: int,
    seed: int,
    market_model: str,
    record_every: int,
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, List[np.ndarray]] = {k: [] for k in ALG_NAMES}
    unstable: Dict[str, List[float]] = {k: [] for k in ALG_NAMES}

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
            )
            for k in ALG_NAMES:
                agg[k].append(one[k].stable_regret)
                unstable[k].append(float(one[k].unstable_count))
    else:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, runs)) as ex:
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
                )
                for r in range(runs)
            ]
            for f in concurrent.futures.as_completed(futures):
                one = f.result()
                for k in ALG_NAMES:
                    agg[k].append(one[k].stable_regret)
                    unstable[k].append(float(one[k].unstable_count))

    out: Dict[str, Dict[str, float]] = {}
    for k in ALG_NAMES:
        mean_sr = np.mean(agg[k], axis=0)
        out[k] = {
            "max_cumulative_stable_regret": float(np.max(mean_sr)),
            "mean_cumulative_stable_regret": float(np.mean(mean_sr)),
            "cumulative_market_unstability": float(np.mean(unstable[k])),
        }
    return out


def maybe_plot(output_dir: Path, data: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip plots.")
        return

    delta_vals = data["delta_sweep"]["delta_values"]
    delta_results = data["delta_sweep"]["results"]
    for metric, fname in [
        ("max_cumulative_stable_regret", "delta_sweep_max_stable_regret.png"),
        ("cumulative_market_unstability", "delta_sweep_unstability.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for alg in ALG_NAMES:
            ys = [entry["metrics"][alg][metric] for entry in delta_results]
            plt.plot(delta_vals, ys, marker="o", label=alg)
        plt.xlabel("Delta")
        plt.ylabel(metric.replace("_", " "))
        plt.title("Appendix E Delta Sweep")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()

    n_vals = data["size_sweep"]["sizes"]
    size_results = data["size_sweep"]["results"]
    for metric, fname in [
        ("max_cumulative_stable_regret", "size_sweep_max_stable_regret.png"),
        ("cumulative_market_unstability", "size_sweep_unstability.png"),
    ]:
        plt.figure(figsize=(7, 4))
        for alg in ALG_NAMES:
            ys = [entry["metrics"][alg][metric] for entry in size_results]
            plt.plot(n_vals, ys, marker="o", label=alg)
        plt.xlabel("N=K")
        plt.ylabel(metric.replace("_", " "))
        plt.title("Appendix E Market-Size Sweep")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=100000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--clip-rewards", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rectify-regret", type=int, choices=[0, 1], default=0)
    parser.add_argument("--market-model", type=str, choices=["paper_rank", "level_uniform"], default="paper_rank")
    parser.add_argument("--record-every", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_rewards = bool(args.clip_rewards)
    rectify_regret = bool(args.rectify_regret)

    delta_values = [0.1, 0.15, 0.2, 0.25]
    size_values = [3, 6, 9, 12]
    payload: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "T": args.T,
            "runs": args.runs,
            "jobs": args.jobs,
            "seed": args.seed,
            "sigma": args.sigma,
            "clip_rewards": int(clip_rewards),
            "rectify_regret": int(rectify_regret),
            "market_model": args.market_model,
            "record_every": args.record_every,
        },
        "delta_sweep": {"delta_values": delta_values, "results": []},
        "size_sweep": {"sizes": size_values, "results": []},
    }

    print("Running delta sweep...")
    for d in delta_values:
        metrics = run_setting(
            5,
            5,
            args.T,
            d,
            args.sigma,
            clip_rewards,
            rectify_regret,
            args.runs,
            args.jobs,
            args.seed + 10000,
            args.market_model,
            args.record_every,
        )
        payload["delta_sweep"]["results"].append({"delta": d, "metrics": metrics})
        print(f"  delta={d} done")

    print("Running market-size sweep...")
    for n in size_values:
        metrics = run_setting(
            n,
            n,
            args.T,
            0.1,
            args.sigma,
            clip_rewards,
            rectify_regret,
            args.runs,
            args.jobs,
            args.seed + 20000,
            args.market_model,
            args.record_every,
        )
        payload["size_sweep"]["results"].append({"N": n, "K": n, "metrics": metrics})
        print(f"  N=K={n} done")

    out_json = output_dir / "appendix_e_sweeps.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved sweep JSON: {out_json}")

    maybe_plot(output_dir, payload)
    print(f"Done. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()

