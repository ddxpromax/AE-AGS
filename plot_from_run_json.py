from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def _get_algorithms(summary: Dict[str, Any], requested: List[str] | None) -> List[str]:
    algs = list(summary.keys())
    if requested:
        out = [a for a in requested if a in summary]
        return out if out else algs
    return algs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True, help="Path to run_experiment --save-json output")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated plots")
    parser.add_argument(
        "--algs",
        type=str,
        default=None,
        help="Comma-separated algorithms to plot, e.g. AE-AGS,C-ETC,P-ETC",
    )
    args = parser.parse_args()

    in_path = Path(args.input_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    summary: Dict[str, Any] = payload["summary"]
    cfg: Dict[str, Any] = payload.get("config", {})

    req_algs = [s.strip() for s in args.algs.split(",")] if args.algs else None
    algs = _get_algorithms(summary, req_algs)

    out_dir = Path(args.output_dir) if args.output_dir else (in_path.parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib required for plotting: {e}") from e

    # Validate curve availability.
    with_curve = [a for a in algs if "curve" in summary[a]]
    if not with_curve:
        raise ValueError("No curve data found in input JSON. Run with --record-every > 0 first.")

    steps = summary[with_curve[0]]["curve"]["steps"]

    # 1) Max cumulative stable regret curve
    plt.figure(figsize=(8, 4.5))
    for a in with_curve:
        c = summary[a]["curve"]
        y = c["max_stable_regret_mean"]
        se = c["max_stable_regret_se"]
        plt.plot(steps, y, label=a)
        lo = [yy - ss for yy, ss in zip(y, se)]
        hi = [yy + ss for yy, ss in zip(y, se)]
        plt.fill_between(steps, lo, hi, alpha=0.15)
    plt.xlabel("Round t")
    plt.ylabel("Max Cumulative Stable Regret")
    plt.title("Run Curves: Max Stable Regret")
    plt.legend()
    plt.tight_layout()
    p1 = out_dir / "run_curve_max_stable_regret.png"
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) Market unstability curve
    plt.figure(figsize=(8, 4.5))
    for a in with_curve:
        c = summary[a]["curve"]
        y = c["unstability_mean"]
        se = c["unstability_se"]
        plt.plot(steps, y, label=a)
        lo = [yy - ss for yy, ss in zip(y, se)]
        hi = [yy + ss for yy, ss in zip(y, se)]
        plt.fill_between(steps, lo, hi, alpha=0.15)
    plt.xlabel("Round t")
    plt.ylabel("Cumulative Market Unstability")
    plt.title("Run Curves: Market Unstability")
    plt.legend()
    plt.tight_layout()
    p2 = out_dir / "run_curve_unstability.png"
    plt.savefig(p2, dpi=150)
    plt.close()

    # Optional text summary for quick glance.
    summary_txt = out_dir / "run_curve_summary.txt"
    lines = []
    lines.append("Config:")
    for k in sorted(cfg.keys()):
        lines.append(f"- {k}: {cfg[k]}")
    lines.append("")
    lines.append("Final metrics (from summary):")
    for a in algs:
        s = summary[a]
        lines.append(
            f"- {a}: max_regret={s['max_cumulative_stable_regret']:.4f}, "
            f"mean_regret={s['mean_cumulative_stable_regret']:.4f}, "
            f"unstability={s['cumulative_market_unstability']:.4f}"
        )
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved plots:\n- {p1}\n- {p2}\nSaved text summary:\n- {summary_txt}")


if __name__ == "__main__":
    main()
