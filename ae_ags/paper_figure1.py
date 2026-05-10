"""
Appendix E Figure 1 layout from run_experiment --save-json output.

Panels (a)-(e): cumulative stable regret per player p1..pN for AE-AGS, C-ETC, P-ETC.
Panel (f): cumulative market unstability.

Random is omitted (not in the paper Figure 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _paper_xticks(max_t: float) -> Tuple[List[int], List[str]]:
    if max_t <= 0:
        return [0], ["0"]
    xs = [0, int(round(0.2 * max_t)), int(round(0.4 * max_t)), int(round(0.6 * max_t)), int(round(0.8 * max_t)), int(max_t)]
    xs = sorted(set(int(max(0, x)) for x in xs))
    labels = []
    for x in xs:
        labels.append(f"{x // 1000}k" if x >= 1000 else str(int(x)))
    return xs, labels


def _paper_algorithms(summary: Dict[str, Any]) -> List[str]:
    order = ["AE-AGS", "C-ETC", "P-ETC"]
    return [a for a in order if a in summary and "curve" in summary[a]]


def _format_axis_k_short(val: float, _pos: Any) -> str:
    """Axis ticks like the PDF: 0, 2k, 10k, -2k (no scientific notation)."""
    if abs(val) < 1e-9:
        return "0"
    sign = "-" if val < 0 else ""
    x = abs(float(val))
    if x >= 1000.0:
        u = x / 1000.0
        if abs(u - round(u)) < 1e-6:
            return f"{sign}{int(round(u))}k"
        return f"{sign}{u:.1f}k"
    if abs(x - round(x)) < 1e-6:
        return f"{sign}{int(round(x))}"
    return f"{sign}{x:.0f}"


def plot_paper_figure1(
    payload: Dict[str, Any],
    out_path: Path,
    dpi: int = 300,
) -> None:
    try:
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "font.family": "serif",
                # Match LaTeX Computer Modern math in proceedings PDFs when possible.
                "mathtext.fontset": "cm",
                "font.size": 8.5,
                "axes.labelsize": 9,
                "axes.titlesize": 9.5,
                "legend.fontsize": 7.5,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.linewidth": 0.75,
                "lines.linewidth": 1.15,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.pad": 2.8,
                "ytick.major.pad": 2.8,
                "legend.frameon": True,
                "legend.edgecolor": "0.82",
                "legend.fancybox": False,
                "figure.constrained_layout.use": False,
            }
        )
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ImportError as e:
        raise RuntimeError(f"matplotlib and numpy required: {e}") from e

    summary: Dict[str, Any] = payload["summary"]
    cfg: Dict[str, Any] = payload.get("config", {})
    n_players = int(cfg.get("N", 5))

    algs = _paper_algorithms(summary)
    if len(algs) < 3:
        raise ValueError("Need AE-AGS, C-ETC, P-ETC with curve data in summary for paper Figure 1.")

    first_curve = summary[algs[0]]["curve"]
    if "per_player_stable_regret_mean" not in first_curve:
        raise ValueError(
            "JSON lacks per-player curve fields; re-save with current run_experiment "
            "(per_player_stable_regret_mean / …_se in curve)."
        )

    ppm, ppe = {}, {}
    for a in algs:
        c = summary[a]["curve"]
        ppm[a] = np.array(c["per_player_stable_regret_mean"], dtype=float)
        ppe[a] = np.array(c["per_player_stable_regret_se"], dtype=float)

    steps_arr = np.array(first_curve["steps"], dtype=int)
    prepend_t0 = bool(steps_arr.size > 0 and int(steps_arr[0]) > 0)
    steps = (
        np.concatenate([[0], steps_arr])
        if prepend_t0
        else steps_arr
    )
    if prepend_t0:
        for a in algs:
            z = np.zeros((ppm[a].shape[0], 1), dtype=float)
            ppm[a] = np.hstack([z, ppm[a]])
            ppe[a] = np.hstack([z, ppe[a]])

    # Okabe–Ito (color-blind-safe), fixed across panels like camera-ready PDFs.
    color_map = {
        "AE-AGS": "#0072B2",
        "C-ETC": "#D55E00",
        "P-ETC": "#009E73",
    }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(7.1, 7.55),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.ravel()

    n_pp = ppm[algs[0]].shape[0]
    n_plot_players = min(5, n_players, n_pp)
    fmt_y = FuncFormatter(_format_axis_k_short)

    def style_axis(ax: Any, ylabel: str) -> None:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.65, color="#9a9a9a", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3.8, labelsize=8)
        ax.yaxis.set_major_formatter(fmt_y)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax.set_ylabel(ylabel, fontsize=9, labelpad=5)

    for pi in range(n_plot_players):
        ax = axes_flat[pi]
        letter = chr(ord("a") + pi)
        ax.set_title(
            rf"$\mathbf{{({letter})}}$ Player $p_{{{pi + 1}}}$",
            fontsize=9.5,
            pad=5,
            loc="center",
        )
        for alg in algs:
            color = color_map.get(alg, None)
            y = ppm[alg][pi]
            se = ppe[alg][pi]
            lbl = f"{alg}, Player p{pi + 1}"
            ax.plot(steps, y, label=lbl, color=color, lw=1.2, zorder=3)
            ax.fill_between(steps, y - se, y + se, alpha=0.12, color=color, linewidth=0, zorder=1)
        style_axis(ax, "Cumulative stable regret")
        lg = ax.legend(
            loc="upper left",
            fontsize=7,
            framealpha=0.94,
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=1.9,
        )
        if lg.get_frame():
            lg.get_frame().set_linewidth(0.55)

    for hide_i in range(n_plot_players, 5):
        axes_flat[hide_i].axis("off")

    ax_f = axes_flat[5]
    ax_f.set_title(
        r"$\mathbf{(f)}$ Cumulative market unstability",
        fontsize=9.5,
        pad=5,
        loc="center",
    )
    for alg in algs:
        color = color_map.get(alg, None)
        c = summary[alg]["curve"]
        y_raw = np.array(c["unstability_mean"], dtype=float)
        se_raw = np.array(c["unstability_se"], dtype=float)
        if prepend_t0:
            y_raw = np.concatenate([[0.0], y_raw])
            se_raw = np.concatenate([[0.0], se_raw])
        ax_f.plot(steps, y_raw, label=alg, color=color, lw=1.2, zorder=3)
        ax_f.fill_between(steps, y_raw - se_raw, y_raw + se_raw, alpha=0.12, color=color, linewidth=0, zorder=1)
    style_axis(ax_f, "Cumulative market unstability")
    ax_f.set_ylim(bottom=0)
    leg_f = ax_f.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.94,
        borderpad=0.35,
        labelspacing=0.32,
        handlelength=2.1,
    )
    if leg_f.get_frame():
        leg_f.get_frame().set_linewidth(0.55)

    max_t = float(steps[-1])
    xticks, xticklabels = _paper_xticks(max_t)
    for idx, ax in enumerate(axes_flat):
        if idx < 5 and idx >= n_plot_players:
            continue
        if not ax.has_data() and idx < 5:
            continue
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.93, bottom=0.175, hspace=0.34, wspace=0.28)
    fig.supxlabel("Round $t$", fontsize=9.5, y=0.065)

    ae_c = cfg.get("aeags_confidence_factor")
    cet = cfg.get("c_etc_log_coeff")
    pet = cfg.get("p_etc_explore_coef")
    arm_s = cfg.get("aeags_arm_schedule", "?")
    rnm = str(cfg.get("reward_noise_mode", "shared")).lower()
    apt = cfg.get("aeags_player_pull_tiebreak", "?")
    ucb_ts = cfg.get("aeags_ucb_time_scale", "?")
    footer = (
        f"Appendix E setup: N=K={cfg.get('N', '?')}, delta={cfg.get('delta', '?')}, "
        f"sigma={cfg.get('sigma', '?')}, T={cfg.get('T', '?')}, runs={cfg.get('runs', '?')}, "
        f"market_model={cfg.get('market_model', '?')}."
    )
    if ae_c is not None and cet is not None and pet is not None:
        noise_note = (
            "per-algorithm deterministic streams (--reward-noise-mode independent)."
            if rnm == "independent"
            else "(seed,t,i,a) keyed; shared across algos when matchings agree (--reward-noise-mode shared)."
        )
        ucb_note = "ln(t) radius (empirical)" if str(ucb_ts).lower() == "elapsed" else "ln(T) radius (paper)"
        footer += (
            f" Repro knobs: aeags_rad_factor={ae_c} sqrt(factor·{ucb_note}/T_ij); "
            f"c_etc_log_coeff={cet}; p_etc_explore_coef={pet}; "
            f"AE-AGS arm_schedule={arm_s}; pull_tiebreak={apt}; reward noise: {noise_note}"
        )
    fig.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=6.5, transform=fig.transFigure)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Figure 1 (six panels) from experiment JSON.")
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="PNG path")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    in_path = Path(args.input_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    out = Path(args.output) if args.output else (in_path.parent / "figure1_paper_sixpanels.png")
    plot_paper_figure1(payload, out_path=out, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
