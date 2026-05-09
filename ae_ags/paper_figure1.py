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
from typing import Any, Dict, List


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


def plot_paper_figure1(
    payload: Dict[str, Any],
    out_path: Path,
    dpi: int = 220,
) -> None:
    try:
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "font.family": "serif",
                "font.size": 8.5,
                "axes.labelsize": 9,
                "axes.titlesize": 9,
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.linewidth": 0.9,
                "lines.linewidth": 1.25,
                "figure.constrained_layout.use": False,
            }
        )
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"matplotlib and numpy required: {e}") from e

    summary: Dict[str, Any] = payload["summary"]
    cfg: Dict[str, Any] = payload.get("config", {})
    n_players = int(cfg.get("N", 5))

    algs = _paper_algorithms(summary)
    if len(algs) < 3:
        raise ValueError("Need AE-AGS, C-ETC, P-ETC with curve data in summary for paper Figure 1.")

    first_curve = summary[algs[0]]["curve"]
    steps = np.array(first_curve["steps"], dtype=int)
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

    # Stable colors consistent across panels (common in proceedings figures).
    color_map = {
        "AE-AGS": "#1f77b4",
        "C-ETC": "#ff7f0e",
        "P-ETC": "#2ca02c",
    }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(6.875, 7.05),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.ravel()

    n_pp = ppm[algs[0]].shape[0]
    n_plot_players = min(5, n_players, n_pp)

    def panel_label(ax, letter: str) -> None:
        ax.text(
            0.02,
            0.98,
            f"({letter})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
        )

    handles = []

    def style_axis(ax: Any, show_ylabel: bool) -> None:
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.68)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_ylabel:
            ax.set_ylabel("Cumulative Stable Regret")

    for pi in range(n_plot_players):
        ax = axes_flat[pi]
        letter = chr(ord("a") + pi)
        panel_label(ax, letter)
        ax.set_title(f"Player $p_{{{pi + 1}}}$", fontsize=8.8, pad=3)
        for alg in algs:
            color = color_map.get(alg, None)
            y = ppm[alg][pi]
            se = ppe[alg][pi]
            (line,) = ax.plot(steps, y, label=alg, color=color, zorder=2)
            ax.fill_between(steps, y - se, y + se, alpha=0.15, color=color, linewidth=0, zorder=1)
            if pi == 0:
                handles.append(mlines.Line2D([], [], color=color, label=alg, lw=2))
        style_axis(ax, show_ylabel=True)

    for hide_i in range(n_plot_players, 5):
        axes_flat[hide_i].axis("off")

    ax_f = axes_flat[5]
    panel_label(ax_f, "f")
    ax_f.set_title("Cumulative market unstability", fontsize=8.8, pad=3)
    for alg in algs:
        color = color_map.get(alg, None)
        c = summary[alg]["curve"]
        y = np.array(c["unstability_mean"], dtype=float)
        se = np.array(c["unstability_se"], dtype=float)
        ax_f.plot(steps, y, label=alg, color=color, zorder=2)
        ax_f.fill_between(steps, y - se, y + se, alpha=0.15, color=color, linewidth=0, zorder=1)
    style_axis(ax_f, show_ylabel=True)
    ax_f.set_ylabel("Cumulative Market Unstability")

    max_t = float(steps[-1])
    xticks, xticklabels = _paper_xticks(max_t)
    for idx, ax in enumerate(axes_flat):
        if idx < 5 and idx >= n_plot_players:
            continue
        if not ax.has_data() and idx < 5:
            continue
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    ncol_alg = len(handles)
    bottom_pad = 0.19 + 0.035 * ((ncol_alg + 2) // 3 if ncol_alg else 0)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=bottom_pad, hspace=0.43, wspace=0.32)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(3, ncol_alg),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
            title="Algorithms",
            title_fontproperties={"weight": "normal", "size": 9},
            columnspacing=1.45,
            handlelength=2.8,
            handletextpad=0.65,
        )
    fig.supxlabel("Round $t$", fontsize=9)

    ae_c = cfg.get("aeags_confidence_factor")
    cet = cfg.get("c_etc_log_coeff")
    pet = cfg.get("p_etc_explore_coef")
    footer = (
        f"Appendix E setup: N=K={cfg.get('N', '?')}, delta={cfg.get('delta', '?')}, "
        f"sigma={cfg.get('sigma', '?')}, T={cfg.get('T', '?')}, runs={cfg.get('runs', '?')}, "
        f"market_model={cfg.get('market_model', '?')}."
    )
    if ae_c is not None and cet is not None and pet is not None:
        footer += (
            f" Repro knobs: aeags_rad_factor={ae_c} in sqrt(factor*ln(T)/T_ij); "
            f"c_etc_log_coeff={cet}; p_etc_explore_coef={pet}."
        )
    fig.text(0.5, -0.012, footer, ha="center", va="top", fontsize=6.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Figure 1 (six panels) from experiment JSON.")
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="PNG path")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    in_path = Path(args.input_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    out = Path(args.output) if args.output else (in_path.parent / "figure1_paper_sixpanels.png")
    plot_paper_figure1(payload, out_path=out, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
