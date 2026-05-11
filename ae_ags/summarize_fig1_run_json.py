"""
Summarize run_experiment --save-json output for Appendix Fig.1 triage:
cumulative market unstability (panel f) vs per-player regret (panels a–e).

Usage:
  python -m ae_ags.summarize_fig1_run_json path/to/run.json [--write path/out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _final_unstability(curve: Dict[str, Any]) -> float | None:
    u = curve.get("unstability_mean")
    if not isinstance(u, list) or not u:
        return None
    return float(u[-1])


def summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = payload.get("config", {})
    summary_in = payload.get("summary", {})
    order: List[str] = []
    for key in ("AE-AGS", "C-ETC", "P-ETC", "Random"):
        if key in summary_in:
            order.append(key)

    rows = []
    for name in order:
        s = summary_in[name]
        cu = s.get("cumulative_market_unstability")
        curve = s.get("curve") or {}
        fin = _final_unstability(curve)
        rows.append(
            {
                "algorithm": name,
                "cumulative_market_unstability_mean": float(cu) if cu is not None else None,
                "curve_unstability_mean_at_last_record": fin,
                "mean_cumulative_stable_regret": (
                    float(s["mean_cumulative_stable_regret"]) if "mean_cumulative_stable_regret" in s else None
                ),
                "max_cumulative_stable_regret": (
                    float(s["max_cumulative_stable_regret"]) if "max_cumulative_stable_regret" in s else None
                ),
            }
        )

    ae = next((r for r in rows if r["algorithm"] == "AE-AGS"), None)
    ce = next((r for r in rows if r["algorithm"] == "C-ETC"), None)
    pe = next((r for r in rows if r["algorithm"] == "P-ETC"), None)
    gap_f = {}
    if ae and ce and ae["cumulative_market_unstability_mean"] is not None:
        gap_f["AE_minus_CETC"] = ae["cumulative_market_unstability_mean"] - ce["cumulative_market_unstability_mean"]
    if pe and ce and pe["cumulative_market_unstability_mean"] is not None:
        gap_f["PETC_minus_CETC"] = pe["cumulative_market_unstability_mean"] - ce["cumulative_market_unstability_mean"]

    note_a_e = (
        "Panels (a)–(e) use stable_regret_reference; best vs worst changes regret only, not panel (f)."
    )
    ref = str(cfg.get("stable_regret_reference", "worst"))

    return {
        "source_config": {
            "stable_regret_reference": ref,
            "c_etc_log_coeff": cfg.get("c_etc_log_coeff"),
            "aeags_confidence_factor": cfg.get("aeags_confidence_factor"),
            "aeags_algo2_outer_loop": cfg.get("aeags_algo2_outer_loop"),
            "aeags_player_pull_tiebreak": cfg.get("aeags_player_pull_tiebreak"),
            "reward_noise_mode": cfg.get("reward_noise_mode"),
            "T": cfg.get("T"),
            "runs": cfg.get("runs"),
        },
        "note": note_a_e,
        "algorithms": rows,
        "gap_panel_f": gap_f,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_json", type=str)
    p.add_argument("--write", type=str, default=None, help="Write summary JSON to this path.")
    args = p.parse_args()
    path = Path(args.input_json)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = summarize_payload(payload)
    text = json.dumps(out, indent=2)
    print(text)
    if args.write:
        wp = Path(args.write)
        wp.parent.mkdir(parents=True, exist_ok=True)
        with wp.open("w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
