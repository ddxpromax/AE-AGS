"""
Regression: AE-AGS Algorithm 3 observe semantics and assign_actions update order.

Run: python -m ae_ags.test_aeags_assign_order
"""

from __future__ import annotations

import numpy as np

from .aeags_centralized import AEAGSCentralized
from .market import MatchingMarket


def test_observe_only_when_matched_to_assigned() -> None:
    mu = np.ones((2, 2), dtype=float)
    arm_rank = np.array([[0, 1], [0, 1]], dtype=int)
    m = MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=1.0)
    pol = AEAGSCentralized(
        2,
        2,
        horizon=10,
        seed=0,
        market=m,
        confidence_factor=6.0,
        arm_schedule="fixed",
        player_pull_tiebreak="random",
        ucb_time_scale="horizon",
        algo2_outer_loop="pick_one",
        arm_rank_jitter_scale=0.0,
    )
    assigned = np.array([0, 1], dtype=int)
    matched = np.array([-1, 1], dtype=int)
    rewards = np.array([0.7, 0.3])
    pol.observe(assigned, matched, rewards)
    assert int(pol.state.counts[0, 0]) == 0, "no update when unmatched to assigned arm"
    assert int(pol.state.counts[1, 1]) == 1
    assert abs(float(pol.state.mu_hat[1, 1]) - 0.3) < 1e-9


def test_arm_rank_jitter_changes_propose_order_sometimes() -> None:
    """With jitter > 0, per-round perturbed sorting can reorder tied arm-side ranks."""
    mu = np.ones((2, 2), dtype=float)
    arm_rank = np.array([[0, 0], [0, 0]], dtype=int)  # full ties each row
    m = MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=1.0)
    pol2 = AEAGSCentralized(
        2,
        2,
        horizon=5,
        seed=42,
        market=m,
        confidence_factor=6.0,
        arm_schedule="fixed",
        arm_rank_jitter_scale=1e-6,
    )
    row0_orders: set[tuple[int, ...]] = set()
    for _ in range(200):
        eff = arm_rank.astype(np.float64, copy=False) + 1e-6 * pol2.rng.normal(size=arm_rank.shape)
        o0 = tuple(np.argsort(eff[0], kind="stable").tolist())
        row0_orders.add(o0)
        if len(row0_orders) >= 2:
            break
    assert len(row0_orders) >= 2, "expected jitter to break ties into at least two orderings"


def test_better_updated_before_matching_each_round() -> None:
    """Sanity: assign_actions increments round then updates Better before subroutine matching."""
    mu = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    arm_rank = np.array([[0, 1], [0, 1]], dtype=int)
    m = MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=1.0)
    pol = AEAGSCentralized(2, 2, horizon=3, seed=0, market=m, algo2_outer_loop="pick_one")
    assert pol._round == 0
    pol.assign_actions(m.arm_rank)
    assert pol._round == 1


def main() -> None:
    test_observe_only_when_matched_to_assigned()
    test_arm_rank_jitter_changes_propose_order_sometimes()
    test_better_updated_before_matching_each_round()
    print("ae_ags.test_aeags_assign_order: ok")


if __name__ == "__main__":
    main()
