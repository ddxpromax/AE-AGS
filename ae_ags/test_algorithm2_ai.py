"""
Regression: Algorithm 2 A_i must include the incumbent tentative arm when a new arm proposes.

Run: python -m ae_ags.test_algorithm2_ai
"""

from __future__ import annotations

import numpy as np

from .aeags_centralized import AEAGSCentralized
from .market import MatchingMarket


def test_two_arms_compete_on_same_player_uses_min_pulls() -> None:
    """
    N=K=2. Both arms first-preference player 0. Player 0 should compare both once arm 1 proposes,
    and pick the arm with smaller T_{0,j} (ties: player_pull_tiebreak).
    """
    mu = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)
    arm_rank = np.array([[0, 1], [0, 1]], dtype=int)
    m = MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=1.0)
    pol = AEAGSCentralized(
        2,
        2,
        horizon=100,
        seed=0,
        market=m,
        confidence_factor=6.0,
        arm_schedule="fixed",
        player_pull_tiebreak="smallest_arm",
        ucb_time_scale="horizon",
    )
    # No Better flags; player 0 should break ties by min count only.
    pol.state.better.fill(0)
    pol.state.counts[0, 0] = 5
    pol.state.counts[0, 1] = 1
    pol.state.counts[1, :] = 0

    # Both arms rank player 0 first: rows are arms, columns proposal order.
    order = np.array([[0, 1], [0, 1]], dtype=np.int32)
    out = pol._subroutine_matching(order)
    # Arm 1 has strictly smaller T_{0,1} than T_{0,0} -> player 0 should end with arm 1.
    assert int(out[0]) == 1, f"expected arm 1 for player 0, got {out[0]}"
    assert int(out[1]) == 0, f"expected arm 0 for player 1, got {out[1]}"


def test_two_arms_same_with_round_sweep_outer() -> None:
    mu = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)
    arm_rank = np.array([[0, 1], [0, 1]], dtype=int)
    m = MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=1.0)
    pol = AEAGSCentralized(
        2,
        2,
        horizon=100,
        seed=0,
        market=m,
        confidence_factor=6.0,
        arm_schedule="fixed",
        player_pull_tiebreak="smallest_arm",
        ucb_time_scale="horizon",
        algo2_outer_loop="round_sweep",
    )
    pol.state.better.fill(0)
    pol.state.counts[0, 0] = 5
    pol.state.counts[0, 1] = 1
    pol.state.counts[1, :] = 0
    order = np.array([[0, 1], [0, 1]], dtype=np.int32)
    out = pol._subroutine_matching(order)
    assert int(out[0]) == 1
    assert int(out[1]) == 0


def main() -> None:
    test_two_arms_compete_on_same_player_uses_min_pulls()
    test_two_arms_same_with_round_sweep_outer()
    print("ae_ags.test_algorithm2_ai: ok")


if __name__ == "__main__":
    main()
