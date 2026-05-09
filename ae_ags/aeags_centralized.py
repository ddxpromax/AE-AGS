from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .market import MatchingMarket


@dataclass
class AEAGSState:
    mu_hat: np.ndarray  # [N, K]
    counts: np.ndarray  # [N, K]
    better: np.ndarray  # [N, K, K], boolean encoded as {0,1}


class AEAGSCentralized:
    """
    Centralized AE-AGS (Algorithm 1/2/3 style).
    """

    def __init__(
        self,
        n_players: int,
        n_arms: int,
        horizon: int,
        seed: int = 0,
        market: Optional["MatchingMarket"] = None,
    ):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self._log_T = float(np.log(self.T))
        self.rng = np.random.default_rng(seed)
        self.state = AEAGSState(
            mu_hat=np.zeros((self.N, self.K), dtype=float),
            counts=np.zeros((self.N, self.K), dtype=int),
            better=np.zeros((self.N, self.K, self.K), dtype=np.int8),
        )
        # Fixed for the run; avoids argsort(arm_rank) every round.
        self._arm_propose_order: Optional[np.ndarray] = (
            np.asarray(market.arm_propose_player_idx, dtype=np.int32) if market is not None else None
        )

    def _compute_confidence_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        counts = self.state.counts
        mu_hat = self.state.mu_hat
        rad = np.zeros_like(mu_hat)
        positive = counts > 0
        rad[positive] = np.sqrt(6.0 * self._log_T / counts[positive])
        ucb = np.where(positive, mu_hat + rad, np.inf)
        lcb = np.where(positive, mu_hat - rad, -np.inf)
        return ucb, lcb

    def _update_better(self, ucb: np.ndarray, lcb: np.ndarray) -> None:
        # Better(i,j,j') = 1 iff LCB_{i,j} > UCB_{i,j'} (Algorithm 3); monotonic OR with past.
        flags = lcb[:, :, None] > ucb[:, None, :]
        self.state.better |= flags.astype(np.int8, copy=False)

    def _subroutine_matching(self, propose_order: np.ndarray) -> np.ndarray:
        """
        Arm-guided GS with adaptive elimination.
        propose_order: [K, N] int, for each arm row: player indices best → worst.
        """
        available = [set() for _ in range(self.N)]  # A_i in the paper.
        player_match = np.full(self.N, -1, dtype=int)  # m_i
        arm_match = np.full(self.K, -1, dtype=int)  # m_j^{-1}
        s = np.zeros(self.K, dtype=int)  # current proposing rank s_j (0-indexed)

        order = propose_order

        def choose_player_arm(i: int) -> int:
            candidates = sorted(available[i])
            if not candidates:
                return -1
            # Estimated suboptimal set.
            subopt = set()
            for j in candidates:
                for jp in candidates:
                    if self.state.better[i, jp, j] == 1:
                        subopt.add(j)
                        break
            valid = [j for j in candidates if j not in subopt]
            if not valid:
                valid = candidates
            # Pick least explored among valid arms.
            cnt = self.state.counts[i, valid]
            best_idx = int(np.argmin(cnt))
            return int(valid[best_idx])

        while True:
            # Find an unmatched arm that still has players to propose.
            a = -1
            for j in range(self.K):
                if arm_match[j] == -1 and s[j] < self.N:
                    a = j
                    break
            if a == -1:
                break

            # Arm a proposes to its s[a]-th preferred player.
            i = int(order[a, s[a]])
            available[i].add(a)

            # Player i chooses among available arms with AE-AGS rule.
            chosen = choose_player_arm(i)
            old = int(player_match[i])
            player_match[i] = chosen
            arm_match[chosen] = i

            # Old matched arm of i gets rejected and moves to next player.
            if old != -1 and old != chosen:
                arm_match[old] = -1
                s[old] += 1

            # All non-chosen available arms are rejected by i.
            for rej in list(available[i]):
                if rej == chosen:
                    continue
                arm_match[rej] = -1
                s[rej] += 1
        return player_match

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        if self._arm_propose_order is None:
            self._arm_propose_order = np.argsort(arm_rank, axis=1, kind="stable").astype(np.int32, copy=False)
        ucb, lcb = self._compute_confidence_bounds()
        self._update_better(ucb, lcb)
        return self._subroutine_matching(self._arm_propose_order)

    def observe(self, assigned_arm: np.ndarray, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        # Algorithm 3: update only if p_i is successfully matched to the assigned A_i(t).
        for i in range(self.N):
            a = int(assigned_arm[i])
            if a < 0 or int(matched_arm[i]) != a:
                continue
            c = self.state.counts[i, a]
            new_c = c + 1
            self.state.mu_hat[i, a] = (self.state.mu_hat[i, a] * c + rewards[i]) / new_c
            self.state.counts[i, a] = new_c

