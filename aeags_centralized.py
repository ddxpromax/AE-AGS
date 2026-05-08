from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AEAGSState:
    mu_hat: np.ndarray  # [N, K]
    counts: np.ndarray  # [N, K]
    better: np.ndarray  # [N, K, K], boolean encoded as {0,1}


class AEAGSCentralized:
    """
    Centralized AE-AGS (Algorithm 1/2/3 style).
    """

    def __init__(self, n_players: int, n_arms: int, horizon: int, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self.rng = np.random.default_rng(seed)
        self.state = AEAGSState(
            mu_hat=np.zeros((self.N, self.K), dtype=float),
            counts=np.zeros((self.N, self.K), dtype=int),
            better=np.zeros((self.N, self.K, self.K), dtype=np.int8),
        )

    def _compute_confidence_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        counts = self.state.counts
        mu_hat = self.state.mu_hat
        rad = np.zeros_like(mu_hat)
        positive = counts > 0
        rad[positive] = np.sqrt(6.0 * np.log(self.T) / counts[positive])
        ucb = np.where(positive, mu_hat + rad, np.inf)
        lcb = np.where(positive, mu_hat - rad, -np.inf)
        return ucb, lcb

    def _update_better(self, ucb: np.ndarray, lcb: np.ndarray) -> None:
        for i in range(self.N):
            for j in range(self.K):
                for jp in range(self.K):
                    if lcb[i, j] > ucb[i, jp]:
                        self.state.better[i, j, jp] = 1

    def _subroutine_matching(self, arm_rank: np.ndarray) -> np.ndarray:
        """
        Arm-guided GS with adaptive elimination.
        arm_rank: [K, N], lower rank means more preferred.
        """
        available = [set() for _ in range(self.N)]
        player_match = np.full(self.N, -1, dtype=int)
        arm_match = np.full(self.K, -1, dtype=int)
        s = np.zeros(self.K, dtype=int)

        # Build inverse index: position -> player for each arm.
        order = np.argsort(arm_rank, axis=1, kind="stable")

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
            free_arm = None
            for a in range(self.K):
                if arm_match[a] == -1 and s[a] < self.N:
                    free_arm = a
                    break
            if free_arm is None:
                break

            a = free_arm
            i = int(order[a, s[a]])
            available[i].add(a)
            chosen = choose_player_arm(i)
            old = player_match[i]
            player_match[i] = chosen
            arm_match[chosen] = i
            if old != -1 and old != chosen:
                arm_match[old] = -1
                s[old] += 1

            # Any non-chosen arm in available set gets rejected this iteration.
            for rej in list(available[i]):
                if rej != chosen and arm_match[rej] == -1:
                    s[rej] += 1
        return player_match

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        ucb, lcb = self._compute_confidence_bounds()
        self._update_better(ucb, lcb)
        return self._subroutine_matching(arm_rank)

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(self.N):
            a = int(matched_arm[i])
            if a < 0:
                continue
            c = self.state.counts[i, a]
            new_c = c + 1
            self.state.mu_hat[i, a] = (self.state.mu_hat[i, a] * c + rewards[i]) / new_c
            self.state.counts[i, a] = new_c
