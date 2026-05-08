from __future__ import annotations

import numpy as np


class RandomMatchingPolicy:
    """Simple baseline: random one-to-one assignment each round."""

    def __init__(self, n_players: int, n_arms: int, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.rng = np.random.default_rng(seed)

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        arms = self.rng.permutation(self.K)[: self.N]
        return np.asarray(arms, dtype=int)

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        return


class ExploreThenCommit:
    """
    Simplified centralized C-ETC-like baseline.
    - explore each (player, arm) pair for m rounds (as feasible)
    - then commit to repeated GS matching using empirical means
    """

    def __init__(self, n_players: int, n_arms: int, explore_rounds: int = 30, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.m = explore_rounds
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.mu_hat = np.zeros((self.N, self.K), dtype=float)
        self.counts = np.zeros((self.N, self.K), dtype=int)

    def _gs_with_mu(self, arm_rank: np.ndarray) -> np.ndarray:
        # Player-proposing GS using estimated player preference.
        order = np.argsort(-self.mu_hat, axis=1, kind="stable")
        next_idx = np.zeros(self.N, dtype=int)
        arm_match = np.full(self.K, -1, dtype=int)
        player_match = np.full(self.N, -1, dtype=int)
        free = list(range(self.N))
        while free:
            i = free.pop()
            if next_idx[i] >= self.K:
                continue
            a = int(order[i, next_idx[i]])
            next_idx[i] += 1
            cur = arm_match[a]
            if cur == -1:
                arm_match[a] = i
                player_match[i] = a
            else:
                if arm_rank[a, i] < arm_rank[a, cur]:
                    arm_match[a] = i
                    player_match[i] = a
                    player_match[cur] = -1
                    free.append(cur)
                else:
                    free.append(i)
        return player_match

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        self.t += 1
        # Exploration schedule: cycle players and arms.
        if np.min(self.counts) < self.m:
            target = np.full(self.N, -1, dtype=int)
            used = set()
            for i in range(self.N):
                for a in range(self.K):
                    if self.counts[i, a] < self.m and a not in used:
                        target[i] = a
                        used.add(a)
                        break
                if target[i] == -1:
                    # fallback distinct random arm
                    cand = [a for a in range(self.K) if a not in used]
                    if cand:
                        pick = int(self.rng.choice(cand))
                        target[i] = pick
                        used.add(pick)
            return target
        return self._gs_with_mu(arm_rank)

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(self.N):
            a = int(matched_arm[i])
            if a < 0:
                continue
            c = self.counts[i, a]
            nc = c + 1
            self.mu_hat[i, a] = (self.mu_hat[i, a] * c + rewards[i]) / nc
            self.counts[i, a] = nc
