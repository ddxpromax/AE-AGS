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


def _player_proposing_gs(score: np.ndarray, arm_rank: np.ndarray) -> np.ndarray:
    """
    Player-proposing GS using score[i, a] as player-side preference proxy.
    """
    n_players, n_arms = score.shape
    order = np.argsort(-score, axis=1, kind="stable")
    next_idx = np.zeros(n_players, dtype=int)
    arm_match = np.full(n_arms, -1, dtype=int)
    player_match = np.full(n_players, -1, dtype=int)
    free = list(range(n_players))
    while free:
        i = free.pop()
        if next_idx[i] >= n_arms:
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


class CETCKnownDelta:
    """
    Closer-to-paper centralized ETC with known delta.
    Exploration budget m scales with log(T)/delta^2, then commit by GS.
    """

    def __init__(self, n_players: int, n_arms: int, horizon: int, delta: float, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self.delta = max(1e-6, float(delta))
        self.rng = np.random.default_rng(seed)
        self.mu_hat = np.zeros((self.N, self.K), dtype=float)
        self.counts = np.zeros((self.N, self.K), dtype=int)
        self.committed = False
        self.commit_actions = np.full(self.N, -1, dtype=int)

        # Exploration rounds per pair.
        self.m = int(np.ceil(16.0 * np.log(self.T) / (self.delta ** 2)))

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        if self.committed:
            return self.commit_actions.copy()

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
                    cand = [a for a in range(self.K) if a not in used]
                    if cand:
                        pick = int(self.rng.choice(cand))
                        target[i] = pick
                        used.add(pick)
            return target

        self.commit_actions = _player_proposing_gs(self.mu_hat, arm_rank)
        self.committed = True
        return self.commit_actions.copy()

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(self.N):
            a = int(matched_arm[i])
            if a < 0:
                continue
            c = self.counts[i, a]
            nc = c + 1
            self.mu_hat[i, a] = (self.mu_hat[i, a] * c + rewards[i]) / nc
            self.counts[i, a] = nc


class PhasedETC:
    """
    Practical phased ETC-style baseline:
    repeatedly explore for a short phase, then commit by GS for a longer phase.
    """

    def __init__(
        self,
        n_players: int,
        n_arms: int,
        horizon: int,
        phase_base: int = 16,
        seed: int = 0,
    ):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self.phase_base = max(2, int(phase_base))
        self.rng = np.random.default_rng(seed)
        self.mu_hat = np.zeros((self.N, self.K), dtype=float)
        self.counts = np.zeros((self.N, self.K), dtype=int)
        self.t = 0
        self.phase_id = 0
        self.explore_len = self.phase_base
        self.commit_len = self.phase_base
        self.phase_t = 0
        self.in_explore = True
        self.commit_actions = np.full(self.N, -1, dtype=int)

    def _next_phase(self, arm_rank: np.ndarray) -> None:
        self.phase_id += 1
        self.explore_len = self.phase_base
        self.commit_len = self.phase_base * (2 ** self.phase_id)
        self.phase_t = 0
        self.in_explore = True
        self.commit_actions = _player_proposing_gs(self.mu_hat, arm_rank)

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.phase_id == 0 and self.phase_t == 0:
            self._next_phase(arm_rank)

        if self.in_explore:
            target = np.full(self.N, -1, dtype=int)
            used = set()
            for i in range(self.N):
                for a in range(self.K):
                    if a not in used:
                        target[i] = a
                        used.add(a)
                        break
                if target[i] == -1:
                    cand = [a for a in range(self.K) if a not in used]
                    if cand:
                        pick = int(self.rng.choice(cand))
                        target[i] = pick
                        used.add(pick)
            self.phase_t += 1
            if self.phase_t >= self.explore_len:
                self.in_explore = False
                self.phase_t = 0
                self.commit_actions = _player_proposing_gs(self.mu_hat, arm_rank)
            return target

        self.phase_t += 1
        if self.phase_t >= self.commit_len:
            self._next_phase(arm_rank)
        return self.commit_actions.copy()

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        for i in range(self.N):
            a = int(matched_arm[i])
            if a < 0:
                continue
            c = self.counts[i, a]
            nc = c + 1
            self.mu_hat[i, a] = (self.mu_hat[i, a] * c + rewards[i]) / nc
            self.counts[i, a] = nc


# Backward-compatible alias for existing code.
ExploreThenCommit = CETCKnownDelta
