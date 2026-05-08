from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def _player_proposing_gs(player_rank: np.ndarray, arm_rank: np.ndarray) -> np.ndarray:
    """
    Standard player-proposing Gale-Shapley with deterministic tie-breaking.
    player_rank: [N,K], lower is better.
    arm_rank: [K,N], lower is better.
    Returns matched arm per player shape [N].
    """
    N, K = player_rank.shape
    pref_lists = np.argsort(player_rank, axis=1, kind="stable")
    arm_pos = np.full((K, N), N, dtype=int)
    for a in range(K):
        order = np.argsort(arm_rank[a], kind="stable")
        for pos, p in enumerate(order):
            arm_pos[a, p] = pos

    next_idx = np.zeros(N, dtype=int)
    player_match = np.full(N, -1, dtype=int)
    arm_match = np.full(K, -1, dtype=int)
    free = list(range(N))
    while free:
        i = free.pop()
        while next_idx[i] < K and player_match[i] == -1:
            a = int(pref_lists[i, next_idx[i]])
            next_idx[i] += 1
            cur = arm_match[a]
            if cur == -1:
                arm_match[a] = i
                player_match[i] = a
            else:
                if arm_pos[a, i] < arm_pos[a, cur]:
                    arm_match[a] = i
                    player_match[i] = a
                    player_match[cur] = -1
                    free.append(int(cur))
        # if exhausted options, player stays unmatched
    return player_match


@dataclass
class RandomMatchingPolicy:
    N: int
    K: int
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        arms = np.arange(self.K)
        self.rng.shuffle(arms)
        m = np.full(self.N, -1, dtype=int)
        for i in range(min(self.N, self.K)):
            m[i] = arms[i]
        return m

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        return


class CETCKnownDelta:
    """
    A simple C-ETC style baseline with known Delta:
    - Explore each (player,arm) pair m times (round-robin style)
    - Commit to GS matching on estimated means afterwards.
    """

    def __init__(self, n_players: int, n_arms: int, horizon: int, delta: float, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self.delta = max(delta, 1e-6)
        self.rng = np.random.default_rng(seed)

        self.mu_hat = np.zeros((self.N, self.K), dtype=float)
        self.counts = np.zeros((self.N, self.K), dtype=int)
        self.phase = "explore"
        self.ptr = 0
        self.commit_matching: Optional[np.ndarray] = None

        self.m = int(math.ceil(4.0 * math.log(self.T) / (self.delta**2)))
        self.total_pairs = self.N * self.K

    def _explore_action(self) -> np.ndarray:
        acts = np.full(self.N, -1, dtype=int)
        base = self.ptr
        for i in range(self.N):
            a = (base + i) % self.K
            acts[i] = a
        self.ptr = (self.ptr + 1) % self.K
        return acts

    def _ready_to_commit(self) -> bool:
        return np.all(self.counts >= self.m)

    def _build_commit(self, arm_rank: np.ndarray) -> np.ndarray:
        player_rank = np.argsort(-self.mu_hat, axis=1, kind="stable")
        rank_matrix = np.empty_like(player_rank)
        for i in range(self.N):
            for pos, a in enumerate(player_rank[i]):
                rank_matrix[i, a] = pos
        return _player_proposing_gs(rank_matrix, arm_rank)

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        if self.phase == "explore":
            if self._ready_to_commit():
                self.phase = "commit"
                self.commit_matching = self._build_commit(arm_rank)
            else:
                return self._explore_action()

        if self.commit_matching is None:
            self.commit_matching = self._build_commit(arm_rank)
        return self.commit_matching

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        if self.phase != "explore":
            return
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
    A phased ETC baseline:
    - phase s explores each pair m_s times
    - then exploits a GS matching built from current estimates for 2^s rounds
    """

    def __init__(self, n_players: int, n_arms: int, horizon: int, delta: float, seed: int = 0):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self.delta = max(delta, 1e-6)
        self.rng = np.random.default_rng(seed)

        self.mu_hat = np.zeros((self.N, self.K), dtype=float)
        self.counts = np.zeros((self.N, self.K), dtype=int)

        self.phase_idx = 0
        self.phase_mode = "explore"
        self.phase_round = 0
        self.ptr = 0
        self.current_commit: Optional[np.ndarray] = None

    def _m_s(self, s: int) -> int:
        return max(1, int(math.ceil((s + 1) * math.log(self.T) / (self.delta**2))))

    def _explore_len(self, s: int) -> int:
        return self._m_s(s) * self.K

    def _exploit_len(self, s: int) -> int:
        return 2**s

    def _explore_action(self) -> np.ndarray:
        acts = np.full(self.N, -1, dtype=int)
        base = self.ptr
        for i in range(self.N):
            acts[i] = (base + i) % self.K
        self.ptr = (self.ptr + 1) % self.K
        return acts

    def _build_commit(self, arm_rank: np.ndarray) -> np.ndarray:
        player_rank = np.argsort(-self.mu_hat, axis=1, kind="stable")
        rank_matrix = np.empty_like(player_rank)
        for i in range(self.N):
            for pos, a in enumerate(player_rank[i]):
                rank_matrix[i, a] = pos
        return _player_proposing_gs(rank_matrix, arm_rank)

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        if self.phase_mode == "explore":
            if self.phase_round >= self._explore_len(self.phase_idx):
                self.phase_mode = "exploit"
                self.phase_round = 0
                self.current_commit = self._build_commit(arm_rank)
            else:
                self.phase_round += 1
                return self._explore_action()

        if self.phase_mode == "exploit":
            if self.current_commit is None:
                self.current_commit = self._build_commit(arm_rank)
            if self.phase_round < self._exploit_len(self.phase_idx):
                self.phase_round += 1
                return self.current_commit
            self.phase_idx += 1
            self.phase_mode = "explore"
            self.phase_round = 0
            self.current_commit = None
            return self._explore_action()

        return self._explore_action()

    def observe(self, matched_arm: np.ndarray, rewards: np.ndarray) -> None:
        if self.phase_mode != "explore":
            return
        for i in range(self.N):
            a = int(matched_arm[i])
            if a < 0:
                continue
            c = self.counts[i, a]
            nc = c + 1
            self.mu_hat[i, a] = (self.mu_hat[i, a] * c + rewards[i]) / nc
            self.counts[i, a] = nc

