from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MatchingMarket:
    """One-to-one matching market with player utilities and arm rankings."""

    mu: np.ndarray  # shape [N, K], player utilities for arms
    arm_rank: np.ndarray  # shape [K, N], lower is better rank for arms over players
    sigma: float = 1.0
    clip_rewards: bool = False
    reward_min: float = 0.0
    reward_max: float = 1.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.arm_rank = np.asarray(self.arm_rank, dtype=int)
        self.N, self.K = self.mu.shape
        assert self.arm_rank.shape == (self.K, self.N)
        assert self.N <= self.K, "Require N <= K"
        self.rng = np.random.default_rng(self.rng_seed)

    def resolve_round(self, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve one round from player proposals.
        actions[i] in [0, K-1], or -1 means no proposal.
        Returns:
          matched_arm: shape [N], matched arm for each player, -1 if rejected
          rewards: shape [N], observed reward (0 if rejected)
        """
        matched_arm = np.full(self.N, -1, dtype=int)
        matched_player = np.full(self.K, -1, dtype=int)

        # Collect proposals per arm.
        proposers: List[List[int]] = [[] for _ in range(self.K)]
        for i, a in enumerate(actions):
            if 0 <= a < self.K:
                proposers[a].append(i)

        # Arms accept one best-ranked proposer; tie broken uniformly.
        for a in range(self.K):
            if not proposers[a]:
                continue
            ranks = np.array([self.arm_rank[a, i] for i in proposers[a]])
            best_rank = ranks.min()
            best_indices = [idx for idx, r in enumerate(ranks) if r == best_rank]
            chosen_local = int(self.rng.choice(best_indices))
            chosen_player = proposers[a][chosen_local]
            matched_player[a] = chosen_player
            matched_arm[chosen_player] = a

        rewards = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            a = matched_arm[i]
            if a >= 0:
                sampled = float(self.rng.normal(self.mu[i, a], self.sigma))
                if self.clip_rewards:
                    sampled = float(np.clip(sampled, self.reward_min, self.reward_max))
                rewards[i] = sampled
        return matched_arm, rewards

    def is_stable_matching(self, matching: Sequence[int]) -> bool:
        """Weak stability under indifference as in the paper."""
        m = np.asarray(matching, dtype=int)
        if m.shape != (self.N,):
            return False

        arm_partner = np.full(self.K, -1, dtype=int)
        for i, a in enumerate(m):
            if a >= 0:
                if arm_partner[a] != -1:
                    return False
                arm_partner[a] = i

        for i in range(self.N):
            current_a = m[i]
            current_u = self.mu[i, current_a] if current_a >= 0 else 0.0
            for a in range(self.K):
                if self.mu[i, a] <= current_u:
                    continue
                partner = arm_partner[a]
                if partner == -1:
                    # If arm is unmatched, player-arm forms a blocking pair.
                    return False
                if self.arm_rank[a, i] < self.arm_rank[a, partner]:
                    return False
        return True

    def stable_baseline_reward(self) -> np.ndarray:
        """
        For each player i, baseline is min reward among all stable matchings.
        For small N only (used in toy experiments).
        """
        from itertools import permutations

        min_reward = np.full(self.N, np.inf, dtype=float)
        arms = list(range(self.K))
        for picked in permutations(arms, self.N):
            m = np.array(picked, dtype=int)
            if self.is_stable_matching(m):
                min_reward = np.minimum(min_reward, self.mu[np.arange(self.N), m])
        if np.any(~np.isfinite(min_reward)):
            # Fallback for unexpected degenerate cases.
            return np.zeros(self.N, dtype=float)
        return min_reward


def make_random_market(
    n_players: int,
    n_arms: int,
    delta: float = 0.1,
    levels: int = 3,
    sigma: float = 1.0,
    clip_rewards: bool = False,
    seed: int = 0,
) -> MatchingMarket:
    """
    Build a synthetic indifferent-preference market.
    Players' utilities are sampled via rank levels; same level => equal utility.
    """
    rng = np.random.default_rng(seed)
    utility_levels = np.linspace(0.2, 0.2 + delta * (levels - 1), levels)
    mu = np.zeros((n_players, n_arms), dtype=float)
    for i in range(n_players):
        level_ids = rng.integers(0, levels, size=n_arms)
        mu[i] = utility_levels[level_ids]

    # Arm ranking positions (ties allowed).
    arm_rank = np.zeros((n_arms, n_players), dtype=int)
    for a in range(n_arms):
        arm_rank[a] = rng.integers(0, levels, size=n_players)
    return MatchingMarket(
        mu=mu,
        arm_rank=arm_rank,
        sigma=sigma,
        clip_rewards=clip_rewards,
        rng_seed=seed + 999,
    )
