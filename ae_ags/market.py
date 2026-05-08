from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Optional

import numpy as np


@dataclass
class MatchingMarket:
    """
    Static market instance for one experiment run.
    - mu[i, a]: true expected reward for player i on arm a
    - arm_rank[a, i]: arm a's preference rank over players (lower is better)
    """

    mu: np.ndarray  # shape [N, K]
    arm_rank: np.ndarray  # shape [K, N]
    sigma: float = 1.0
    clip_rewards: bool = False
    reward_min: float = -10.0
    reward_max: float = 10.0

    def resolve_round(self, chosen_arm: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """
        Resolve one round under one-to-one capacity constraints.
        Players propose to chosen arms; each arm accepts the most preferred proposer.
        Returns:
          matched_arm: shape [N], matched arm id or -1
          rewards: shape [N], sampled reward if matched else 0.0
        """
        N, K = self.mu.shape
        proposers = [[] for _ in range(K)]
        for i, a in enumerate(chosen_arm):
            if 0 <= a < K:
                proposers[a].append(i)

        matched_arm = np.full(N, -1, dtype=int)
        rewards = np.zeros(N, dtype=float)

        for a in range(K):
            if not proposers[a]:
                continue
            best_i = min(proposers[a], key=lambda i: self.arm_rank[a, i])
            matched_arm[best_i] = a
            sampled = float(rng.normal(self.mu[best_i, a], self.sigma))
            if self.clip_rewards:
                sampled = float(np.clip(sampled, self.reward_min, self.reward_max))
            rewards[best_i] = sampled
        return matched_arm, rewards

    def is_stable_matching(self, matched_arm: np.ndarray) -> bool:
        """
        Weak stability check.
        """
        N, K = self.mu.shape
        arm_partner = np.full(K, -1, dtype=int)
        for i, a in enumerate(matched_arm):
            if 0 <= a < K:
                arm_partner[a] = i

        player_rank = np.argsort(-self.mu, axis=1, kind="stable")
        inv_rank = np.empty_like(player_rank)
        for i in range(N):
            inv_rank[i, player_rank[i]] = np.arange(K)

        for i in range(N):
            current = matched_arm[i]
            current_rank = inv_rank[i, current] if current >= 0 else K
            for a in range(K):
                if inv_rank[i, a] >= current_rank:
                    continue
                partner = arm_partner[a]
                if partner == -1:
                    return False
                if self.arm_rank[a, i] < self.arm_rank[a, partner]:
                    return False
        return True

    def stable_baseline_reward(
        self,
        exact_cutoff: int = 8,
        approx_samples: int = 256,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Minimum total expected reward among stable matchings.
        Exact enumeration for small N; sampled GS for larger N.
        """
        N, K = self.mu.shape
        rng = np.random.default_rng() if rng is None else rng

        if N <= exact_cutoff:
            best = np.inf
            for perm in itertools.permutations(range(K), N):
                m = np.array(perm, dtype=int)
                if self.is_stable_matching(m):
                    total = float(np.sum(self.mu[np.arange(N), m]))
                    best = min(best, total)
            if np.isfinite(best):
                return best

        def sampled_player_gs() -> np.ndarray:
            player_rank = np.argsort(-self.mu + 1e-9 * rng.normal(size=self.mu.shape), axis=1)
            arm_rank = np.argsort(self.arm_rank + 1e-9 * rng.normal(size=self.arm_rank.shape), axis=1)

            next_idx = np.zeros(N, dtype=int)
            player_match = np.full(N, -1, dtype=int)
            arm_match = np.full(K, -1, dtype=int)
            free = list(range(N))

            arm_pos = np.full((K, N), N, dtype=int)
            for a in range(K):
                for pos, p in enumerate(arm_rank[a]):
                    arm_pos[a, p] = pos

            while free:
                i = free.pop()
                while next_idx[i] < K and player_match[i] == -1:
                    a = int(player_rank[i, next_idx[i]])
                    next_idx[i] += 1
                    cur = arm_match[a]
                    if cur == -1:
                        arm_match[a] = i
                        player_match[i] = a
                    elif arm_pos[a, i] < arm_pos[a, cur]:
                        arm_match[a] = i
                        player_match[i] = a
                        player_match[cur] = -1
                        free.append(cur)
            return player_match

        best = np.inf
        for _ in range(max(1, approx_samples)):
            m = sampled_player_gs()
            if self.is_stable_matching(m):
                total = float(np.sum(self.mu[np.arange(N), m]))
                if total < best:
                    best = total

        if np.isfinite(best):
            return best

        return float(np.sum(np.max(self.mu, axis=1)))


def make_random_market(
    N: int,
    K: int,
    delta: float,
    seed: int = 0,
    sigma: float = 1.0,
    clip_rewards: bool = False,
    model: str = "level_uniform",
) -> MatchingMarket:
    """
    Generate a synthetic market with indifference.
    model:
      - "level_uniform": legacy uniform-level model.
      - "paper_rank": rank-position model aligned with Appendix E description.
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros((N, K), dtype=float)
    arm_rank = np.zeros((K, N), dtype=int)

    if model == "paper_rank":
        for i in range(N):
            positions = np.arange(K)
            rng.shuffle(positions)
            for a in range(K):
                rank_pos = int(positions[a]) + 1
                mu[i, a] = 1.0 - delta * rank_pos
            tie_mask = rng.random(K) < 0.2
            if np.any(tie_mask):
                mu[i, tie_mask] = np.round(mu[i, tie_mask] / max(delta, 1e-6)) * delta

        for a in range(K):
            ranks = np.arange(N)
            rng.shuffle(ranks)
            arm_rank[a] = ranks
    else:
        for i in range(N):
            best = rng.uniform(0.6, 1.0)
            levels = np.array([best - j * delta for j in range(K)], dtype=float)
            levels = np.clip(levels, 0.0, 1.0)
            perm = rng.permutation(K)
            mu[i] = levels[perm]
            tie_mask = rng.random(K) < 0.2
            if np.any(tie_mask):
                tie_val = rng.choice(levels)
                mu[i, tie_mask] = tie_val

        for a in range(K):
            levels = np.arange(N)
            rng.shuffle(levels)
            arm_rank[a] = levels

    return MatchingMarket(mu=mu, arm_rank=arm_rank, sigma=sigma, clip_rewards=clip_rewards)

