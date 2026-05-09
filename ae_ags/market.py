from __future__ import annotations

from dataclasses import dataclass, field
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
    # Cached for is_stable_matching (μ is fixed for the run).
    _player_inv_rank: np.ndarray = field(init=False, repr=False)
    # Cached arm-side propose order: for each arm, player indices best→worst.
    _arm_propose_player_idx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        N, K = self.mu.shape
        player_rank = np.argsort(-self.mu, axis=1, kind="stable")
        inv = np.empty((N, K), dtype=np.int32)
        inv[np.arange(N, dtype=np.int32)[:, None], player_rank.astype(np.int32, copy=False)] = np.arange(
            K, dtype=np.int32
        )
        object.__setattr__(self, "_player_inv_rank", inv)
        arm_order = np.argsort(self.arm_rank, axis=1, kind="stable").astype(np.int32, copy=False)
        object.__setattr__(self, "_arm_propose_player_idx", arm_order)

    @property
    def arm_propose_player_idx(self) -> np.ndarray:
        """For each arm, player indices from most to least preferred (Algorithm 2 / resolve_round)."""
        return self._arm_propose_player_idx

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
            plist = list(proposers[a])
            best_rank = min(self.arm_rank[a, i] for i in plist)
            finalists = [i for i in plist if self.arm_rank[a, i] == best_rank]
            best_i = int(finalists[int(rng.integers(0, len(finalists)))]) if len(finalists) > 1 else int(finalists[0])
            matched_arm[best_i] = a
            sampled = float(rng.normal(self.mu[best_i, a], self.sigma))
            if self.clip_rewards:
                sampled = float(np.clip(sampled, self.reward_min, self.reward_max))
            rewards[best_i] = sampled
        return matched_arm, rewards

    def is_stable_matching(self, matched_arm: np.ndarray) -> bool:
        """
        Stability as in the paper §3 / Appendix E experiments: blocking pair iff
        µ_{i,j} > µ_{i,Āᵢ} strictly and π_{j,i} ≺ π_{j,Ā^{-1}_j} strictly (ties are never strict).

        This must not break ties among equal µᵢ using an arbitrary total order — that falsely
        flags many unstable rounds under indifferences.
        """
        N, K = self.mu.shape
        arm_partner = np.full(K, -1, dtype=int)
        mu = self.mu.astype(float)
        pi = self.arm_rank

        for i, a in enumerate(matched_arm):
            if 0 <= a < K:
                if arm_partner[a] != -1:
                    # two players claiming the same arm: not a feasible matching outcome
                    return False
                arm_partner[a] = i

        for i in range(N):
            curr = int(matched_arm[i])
            u_curr = float(mu[i, curr]) if curr >= 0 else float("-inf")
            for j in range(K):
                if not (float(mu[i, j]) > u_curr):
                    continue
                partner = int(arm_partner[j])
                if partner == -1:
                    return False
                if int(pi[j, i]) < int(pi[j, partner]):
                    return False
        return True

    def _sampled_player_gs(self, rng: np.random.Generator) -> np.ndarray:
        """Player-proposing GS with random tie perturbations (for baseline sampling)."""
        N, K = self.mu.shape
        player_rank = np.argsort(-self.mu + 1e-9 * rng.normal(size=self.mu.shape), axis=1)
        perturbed_arm = np.argsort(self.arm_rank + 1e-9 * rng.normal(size=self.arm_rank.shape), axis=1)

        next_idx = np.zeros(N, dtype=int)
        player_match = np.full(N, -1, dtype=int)
        arm_match = np.full(K, -1, dtype=int)
        free = list(range(N))

        arm_pos = np.full((K, N), N, dtype=int)
        for a in range(K):
            for pos, p in enumerate(perturbed_arm[a]):
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

    def stable_regret_reference_per_player(
        self,
        exact_cutoff: int = 8,
        approx_samples: int = 256,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Reference vector for stable regret (paper Eq. (1)), not related to baseline *algorithms*.

        For each player i: μ_{i,m_i} = min_{m' stable} μ_{i,m'(i)}.
        Exact over all stable matchings when N ≤ exact_cutoff; otherwise approximate via
        random player-proposing GS draws.
        """
        N, K = self.mu.shape
        rng = np.random.default_rng() if rng is None else rng
        idx = np.arange(N, dtype=int)
        fallback = np.min(self.mu, axis=1).astype(float)

        per = np.full(N, np.inf, dtype=float)

        if N <= exact_cutoff:
            for perm in itertools.permutations(range(K), N):
                m = np.array(perm, dtype=int)
                if self.is_stable_matching(m):
                    vals = self.mu[idx, m].astype(float)
                    per = np.minimum(per, vals)
            if np.all(np.isfinite(per)):
                return per
        else:
            for _ in range(max(1, approx_samples)):
                m = self._sampled_player_gs(rng)
                if self.is_stable_matching(m):
                    vals = self.mu[idx, m].astype(float)
                    per = np.minimum(per, vals)
            if np.all(np.isfinite(per)):
                return per

        return np.where(np.isfinite(per), per, fallback)


def _values_from_discrete_positions(pos: np.ndarray, delta: float) -> np.ndarray:
    """
    Appendix E style: each coordinate is a rank bucket in {1,...,Pmax}.
    Same bucket → same latent value; successive distinct buckets differ by Δ (μ decreases).
    """
    pos = pos.astype(np.int64, copy=False)
    uniq = np.sort(np.unique(pos))
    tier_mu = {int(p): float(1.0 - t * delta) for t, p in enumerate(uniq)}
    return np.array([tier_mu[int(p)] for p in pos], dtype=float)


def _ordinal_tiers_low_best(pos: np.ndarray) -> np.ndarray:
    """Integer ranks: smaller = more preferred (ties identical). Sorted unique ascending → tier 0,1,..."""
    pos = pos.astype(np.int64, copy=False)
    uniq = np.sort(np.unique(pos))
    tier_idx = {int(p): np.int32(t) for t, p in enumerate(uniq)}
    return np.array([tier_idx[int(p)] for p in pos], dtype=np.int32)


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
      - "paper_rank": Appendix E (Fig.1/2): iid positions in {1,..,K} per (player,arm); same
        bucket shares μ; tiers decrease by Δ; arms rank players symmetrically (positions in {1,..,N}).
      - "paper_strict_perm": legacy strict permutation-of-slots (backward compatibility / ablation).
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros((N, K), dtype=float)
    arm_rank = np.zeros((K, N), dtype=int)

    if model == "paper_rank":
        # Text: "The position of each arm in a player's preference ranking is a random number
        # in {1, 2, . . . , K} ... Arms sharing the same position ... same preference values ...
        # gap ... between arms ranked in adjacent positions is set to ∆ = 0.1."
        for i in range(N):
            pos = rng.integers(1, K + 1, size=K, endpoint=False)
            mu[i, :] = _values_from_discrete_positions(pos, delta)
        for j in range(K):
            pos_players = rng.integers(1, N + 1, size=N, endpoint=False)
            arm_rank[j, :] = _ordinal_tiers_low_best(pos_players)
    elif model == "paper_strict_perm":
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

