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
        confidence_factor: float = 6.0,
        arm_schedule: str = "fixed",
        player_pull_tiebreak: str = "random",
        ucb_time_scale: str = "horizon",
        algo2_outer_loop: str = "pick_one",
        arm_rank_jitter_scale: float = 0.0,
    ):
        self.N = n_players
        self.K = n_arms
        self.T = max(2, horizon)
        self._log_T = float(np.log(self.T))
        self._confidence_factor = float(confidence_factor)
        self._round = 0
        sched = str(arm_schedule).lower().replace("-", "_")
        if sched not in ("fixed", "random", "round_robin"):
            raise ValueError("arm_schedule must be 'fixed', 'random', or 'round_robin'.")
        self._arm_schedule = sched
        ptb = str(player_pull_tiebreak).lower().replace("-", "_")
        if ptb not in ("random", "smallest_arm"):
            raise ValueError("player_pull_tiebreak must be 'random' or 'smallest_arm'.")
        self._player_pull_tiebreak = ptb
        uts = str(ucb_time_scale).lower().replace("-", "_")
        if uts not in ("horizon", "elapsed"):
            raise ValueError("ucb_time_scale must be 'horizon' or 'elapsed'.")
        self._ucb_time_scale = uts
        a2 = str(algo2_outer_loop).lower().replace("-", "_")
        if a2 not in ("pick_one", "round_sweep"):
            raise ValueError("algo2_outer_loop must be 'pick_one' or 'round_sweep'.")
        self._algo2_outer_loop = a2
        jitter = float(arm_rank_jitter_scale)
        if jitter < 0.0:
            raise ValueError("arm_rank_jitter_scale must be non-negative.")
        self._arm_rank_jitter_scale = jitter
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
        log_scale = (
            float(np.log(max(2.0, float(self._round))))
            if self._ucb_time_scale == "elapsed"
            else self._log_T
        )
        rad[positive] = np.sqrt(self._confidence_factor * log_scale / counts[positive])
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
        rr_cursor = 0

        def pick_proposing_arm(candidates_a: list[int]) -> int:
            nonlocal rr_cursor
            if self._arm_schedule == "random":
                return int(self.rng.choice(candidates_a))
            if self._arm_schedule == "fixed":
                return int(min(candidates_a))
            cand_set = set(candidates_a)
            for step in range(self.K):
                a_try = (rr_cursor + step) % self.K
                if a_try in cand_set:
                    rr_cursor = (a_try + 1) % self.K
                    return int(a_try)
            return int(min(candidates_a))

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
            cnt = np.array([self.state.counts[i, j] for j in valid])
            vmin = int(np.min(cnt))
            tied = np.flatnonzero(cnt == vmin).astype(int)
            if self._player_pull_tiebreak == "smallest_arm":
                arms_tied = [valid[int(idx)] for idx in tied]
                return int(min(arms_tied))
            pick = tied[int(self.rng.integers(0, len(tied)))]
            return int(valid[int(pick)])

        def proposal_step(a: int) -> None:
            """One Algorithm-2 iteration for arm ``a``: propose, build A_i, player choice, reject others."""
            i = int(order[a, s[a]])
            available[i].add(a)
            inc = int(player_match[i])
            if inc != -1:
                available[i].add(inc)

            chosen = choose_player_arm(i)
            prev_holder = int(arm_match[chosen])
            if prev_holder != -1 and prev_holder != i:
                player_match[prev_holder] = -1

            player_match[i] = chosen
            arm_match[chosen] = i

            for rej in list(available[i]):
                if rej == chosen:
                    continue
                arm_match[rej] = -1
                s[rej] += 1
            available[i].clear()

        if self._algo2_outer_loop == "round_sweep":
            # Repeated sweeps arm 0..K-1; each eligible unmatched arm performs at most one step per sweep.
            # Ignores arm_schedule among simultaneous eligibles (order is forced by index).
            while True:
                progressed = False
                for j in range(self.K):
                    if arm_match[j] != -1 or s[j] >= self.N:
                        continue
                    proposal_step(j)
                    progressed = True
                if not progressed:
                    break
        else:
            while True:
                candidates_a = [
                    int(j)
                    for j in range(self.K)
                    if arm_match[j] == -1 and s[j] < self.N
                ]
                if not candidates_a:
                    break
                a = pick_proposing_arm(candidates_a)
                proposal_step(a)

        return player_match

    def assign_actions(self, arm_rank: np.ndarray) -> np.ndarray:
        self._round += 1
        # Appendix B-style random tie-breaking on arm-side indifferences: perturb integer ranks before sorting.
        if self._arm_rank_jitter_scale > 0.0:
            pert = arm_rank.astype(np.float64, copy=False)
            pert += self._arm_rank_jitter_scale * self.rng.normal(size=arm_rank.shape)
            propose_order = np.argsort(pert, axis=1, kind="stable").astype(np.int32, copy=False)
        else:
            if self._arm_propose_order is None:
                self._arm_propose_order = np.argsort(arm_rank, axis=1, kind="stable").astype(
                    np.int32, copy=False
                )
            propose_order = self._arm_propose_order
        ucb, lcb = self._compute_confidence_bounds()
        self._update_better(ucb, lcb)
        return self._subroutine_matching(propose_order)

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

