"""
Budget Manager: Allocates expensive simulation calls strategically.

Implements the "sparse truth evaluation" principle by selecting candidates
based on: best predicted, highest uncertainty, and disagreement.

Usage:
    from campro.orchestration import BudgetManager

    budget = BudgetManager(total_sim_calls=100)
    to_evaluate = budget.select(candidates, predictions, uncertainty)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class SelectionStrategy(Enum):
    """Strategies for selecting candidates for truth evaluation."""

    BEST_PREDICTED = "best"  # Highest predicted performance
    HIGH_UNCERTAINTY = "uncertain"  # Highest surrogate uncertainty
    DISAGREEMENT = "disagreement"  # Model vs CEM disagreement
    RANDOM = "random"  # Random exploration


@dataclass
class BudgetAllocation:
    """How to split budget across selection strategies."""

    best_fraction: float = 0.4  # 40% to best predicted
    uncertain_fraction: float = 0.3  # 30% to high uncertainty
    disagreement_fraction: float = 0.2  # 20% to disagreement
    random_fraction: float = 0.1  # 10% random

    def __post_init__(self):
        total = (
            self.best_fraction
            + self.uncertain_fraction
            + self.disagreement_fraction
            + self.random_fraction
        )
        if not np.isclose(total, 1.0):
            log.warning(f"Budget fractions sum to {total}, normalizing")
            self.best_fraction /= total
            self.uncertain_fraction /= total
            self.disagreement_fraction /= total
            self.random_fraction /= total


@dataclass
class BudgetState:
    """Tracks budget consumption over time."""

    total: int
    used: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def consume(self, count: int, reason: str = "") -> None:
        self.used += count
        self.history.append(
            {
                "count": count,
                "reason": reason,
                "remaining": self.remaining,
            }
        )


class BudgetManager:
    """
    Manages expensive simulation call budget.

    Implements the active learning principle: evaluate where it matters most.
    Allocates calls across best predicted, uncertain, and disagreement regions.
    """

    def __init__(
        self,
        total_sim_calls: int,
        allocation: BudgetAllocation | None = None,
        reserve_for_validation: float = 0.1,
    ):
        """
        Initialize budget manager.

        Args:
            total_sim_calls: Maximum number of expensive simulation calls
            allocation: How to split budget across strategies
            reserve_for_validation: Fraction to reserve for final validation
        """
        self.allocation = allocation or BudgetAllocation()
        self.reserve_fraction = reserve_for_validation

        # Reserve some budget for final validation
        reserve = int(total_sim_calls * reserve_for_validation)
        active_budget = total_sim_calls - reserve

        self.state = BudgetState(total=active_budget)
        self.validation_budget = reserve
        self._rng = np.random.default_rng(42)

        log.info(f"Budget: {active_budget} active + {reserve} validation = {total_sim_calls} total")

    def remaining(self) -> int:
        """Get remaining budget."""
        return self.state.remaining

    def exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.state.exhausted

    def select(
        self,
        candidates: list[Any],
        predictions: np.ndarray,
        uncertainty: np.ndarray,
        cem_feasibility: np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> list[int]:
        """
        Select candidates for truth evaluation.

        Args:
            candidates: List of candidate designs
            predictions: Predicted objective values (higher = better)
            uncertainty: Surrogate uncertainty estimates
            cem_feasibility: Optional CEM feasibility scores (for disagreement)
            batch_size: Number to select (default: min(remaining, reasonable batch))

        Returns:
            Indices of selected candidates
        """
        n = len(candidates)
        if n == 0:
            return []

        # Determine batch size
        if batch_size is None:
            batch_size = min(self.state.remaining, max(1, n // 10))
        batch_size = min(batch_size, self.state.remaining, n)

        if batch_size <= 0:
            log.warning("No budget remaining for selection")
            return []

        # Allocate slots to each strategy
        n_best = max(1, int(batch_size * self.allocation.best_fraction))
        n_uncertain = max(1, int(batch_size * self.allocation.uncertain_fraction))
        n_disagree = int(batch_size * self.allocation.disagreement_fraction)
        n_random = batch_size - n_best - n_uncertain - n_disagree

        selected: set[int] = set()

        # 1. Best predicted
        best_indices = np.argsort(predictions)[-n_best:][::-1]
        selected.update(best_indices.tolist())

        # 2. Highest uncertainty
        available = [i for i in range(n) if i not in selected]
        if available and n_uncertain > 0:
            uncertain_scores = uncertainty[available]
            top_uncertain = np.argsort(uncertain_scores)[-n_uncertain:][::-1]
            selected.update([available[i] for i in top_uncertain])

        # 3. Disagreement (if CEM feasibility provided)
        if cem_feasibility is not None and n_disagree > 0:
            available = [i for i in range(n) if i not in selected]
            if available:
                # Disagreement = high predicted but low CEM feasibility
                disagree_score = predictions[available] * (1 - cem_feasibility[available])
                top_disagree = np.argsort(disagree_score)[-n_disagree:][::-1]
                selected.update([available[i] for i in top_disagree])

        # 4. Random exploration
        available = [i for i in range(n) if i not in selected]
        if available and n_random > 0:
            random_picks = self._rng.choice(
                available, size=min(n_random, len(available)), replace=False
            )
            selected.update(random_picks.tolist())

        result = list(selected)[:batch_size]
        self.state.consume(len(result), "select_batch")

        log.debug(
            f"Selected {len(result)} candidates: "
            f"{n_best} best, {n_uncertain} uncertain, {n_disagree} disagree, {n_random} random"
        )

        return result

    def select_for_validation(
        self,
        candidates: list[Any],
        predictions: np.ndarray,
    ) -> list[int]:
        """
        Select final candidates for validation (uses reserved budget).

        Args:
            candidates: List of candidate designs
            predictions: Predicted objective values

        Returns:
            Indices of candidates for final validation
        """
        n = len(candidates)
        n_select = min(self.validation_budget, n)

        if n_select <= 0:
            return []

        # Select top predicted
        top_indices = np.argsort(predictions)[-n_select:][::-1]

        self.validation_budget -= n_select
        log.info(f"Selected {n_select} candidates for final validation")

        return top_indices.tolist()

    def get_statistics(self) -> dict[str, Any]:
        """Get budget usage statistics."""
        return {
            "total": self.state.total,
            "used": self.state.used,
            "remaining": self.state.remaining,
            "validation_remaining": self.validation_budget,
            "efficiency": self.state.used / max(1, self.state.total),
        }


__all__ = [
    "BudgetAllocation",
    "BudgetManager",
    "BudgetState",
    "SelectionStrategy",
]
