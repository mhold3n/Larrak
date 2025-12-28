"""
CEM Adaptive Rules: Framework for rules that learn from HiFi feedback.

This module introduces adaptive validation rules whose limits update
based on observed margins from HiFi simulation results.

Pattern: After each generation of optimal profiles, rules:
1. Tighten limits when HiFi shows consistent safety headroom
2. Relax limits when consistently underestimating margins
3. Log all adaptations to Weaviate for provenance tracking

Usage:
    from truthmaker.cem.rules.adaptation import AdaptiveRuleBase, AdaptiveRuleState

    class MyAdaptiveRule(AdaptiveRuleBase):
        def adapt(self, hifi_result: dict, predicted: float) -> float:
            # Return delta (negative = tighten, positive = relax)
            ...
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import RuleBase, RuleCategory, RuleResult, RuleSeverity

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


@dataclass
class AdaptiveRuleState:
    """Persistent state for an adaptive rule.

    Tracks the learned limit and history for save/restore.
    """

    limit: float  # Current learned limit
    initial_limit: float  # Original design limit (never changes)
    n_observations: int = 0  # Total HiFi samples observed
    margin_history: list[float] = field(default_factory=list)
    last_adapted: Optional[datetime] = None
    total_delta: float = 0.0  # Cumulative change from initial
    regime_states: dict[int, float] = field(default_factory=dict)  # Per-regime limits

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "limit": self.limit,
            "initial_limit": self.initial_limit,
            "n_observations": self.n_observations,
            "margin_history": self.margin_history[-100:],  # Keep last 100 only
            "last_adapted": self.last_adapted.isoformat() if self.last_adapted else None,
            "total_delta": self.total_delta,
            "regime_states": self.regime_states,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdaptiveRuleState":
        """Deserialize from JSON."""
        return cls(
            limit=data["limit"],
            initial_limit=data["initial_limit"],
            n_observations=data.get("n_observations", 0),
            margin_history=data.get("margin_history", []),
            last_adapted=datetime.fromisoformat(data["last_adapted"])
            if data.get("last_adapted")
            else None,
            total_delta=data.get("total_delta", 0.0),
            regime_states=data.get("regime_states", {}),
        )


@dataclass
class AdaptationReport:
    """Summary of rule adaptations from a single feedback cycle."""

    adapted_rules: list[tuple[str, float]]  # (rule_name, delta)
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None

    @property
    def any_adapted(self) -> bool:
        return len(self.adapted_rules) > 0

    @property
    def total_rules_adapted(self) -> int:
        return len(self.adapted_rules)

    def summary(self) -> str:
        if not self.any_adapted:
            return "No rules adapted"
        lines = [f"Adapted {len(self.adapted_rules)} rules:"]
        for name, delta in self.adapted_rules:
            direction = "tightened" if delta < 0 else "relaxed"
            lines.append(f"  - {name}: {direction} by {abs(delta):.4f}")
        return "\n".join(lines)


class AdaptiveRuleBase(RuleBase):
    """
    Base class for rules that adapt their limits based on HiFi feedback.

    Subclasses implement `adapt()` which receives HiFi results and returns
    the delta (change) to apply to the rule's limit. The base class handles
    state management, persistence, and logging.

    Attributes:
        learning_rate: Factor controlling adaptation speed (0.01 - 0.2 typical)
        min_observations: Minimum HiFi samples before adaptation starts
        window_size: Rolling window size for margin history
        margin_threshold_tight: Avg margin > this triggers tightening
        margin_threshold_relax: Avg margin < this triggers relaxation
    """

    # Adaptation hyperparameters (override in subclass if needed)
    learning_rate: float = 0.05
    min_observations: int = 10
    window_size: int = 50
    margin_threshold_tight: float = 0.20  # 20% headroom -> tighten
    margin_threshold_relax: float = 0.05  # 5% headroom -> relax

    def __init__(self, initial_limit: float, **kwargs):
        """
        Initialize an adaptive rule.

        Args:
            initial_limit: Starting constraint value
            **kwargs: Override hyperparameters (learning_rate, min_observations, etc.)
        """
        self._state = AdaptiveRuleState(
            limit=initial_limit,
            initial_limit=initial_limit,
        )

        # Allow hyperparameter overrides
        for key in [
            "learning_rate",
            "min_observations",
            "window_size",
            "margin_threshold_tight",
            "margin_threshold_relax",
        ]:
            if key in kwargs:
                setattr(self, key, kwargs[key])

    @property
    def limit(self) -> float:
        """Current (potentially adapted) limit value."""
        return self._state.limit

    @limit.setter
    def limit(self, value: float) -> None:
        """Update the limit (tracks delta automatically)."""
        delta = value - self._state.limit
        self._state.limit = value
        self._state.total_delta += delta
        self._state.last_adapted = datetime.now()

    @property
    def initial_limit(self) -> float:
        """Original design limit (immutable reference)."""
        return self._state.initial_limit

    @abstractmethod
    def adapt(self, hifi_result: dict[str, Any], predicted: float) -> float:
        """
        Update internal limit based on HiFi vs predicted comparison.

        Called after each HiFi simulation result is available.

        Args:
            hifi_result: Dict containing actual simulation outputs
            predicted: The surrogate's prediction for this candidate

        Returns:
            delta: Change applied to limit (negative=tighten, positive=relax, 0=no change)
        """
        ...

    def get_state(self) -> AdaptiveRuleState:
        """Export current state for persistence."""
        return self._state

    def load_state(self, state: AdaptiveRuleState) -> None:
        """Restore state from persistence."""
        self._state = state

    def get_regime_limit(self, regime_id: int) -> float:
        """Get limit for a specific operating regime (or default)."""
        return self._state.regime_states.get(regime_id, self._state.limit)

    def set_regime_limit(self, regime_id: int, value: float) -> None:
        """Set per-regime limit."""
        self._state.regime_states[regime_id] = value

    def _update_margin_history(self, margin: float) -> None:
        """Add margin to rolling history, trimming if needed."""
        self._state.margin_history.append(margin)
        if len(self._state.margin_history) > self.window_size:
            self._state.margin_history.pop(0)
        self._state.n_observations += 1

    def _should_adapt(self) -> bool:
        """Check if enough observations for adaptation."""
        return self._state.n_observations >= self.min_observations

    def _get_avg_margin(self) -> float:
        """Compute average margin over history window."""
        if not self._state.margin_history:
            return 0.0
        return float(np.mean(self._state.margin_history))

    def reset_to_initial(self) -> None:
        """Reset rule to initial limit (discards learned state)."""
        self._state.limit = self._state.initial_limit
        self._state.margin_history.clear()
        self._state.n_observations = 0
        self._state.total_delta = 0.0
        self._state.last_adapted = None
        self._state.regime_states.clear()
        log.info(f"Reset {self.name} to initial limit {self.initial_limit}")


class MaxCrownTemperatureAdaptive(AdaptiveRuleBase):
    """
    Adaptive thermal constraint that tightens/relaxes based on FEA feedback.

    Monitors T_crown_max from HiFi thermal simulations and adjusts the
    temperature limit to find the true safe operating boundary.
    """

    category = RuleCategory.THERMODYNAMIC
    name = "max_crown_temperature_adaptive"
    default_severity = RuleSeverity.ERROR

    def __init__(self, limit_k: float = 573.0, **kwargs):
        """
        Args:
            limit_k: Initial crown temperature limit in Kelvin (default 300°C)
        """
        super().__init__(initial_limit=limit_k, **kwargs)

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        """Check if crown temperature is within adaptive limit."""
        t_crown = context.get("T_crown_max", context.get("T_crown", 0.0))

        margin = (self.limit - t_crown) / self.limit if self.limit > 0 else 0.0
        passed = t_crown <= self.limit

        return RuleResult(
            rule_name=self.name,
            passed=passed,
            margin=margin,
            message=f"T_crown={t_crown:.1f}K vs limit={self.limit:.1f}K (margin={margin:.1%})",
            severity=self.default_severity if not passed else RuleSeverity.INFO,
            context={"T_crown_max": t_crown, "limit_k": self.limit, "margin": margin},
        )

    def adapt(self, hifi_result: dict[str, Any], predicted: float) -> float:
        """
        Adjust limit based on HiFi T_crown_max vs prediction.

        Strategy:
        - If avg margin > 20%: tighten (allow more aggressive designs)
        - If avg margin < 5%: relax (was rejecting valid designs)
        """
        actual = hifi_result.get("T_crown_max", predicted)
        margin = (self.limit - actual) / self.limit if self.limit > 0 else 0.0

        self._update_margin_history(margin)

        if not self._should_adapt():
            return 0.0

        avg_margin = self._get_avg_margin()
        delta = 0.0

        # Tighten if consistent headroom
        if avg_margin > self.margin_threshold_tight:
            delta = -self.learning_rate * self.limit * avg_margin
            self.limit = self.limit + delta
            log.info(f"{self.name}: Tightened by {abs(delta):.2f}K (avg_margin={avg_margin:.1%})")

        # Relax if too tight
        elif avg_margin < self.margin_threshold_relax:
            delta = self.learning_rate * self.limit * 0.05
            self.limit = self.limit + delta
            log.info(f"{self.name}: Relaxed by {delta:.2f}K (avg_margin={avg_margin:.1%})")

        return delta


class MaxContactStressAdaptive(AdaptiveRuleBase):
    """
    Adaptive mechanical constraint for gear contact stress.

    Monitors von Mises max stress from CalculiX FEA and adjusts
    the Hertzian contact stress limit.
    """

    category = RuleCategory.MECHANICAL
    name = "max_contact_stress_adaptive"
    default_severity = RuleSeverity.ERROR

    def __init__(self, limit_mpa: float = 1500.0, **kwargs):
        """
        Args:
            limit_mpa: Initial contact stress limit in MPa
        """
        super().__init__(initial_limit=limit_mpa, **kwargs)

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        """Check if contact stress is within adaptive limit."""
        stress = context.get("von_mises_max", context.get("contact_stress", 0.0))

        margin = (self.limit - stress) / self.limit if self.limit > 0 else 0.0
        passed = stress <= self.limit

        return RuleResult(
            rule_name=self.name,
            passed=passed,
            margin=margin,
            message=f"σ_max={stress:.1f}MPa vs limit={self.limit:.1f}MPa",
            severity=self.default_severity if not passed else RuleSeverity.INFO,
            context={"von_mises_max": stress, "limit_mpa": self.limit, "margin": margin},
        )

    def adapt(self, hifi_result: dict[str, Any], predicted: float) -> float:
        """Adjust limit based on HiFi stress results."""
        actual = hifi_result.get("von_mises_max", predicted)
        margin = (self.limit - actual) / self.limit if self.limit > 0 else 0.0

        self._update_margin_history(margin)

        if not self._should_adapt():
            return 0.0

        avg_margin = self._get_avg_margin()
        delta = 0.0

        if avg_margin > self.margin_threshold_tight:
            delta = -self.learning_rate * self.limit * avg_margin
            self.limit = self.limit + delta
            log.info(f"{self.name}: Tightened by {abs(delta):.2f}MPa")

        elif avg_margin < self.margin_threshold_relax:
            delta = self.learning_rate * self.limit * 0.05
            self.limit = self.limit + delta
            log.info(f"{self.name}: Relaxed by {delta:.2f}MPa")

        return delta


__all__ = [
    "AdaptiveRuleState",
    "AdaptiveRuleBase",
    "AdaptationReport",
    "MaxCrownTemperatureAdaptive",
    "MaxContactStressAdaptive",
]
