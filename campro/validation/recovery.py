"""
CEM Violation Recovery: Map violations to corrective actions.

This module provides recovery strategies for CEM constraint violations.
When optimization or validation fails, this module suggests how to
adjust parameters to achieve feasibility.

Usage:
    from campro.validation.recovery import RecoveryEngine

    engine = RecoveryEngine()
    actions = engine.suggest_recovery(report.violations)
    adjusted_params = engine.apply_recovery(params, actions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)

# Try to import CEM types
try:
    from truthmaker.cem import (
        ConstraintViolation,
        SuggestedActionCode,
        ViolationCode,
        ViolationSeverity,
    )

    CEM_AVAILABLE = True
except ImportError:
    CEM_AVAILABLE = False

    # Define minimal stubs for type checking
    class SuggestedActionCode(Enum):  # type: ignore
        NONE = 0
        INCREASE_SMOOTHING = 100
        REDUCE_STROKE = 101
        ADJUST_PHASE = 102
        TIGHTEN_BOUNDS = 200
        RELAX_BOUNDS = 201
        IMPROVE_INITIAL_GUESS = 400
        MANUAL_REVIEW_REQUIRED = 900


@dataclass
class RecoveryAction:
    """A concrete action to recover from a violation."""

    parameter: str
    adjustment: str  # "increase", "decrease", "set"
    value: float | None = None  # Target value for "set"
    factor: float = 1.0  # Multiplier for increase/decrease
    reason: str = ""


@dataclass
class RecoveryPlan:
    """Collection of recovery actions for a set of violations."""

    actions: list[RecoveryAction] = field(default_factory=list)
    requires_manual_review: bool = False
    confidence: float = 1.0  # 0-1, how likely actions will resolve issues
    notes: list[str] = field(default_factory=list)


class RecoveryEngine:
    """
    Engine that maps CEM violations to concrete recovery actions.

    This class interprets SuggestedActionCode from CEM violations and
    produces specific parameter adjustments that can be applied to
    the optimization configuration.
    """

    def __init__(self, conservative: bool = True):
        """
        Initialize recovery engine.

        Args:
            conservative: If True, make smaller adjustments
        """
        self.conservative = conservative
        self._adjustment_factor = 0.1 if conservative else 0.2

    def suggest_recovery(
        self,
        violations: list[Any],
    ) -> RecoveryPlan:
        """
        Generate recovery plan from violations.

        Args:
            violations: List of ConstraintViolation from CEM

        Returns:
            RecoveryPlan with suggested actions
        """
        plan = RecoveryPlan()

        for v in violations:
            action_code = getattr(v, "suggested_action", SuggestedActionCode.NONE)
            margin = getattr(v, "margin", None)

            actions = self._map_action_code(action_code, margin, v)
            plan.actions.extend(actions)

            if action_code == SuggestedActionCode.MANUAL_REVIEW_REQUIRED:
                plan.requires_manual_review = True
                plan.confidence *= 0.5

        # Deduplicate actions on same parameter
        plan.actions = self._deduplicate_actions(plan.actions)

        return plan

    def _map_action_code(
        self,
        code: SuggestedActionCode,
        margin: float | None,
        violation: Any,
    ) -> list[RecoveryAction]:
        """Map a SuggestedActionCode to concrete actions."""
        actions = []

        if code == SuggestedActionCode.INCREASE_SMOOTHING:
            actions.append(
                RecoveryAction(
                    parameter="motion.smoothing_factor",
                    adjustment="increase",
                    factor=1.0 + self._adjustment_factor,
                    reason="Reduce profile curvature to meet kinematic limits",
                )
            )

        elif code == SuggestedActionCode.REDUCE_STROKE:
            actions.append(
                RecoveryAction(
                    parameter="geometry.stroke",
                    adjustment="decrease",
                    factor=1.0 - self._adjustment_factor,
                    reason="Shorten stroke to reduce motion envelope",
                )
            )

        elif code == SuggestedActionCode.ADJUST_PHASE:
            actions.append(
                RecoveryAction(
                    parameter="motion.phase_offset",
                    adjustment="set",
                    value=0.0,  # Reset to default
                    reason="Adjust phase timing for scavenging",
                )
            )

        elif code == SuggestedActionCode.TIGHTEN_BOUNDS:
            # Tighten the violated bound
            affected = getattr(violation, "affected_variables", []) or []
            for var in affected:
                actions.append(
                    RecoveryAction(
                        parameter=f"bounds.{var}",
                        adjustment="decrease" if "max" in var else "increase",
                        factor=1.0 - self._adjustment_factor * 0.5,
                        reason=f"Tighten {var} bound to improve feasibility",
                    )
                )

        elif code == SuggestedActionCode.RELAX_BOUNDS:
            affected = getattr(violation, "affected_variables", []) or []
            for var in affected:
                actions.append(
                    RecoveryAction(
                        parameter=f"bounds.{var}",
                        adjustment="increase" if "max" in var else "decrease",
                        factor=1.0 + self._adjustment_factor * 0.5,
                        reason=f"Relax {var} bound to restore feasibility",
                    )
                )

        elif code == SuggestedActionCode.IMPROVE_INITIAL_GUESS:
            actions.append(
                RecoveryAction(
                    parameter="solver.use_warm_start",
                    adjustment="set",
                    value=1.0,  # Enable
                    reason="Use warm start from previous solution",
                )
            )

        elif code == SuggestedActionCode.MANUAL_REVIEW_REQUIRED:
            log.warning("Violation requires manual review, no automatic recovery")

        return actions

    def _deduplicate_actions(
        self,
        actions: list[RecoveryAction],
    ) -> list[RecoveryAction]:
        """Remove duplicate actions on same parameter, keeping most aggressive."""
        seen: dict[str, RecoveryAction] = {}
        for action in actions:
            key = action.parameter
            if key not in seen:
                seen[key] = action
            else:
                # Keep the more aggressive adjustment
                existing = seen[key]
                if action.adjustment == existing.adjustment:
                    if action.adjustment == "increase":
                        seen[key] = action if action.factor > existing.factor else existing
                    elif action.adjustment == "decrease":
                        seen[key] = action if action.factor < existing.factor else existing
        return list(seen.values())

    def apply_recovery(
        self,
        params: dict[str, Any],
        plan: RecoveryPlan,
    ) -> dict[str, Any]:
        """
        Apply recovery actions to parameter dictionary.

        Args:
            params: Original optimization parameters
            plan: Recovery plan from suggest_recovery

        Returns:
            Modified parameters with recovery actions applied
        """
        # Deep copy to avoid mutation
        import copy

        result = copy.deepcopy(params)

        for action in plan.actions:
            try:
                self._apply_single_action(result, action)
            except Exception as e:
                log.warning(f"Failed to apply recovery action {action.parameter}: {e}")

        return result

    def _apply_single_action(
        self,
        params: dict[str, Any],
        action: RecoveryAction,
    ) -> None:
        """Apply a single recovery action to params in-place."""
        # Parse dotted parameter path
        parts = action.parameter.split(".")
        obj = params

        # Navigate to parent
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]

        key = parts[-1]
        current = obj.get(key)

        if action.adjustment == "set":
            obj[key] = action.value
            log.debug(f"Recovery: set {action.parameter} = {action.value}")

        elif action.adjustment == "increase":
            if current is not None and isinstance(current, (int, float)):
                obj[key] = current * action.factor
                log.debug(f"Recovery: {action.parameter} *= {action.factor}")

        elif action.adjustment == "decrease":
            if current is not None and isinstance(current, (int, float)):
                obj[key] = current * action.factor
                log.debug(f"Recovery: {action.parameter} *= {action.factor}")


__all__ = [
    "RecoveryAction",
    "RecoveryEngine",
    "RecoveryPlan",
]
