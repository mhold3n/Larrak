"""
Base constraint classes and interfaces.

This module defines the fundamental constraint system that all other
constraint types inherit from, providing a consistent interface for
constraint definition, validation, and violation checking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class ConstraintType(Enum):
    """Types of constraints that can be applied."""

    # State constraints
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    JERK = "jerk"

    # Control constraints
    CONTROL = "control"

    # Boundary constraints
    INITIAL_STATE = "initial_state"
    FINAL_STATE = "final_state"

    # Custom constraints
    CUSTOM = "custom"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint_type: ConstraintType
    constraint_name: str
    violation_value: float
    limit_value: float
    time_index: Optional[int] = None
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = f"{self.constraint_name} violation: {self.violation_value:.3f} exceeds limit {self.limit_value:.3f}"


class BaseConstraints(ABC):
    """
    Base class for all constraint systems.

    Provides a common interface for constraint definition, validation,
    and violation checking across different domains (motion, cam, physics).
    """

    def __init__(self):
        self._constraints: Dict[str, Any] = {}
        self._violations: List[ConstraintViolation] = []

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate all constraints.

        Returns:
            bool: True if all constraints are valid, False otherwise
        """

    @abstractmethod
    def check_violations(
        self, solution: Dict[str, np.ndarray],
    ) -> List[ConstraintViolation]:
        """
        Check for constraint violations in a solution.

        Args:
            solution: Dictionary containing solution arrays (time, position, velocity, etc.)

        Returns:
            List of constraint violations found
        """

    def add_constraint(self, name: str, constraint: Any) -> None:
        """Add a constraint to the constraint set."""
        self._constraints[name] = constraint
        log.debug(f"Added constraint: {name}")

    def get_constraint(self, name: str) -> Any:
        """Get a constraint by name."""
        return self._constraints.get(name)

    def remove_constraint(self, name: str) -> None:
        """Remove a constraint by name."""
        if name in self._constraints:
            del self._constraints[name]
            log.debug(f"Removed constraint: {name}")

    def list_constraints(self) -> List[str]:
        """List all constraint names."""
        return list(self._constraints.keys())

    def clear_violations(self) -> None:
        """Clear all constraint violations."""
        self._violations.clear()

    def get_violations(self) -> List[ConstraintViolation]:
        """Get all constraint violations."""
        return self._violations.copy()

    def has_violations(self) -> bool:
        """Check if there are any constraint violations."""
        return len(self._violations) > 0

    def _add_violation(self, violation: ConstraintViolation) -> None:
        """Add a constraint violation."""
        self._violations.append(violation)
        log.warning(f"Constraint violation: {violation.message}")

    def _check_bounds(
        self,
        values: np.ndarray,
        bounds: Tuple[float, float],
        constraint_type: ConstraintType,
        name: str,
    ) -> List[ConstraintViolation]:
        """
        Check if values are within specified bounds.

        Args:
            values: Array of values to check
            bounds: (min, max) bounds
            constraint_type: Type of constraint being checked
            name: Name of the constraint

        Returns:
            List of violations found
        """
        violations = []

        if bounds is None:
            return violations

        min_val, max_val = bounds

        # Check lower bound
        if min_val is not None:
            min_violations = np.where(values < min_val)[0]
            for idx in min_violations:
                violation = ConstraintViolation(
                    constraint_type=constraint_type,
                    constraint_name=f"{name}_min",
                    violation_value=values[idx],
                    limit_value=min_val,
                    time_index=int(idx),
                    message=f"{name} minimum violation at index {idx}: {values[idx]:.3f} < {min_val:.3f}",
                )
                violations.append(violation)

        # Check upper bound
        if max_val is not None:
            max_violations = np.where(values > max_val)[0]
            for idx in max_violations:
                violation = ConstraintViolation(
                    constraint_type=constraint_type,
                    constraint_name=f"{name}_max",
                    violation_value=values[idx],
                    limit_value=max_val,
                    time_index=int(idx),
                    message=f"{name} maximum violation at index {idx}: {values[idx]:.3f} > {max_val:.3f}",
                )
                violations.append(violation)

        return violations

    def _check_boundary_condition(
        self,
        actual: float,
        expected: float,
        constraint_type: ConstraintType,
        name: str,
        tolerance: float = 1e-6,
    ) -> Optional[ConstraintViolation]:
        """
        Check if a boundary condition is satisfied.

        Args:
            actual: Actual value
            expected: Expected value
            constraint_type: Type of constraint
            name: Name of the constraint
            tolerance: Tolerance for equality check

        Returns:
            ConstraintViolation if violated, None otherwise
        """
        if abs(actual - expected) > tolerance:
            return ConstraintViolation(
                constraint_type=constraint_type,
                constraint_name=name,
                violation_value=actual,
                limit_value=expected,
                message=f"{name} boundary condition violation: {actual:.3f} != {expected:.3f}",
            )
        return None
