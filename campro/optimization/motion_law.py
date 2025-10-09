"""
Motion law optimization for linear followers.

This module provides proper motion law optimization using cam angle as the
independent variable, with real optimization methods instead of analytical
solutions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class MotionType(Enum):
    """Types of motion law optimization."""
    MINIMUM_JERK = "minimum_jerk"
    MINIMUM_TIME = "minimum_time"
    MINIMUM_ENERGY = "minimum_energy"


@dataclass
class MotionLawConstraints:
    """Constraints for motion law optimization."""

    # User-defined constraints
    stroke: float                    # Total displacement (mm)
    upstroke_duration_percent: float # % of cycle for upstroke (0-100)
    zero_accel_duration_percent: float # % of cycle with zero acceleration (0-100)

    # Physical limits (optional)
    max_velocity: Optional[float] = None      # mm/rad
    max_acceleration: Optional[float] = None  # mm/rad²
    max_jerk: Optional[float] = None          # mm/rad³

    def __post_init__(self):
        """Validate and sanitize constraints after initialization."""
        # Be lenient: clamp invalid values rather than raising, to keep integrations robust
        if self.stroke <= 0:
            log.warning("Stroke must be positive; clamping to small positive value")
            self.stroke = 1e-6
        # Clamp percentages to [0, 100]
        if self.upstroke_duration_percent < 0 or self.upstroke_duration_percent > 100:
            log.warning("Upstroke duration percent out of range; clamping to [0, 100]")
            self.upstroke_duration_percent = float(np.clip(self.upstroke_duration_percent, 0.0, 100.0))
        if self.zero_accel_duration_percent < 0 or self.zero_accel_duration_percent > 100:
            log.warning("Zero acceleration duration percent out of range; clamping to [0, 100]")
            self.zero_accel_duration_percent = float(np.clip(self.zero_accel_duration_percent, 0.0, 100.0))
        # Ensure total does not exceed 100%
        total = self.upstroke_duration_percent + self.zero_accel_duration_percent
        if total > 100.0:
            excess = total - 100.0
            log.warning("Sum of upstroke and zero-accel durations exceeds 100%; reducing zero-accel by %.2f%%", excess)
            self.zero_accel_duration_percent = max(0.0, self.zero_accel_duration_percent - excess)

    @property
    def upstroke_angle(self) -> float:
        """Upstroke angle in radians."""
        return 2 * np.pi * self.upstroke_duration_percent / 100.0

    @property
    def downstroke_angle(self) -> float:
        """Downstroke angle in radians."""
        return 2 * np.pi - self.upstroke_angle

    @property
    def zero_accel_angle(self) -> float:
        """Zero acceleration angle in radians."""
        return 2 * np.pi * self.zero_accel_duration_percent / 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraints to dictionary format."""
        return {
            "stroke": self.stroke,
            "upstroke_duration_percent": self.upstroke_duration_percent,
            "zero_accel_duration_percent": self.zero_accel_duration_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
        }


@dataclass
class MotionLawResult:
    """Result of motion law optimization."""

    # Motion law data (vs cam angle)
    cam_angle: np.ndarray        # Cam angles (0° to 360°)
    position: np.ndarray         # Linear follower position (mm)
    velocity: np.ndarray         # Linear follower velocity (mm/rad)
    acceleration: np.ndarray     # Linear follower acceleration (mm/rad²)
    jerk: np.ndarray            # Linear follower jerk (mm/rad³)

    # Optimization info
    objective_value: float       # Final objective function value
    convergence_status: str      # "converged", "failed", etc.
    solve_time: float           # Time to solve (seconds)
    iterations: int             # Number of iterations

    # User constraints used
    stroke: float
    upstroke_duration_percent: float
    zero_accel_duration_percent: float
    motion_type: str

    def __post_init__(self):
        """Validate result after initialization."""
        if len(self.cam_angle) != len(self.position):
            raise ValueError("All arrays must have the same length")
        if len(self.cam_angle) != len(self.velocity):
            raise ValueError("All arrays must have the same length")
        if len(self.cam_angle) != len(self.acceleration):
            raise ValueError("All arrays must have the same length")
        if len(self.cam_angle) != len(self.jerk):
            raise ValueError("All arrays must have the same length")
        # No additional validation here; compatibility helpers below

    # Compatibility aliases for legacy/tests expecting different names
    @property
    def theta(self) -> np.ndarray:
        return self.cam_angle

    @property
    def x(self) -> np.ndarray:
        return self.position

    @property
    def v(self) -> np.ndarray:
        return self.velocity

    @property
    def a(self) -> np.ndarray:
        return self.acceleration

    @property
    def j(self) -> np.ndarray:
        return self.jerk

    @property
    def constraints(self) -> MotionLawConstraints:
        # Provide a lightweight view using values embedded in the result
        return MotionLawConstraints(
            stroke=self.stroke,
            upstroke_duration_percent=self.upstroke_duration_percent,
            zero_accel_duration_percent=self.zero_accel_duration_percent,
        )

    @property
    def cam_angle_degrees(self) -> np.ndarray:
        """Cam angles in degrees."""
        return np.degrees(self.cam_angle)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "cam_angle": self.cam_angle,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "jerk": self.jerk,
            "objective_value": self.objective_value,
            "convergence_status": self.convergence_status,
            "solve_time": self.solve_time,
            "iterations": self.iterations,
            "stroke": self.stroke,
            "upstroke_duration_percent": self.upstroke_duration_percent,
            "zero_accel_duration_percent": self.zero_accel_duration_percent,
            "motion_type": self.motion_type,
        }


@dataclass
class ValidationResult:
    """Result of motion law validation."""

    valid: bool
    issues: List[str]

    def __str__(self) -> str:
        if self.valid:
            return "Motion law validation: PASSED"
        return f"Motion law validation: FAILED - {', '.join(self.issues)}"


class MotionLawValidator:
    """Validate motion law results for physical feasibility."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def validate(self, result: MotionLawResult) -> ValidationResult:
        """
        Validate the motion law result.
        
        Args:
            result: Motion law result to validate
            
        Returns:
            ValidationResult with validation status and issues
        """
        issues = []

        # Check boundary conditions
        if not self._check_boundary_conditions(result):
            issues.append("Boundary conditions not satisfied")

        # Check continuity
        if not self._check_continuity(result):
            issues.append("Motion law not continuous")

        # Check user constraints
        if not self._check_user_constraints(result):
            issues.append("User constraints not satisfied")

        # Check physical feasibility
        if not self._check_physical_feasibility(result):
            issues.append("Motion law not physically feasible")

        return ValidationResult(valid=len(issues) == 0, issues=issues)

    def _check_boundary_conditions(self, result: MotionLawResult) -> bool:
        """Check that boundary conditions are satisfied."""
        try:
            # Check position boundaries
            if abs(result.position[0]) > self.tolerance:
                log.warning(f"Position at start: {result.position[0]}, expected: 0")
                return False
            if abs(result.position[-1]) > self.tolerance:
                log.warning(f"Position at end: {result.position[-1]}, expected: 0")
                return False

            # Check velocity boundaries
            if abs(result.velocity[0]) > self.tolerance:
                log.warning(f"Velocity at start: {result.velocity[0]}, expected: 0")
                return False
            if abs(result.velocity[-1]) > self.tolerance:
                log.warning(f"Velocity at end: {result.velocity[-1]}, expected: 0")
                return False

            # Check acceleration boundaries
            if abs(result.acceleration[0]) > self.tolerance:
                log.warning(f"Acceleration at start: {result.acceleration[0]}, expected: 0")
                return False
            if abs(result.acceleration[-1]) > self.tolerance:
                log.warning(f"Acceleration at end: {result.acceleration[-1]}, expected: 0")
                return False

            return True
        except Exception as e:
            log.error(f"Error checking boundary conditions: {e}")
            return False

    def _check_continuity(self, result: MotionLawResult) -> bool:
        """Check that motion law is continuous."""
        try:
            # Check position continuity (more lenient for optimization results)
            position_diff = np.diff(result.position)
            if np.any(np.abs(position_diff) > 100 * self.tolerance):
                log.warning("Position not continuous")
                return False

            # Check velocity continuity (more lenient for optimization results)
            velocity_diff = np.diff(result.velocity)
            if np.any(np.abs(velocity_diff) > 100 * self.tolerance):
                log.warning("Velocity not continuous")
                return False

            # Check acceleration continuity (more lenient for optimization results)
            acceleration_diff = np.diff(result.acceleration)
            if np.any(np.abs(acceleration_diff) > 100 * self.tolerance):
                log.warning("Acceleration not continuous")
                return False

            return True
        except Exception as e:
            log.error(f"Error checking continuity: {e}")
            return False

    def _check_user_constraints(self, result: MotionLawResult) -> bool:
        """Check that user constraints are satisfied."""
        try:
            # Check stroke constraint (more lenient for optimization results)
            max_position = np.max(result.position)
            if abs(max_position - result.stroke) > 0.1:  # Allow 0.1mm tolerance
                log.warning(f"Max position: {max_position}, expected stroke: {result.stroke}")
                return False

            # Check upstroke timing constraint (more lenient for optimization results)
            upstroke_angle = 2 * np.pi * result.upstroke_duration_percent / 100.0
            upstroke_idx = np.argmin(np.abs(result.cam_angle - upstroke_angle))
            upstroke_position = result.position[upstroke_idx]
            if abs(upstroke_position - result.stroke) > 0.1:  # Allow 0.1mm tolerance
                log.warning(f"Position at upstroke end: {upstroke_position}, expected: {result.stroke}")
                return False

            return True
        except Exception as e:
            log.error(f"Error checking user constraints: {e}")
            return False

    def _check_physical_feasibility(self, result: MotionLawResult) -> bool:
        """Check that motion law is physically feasible."""
        try:
            # Check for negative positions (should not happen for linear follower)
            if np.any(result.position < -self.tolerance):
                log.warning("Negative positions detected")
                return False

            # Check for infinite or NaN values
            if np.any(np.isinf(result.position)) or np.any(np.isnan(result.position)):
                log.warning("Infinite or NaN positions detected")
                return False
            if np.any(np.isinf(result.velocity)) or np.any(np.isnan(result.velocity)):
                log.warning("Infinite or NaN velocities detected")
                return False
            if np.any(np.isinf(result.acceleration)) or np.any(np.isnan(result.acceleration)):
                log.warning("Infinite or NaN accelerations detected")
                return False
            if np.any(np.isinf(result.jerk)) or np.any(np.isnan(result.jerk)):
                log.warning("Infinite or NaN jerks detected")
                return False

            return True
        except Exception as e:
            log.error(f"Error checking physical feasibility: {e}")
            return False
