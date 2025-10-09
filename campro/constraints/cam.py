"""
Cam-specific constraint definitions.

This module defines constraints specific to cam follower motion law problems,
including stroke, timing, and cam-specific boundary conditions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from campro.logging import get_logger

from .base import BaseConstraints, ConstraintType, ConstraintViolation
from .motion import MotionConstraints

log = get_logger(__name__)


class CamConstraintType(Enum):
    """Types of cam-specific constraints."""

    STROKE = "stroke"
    UPSTROKE_DURATION = "upstroke_duration"
    ZERO_ACCEL_DURATION = "zero_accel_duration"
    CYCLE_TIME = "cycle_time"
    DWELL_AT_TDC = "dwell_at_tdc"
    DWELL_AT_BDC = "dwell_at_bdc"
    MAX_VELOCITY = "max_velocity"
    MAX_ACCELERATION = "max_acceleration"
    MAX_JERK = "max_jerk"


@dataclass
class CamMotionConstraints(BaseConstraints):
    """
    Simplified constraints for cam follower motion law problems.
    
    This class provides intuitive cam-specific constraints that are easier to use
    than the general motion constraints.
    """

    # Core cam parameters
    stroke: float  # Total follower stroke (required)
    upstroke_duration_percent: float  # % of cycle for upstroke (0-100)
    zero_accel_duration_percent: Optional[float] = None  # % of cycle with zero acceleration (can be anywhere in cycle)

    # Optional constraints
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    max_jerk: Optional[float] = None

    # Boundary conditions (optional - defaults to dwell at TDC and BDC)
    dwell_at_tdc: bool = True  # Zero velocity at TDC (0°)
    dwell_at_bdc: bool = True  # Zero velocity at BDC (180°)

    def __post_init__(self):
        """Validate cam constraint parameters and initialize constraint system."""
        super().__init__()
        self._register_constraints()

        # Validate parameters
        if self.stroke <= 0:
            raise ValueError("Stroke must be positive")
        if not 0 <= self.upstroke_duration_percent <= 100:
            raise ValueError("Upstroke duration percent must be between 0 and 100")
        if self.zero_accel_duration_percent is not None:
            if not 0 <= self.zero_accel_duration_percent <= 100:
                raise ValueError("Zero acceleration duration percent must be between 0 and 100")
            # Note: Zero acceleration duration can be anywhere in the cycle
            # and is not limited by upstroke duration

    def _register_constraints(self):
        """Register all cam constraints in the constraint system."""
        constraints = {
            "stroke": self.stroke,
            "upstroke_duration_percent": self.upstroke_duration_percent,
            "zero_accel_duration_percent": self.zero_accel_duration_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
            "dwell_at_tdc": self.dwell_at_tdc,
            "dwell_at_bdc": self.dwell_at_bdc,
        }

        for name, value in constraints.items():
            if value is not None:
                self.add_constraint(name, value)

    def validate(self) -> bool:
        """
        Validate all cam constraints.
        
        Returns:
            bool: True if all constraints are valid, False otherwise
        """
        self.clear_violations()
        is_valid = True

        # Validate stroke
        if self.stroke <= 0:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.CUSTOM,
                constraint_name="stroke_positive",
                violation_value=self.stroke,
                limit_value=0.0,
                message=f"Stroke must be positive, got {self.stroke}",
            )
            self._add_violation(violation)
            is_valid = False

        # Validate upstroke duration
        if not 0 <= self.upstroke_duration_percent <= 100:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.CUSTOM,
                constraint_name="upstroke_duration_range",
                violation_value=self.upstroke_duration_percent,
                limit_value=100.0,
                message=f"Upstroke duration must be between 0 and 100%, got {self.upstroke_duration_percent}",
            )
            self._add_violation(violation)
            is_valid = False

        # Validate zero acceleration duration
        if self.zero_accel_duration_percent is not None:
            if not 0 <= self.zero_accel_duration_percent <= 100:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOM,
                    constraint_name="zero_accel_duration_range",
                    violation_value=self.zero_accel_duration_percent,
                    limit_value=100.0,
                    message=f"Zero acceleration duration must be between 0 and 100%, got {self.zero_accel_duration_percent}",
                )
                self._add_violation(violation)
                is_valid = False

        # Validate max constraints
        for constraint_name, value in [
            ("max_velocity", self.max_velocity),
            ("max_acceleration", self.max_acceleration),
            ("max_jerk", self.max_jerk),
        ]:
            if value is not None and value <= 0:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOM,
                    constraint_name=f"{constraint_name}_positive",
                    violation_value=value,
                    limit_value=0.0,
                    message=f"{constraint_name} must be positive, got {value}",
                )
                self._add_violation(violation)
                is_valid = False

        log.info(f"Cam constraints validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid

    def check_violations(self, solution: Dict[str, np.ndarray]) -> List[ConstraintViolation]:
        """
        Check for constraint violations in a cam motion law solution.
        
        Args:
            solution: Dictionary containing solution arrays
                - 'time': Time array
                - 'cam_angle': Cam angle array (0-360°)
                - 'position': Position array
                - 'velocity': Velocity array
                - 'acceleration': Acceleration array
                - 'control': Control (jerk) array
                
        Returns:
            List of constraint violations found
        """
        self.clear_violations()

        # Check stroke constraint
        if "position" in solution:
            max_position = np.max(solution["position"])
            if abs(max_position - self.stroke) > 1e-6:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOM,
                    constraint_name="stroke_achievement",
                    violation_value=max_position,
                    limit_value=self.stroke,
                    message=f"Maximum position {max_position:.3f} does not match stroke {self.stroke:.3f}",
                )
                self._violations.append(violation)

        # Check velocity constraints
        if "velocity" in solution and self.max_velocity is not None:
            max_velocity = np.max(np.abs(solution["velocity"]))
            if max_velocity > self.max_velocity:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.VELOCITY,
                    constraint_name="max_velocity",
                    violation_value=max_velocity,
                    limit_value=self.max_velocity,
                    message=f"Maximum velocity {max_velocity:.3f} exceeds limit {self.max_velocity:.3f}",
                )
                self._violations.append(violation)

        # Check acceleration constraints
        if "acceleration" in solution and self.max_acceleration is not None:
            max_acceleration = np.max(np.abs(solution["acceleration"]))
            if max_acceleration > self.max_acceleration:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.ACCELERATION,
                    constraint_name="max_acceleration",
                    violation_value=max_acceleration,
                    limit_value=self.max_acceleration,
                    message=f"Maximum acceleration {max_acceleration:.3f} exceeds limit {self.max_acceleration:.3f}",
                )
                self._violations.append(violation)

        # Check jerk constraints
        if "control" in solution and self.max_jerk is not None:
            max_jerk = np.max(np.abs(solution["control"]))
            if max_jerk > self.max_jerk:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.JERK,
                    constraint_name="max_jerk",
                    violation_value=max_jerk,
                    limit_value=self.max_jerk,
                    message=f"Maximum jerk {max_jerk:.3f} exceeds limit {self.max_jerk:.3f}",
                )
                self._violations.append(violation)

        # Check dwell conditions
        if "velocity" in solution and len(solution["velocity"]) > 0:
            # Check TDC dwell (velocity at start should be zero)
            if self.dwell_at_tdc and abs(solution["velocity"][0]) > 1e-6:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.INITIAL_STATE,
                    constraint_name="dwell_at_tdc",
                    violation_value=solution["velocity"][0],
                    limit_value=0.0,
                    message=f"Velocity at TDC should be zero, got {solution['velocity'][0]:.3f}",
                )
                self._violations.append(violation)

            # Check BDC dwell (velocity at end should be zero)
            if self.dwell_at_bdc and abs(solution["velocity"][-1]) > 1e-6:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.FINAL_STATE,
                    constraint_name="dwell_at_bdc",
                    violation_value=solution["velocity"][-1],
                    limit_value=0.0,
                    message=f"Velocity at BDC should be zero, got {solution['velocity'][-1]:.3f}",
                )
                self._violations.append(violation)

        log.info(f"Found {len(self._violations)} cam constraint violations")
        return self._violations.copy()

    def to_motion_constraints(self, cycle_time: float = 1.0) -> MotionConstraints:
        """
        Convert cam constraints to general motion constraints.
        
        Args:
            cycle_time: Total cycle time in seconds
            
        Returns:
            MotionConstraints object
        """
        # Calculate time segments
        upstroke_time = cycle_time * self.upstroke_duration_percent / 100.0
        downstroke_time = cycle_time - upstroke_time

        # Create motion constraints
        motion_constraints = MotionConstraints()

        # Set velocity bounds if specified
        if self.max_velocity is not None:
            motion_constraints.velocity_bounds = (-self.max_velocity, self.max_velocity)

        # Set acceleration bounds if specified
        if self.max_acceleration is not None:
            motion_constraints.acceleration_bounds = (-self.max_acceleration, self.max_acceleration)

        # Set jerk bounds if specified
        if self.max_jerk is not None:
            motion_constraints.jerk_bounds = (-self.max_jerk, self.max_jerk)
            motion_constraints.control_bounds = (-self.max_jerk, self.max_jerk)

        # Set boundary conditions based on dwell settings
        if self.dwell_at_tdc:
            motion_constraints.initial_velocity = 0.0
            motion_constraints.initial_acceleration = 0.0

        if self.dwell_at_bdc:
            motion_constraints.final_velocity = 0.0
            motion_constraints.final_acceleration = 0.0

        return motion_constraints

    def to_dict(self) -> Dict[str, any]:
        """Convert cam constraints to dictionary format."""
        return {
            "stroke": self.stroke,
            "upstroke_duration_percent": self.upstroke_duration_percent,
            "zero_accel_duration_percent": self.zero_accel_duration_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
            "dwell_at_tdc": self.dwell_at_tdc,
            "dwell_at_bdc": self.dwell_at_bdc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "CamMotionConstraints":
        """Create cam constraints from dictionary format."""
        return cls(**data)


