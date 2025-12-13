"""
Cam-specific constraint definitions.

This module defines constraints specific to cam follower motion law problems,
including stroke, timing, and cam-specific boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

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
    zero_accel_duration_percent: float | None = (
        None  # % of cycle with zero acceleration (can be anywhere in cycle)
    )

    # Optional constraints
    max_velocity: float | None = None
    max_acceleration: float | None = None
    max_jerk: float | None = None

    # Boundary conditions (optional - defaults to dwell at TDC and BDC)
    dwell_at_tdc: bool = True  # Zero velocity at TDC (0°)
    dwell_at_bdc: bool = True  # Zero velocity at BDC (180°)

    def __post_init__(self) -> None:
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
                raise ValueError(
                    "Zero acceleration duration percent must be between 0 and 100",
                )
            # Note: Zero acceleration duration can be anywhere in the cycle
            # and is not limited by upstroke duration

    def _register_constraints(self) -> None:
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

    # ... (skipping method bodies)

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> CamMotionConstraints:
        """Create cam constraints from dictionary format."""
        return cls(**data)
