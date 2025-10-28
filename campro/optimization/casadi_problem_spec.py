"""
Problem specification interface for CasADi optimization.

This module provides clean interfaces for defining optimization problems
with proper validation and conversion utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class OptimizationObjective(Enum):
    """Available optimization objectives."""

    MINIMIZE_JERK = "minimize_jerk"
    MAXIMIZE_THERMAL_EFFICIENCY = "maximize_thermal_efficiency"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_ENERGY = "minimize_energy"
    SMOOTHNESS = "smoothness"


class CollocationMethod(Enum):
    """Available collocation methods."""

    LEGENDRE = "legendre"
    RADAU = "radau"
    TRAPEZOIDAL = "trapezoidal"


@dataclass
class CasADiMotionProblem:
    """
    Problem specification for CasADi motion law optimization.

    This class provides a clean interface for defining optimization problems
    with proper validation and default values.
    """

    # Boundary conditions
    stroke: float
    cycle_time: float
    upstroke_percent: float

    # Physical constraints
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    compression_ratio_limits: tuple[float, float] = (20.0, 70.0)

    # Optimization objectives
    objectives: list[OptimizationObjective] = field(
        default_factory=lambda: [
            OptimizationObjective.MINIMIZE_JERK,
            OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
        ],
    )

    # Objective weights
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "jerk": 1.0,
            "thermal_efficiency": 0.1,
            "smoothness": 0.01,
            "time": 0.0,
            "energy": 0.0,
        },
    )

    # Collocation settings
    n_segments: int = 50
    poly_order: int = 3
    collocation_method: CollocationMethod = CollocationMethod.LEGENDRE

    # Solver settings
    solver_options: dict[str, Any] = field(
        default_factory=lambda: {
            "ipopt.linear_solver": "ma57",
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-6,
            "ipopt.print_level": 0,
            "ipopt.warm_start_init_point": "yes",
        },
    )

    # Thermal efficiency settings
    thermal_efficiency_target: float = 0.55
    heat_transfer_coeff: float = 0.1
    friction_coeff: float = 0.01

    def __post_init__(self):
        """Validate problem specification after initialization."""
        self._validate_parameters()
        self._normalize_weights()

    def _validate_parameters(self) -> None:
        """Validate problem parameters."""
        if self.stroke <= 0:
            raise ValueError("Stroke must be positive")

        if self.cycle_time <= 0:
            raise ValueError("Cycle time must be positive")

        if not 0 < self.upstroke_percent < 100:
            raise ValueError("Upstroke percent must be between 0 and 100")

        if self.max_velocity <= 0:
            raise ValueError("Max velocity must be positive")

        if self.max_acceleration <= 0:
            raise ValueError("Max acceleration must be positive")

        if self.max_jerk <= 0:
            raise ValueError("Max jerk must be positive")

        if self.compression_ratio_limits[0] >= self.compression_ratio_limits[1]:
            raise ValueError("Compression ratio limits must be ordered")

        if self.n_segments <= 0:
            raise ValueError("Number of segments must be positive")

        if self.poly_order < 1:
            raise ValueError("Polynomial order must be at least 1")

        if not 0 < self.thermal_efficiency_target < 1:
            raise ValueError("Thermal efficiency target must be between 0 and 1")

    def _normalize_weights(self) -> None:
        """Normalize objective weights."""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for key in self.weights:
                self.weights[key] /= total_weight

    def to_dict(self) -> dict[str, Any]:
        """Convert problem to dictionary."""
        return {
            "stroke": self.stroke,
            "cycle_time": self.cycle_time,
            "upstroke_percent": self.upstroke_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
            "compression_ratio_limits": self.compression_ratio_limits,
            "objectives": [obj.value for obj in self.objectives],
            "weights": self.weights,
            "n_segments": self.n_segments,
            "poly_order": self.poly_order,
            "collocation_method": self.collocation_method.value,
            "solver_options": self.solver_options,
            "thermal_efficiency_target": self.thermal_efficiency_target,
            "heat_transfer_coeff": self.heat_transfer_coeff,
            "friction_coeff": self.friction_coeff,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CasADiMotionProblem:
        """Create problem from dictionary."""
        # Convert objectives back to enum
        objectives = [OptimizationObjective(obj) for obj in data.get("objectives", [])]

        # Convert collocation method back to enum
        collocation_method = CollocationMethod(
            data.get("collocation_method", "legendre"),
        )

        return cls(
            stroke=data["stroke"],
            cycle_time=data["cycle_time"],
            upstroke_percent=data["upstroke_percent"],
            max_velocity=data["max_velocity"],
            max_acceleration=data["max_acceleration"],
            max_jerk=data["max_jerk"],
            compression_ratio_limits=tuple(
                data.get("compression_ratio_limits", (20.0, 70.0)),
            ),
            objectives=objectives,
            weights=data.get("weights", {}),
            n_segments=data.get("n_segments", 50),
            poly_order=data.get("poly_order", 3),
            collocation_method=collocation_method,
            solver_options=data.get("solver_options", {}),
            thermal_efficiency_target=data.get("thermal_efficiency_target", 0.55),
            heat_transfer_coeff=data.get("heat_transfer_coeff", 0.1),
            friction_coeff=data.get("friction_coeff", 0.01),
        )

    def update_weights(self, **weights: float) -> None:
        """Update objective weights."""
        self.weights.update(weights)
        self._normalize_weights()

    def add_objective(
        self, objective: OptimizationObjective, weight: float = 1.0,
    ) -> None:
        """Add optimization objective."""
        if objective not in self.objectives:
            self.objectives.append(objective)
        self.weights[objective.value] = weight
        self._normalize_weights()

    def remove_objective(self, objective: OptimizationObjective) -> None:
        """Remove optimization objective."""
        if objective in self.objectives:
            self.objectives.remove(objective)
        if objective.value in self.weights:
            del self.weights[objective.value]
        self._normalize_weights()

    def get_frequency(self) -> float:
        """Get engine frequency in Hz."""
        return 1.0 / self.cycle_time

    def get_upstroke_time(self) -> float:
        """Get upstroke time in seconds."""
        return self.cycle_time * self.upstroke_percent / 100.0

    def get_downstroke_time(self) -> float:
        """Get downstroke time in seconds."""
        return self.cycle_time * (100 - self.upstroke_percent) / 100.0

    def get_max_compression_ratio(self) -> float:
        """Get maximum possible compression ratio."""
        clearance = 0.002  # 2mm clearance
        return (self.stroke + clearance) / clearance

    def is_feasible(self) -> bool:
        """Check if problem is feasible."""
        try:
            self._validate_parameters()
            return True
        except ValueError:
            return False

    def get_problem_summary(self) -> str:
        """Get human-readable problem summary."""
        return (
            f"CasADi Motion Problem: "
            f"stroke={self.stroke:.3f}m, "
            f"cycle_time={self.cycle_time:.3f}s, "
            f"frequency={self.get_frequency():.1f}Hz, "
            f"upstroke={self.upstroke_percent:.1f}%, "
            f"objectives={[obj.value for obj in self.objectives]}"
        )


@dataclass
class CasADiOptimizationResult:
    """Result from CasADi optimization."""

    # Optimization status
    successful: bool
    solve_time: float
    objective_value: float
    n_iterations: int

    # Solution variables
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    # Metadata
    problem_spec: CasADiMotionProblem
    solver_stats: dict[str, Any]
    thermal_efficiency: float | None = None

    def get_motion_profile(self) -> dict[str, np.ndarray]:
        """Get complete motion profile."""
        return {
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "jerk": self.jerk,
        }

    def get_efficiency_metrics(self) -> dict[str, float]:
        """Get thermal efficiency metrics."""
        if self.thermal_efficiency is None:
            return {}

        return {
            "thermal_efficiency": self.thermal_efficiency,
            "efficiency_target": self.problem_spec.thermal_efficiency_target,
            "efficiency_achieved": self.thermal_efficiency
            >= self.problem_spec.thermal_efficiency_target,
        }

    def get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics."""
        return {
            "solve_time": self.solve_time,
            "n_iterations": self.n_iterations,
            "objective_value": self.objective_value,
            "max_velocity": float(np.max(np.abs(self.velocity))),
            "max_acceleration": float(np.max(np.abs(self.acceleration))),
            "max_jerk": float(np.max(np.abs(self.jerk))),
        }


def create_default_problem(
    stroke: float = 0.1, cycle_time: float = 0.0385, upstroke_percent: float = 50.0,
) -> CasADiMotionProblem:
    """
    Create a default optimization problem.

    Parameters
    ----------
    stroke : float
        Stroke length in meters
    cycle_time : float
        Cycle time in seconds
    upstroke_percent : float
        Upstroke percentage

    Returns
    -------
    CasADiMotionProblem
        Default problem specification
    """
    return CasADiMotionProblem(
        stroke=stroke,
        cycle_time=cycle_time,
        upstroke_percent=upstroke_percent,
        max_velocity=5.0,
        max_acceleration=500.0,
        max_jerk=50000.0,
        compression_ratio_limits=(20.0, 70.0),
        objectives=[
            OptimizationObjective.MINIMIZE_JERK,
            OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
        ],
        weights={
            "jerk": 1.0,
            "thermal_efficiency": 0.1,
            "smoothness": 0.01,
        },
    )


def create_high_efficiency_problem(
    stroke: float = 0.1, cycle_time: float = 0.0385,
) -> CasADiMotionProblem:
    """
    Create a high-efficiency optimization problem.

    Parameters
    ----------
    stroke : float
        Stroke length in meters
    cycle_time : float
        Cycle time in seconds

    Returns
    -------
    CasADiMotionProblem
        High-efficiency problem specification
    """
    return CasADiMotionProblem(
        stroke=stroke,
        cycle_time=cycle_time,
        upstroke_percent=45.0,  # Slightly shorter upstroke for efficiency
        max_velocity=4.0,  # Lower velocity for efficiency
        max_acceleration=400.0,  # Lower acceleration
        max_jerk=40000.0,  # Lower jerk
        compression_ratio_limits=(25.0, 70.0),  # Higher minimum CR
        objectives=[
            OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
            OptimizationObjective.MINIMIZE_JERK,
        ],
        weights={
            "thermal_efficiency": 1.0,
            "jerk": 0.1,
            "smoothness": 0.01,
        },
        thermal_efficiency_target=0.60,  # Higher target
    )


def create_smooth_motion_problem(
    stroke: float = 0.1, cycle_time: float = 0.0385,
) -> CasADiMotionProblem:
    """
    Create a smooth motion optimization problem.

    Parameters
    ----------
    stroke : float
        Stroke length in meters
    cycle_time : float
        Cycle time in seconds

    Returns
    -------
    CasADiMotionProblem
        Smooth motion problem specification
    """
    return CasADiMotionProblem(
        stroke=stroke,
        cycle_time=cycle_time,
        upstroke_percent=50.0,
        max_velocity=3.0,  # Lower velocity for smoothness
        max_acceleration=300.0,  # Lower acceleration
        max_jerk=30000.0,  # Lower jerk
        compression_ratio_limits=(20.0, 50.0),  # Lower CR range
        objectives=[
            OptimizationObjective.SMOOTHNESS,
            OptimizationObjective.MINIMIZE_JERK,
        ],
        weights={
            "smoothness": 1.0,
            "jerk": 0.5,
            "thermal_efficiency": 0.01,
        },
    )
