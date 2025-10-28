"""
Warm-start manager for CasADi optimization.

This module implements solution history management and initial guess generation
for warm-starting CasADi optimization problems.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class SolutionRecord:
    """Record of a solved optimization problem and its solution."""

    # Problem parameters
    stroke: float
    cycle_time: float
    upstroke_percent: float
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    compression_ratio_limits: tuple[float, float]

    # Solution data
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray

    # Metadata
    solve_time: float
    objective_value: float
    n_segments: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stroke": self.stroke,
            "cycle_time": self.cycle_time,
            "upstroke_percent": self.upstroke_percent,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "max_jerk": self.max_jerk,
            "compression_ratio_limits": self.compression_ratio_limits,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "acceleration": self.acceleration.tolist(),
            "jerk": self.jerk.tolist(),
            "solve_time": self.solve_time,
            "objective_value": self.objective_value,
            "n_segments": self.n_segments,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolutionRecord:
        """Create from dictionary."""
        return cls(
            stroke=data["stroke"],
            cycle_time=data["cycle_time"],
            upstroke_percent=data["upstroke_percent"],
            max_velocity=data["max_velocity"],
            max_acceleration=data["max_acceleration"],
            max_jerk=data["max_jerk"],
            compression_ratio_limits=tuple(data["compression_ratio_limits"]),
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            acceleration=np.array(data["acceleration"]),
            jerk=np.array(data["jerk"]),
            solve_time=data["solve_time"],
            objective_value=data["objective_value"],
            n_segments=data["n_segments"],
            timestamp=data["timestamp"],
        )


class WarmStartManager:
    """
    Manages solution history and generates initial guesses for warm-starting.

    Implements three strategies:
    1. Primary: Match parameters to previous solution (±10% tolerance)
    2. Secondary: Interpolate between bracketing solutions
    3. Fallback: Generate from polynomial interpolation
    """

    def __init__(
        self,
        max_history: int = 50,
        tolerance: float = 0.1,
        storage_path: str | None = None,
    ):
        """
        Initialize warm-start manager.

        Parameters
        ----------
        max_history : int
            Maximum number of solutions to keep in history
        tolerance : float
            Parameter tolerance for matching solutions (10% default)
        storage_path : Optional[str]
            Path to persistent storage for solution history
        """
        self.max_history = max_history
        self.tolerance = tolerance
        self.storage_path = Path(storage_path) if storage_path else None

        # In-memory solution history
        self.solution_history: list[SolutionRecord] = []

        # Load existing history if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load_history()

        log.info(
            f"Initialized WarmStartManager: max_history={max_history}, "
            f"tolerance={tolerance}, storage_path={storage_path}",
        )

    def store_solution(
        self,
        problem_params: dict[str, Any],
        solution_data: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> None:
        """
        Store a solution in the history.

        Parameters
        ----------
        problem_params : Dict[str, Any]
            Problem parameters (stroke, cycle_time, etc.)
        solution_data : Dict[str, np.ndarray]
            Solution variables (position, velocity, acceleration, jerk)
        metadata : Dict[str, Any]
            Additional metadata (solve_time, objective_value, etc.)
        """
        # Create solution record
        record = SolutionRecord(
            stroke=problem_params["stroke"],
            cycle_time=problem_params["cycle_time"],
            upstroke_percent=problem_params["upstroke_percent"],
            max_velocity=problem_params["max_velocity"],
            max_acceleration=problem_params["max_acceleration"],
            max_jerk=problem_params["max_jerk"],
            compression_ratio_limits=problem_params.get(
                "compression_ratio_limits", (20.0, 70.0),
            ),
            position=solution_data["position"],
            velocity=solution_data["velocity"],
            acceleration=solution_data["acceleration"],
            jerk=solution_data["jerk"],
            solve_time=metadata.get("solve_time", 0.0),
            objective_value=metadata.get("objective_value", 0.0),
            n_segments=metadata.get("n_segments", 50),
            timestamp=metadata.get("timestamp", 0.0),
        )

        # Add to history
        self.solution_history.append(record)

        # Maintain max history limit
        if len(self.solution_history) > self.max_history:
            self.solution_history.pop(0)

        # Save to persistent storage
        if self.storage_path:
            self.save_history()

        log.debug(
            f"Stored solution: stroke={record.stroke}, "
            f"cycle_time={record.cycle_time}, solve_time={record.solve_time:.3f}s",
        )

    def get_initial_guess(
        self, problem_params: dict[str, Any],
    ) -> dict[str, np.ndarray] | None:
        """
        Get initial guess for optimization problem.

        Parameters
        ----------
        problem_params : Dict[str, Any]
            Problem parameters to match

        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            Initial guess variables or None if no suitable match
        """
        if not self.solution_history:
            log.debug("No solution history available for warm-start")
            return None

        # Strategy 1: Find closest match by parameter distance
        closest_record = self._find_closest_solution(problem_params)

        if closest_record:
            log.debug(
                f"Found closest solution for warm-start: "
                f"stroke={closest_record.stroke}, cycle_time={closest_record.cycle_time}",
            )
            return self._interpolate_solution(closest_record, problem_params)

        # Strategy 2: Interpolate between bracketing solutions
        bracketing_solutions = self._find_bracketing_solutions(problem_params)

        if len(bracketing_solutions) >= 2:
            log.debug(
                f"Found {len(bracketing_solutions)} bracketing solutions for interpolation",
            )
            return self._interpolate_between_solutions(
                bracketing_solutions, problem_params,
            )

        # Strategy 3: Fallback - generate from simple motion profiles
        log.debug("Using fallback initial guess generation")
        return self._generate_fallback_guess(problem_params)

    def _find_closest_solution(
        self, problem_params: dict[str, Any],
    ) -> SolutionRecord | None:
        """Find the closest solution by parameter distance."""
        if not self.solution_history:
            return None

        # Normalize parameters for distance calculation
        target_params = np.array(
            [
                problem_params["stroke"],
                problem_params["cycle_time"],
                problem_params["upstroke_percent"],
                problem_params["max_velocity"],
                problem_params["max_acceleration"],
                problem_params["max_jerk"],
            ],
        )

        min_distance = float("inf")
        closest_record = None

        for record in self.solution_history:
            record_params = np.array(
                [
                    record.stroke,
                    record.cycle_time,
                    record.upstroke_percent,
                    record.max_velocity,
                    record.max_acceleration,
                    record.max_jerk,
                ],
            )

            # Normalized distance
            distance = np.linalg.norm((target_params - record_params) / target_params)

            if distance < min_distance and distance <= self.tolerance:
                min_distance = distance
                closest_record = record

        return closest_record

    def _find_bracketing_solutions(
        self, problem_params: dict[str, Any],
    ) -> list[SolutionRecord]:
        """Find solutions that bracket the target parameters."""
        target_stroke = problem_params["stroke"]
        target_cycle_time = problem_params["cycle_time"]

        # Sort by stroke and cycle_time
        sorted_solutions = sorted(
            self.solution_history, key=lambda r: (r.stroke, r.cycle_time),
        )

        bracketing = []

        for i, record in enumerate(sorted_solutions):
            # Check if this solution is close to target
            stroke_diff = abs(record.stroke - target_stroke) / target_stroke
            time_diff = abs(record.cycle_time - target_cycle_time) / target_cycle_time

            if stroke_diff <= self.tolerance and time_diff <= self.tolerance:
                bracketing.append(record)

        return bracketing[:4]  # Limit to 4 solutions for interpolation

    def _interpolate_solution(
        self, record: SolutionRecord, problem_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Interpolate solution to match target problem parameters."""
        # Simple linear scaling for now
        scale_factor = problem_params["stroke"] / record.stroke
        time_scale = problem_params["cycle_time"] / record.cycle_time

        # Scale position and velocity
        scaled_position = record.position * scale_factor
        scaled_velocity = record.velocity * scale_factor / time_scale
        scaled_acceleration = record.acceleration * scale_factor / (time_scale**2)
        scaled_jerk = record.jerk * scale_factor / (time_scale**3)

        return {
            "x": scaled_position,
            "v": scaled_velocity,
            "a": scaled_acceleration,
            "j": scaled_jerk,
        }

    def _interpolate_between_solutions(
        self, solutions: list[SolutionRecord], problem_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Interpolate between multiple solutions."""
        if len(solutions) < 2:
            return self._interpolate_solution(solutions[0], problem_params)

        # Weight solutions by parameter distance
        target_stroke = problem_params["stroke"]
        target_cycle_time = problem_params["cycle_time"]

        weights = []
        for sol in solutions:
            stroke_diff = abs(sol.stroke - target_stroke) / target_stroke
            time_diff = abs(sol.cycle_time - target_cycle_time) / target_cycle_time
            weight = 1.0 / (1.0 + stroke_diff + time_diff)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Interpolate variables
        n_points = len(solutions[0].position)
        interpolated = {
            "x": np.zeros(n_points),
            "v": np.zeros(n_points),
            "a": np.zeros(n_points),
            "j": np.zeros(n_points),
        }

        for i, (sol, weight) in enumerate(zip(solutions, weights)):
            interpolated["x"] += weight * sol.position
            interpolated["v"] += weight * sol.velocity
            interpolated["a"] += weight * sol.acceleration
            interpolated["j"] += weight * sol.jerk

        return interpolated

    def _generate_fallback_guess(
        self, problem_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Generate fallback initial guess from simple motion profiles."""
        stroke = problem_params["stroke"]
        cycle_time = problem_params["cycle_time"]
        upstroke_percent = problem_params["upstroke_percent"]

        # Simple polynomial motion profile
        n_points = 51  # Default number of points

        # Time vector
        t = np.linspace(0, cycle_time, n_points)

        # Upstroke phase
        upstroke_time = cycle_time * upstroke_percent / 100.0
        upstroke_index = int(upstroke_percent * n_points / 100.0)

        # Position profile (smooth S-curve)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points - 1)

        for i in range(n_points):
            if i <= upstroke_index:
                # Upstroke: smooth acceleration
                phase = i / upstroke_index
                position[i] = stroke * (3 * phase**2 - 2 * phase**3)
                velocity[i] = stroke * (6 * phase - 6 * phase**2) / upstroke_time
                acceleration[i] = stroke * (6 - 12 * phase) / (upstroke_time**2)
            else:
                # Downstroke: smooth deceleration
                phase = (i - upstroke_index) / (n_points - upstroke_index - 1)
                position[i] = stroke * (1 - 3 * phase**2 + 2 * phase**3)
                velocity[i] = (
                    -stroke * (6 * phase - 6 * phase**2) / (cycle_time - upstroke_time)
                )
                acceleration[i] = (
                    -stroke * (6 - 12 * phase) / ((cycle_time - upstroke_time) ** 2)
                )

        # Jerk (derivative of acceleration)
        for i in range(n_points - 1):
            jerk[i] = (acceleration[i + 1] - acceleration[i]) / (
                cycle_time / (n_points - 1)
            )

        return {
            "x": position,
            "v": velocity,
            "a": acceleration,
            "j": jerk,
        }

    def save_history(self) -> None:
        """Save solution history to persistent storage."""
        if not self.storage_path:
            return

        # Create directory if it doesn't exist
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        history_data = [record.to_dict() for record in self.solution_history]

        # Save as JSON
        with open(self.storage_path, "w") as f:
            json.dump(history_data, f, indent=2)

        log.debug(
            f"Saved {len(self.solution_history)} solutions to {self.storage_path}",
        )

    def load_history(self) -> None:
        """Load solution history from persistent storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                history_data = json.load(f)

            self.solution_history = [
                SolutionRecord.from_dict(data) for data in history_data
            ]
            log.debug(
                f"Loaded {len(self.solution_history)} solutions from {self.storage_path}",
            )

        except Exception as e:
            log.warning(f"Failed to load solution history: {e}")
            self.solution_history = []

    def clear_history(self) -> None:
        """Clear solution history."""
        self.solution_history.clear()

        if self.storage_path and self.storage_path.exists():
            self.storage_path.unlink()

        log.info("Cleared solution history")

    def get_history_stats(self) -> dict[str, Any]:
        """Get statistics about solution history."""
        if not self.solution_history:
            return {"count": 0}

        strokes = [r.stroke for r in self.solution_history]
        cycle_times = [r.cycle_time for r in self.solution_history]
        solve_times = [r.solve_time for r in self.solution_history]

        return {
            "count": len(self.solution_history),
            "stroke_range": (min(strokes), max(strokes)),
            "cycle_time_range": (min(cycle_times), max(cycle_times)),
            "avg_solve_time": np.mean(solve_times),
            "min_solve_time": min(solve_times),
            "max_solve_time": max(solve_times),
        }
