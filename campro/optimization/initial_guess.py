"""
Deterministic initial-guess generation for CasADi motion optimization.

Provides a physics-aware seed plus an optional polishing pass that enforces
basic boundary/constraint consistency without relying on historical solutions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class _MotionProblem(Protocol):
    """Protocol for motion optimization problems.
    
    All velocity, acceleration, and jerk constraints are optional and must be in per-degree units:
    - max_velocity: mm/deg (or m/deg in SI), or None for unbounded
    - max_acceleration: mm/deg² (or m/deg² in SI), or None for unbounded
    - max_jerk: mm/deg³ (or m/deg³ in SI), or None for unbounded
    """
    stroke: float
    cycle_time: float
    duration_angle_deg: float
    upstroke_percent: float
    max_velocity: float | None  # Per-degree units: mm/deg (or m/deg in SI), optional
    max_acceleration: float | None  # Per-degree units: mm/deg² (or m/deg² in SI), optional
    max_jerk: float | None  # Per-degree units: mm/deg³ (or m/deg³ in SI), optional


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Standard 5th-order smoothstep (0≤t≤1)."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _smoothstep_derivatives(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first/second/third derivatives of smoothstep."""
    first = 30 * t**2 - 60 * t**3 + 30 * t**4
    second = 60 * t - 180 * t**2 + 120 * t**3
    third = 60 - 360 * t + 360 * t**2
    return first, second, third


@dataclass
class InitialGuessBuilder:
    """Construct baseline and polished initial guesses for the CasADi solver."""

    n_segments: int

    def __post_init__(self) -> None:
        self.n_segments = max(4, int(self.n_segments))

    def update_segments(self, n_segments: int) -> None:
        """Keep builder aligned with solver discretization."""
        self.n_segments = max(4, int(n_segments))

    def build_seed(self, problem: _MotionProblem) -> dict[str, np.ndarray]:
        """
        Generate a deterministic S-curve seed that respects boundary conditions.

        The profile uses a 5th-order polynomial to move from 0 → stroke during
        the configured upstroke phase, then dwells at stroke for the remainder
        of the cycle.

        All outputs are in per-degree units:
        - x: position (mm or m)
        - v: velocity (mm/deg or m/deg)
        - a: acceleration (mm/deg² or m/deg²)
        - j: jerk (mm/deg³ or m/deg³)

        Parameters
        ----------
        problem : _MotionProblem
            Problem specification with constraints in per-degree units.
            Must have duration_angle_deg attribute (required, no fallback to cycle_time).
            max_velocity, max_acceleration, max_jerk are optional (None for unbounded).

        Returns
        -------
        dict[str, np.ndarray]
            Seed profile with keys: 'x', 'v', 'a', 'j'
            All derivatives are in per-degree units.

        Raises
        ------
        ValueError
            If duration_angle_deg is missing or invalid (<= 0).
        """
        n_points = self.n_segments + 1
        stroke = float(problem.stroke)
        
        # Require duration_angle_deg
        if not hasattr(problem, "duration_angle_deg"):
            raise ValueError(
                "duration_angle_deg is required for Phase 1 per-degree optimization. "
                "The problem must provide duration_angle_deg (in degrees)."
            )
        
        duration_angle = float(problem.duration_angle_deg)
        if duration_angle <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {duration_angle}. "
                "Phase 1 optimization requires angle-based units, not time-based."
            )
        up_frac = np.clip(problem.upstroke_percent / 100.0, 0.05, 0.95)
        up_angle = duration_angle * up_frac
        dtheta = duration_angle / self.n_segments

        theta = np.linspace(0.0, duration_angle, n_points)
        upstroke_limit = max(1, min(self.n_segments - 1, int(round(up_frac * self.n_segments))))

        x = np.full(n_points, stroke)
        v = np.zeros(n_points)
        a = np.zeros(n_points)

        if upstroke_limit <= 1:
            x[:upstroke_limit] = 0.0
        else:
            up_angles = theta[: upstroke_limit + 1]
            phase = np.clip(up_angles / max(up_angle, 1e-6), 0.0, 1.0)
            s = _smoothstep(phase)
            ds, d2s, d3s = _smoothstep_derivatives(phase)

            x[: upstroke_limit + 1] = stroke * s
            # Derivatives computed in per-degree units (using dtheta)
            # stroke is already in meters, so derivatives are in m/deg, m/deg², etc.
            v[: upstroke_limit + 1] = stroke / max(up_angle, 1e-6) * ds  # m/deg
            a[: upstroke_limit + 1] = stroke / max(up_angle, 1e-6) ** 2 * d2s  # m/deg²

        # Dwell portion: hold at stroke with zero derivatives
        x[-1] = stroke
        v[0] = 0.0
        v[-1] = 0.0
        a[0] = 0.0
        a[-1] = 0.0

        # Jerk computed in per-degree units (using dtheta)
        jerk_nodes = np.gradient(a, dtheta)  # m/deg³ (stroke is already in meters)
        j = 0.5 * (jerk_nodes[:-1] + jerk_nodes[1:])

        # problem.stroke is already in meters (SI units), so no conversion needed
        # All values are already in the correct units
        return {
            "x": x.astype(float),  # Position in meters
            "v": v.astype(float),  # Velocity in m/deg
            "a": a.astype(float),  # Acceleration in m/deg²
            "j": j.astype(float),  # Jerk in m/deg³
        }

    def polish_seed(
        self,
        problem: _MotionProblem,
        guess: dict[str, np.ndarray],
        *,
        smoothing_passes: int = 2,
    ) -> dict[str, np.ndarray]:
        """
        Smooth and clip the deterministic seed to satisfy basic constraints.

        - Applies a moving-average filter to the position profile
        - Re-imposes monotonicity and boundary conditions
        - Recomputes velocity/acceleration/jerk from the smoothed profile
        - Clips derivatives to the provided motion constraints (in per-degree units)

        Parameters
        ----------
        problem : _MotionProblem
            Problem specification with constraints in per-degree units.
            Must have duration_angle_deg attribute (required).
            max_velocity, max_acceleration, max_jerk are optional (None for unbounded).
            If provided, must be in per-degree units (mm/deg, mm/deg², mm/deg³ respectively).
        guess : dict[str, np.ndarray]
            Initial seed profile with keys 'x', 'v', 'a', 'j'
        smoothing_passes : int, optional
            Number of smoothing passes to apply, by default 2

        Returns
        -------
        dict[str, np.ndarray]
            Polished seed profile with keys: 'x', 'v', 'a', 'j'
            All derivatives are in per-degree units.
            Derivatives are clipped to problem constraints only if constraints are provided (not None).

        Raises
        ------
        ValueError
            If duration_angle_deg is missing or invalid (<= 0).
        """
        stroke = float(problem.stroke)
        
        # Require duration_angle_deg
        if not hasattr(problem, "duration_angle_deg"):
            raise ValueError(
                "duration_angle_deg is required for Phase 1 per-degree optimization. "
                "The problem must provide duration_angle_deg (in degrees)."
            )
        
        duration_angle = float(problem.duration_angle_deg)
        if duration_angle <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {duration_angle}. "
                "Phase 1 optimization requires angle-based units, not time-based."
            )
        dtheta = duration_angle / self.n_segments

        position = np.array(guess["x"], dtype=float)
        for _ in range(max(0, smoothing_passes)):
            position = self._moving_average(position, window=5)
            # Keep monotonic rise toward final stroke value
            position[0] = 0.0
            position = np.maximum.accumulate(position)
            position = np.clip(position, 0.0, stroke)
            if position[-1] == 0.0:
                continue
            scale = stroke / position[-1]
            position *= scale
        position[-1] = stroke

        # Recompute derivatives in per-degree units (using dtheta)
        # position is already in meters (from problem.stroke), so derivatives are in m/deg, m/deg², etc.
        velocity = np.gradient(position, dtheta)  # m/deg
        # Clamp to per-degree velocity constraint (only if provided)
        if problem.max_velocity is not None:
            max_vel = max(1e-6, float(problem.max_velocity))  # Expected in m/deg
            velocity = np.clip(velocity, -max_vel, max_vel)
        velocity[0] = 0.0
        velocity[-1] = 0.0

        acceleration = np.gradient(velocity, dtheta)  # m/deg²
        # Clamp to per-degree acceleration constraint (only if provided)
        if problem.max_acceleration is not None:
            max_acc = max(1e-6, float(problem.max_acceleration))  # Expected in m/deg²
            acceleration = np.clip(acceleration, -max_acc, max_acc)
        acceleration[0] = 0.0
        acceleration[-1] = 0.0

        jerk_nodes = np.gradient(acceleration, dtheta)  # m/deg³
        jerk = 0.5 * (jerk_nodes[:-1] + jerk_nodes[1:])
        # Clamp to per-degree jerk constraint (only if provided)
        if problem.max_jerk is not None:
            max_jerk = max(1e-6, float(problem.max_jerk))  # Expected in m/deg³
            jerk = np.clip(jerk, -max_jerk, max_jerk)

        # problem.stroke is already in meters (SI units), so no conversion needed
        return {
            "x": position.astype(float),  # Position in meters
            "v": velocity.astype(float),  # Velocity in m/deg
            "a": acceleration.astype(float),  # Acceleration in m/deg²
            "j": jerk.astype(float),  # Jerk in m/deg³
        }

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Simple centered moving average keeping array length constant."""
        if window <= 1:
            return data
        pad = window // 2
        padded = np.pad(data, (pad, pad), mode="edge")
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: data.shape[0]]
