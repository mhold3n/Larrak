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
    stroke: float
    cycle_time: float
    duration_angle_deg: float
    upstroke_percent: float
    max_velocity: float
    max_acceleration: float
    max_jerk: float


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
        """
        n_points = self.n_segments + 1
        stroke = float(problem.stroke)
        duration_angle = float(getattr(problem, "duration_angle_deg", getattr(problem, "cycle_time", 360.0)))
        duration_angle = max(duration_angle, 1e-6)
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
            v[: upstroke_limit + 1] = stroke / max(up_angle, 1e-6) * ds
            a[: upstroke_limit + 1] = stroke / max(up_angle, 1e-6) ** 2 * d2s

        # Dwell portion: hold at stroke with zero derivatives
        x[-1] = stroke
        v[0] = 0.0
        v[-1] = 0.0
        a[0] = 0.0
        a[-1] = 0.0

        jerk_nodes = np.gradient(a, dtheta)
        j = 0.5 * (jerk_nodes[:-1] + jerk_nodes[1:])

        return {
            "x": x.astype(float),
            "v": v.astype(float),
            "a": a.astype(float),
            "j": j.astype(float),
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
        - Clips derivatives to the provided motion constraints
        """
        stroke = float(problem.stroke)
        duration_angle = float(getattr(problem, "duration_angle_deg", getattr(problem, "cycle_time", 360.0)))
        duration_angle = max(duration_angle, 1e-6)
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

        velocity = np.gradient(position, dtheta)
        max_vel = max(1e-6, float(problem.max_velocity))
        velocity = np.clip(velocity, -max_vel, max_vel)
        velocity[0] = 0.0
        velocity[-1] = 0.0

        acceleration = np.gradient(velocity, dtheta)
        max_acc = max(1e-6, float(problem.max_acceleration))
        acceleration = np.clip(acceleration, -max_acc, max_acc)
        acceleration[0] = 0.0
        acceleration[-1] = 0.0

        jerk_nodes = np.gradient(acceleration, dtheta)
        jerk = 0.5 * (jerk_nodes[:-1] + jerk_nodes[1:])
        max_jerk = max(1e-6, float(problem.max_jerk))
        jerk = np.clip(jerk, -max_jerk, max_jerk)

        return {
            "x": position,
            "v": velocity,
            "a": acceleration,
            "j": jerk,
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
