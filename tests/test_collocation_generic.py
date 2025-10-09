"""
Tests for generic collocation support and objective computation.
"""

import numpy as np

from campro.optimization.base import OptimizationStatus
from campro.optimization.collocation import CollocationOptimizer, CollocationSettings


class DummyConstraints:
    """Constraints providing a generic collocation problem builder hook."""

    def build_collocation_problem(self, objective, time_horizon: float, n_points: int, initial_guess=None):
        theta = np.linspace(0.0, 2 * np.pi, n_points)
        # Constant jerk profile of 1 over [0, 2π] so integral j^2 dθ = 2π
        jerk = np.ones_like(theta)
        return {
            "cam_angle": theta,
            "time": np.linspace(0.0, time_horizon, n_points),
            "position": np.zeros_like(theta),
            "velocity": np.zeros_like(theta),
            "acceleration": np.zeros_like(theta),
            "jerk": jerk,
        }


def test_collocation_generic_builder_and_objective_integral():
    optimizer = CollocationOptimizer(CollocationSettings())
    # Provide a dummy objective callable as required by the base optimizer
    dummy_objective = lambda sol: 0.0
    constraints = DummyConstraints()

    result = optimizer.optimize(dummy_objective, constraints, time_horizon=1.0, n_points=200)

    assert result.status == OptimizationStatus.CONVERGED
    assert result.objective_value is not None
    # Integral of 1^2 over [0, 2π] should be 2π
    assert result.objective_value == np.pi * 2


