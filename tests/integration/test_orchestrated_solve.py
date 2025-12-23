"""
Integration test for orchestrated optimization.
"""

import numpy as np
import pytest

from campro.optimization.driver import solve_cycle_orchestrated


class TestOrchestratedSolve:
    """End-to-end tests for orchestrated optimization."""

    def test_solve_cycle_orchestrated_mock(self):
        """Can run orchestrated solve with mock components."""
        # Simple problem params
        params = {
            "rpm": 3000.0,
            "p_intake_bar": 1.5,
            "fuel_mass_kg": 5e-5,
            # Solver control
            "max_iter": 5,
        }

        # Run with mocks (use_cem=False mocks CEM, use_surrogate=False mocks surrogate)
        result = solve_cycle_orchestrated(
            params,
            budget=5,  # Small budget for fast test
            use_cem=False,
            use_surrogate=False,
            mock_simulation=True,
            mock_solver=True,
        )

        assert result is not None
        # Check for convergence or at least a valid result structure
        assert hasattr(result, "objective_value") or hasattr(result, "eta_thermal")

    def test_solve_cycle_orchestrated_integration(self):
        """Can run with real adapters (if visible)."""
        # This tests the full wiring including adapter logic
        params = {
            "rpm": 3000.0,
            "p_intake_bar": 1.5,
        }

        result = solve_cycle_orchestrated(
            params,
            budget=2,
            use_cem=False,  # Still default to mock CEM if no service
            use_surrogate=False,
        )

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
