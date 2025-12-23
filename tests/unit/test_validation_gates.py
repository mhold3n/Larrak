"""
Unit tests for CEM validation gates and recovery.
"""

import numpy as np
import pytest

from campro.validation import (
    CEM_AVAILABLE,
    RecoveryAction,
    RecoveryEngine,
    RecoveryPlan,
    check_gear_feasibility,
    check_motion_feasibility,
    check_thermo_feasibility,
    get_operating_envelope,
)


class TestCEMGates:
    """Tests for CEM validation gate functions."""

    def test_check_motion_feasibility_returns_report(self):
        """Motion feasibility check returns a report object."""
        x_profile = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.05 + 0.1
        report = check_motion_feasibility(x_profile)

        assert hasattr(report, "is_valid")
        assert hasattr(report, "violations")

    def test_check_motion_feasibility_with_theta(self):
        """Motion feasibility works with explicit theta."""
        x_profile = np.linspace(0.05, 0.15, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        report = check_motion_feasibility(x_profile, theta)

        assert report.is_valid is True or report.is_valid is False

    def test_check_thermo_feasibility_returns_report(self):
        """Thermo feasibility check returns a report."""
        report = check_thermo_feasibility(
            rpm=3000.0,
            p_intake_bar=1.5,
            fuel_mass_kg=5e-5,
        )
        assert hasattr(report, "is_valid")

    def test_check_gear_feasibility_mock(self):
        """Gear feasibility returns valid in mock mode."""
        config = {"module": 2.0, "z_sun": 20, "z_planet": 30, "z_ring": 80}
        report = check_gear_feasibility(config, mock=True)

        assert report.is_valid is True

    def test_get_operating_envelope(self):
        """Operating envelope returns bounds."""
        envelope = get_operating_envelope(
            bore=0.1,
            stroke=0.1,
            cr=15.0,
            rpm=3000.0,
        )

        assert hasattr(envelope, "boost_range")
        assert hasattr(envelope, "fuel_range")
        assert len(envelope.boost_range) == 2
        assert envelope.boost_range[0] < envelope.boost_range[1]


class TestRecoveryEngine:
    """Tests for RecoveryEngine class."""

    def test_recovery_engine_creation(self):
        """Create recovery engine with default settings."""
        engine = RecoveryEngine()
        assert engine.conservative is True

    def test_recovery_engine_non_conservative(self):
        """Create non-conservative recovery engine."""
        engine = RecoveryEngine(conservative=False)
        assert engine.conservative is False
        assert engine._adjustment_factor == 0.2

    def test_suggest_recovery_empty_violations(self):
        """Empty violations produce empty plan."""
        engine = RecoveryEngine()
        plan = engine.suggest_recovery([])

        assert len(plan.actions) == 0
        assert plan.requires_manual_review is False

    def test_apply_recovery_preserves_params(self):
        """Apply recovery doesn't mutate original params."""
        engine = RecoveryEngine()
        plan = RecoveryPlan(
            actions=[
                RecoveryAction(
                    parameter="geometry.stroke",
                    adjustment="set",
                    value=0.15,
                )
            ]
        )

        original = {"geometry": {"stroke": 0.1}}
        result = engine.apply_recovery(original, plan)

        assert original["geometry"]["stroke"] == 0.1  # Unchanged
        assert result["geometry"]["stroke"] == 0.15

    def test_apply_recovery_set_action(self):
        """Set action overwrites value."""
        engine = RecoveryEngine()
        plan = RecoveryPlan(
            actions=[
                RecoveryAction(
                    parameter="solver.use_warm_start",
                    adjustment="set",
                    value=1.0,
                )
            ]
        )

        params: dict = {}
        result = engine.apply_recovery(params, plan)

        assert result["solver"]["use_warm_start"] == 1.0

    def test_apply_recovery_increase_action(self):
        """Increase action multiplies value."""
        engine = RecoveryEngine()
        plan = RecoveryPlan(
            actions=[
                RecoveryAction(
                    parameter="motion.smoothing_factor",
                    adjustment="increase",
                    factor=1.1,
                )
            ]
        )

        params = {"motion": {"smoothing_factor": 1.0}}
        result = engine.apply_recovery(params, plan)

        assert result["motion"]["smoothing_factor"] == pytest.approx(1.1)


class TestRecoveryAction:
    """Tests for RecoveryAction dataclass."""

    def test_action_defaults(self):
        """RecoveryAction has expected defaults."""
        action = RecoveryAction(
            parameter="test.param",
            adjustment="set",
        )
        assert action.value is None
        assert action.factor == 1.0
        assert action.reason == ""

    def test_action_with_reason(self):
        """RecoveryAction stores reason."""
        action = RecoveryAction(
            parameter="bounds.T_max",
            adjustment="decrease",
            factor=0.9,
            reason="Reduce max temperature to avoid material limits",
        )
        assert "temperature" in action.reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
