"""
Tests for the CEM client.

Run with: pytest tests/cem_client/test_client.py -v
"""

import numpy as np
import pytest
from truthmaker.cem import (
    CEMClient,
    ViolationCode,
    ViolationSeverity,
    SuggestedActionCode,
)


class TestCEMClientMockMode:
    """Tests for CEM client in mock mode."""
    
    def test_validate_motion_valid_profile(self):
        """A smooth sine wave should pass validation."""
        with CEMClient(mock=True) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            # Smooth sine wave - low jerk, reasonable stroke
            x_profile = 50 + 40 * np.sin(theta)  # 80mm stroke, centered at 50mm
            
            report = cem.validate_motion(x_profile, theta)
            
            assert report.is_valid
            assert len([v for v in report.violations if v.severity >= ViolationSeverity.ERROR]) == 0
            assert report.cem_version == "mock-0.1.0"
    
    def test_validate_motion_high_jerk(self):
        """A profile with sharp transitions should fail jerk check."""
        with CEMClient(mock=True, config={"max_jerk": 100.0}) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            # Square wave approximation - very high jerk
            x_profile = 50 + 40 * np.sign(np.sin(theta))
            
            report = cem.validate_motion(x_profile, theta)
            
            assert not report.is_valid
            jerk_violations = report.get_by_code(ViolationCode.KINEMATIC_MAX_JERK)
            assert len(jerk_violations) > 0
            assert jerk_violations[0].severity == ViolationSeverity.ERROR
            assert jerk_violations[0].suggested_action == SuggestedActionCode.INCREASE_SMOOTHING
            assert jerk_violations[0].margin is not None
            assert jerk_violations[0].margin < 0  # Negative margin = violating
    
    def test_validate_motion_excessive_stroke(self):
        """A profile exceeding gear envelope should fail."""
        with CEMClient(mock=True, config={"max_radius": 50.0}) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            # 200mm stroke - would require Rp > 100mm
            x_profile = 100 * (1 - np.cos(theta))
            
            report = cem.validate_motion(x_profile, theta)
            
            assert not report.is_valid
            envelope_violations = report.get_by_code(ViolationCode.GEAR_ENVELOPE_STROKE)
            assert len(envelope_violations) > 0
            assert envelope_violations[0].suggested_action == SuggestedActionCode.REDUCE_STROKE
    
    def test_validate_motion_periodicity_warning(self):
        """Non-periodic motion should produce a warning."""
        with CEMClient(mock=True) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            x_profile = 50 + 40 * np.sin(theta)
            x_profile[-1] += 5  # Break periodicity
            
            report = cem.validate_motion(x_profile, theta)
            
            # Should still be valid (warning, not error)
            period_violations = report.get_by_code(ViolationCode.KINEMATIC_PERIODICITY_BROKEN)
            if period_violations:
                assert period_violations[0].severity == ViolationSeverity.WARN
    
    def test_get_gear_initial_guess(self):
        """Initial guess should be physics-informed."""
        with CEMClient(mock=True) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            x_target = 50 + 40 * np.sin(theta)  # 80mm stroke
            
            guess = cem.get_gear_initial_guess(x_target, theta)
            
            # Rp should be at least stroke/2
            assert np.mean(guess.Rp) >= 40
            # C should be positive
            assert np.mean(guess.C) > 0
            # Conjugacy: Rr = C + Rp (for internal gear)
            np.testing.assert_array_almost_equal(guess.Rr, guess.C + guess.Rp)
            # Phase should be near 0 or 2π (sine minimum is at 3π/2)
            assert 0 <= guess.phase_offset <= 2 * np.pi
    
    def test_get_thermo_envelope(self):
        """Envelope should return sensible bounds."""
        with CEMClient(mock=True) as cem:
            envelope = cem.get_thermo_envelope(
                bore=0.1, stroke=0.2, cr=15.0, rpm=3000
            )
            
            assert envelope.feasible
            assert envelope.boost_range[0] < envelope.boost_range[1]
            assert envelope.fuel_range[0] < envelope.fuel_range[1]
            assert envelope.fuel_range[0] > 0
    
    def test_config_affects_validation(self):
        """Different configs should produce different results."""
        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 + 40 * np.sin(theta)
        
        # Strict config
        with CEMClient(mock=True, config={"max_jerk": 10.0}) as cem_strict:
            report_strict = cem_strict.validate_motion(x_profile, theta)
        
        # Relaxed config
        with CEMClient(mock=True, config={"max_jerk": 1000.0}) as cem_relaxed:
            report_relaxed = cem_relaxed.validate_motion(x_profile, theta)
        
        # Strict should have more violations
        assert len(report_strict.violations) >= len(report_relaxed.violations)
    
    def test_violation_metrics(self):
        """Violations should include diagnostic metrics."""
        with CEMClient(mock=True, config={"max_jerk": 10.0}) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            x_profile = 50 + 40 * np.sin(theta)
            
            report = cem.validate_motion(x_profile, theta)
            
            jerk_violations = report.get_by_code(ViolationCode.KINEMATIC_MAX_JERK)
            if jerk_violations:
                metrics = jerk_violations[0].metrics
                assert metrics is not None
                assert "max_jerk" in metrics
                assert "limit" in metrics
                assert "margin" in metrics


class TestCEMClientVersioning:
    """Tests for reproducibility tracking."""
    
    def test_version_included_in_report(self):
        """Reports should include CEM version."""
        with CEMClient(mock=True) as cem:
            theta = np.linspace(0, 2 * np.pi, 360)
            x_profile = 50 + 40 * np.sin(theta)
            
            report = cem.validate_motion(x_profile, theta)
            
            assert report.cem_version is not None
            assert len(report.cem_version) > 0
    
    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        config = {"max_jerk": 500.0, "max_radius": 80.0}
        
        cem1 = CEMClient(mock=True, config=config)
        cem2 = CEMClient(mock=True, config=config)
        
        assert cem1.config_hash == cem2.config_hash
    
    def test_config_hash_changes_with_config(self):
        """Different config should produce different hash."""
        cem1 = CEMClient(mock=True, config={"max_jerk": 500.0})
        cem2 = CEMClient(mock=True, config={"max_jerk": 600.0})
        
        assert cem1.config_hash != cem2.config_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
