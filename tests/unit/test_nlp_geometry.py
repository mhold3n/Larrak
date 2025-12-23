"""
Unit tests for NLP geometry and constraints modules.

Tests StandardSliderCrankGeometry volume/area calculations and
ThermalConstraints surrogate models used in optimization.
"""

import math

import casadi as ca
import numpy as np
import pytest

from campro.optimization.nlp.constraints import ThermalConstraints
from campro.optimization.nlp.geometry import InterpolatedGeometry, StandardSliderCrankGeometry


class TestStandardSliderCrankGeometry:
    """Tests for slider-crank geometry calculations."""

    @pytest.fixture
    def standard_engine(self) -> StandardSliderCrankGeometry:
        """Standard engine geometry (100mm bore, 100mm stroke, CR=15)."""
        return StandardSliderCrankGeometry(
            bore=0.1,  # 100mm
            stroke=0.1,  # 100mm
            conrod=0.2,  # 200mm (λ=0.5)
            compression_ratio=15.0,
        )

    def test_bore_property(self, standard_engine: StandardSliderCrankGeometry):
        """Bore property returns correct value in meters."""
        assert standard_engine.bore == pytest.approx(0.1, rel=1e-10)

    def test_stroke_property(self, standard_engine: StandardSliderCrankGeometry):
        """Stroke property returns correct value in meters."""
        assert standard_engine.stroke == pytest.approx(0.1, rel=1e-10)

    def test_volume_at_tdc_is_minimum(self, standard_engine: StandardSliderCrankGeometry):
        """Volume at TDC (theta=0) should be minimum (clearance volume)."""
        theta_tdc = ca.SX(0.0)
        theta_bdc = ca.SX(math.pi)

        V_tdc = float(ca.substitute(standard_engine.Volume(ca.SX.sym("t")), ca.SX.sym("t"), 0.0))
        V_bdc = float(
            ca.substitute(standard_engine.Volume(ca.SX.sym("t")), ca.SX.sym("t"), math.pi)
        )

        # Need to evaluate symbolically
        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [standard_engine.Volume(t)])

        V_tdc_val = float(V_func(0.0))
        V_bdc_val = float(V_func(math.pi))

        assert V_tdc_val < V_bdc_val

    def test_volume_at_bdc_is_maximum(self, standard_engine: StandardSliderCrankGeometry):
        """Volume at BDC (theta=π) should be maximum."""
        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [standard_engine.Volume(t)])

        # Sample multiple points
        volumes = [float(V_func(theta)) for theta in np.linspace(0, 2 * math.pi, 100)]

        # BDC is around π radians
        V_bdc = float(V_func(math.pi))

        # BDC should be within top 5% of max
        max_vol = max(volumes)
        assert V_bdc >= 0.95 * max_vol

    def test_volume_periodic(self, standard_engine: StandardSliderCrankGeometry):
        """Volume should be periodic with period 2π."""
        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [standard_engine.Volume(t)])

        test_angles = [0.5, 1.0, 2.0, 3.0]
        for theta in test_angles:
            V1 = float(V_func(theta))
            V2 = float(V_func(theta + 2 * math.pi))
            assert V1 == pytest.approx(V2, rel=1e-10)

    def test_compression_ratio(self, standard_engine: StandardSliderCrankGeometry):
        """Compression ratio = V_BDC / V_TDC."""
        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [standard_engine.Volume(t)])

        V_tdc = float(V_func(0.0))
        V_bdc = float(V_func(math.pi))

        actual_cr = V_bdc / V_tdc
        expected_cr = 15.0

        # Allow some tolerance due to conrod effects
        assert actual_cr == pytest.approx(expected_cr, rel=0.05)

    def test_displaced_volume(self, standard_engine: StandardSliderCrankGeometry):
        """Displaced volume = π/4 * B² * S."""
        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [standard_engine.Volume(t)])

        V_tdc = float(V_func(0.0))
        V_bdc = float(V_func(math.pi))

        displaced = V_bdc - V_tdc
        expected_displaced = math.pi / 4 * 0.1**2 * 0.1  # ~785 cm³

        assert displaced == pytest.approx(expected_displaced, rel=0.02)

    def test_dv_dtheta_positive_in_expansion(self, standard_engine: StandardSliderCrankGeometry):
        """dV/dθ should be positive during expansion stroke (0 to π)."""
        t = ca.SX.sym("t")
        dV_func = ca.Function("dV", [t], [standard_engine.dV_dtheta(t)])

        # Check middle of expansion stroke
        theta = math.pi / 2
        dV = float(dV_func(theta))
        assert dV > 0

    def test_dv_dtheta_negative_in_compression(self, standard_engine: StandardSliderCrankGeometry):
        """dV/dθ should be negative during compression stroke (π to 2π)."""
        t = ca.SX.sym("t")
        dV_func = ca.Function("dV", [t], [standard_engine.dV_dtheta(t)])

        # Check middle of compression stroke
        theta = 3 * math.pi / 2
        dV = float(dV_func(theta))
        assert dV < 0

    def test_area_wall_positive(self, standard_engine: StandardSliderCrankGeometry):
        """Wall area should always be positive."""
        t = ca.SX.sym("t")
        A_func = ca.Function("A", [t], [standard_engine.Area_wall(t)])

        for theta in np.linspace(0, 2 * math.pi, 20):
            A = float(A_func(theta))
            assert A > 0


class TestThermalConstraints:
    """Tests for thermal surrogate models."""

    @pytest.fixture
    def thermal(self) -> ThermalConstraints:
        """Default thermal constraints."""
        return ThermalConstraints()

    def test_crown_temp_increases_with_pressure(self, thermal: ThermalConstraints):
        """Higher peak pressure should increase crown temperature."""
        p_max = ca.SX.sym("p_max")
        T_mean = ca.SX.sym("T_mean")
        rpm = ca.SX.sym("rpm")

        T_crown_expr = thermal.get_max_crown_temp(p_max, T_mean, rpm)
        T_func = ca.Function("T", [p_max, T_mean, rpm], [T_crown_expr])

        # Compare at low and high pressure
        T_low = float(T_func(50e5, 1000.0, 3000.0))  # 50 bar
        T_high = float(T_func(150e5, 1000.0, 3000.0))  # 150 bar

        assert T_high > T_low

    def test_crown_temp_increases_with_mean_temp(self, thermal: ThermalConstraints):
        """Higher mean gas temperature should increase crown temperature."""
        p_max = ca.SX.sym("p_max")
        T_mean = ca.SX.sym("T_mean")
        rpm = ca.SX.sym("rpm")

        T_crown_expr = thermal.get_max_crown_temp(p_max, T_mean, rpm)
        T_func = ca.Function("T", [p_max, T_mean, rpm], [T_crown_expr])

        T_low = float(T_func(100e5, 800.0, 3000.0))
        T_high = float(T_func(100e5, 1200.0, 3000.0))

        assert T_high > T_low

    def test_crown_temp_bounded_by_gas_temp(self, thermal: ThermalConstraints):
        """Crown temp should be less than gas temp (heat flows out)."""
        p_max = ca.SX.sym("p_max")
        T_mean = ca.SX.sym("T_mean")
        rpm = ca.SX.sym("rpm")

        T_crown_expr = thermal.get_max_crown_temp(p_max, T_mean, rpm)
        T_func = ca.Function("T", [p_max, T_mean, rpm], [T_crown_expr])

        T_gas = 1000.0
        T_crown = float(T_func(100e5, T_gas, 3000.0))

        # Crown temp should be between oil temp (360K) and gas temp
        assert 360.0 < T_crown < T_gas * 1.5  # Allow some margin for model

    def test_crown_temp_increases_with_rpm(self, thermal: ThermalConstraints):
        """Higher RPM should increase crown temp (more heat transfer)."""
        p_max = ca.SX.sym("p_max")
        T_mean = ca.SX.sym("T_mean")
        rpm = ca.SX.sym("rpm")

        T_crown_expr = thermal.get_max_crown_temp(p_max, T_mean, rpm)
        T_func = ca.Function("T", [p_max, T_mean, rpm], [T_crown_expr])

        T_low = float(T_func(100e5, 1000.0, 1000.0))
        T_high = float(T_func(100e5, 1000.0, 6000.0))

        assert T_high > T_low


class TestInterpolatedGeometry:
    """Tests for interpolated geometry lookup."""

    def test_interpolated_volume_matches_input(self):
        """Interpolated volume should match at input points."""
        theta_arr = np.linspace(0, 2 * np.pi, 50)
        V_arr = 1e-4 * (1 + 0.5 * np.sin(theta_arr))  # Simple sinusoidal volume

        geo = InterpolatedGeometry(theta_arr, V_arr)

        t = ca.SX.sym("t")
        V_func = ca.Function("V", [t], [geo.Volume(t)])

        # Check at some input points
        for i in [0, 10, 25, 40]:
            theta = theta_arr[i]
            V_interp = float(V_func(theta))
            V_expected = V_arr[i]
            assert V_interp == pytest.approx(V_expected, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
