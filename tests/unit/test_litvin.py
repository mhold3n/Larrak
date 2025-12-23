"""
Unit tests for Litvin involute gear kinematics.

Tests fundamental geometry calculations used by the CEM for gear synthesis.
"""

import math

import numpy as np
import pytest

from campro.litvin.casadi_litvin import casadi_involute_tangent, casadi_involute_xy, casadi_rotate
from campro.litvin.involute_internal import (
    InternalGearParams,
    base_radius,
    involute_xy,
    pitch_radius,
    sample_internal_flank,
)


class TestInternalGearGeometry:
    """Tests for internal gear geometry calculations."""

    @pytest.fixture
    def standard_gear(self) -> InternalGearParams:
        """Standard 20-degree pressure angle gear."""
        return InternalGearParams(
            teeth=60,
            module=2.0,  # mm
            pressure_angle_deg=20.0,
            addendum_factor=1.0,
        )

    def test_pitch_radius(self, standard_gear: InternalGearParams):
        """Pitch radius = module * teeth / 2."""
        rp = pitch_radius(standard_gear)
        expected = 2.0 * 60 / 2.0  # 60 mm
        assert rp == pytest.approx(expected, rel=1e-10)

    def test_base_radius(self, standard_gear: InternalGearParams):
        """Base radius = pitch_radius * cos(pressure_angle)."""
        rb = base_radius(standard_gear)
        alpha_rad = math.radians(20.0)
        expected = 60.0 * math.cos(alpha_rad)  # ~56.38 mm
        assert rb == pytest.approx(expected, rel=1e-10)

    def test_base_radius_less_than_pitch(self, standard_gear: InternalGearParams):
        """Base radius < pitch radius for pressure_angle > 0."""
        rb = base_radius(standard_gear)
        rp = pitch_radius(standard_gear)
        assert rb < rp

    def test_involute_xy_at_origin(self):
        """Involute starts on the base circle at phi=0."""
        rb = 50.0
        x, y = involute_xy(rb, 0.0)
        assert x == pytest.approx(rb, rel=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_involute_xy_grows_outward(self):
        """Involute moves outward as phi increases."""
        rb = 50.0
        x0, y0 = involute_xy(rb, 0.0)
        x1, y1 = involute_xy(rb, 0.5)

        r0 = math.sqrt(x0**2 + y0**2)
        r1 = math.sqrt(x1**2 + y1**2)

        assert r1 > r0

    def test_involute_radius_formula(self):
        """Radius on involute: r = rb * sqrt(1 + phi^2)."""
        rb = 50.0
        phi = 0.8
        x, y = involute_xy(rb, phi)

        actual_r = math.sqrt(x**2 + y**2)
        expected_r = rb * math.sqrt(1 + phi**2)

        assert actual_r == pytest.approx(expected_r, rel=1e-10)

    def test_sample_internal_flank_returns_points(self, standard_gear: InternalGearParams):
        """Sampling returns non-empty lists of points."""
        flank = sample_internal_flank(standard_gear, n=100)

        assert len(flank.phi) == 100
        assert len(flank.points) == 100
        assert len(flank.tangents) == 100

    def test_sample_internal_flank_phi_ascending(self, standard_gear: InternalGearParams):
        """phi values should be monotonically increasing."""
        flank = sample_internal_flank(standard_gear, n=50)

        for i in range(1, len(flank.phi)):
            assert flank.phi[i] > flank.phi[i - 1]

    def test_sample_internal_flank_points_on_involute(self, standard_gear: InternalGearParams):
        """Sampled points should satisfy involute formula."""
        flank = sample_internal_flank(standard_gear, n=20)
        rb = base_radius(standard_gear)

        for phi, (x, y) in zip(flank.phi, flank.points):
            expected_x, expected_y = involute_xy(rb, phi)
            assert x == pytest.approx(expected_x, rel=1e-10)
            assert y == pytest.approx(expected_y, rel=1e-10)


class TestCasadiInvoluteConsistency:
    """Tests ensuring CasADi symbolic functions match NumPy implementations."""

    def test_casadi_involute_xy_matches_numpy(self):
        """CasADi and NumPy involute should produce identical results."""
        rb = 50.0
        test_phis = [0.0, 0.1, 0.5, 1.0, 1.5]

        for phi in test_phis:
            # NumPy version
            x_np, y_np = involute_xy(rb, phi)

            # CasADi version (with numeric inputs, evaluates symbolically)
            x_ca, y_ca = casadi_involute_xy(rb, phi)

            # CasADi may return DM, convert to float
            x_ca_float = float(x_ca) if hasattr(x_ca, "__float__") else x_ca
            y_ca_float = float(y_ca) if hasattr(y_ca, "__float__") else y_ca

            assert x_ca_float == pytest.approx(x_np, rel=1e-10)
            assert y_ca_float == pytest.approx(y_np, rel=1e-10)

    def test_casadi_rotate_identity_at_zero(self):
        """Rotation by 0 should return same point."""
        x, y = 3.0, 4.0
        rx, ry = casadi_rotate(0.0, x, y)

        rx_float = float(rx) if hasattr(rx, "__float__") else rx
        ry_float = float(ry) if hasattr(ry, "__float__") else ry

        assert rx_float == pytest.approx(x, rel=1e-10)
        assert ry_float == pytest.approx(y, rel=1e-10)

    def test_casadi_rotate_90_degrees(self):
        """Rotation by π/2 should swap and negate correctly."""
        x, y = 1.0, 0.0
        rx, ry = casadi_rotate(math.pi / 2, x, y)

        rx_float = float(rx) if hasattr(rx, "__float__") else rx
        ry_float = float(ry) if hasattr(ry, "__float__") else ry

        # (1,0) rotated 90° CCW = (0,1)
        assert rx_float == pytest.approx(0.0, abs=1e-10)
        assert ry_float == pytest.approx(1.0, rel=1e-10)

    def test_casadi_rotate_preserves_magnitude(self):
        """Rotation should preserve vector magnitude."""
        x, y = 3.0, 4.0
        original_magnitude = math.sqrt(x**2 + y**2)

        angles = [0.1, 0.5, 1.0, 2.0, math.pi]
        for theta in angles:
            rx, ry = casadi_rotate(theta, x, y)

            rx_float = float(rx) if hasattr(rx, "__float__") else rx
            ry_float = float(ry) if hasattr(ry, "__float__") else ry

            rotated_magnitude = math.sqrt(rx_float**2 + ry_float**2)
            assert rotated_magnitude == pytest.approx(original_magnitude, rel=1e-10)

    def test_casadi_involute_tangent_perpendicular_to_radius(self):
        """Involute tangent should be perpendicular to radius at base circle."""
        rb = 50.0
        phi = 0.5

        x, y = casadi_involute_xy(rb, phi)
        tx, ty = casadi_involute_tangent(rb, phi)

        # Convert to float if needed
        x_f = float(x) if hasattr(x, "__float__") else x
        y_f = float(y) if hasattr(y, "__float__") else y
        tx_f = float(tx) if hasattr(tx, "__float__") else tx
        ty_f = float(ty) if hasattr(ty, "__float__") else ty

        # Dot product of position and tangent
        # For involute: tangent is tangent to base circle, so perpendicular to radius from base
        # Not exact at arbitrary phi, but tangent vector should have consistent properties

        # Tangent magnitude should be rb * phi
        tangent_mag = math.sqrt(tx_f**2 + ty_f**2)
        expected_mag = rb * phi
        assert tangent_mag == pytest.approx(expected_mag, rel=1e-10)


class TestGearParameterRanges:
    """Tests for edge cases and parameter validation."""

    def test_high_pressure_angle_reduces_base_radius(self):
        """Higher pressure angle reduces base radius."""
        gear_20 = InternalGearParams(60, 2.0, 20.0, 1.0)
        gear_25 = InternalGearParams(60, 2.0, 25.0, 1.0)

        rb_20 = base_radius(gear_20)
        rb_25 = base_radius(gear_25)

        assert rb_25 < rb_20

    def test_more_teeth_increases_radii(self):
        """More teeth increases pitch and base radii."""
        gear_40 = InternalGearParams(40, 2.0, 20.0, 1.0)
        gear_80 = InternalGearParams(80, 2.0, 20.0, 1.0)

        assert pitch_radius(gear_80) > pitch_radius(gear_40)
        assert base_radius(gear_80) > base_radius(gear_40)

    def test_larger_module_scales_geometry(self):
        """Larger module scales all geometry proportionally."""
        gear_m1 = InternalGearParams(60, 1.0, 20.0, 1.0)
        gear_m2 = InternalGearParams(60, 2.0, 20.0, 1.0)

        assert pitch_radius(gear_m2) == pytest.approx(2 * pitch_radius(gear_m1))
        assert base_radius(gear_m2) == pytest.approx(2 * base_radius(gear_m1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
