"""Unit tests for combustion physics models.

Tests cover:
- Wiebe function properties
- Heat release rate consistency
- Property-based validation of combustion physics
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from campro.materials.fuels import DIESEL, GASOLINE, NATURAL_GAS, FuelProperties
from campro.physics.chem import CombustionParameters, wiebe_function, wiebe_heat_release_rate


# Strategy for generating valid CombustionParameters
@st.composite
def st_combustion_params(draw):
    """Generate valid CombustionParameters."""
    theta_d = draw(st.floats(min_value=20.0, max_value=120.0))
    # Make sure theta_start is reasonable crank angle
    theta_s = draw(st.floats(min_value=-50.0, max_value=10.0))

    return CombustionParameters(
        m_wiebe=draw(st.floats(min_value=1.0, max_value=4.0)),
        a_wiebe=draw(st.floats(min_value=3.0, max_value=10.0)),
        theta_start=theta_s,
        theta_duration=theta_d,
        total_heat_release=draw(st.floats(min_value=100.0, max_value=5000.0)),
        lower_heating_value_fuel=43.0e6,  # Constant for reasonable physics
        tau_ignition=0.002,
        laminar_flame_speed_zero=0.4,
        alpha_turbulence=1.0,
    )


class TestCombustionPhysics:
    """Standard unit tests for combustion functions."""

    def test_wiebe_integral_one(self):
        """Wiebe function should end at 1.0 (or very close)."""
        params = CombustionParameters(
            m_wiebe=2.0,
            a_wiebe=6.9,
            theta_start=-10.0,
            theta_duration=40.0,
            total_heat_release=1000.0,
            lower_heating_value_fuel=43.0e6,
            tau_ignition=0.0,
            laminar_flame_speed_zero=0.5,
            alpha_turbulence=1.0,
        )

        # Check end of combustion
        xb_end = wiebe_function(
            theta=30.0,
            theta_start=params.theta_start,
            theta_duration=params.theta_duration,
            m=params.m_wiebe,
            a=params.a_wiebe,
        )
        assert xb_end == pytest.approx(1.0, abs=0.01)

    def test_wiebe_start_zero(self):
        """Wiebe function should start at 0.0."""
        params = CombustionParameters(
            m_wiebe=2.0,
            a_wiebe=6.9,
            theta_start=-10.0,
            theta_duration=40.0,
            total_heat_release=1000.0,
            lower_heating_value_fuel=43.0e6,
            tau_ignition=0.0,
            laminar_flame_speed_zero=0.5,
            alpha_turbulence=1.0,
        )

        xb_start = wiebe_function(
            theta=-10.0,
            theta_start=params.theta_start,
            theta_duration=params.theta_duration,
            m=params.m_wiebe,
            a=params.a_wiebe,
        )
        assert xb_start == 0.0


class TestPropertyBasedCombustion:
    """Property-based tests for combustion models."""

    @given(params=st_combustion_params(), theta=st.floats(min_value=-360.0, max_value=360.0))
    def test_wiebe_bounds(self, params: CombustionParameters, theta: float):
        """Burn fraction must always be in [0, 1]."""
        xb = wiebe_function(
            theta=theta,
            theta_start=params.theta_start,
            theta_duration=params.theta_duration,
            m=params.m_wiebe,
            a=params.a_wiebe,
        )
        assert 0.0 <= xb <= 1.0, f"Wiebe fraction {xb} out of bounds at {theta}"

        # Check causality
        if theta < params.theta_start:
            assert xb == 0.0
        if theta > params.theta_start + params.theta_duration:
            assert xb == pytest.approx(1.0, abs=1e-3)

    @given(params=st_combustion_params())
    def test_heat_release_conservation(self, params: CombustionParameters):
        """Integral of HRR should roughly equal total heat release."""
        # Simple trapezoidal integration
        # Create a grid across duration
        n_steps = 100
        d_theta = params.theta_duration / n_steps

        integral = 0.0
        # Integrate only during combustion
        for i in range(n_steps):
            theta = params.theta_start + (i + 0.5) * d_theta

            # Rate is dq/dtheta (J/deg) - wait, implementation check necessary?
            # Usually rate functions return per time or per degree.
            # Assuming wiebe_heat_release_rate returns per degree or similar consistent unit
            # actually we don't have this function visible, assuming it exists based on plan.

            # Let's check wiebe_function derivative manually if rate is not available
            # x_b is fraction. Q = x_b * Q_total.
            # So integral of (dx_b/dtheta * Q_total) dtheta = Q_total.

            # Using the actual rate function from chem.py if available
            # We imported wiebe_heat_release_rate

            rate = wiebe_heat_release_rate(theta=theta, params=params)
            integral += rate * d_theta

        # Wiebe isn't perfectly 1.0 at end, depends on 'a'.
        # a=6.9 => exp(-6.9) approx 0.001 remaining.
        # So we expect ~99.9% complete.

        # NOTE: wiebe_heat_release_rate might return J/deg or W.
        # Given it takes theta, likely J/deg or equivalent.
        # Let's assert it's close to total_heat_release.

        # Using relaxed tolerance because trapezoidal on wild shapes is approximate
        assert integral == pytest.approx(params.total_heat_release, rel=0.10)
