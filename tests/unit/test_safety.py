"""Unit tests for safety factors and design margins.

Tests cover:
- SafetyFactor dataclass application
- Margin calculations
- Logic robustness via Hypothesis
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from campro.design.safety import FailureMode, SafetyFactor


class TestPropertyBasedSafetyFactors:
    """Property-based tests for safety factor calculations."""

    def test_safety_factor_reduces_allowable(self) -> None:
        """Applying SF should always reduce allowable stress."""

        @given(
            sf_value=st.floats(min_value=1.0, max_value=5.0),
            allowable=st.floats(min_value=1.0, max_value=1e9, allow_nan=False),
        )
        @settings(max_examples=50)
        def check(sf_value: float, allowable: float) -> None:
            sf = SafetyFactor(value=sf_value, failure_mode=FailureMode.YIELDING)
            design_allowable = sf.apply_to_allowable(allowable)
            assert design_allowable <= allowable, "SF must reduce allowable"
            assert design_allowable > 0, "Design allowable must be positive"

        check()

    def test_margin_check_consistent(self) -> None:
        """Margin check must be consistent: passes iff utilization <= 1."""

        @given(
            sf_value=st.floats(min_value=1.0, max_value=3.0),
            actual=st.floats(min_value=1e-3, max_value=1000.0, allow_nan=False),
            allowable=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def check(sf_value: float, actual: float, allowable: float) -> None:
            sf = SafetyFactor(value=sf_value, failure_mode=FailureMode.YIELDING)
            passes, margin = sf.check_margin(actual, allowable)

            design_allowable = sf.apply_to_allowable(allowable)
            utilization = actual / design_allowable if design_allowable > 0 else float("inf")

            # Using slightly fuzzy comparison due to floating point
            assert passes == (utilization <= 1.0 + 1e-9), (
                f"Margin check inconsistent. Passes={passes}, Util={utilization}"
            )

            # Margin should be 1 - utilization ratio
            assert margin == pytest.approx(1.0 - utilization, abs=1e-7)

        check()

    def test_total_factor_includes_uncertainty(self) -> None:
        """Total factor = value * uncertainty_factor."""

        @given(
            sf_value=st.floats(min_value=1.0, max_value=3.0),
            unc_factor=st.floats(min_value=1.0, max_value=2.0),
        )
        @settings(max_examples=50)
        def check(sf_value: float, unc_factor: float) -> None:
            sf = SafetyFactor(
                value=sf_value,
                failure_mode=FailureMode.YIELDING,
                uncertainty_factor=unc_factor,
            )
            assert sf.total_factor == pytest.approx(sf_value * unc_factor)

        check()

    def test_margin_positive_if_compliant(self) -> None:
        """If component is compliant, margin must be non-negative."""

        @given(
            sf_value=st.floats(min_value=1.1, max_value=5.0),
            allowable=st.floats(min_value=100.0, max_value=1000.0),
        )
        def check(sf_value: float, allowable: float):
            sf = SafetyFactor(value=sf_value, failure_mode=FailureMode.YIELDING)
            # Construct minimal passing actual load
            design_allowable = sf.apply_to_allowable(allowable)
            actual = design_allowable * 0.99  # Definitely passes

            passes, margin = sf.check_margin(actual, allowable)
            assert passes
            assert margin >= 0

        check()
