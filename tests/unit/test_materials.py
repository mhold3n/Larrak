"""Unit tests for material properties (gases and fuels).

Tests cover:
- PhysicalConstant dataclass operations (unit checks)
- Fuel property database accuracy
- Gas property correlations (NASA polynomials)
- Temperature-dependent property behavior via Hypothesis
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from campro.materials.fuels import FUEL_DATABASE, create_fuel_database, get_fuel, list_fuels
from campro.materials.fuels import get_lhv as fuel_get_lhv
from campro.materials.gases import (
    GAS_COEFFS,
    get_cp,
    get_enthalpy,
    get_gamma,
    get_molecular_weight,
    list_gases,
)
from campro.materials.gases import R_UNIVERSAL as R_GAS_FLOAT
from campro.units import CASADI_EPSILON, R_UNIVERSAL, PhysicalConstant, get_lhv, get_stoich_afr


class TestPhysicalConstant:
    """Tests for PhysicalConstant dataclass."""

    def test_basic_value_access(self) -> None:
        """Test accessing constant value."""
        pc = PhysicalConstant(
            value=14.7,
            unit="kg_air/kg_fuel",
            source="test",
        )
        assert pc.value == 14.7
        assert pc.unit == "kg_air/kg_fuel"

    def test_float_conversion(self) -> None:
        """Test that PhysicalConstant can be used as float."""
        pc = PhysicalConstant(value=10.0, unit="m", source="test")
        assert float(pc) == 10.0

    def test_arithmetic_operations(self) -> None:
        """Test arithmetic with PhysicalConstant."""
        pc1 = PhysicalConstant(value=10.0, unit="m", source="test")
        pc2 = PhysicalConstant(value=2.0, unit="m", source="test")

        assert pc1 * 2 == 20.0
        assert 2 * pc1 == 20.0
        assert pc1 / 2 == 5.0
        assert pc1 + pc2 == 12.0
        assert pc1 - pc2 == 8.0
        assert pc1 * pc2 == 20.0

    def test_comparison_operations(self) -> None:
        """Test comparison operators."""
        pc1 = PhysicalConstant(value=10.0, unit="m", source="test")
        pc2 = PhysicalConstant(value=5.0, unit="m", source="test")

        assert pc1 > pc2
        assert pc2 < pc1
        assert pc1 >= 10.0
        assert pc2 <= 5.0

    def test_uncertainty_bounds(self) -> None:
        """Test uncertainty calculation."""
        pc = PhysicalConstant(
            value=14.7,
            unit="kg_air/kg_fuel",
            source="test",
            uncertainty=0.1,
        )
        val, unc = pc.with_uncertainty()
        assert val == 14.7
        assert unc == 0.1

        low, high = pc.bounds()
        assert low == pytest.approx(14.6)
        assert high == pytest.approx(14.8)

    def test_validity_range_check(self) -> None:
        """Test valid range checking."""
        pc = PhysicalConstant(
            value=14.7,
            unit="kg_air/kg_fuel",
            source="test",
            valid_range=(14.5, 14.9),
        )
        assert pc.is_valid(14.7)
        assert pc.is_valid(14.5)
        assert not pc.is_valid(14.0)
        assert not pc.is_valid(15.0)

    def test_relative_uncertainty(self) -> None:
        """Test relative uncertainty calculation."""
        pc = PhysicalConstant(
            value=100.0,
            unit="Pa",
            source="test",
            uncertainty=5.0,
        )
        assert pc.relative_uncertainty() == 0.05


class TestUnitsModule:
    """Tests for campro.units module constants."""

    def test_r_universal_value(self) -> None:
        """Verify universal gas constant matches CODATA 2018."""
        # CODATA 2018 exact value
        expected = 8.31446261815324
        assert R_UNIVERSAL.value == expected
        assert R_UNIVERSAL.unit == "J/(mol·K)"

    def test_stoich_afr_gasoline(self) -> None:
        """Test stoichiometric AFR for gasoline."""
        afr = get_stoich_afr("gasoline")
        assert afr.value == pytest.approx(14.7, rel=0.01)
        assert afr.uncertainty is not None
        assert afr.uncertainty <= 0.2

    def test_lhv_hydrogen(self) -> None:
        """Test lower heating value of hydrogen."""
        lhv = get_lhv("hydrogen")
        # Hydrogen LHV should be ~120 MJ/kg
        assert lhv.value == pytest.approx(120.0e6, rel=0.02)

    def test_casadi_epsilon_small_enough(self) -> None:
        """Test that CasADi epsilon prevents division by zero."""
        assert CASADI_EPSILON.value < 1e-10
        assert CASADI_EPSILON.value > 0


class TestFuelDatabase:
    """Tests for fuel property database."""

    def test_list_fuels(self) -> None:
        """Test that fuel database contains expected fuels."""
        fuels = list_fuels()
        assert "gasoline" in fuels
        assert "diesel" in fuels
        assert "hydrogen" in fuels
        assert len(fuels) >= 6

    def test_get_fuel_properties(self) -> None:
        """Test retrieving fuel properties."""
        gasoline = get_fuel("gasoline")
        assert gasoline.lhv > 40e6  # J/kg
        assert gasoline.stoich_afr == pytest.approx(14.7, rel=0.01)
        assert gasoline.density_15c > 700  # kg/m³

    def test_get_fuel_case_insensitive(self) -> None:
        """Test case-insensitive fuel lookup."""
        g1 = get_fuel("gasoline")
        g2 = get_fuel("GASOLINE")
        g3 = get_fuel("Gasoline")
        assert g1.lhv == g2.lhv == g3.lhv

    def test_unknown_fuel_raises(self) -> None:
        """Test that unknown fuel raises KeyError."""
        with pytest.raises(KeyError, match="Unknown fuel"):
            get_fuel("plutonium")

    def test_fuel_lhv_uncertainty(self) -> None:
        """Test that LHV uncertainty is reasonable."""
        for fuel_name in list_fuels():
            fuel = get_fuel(fuel_name)
            rel_unc = fuel.lhv_uncertainty / fuel.lhv
            assert rel_unc < 0.1  # Less than 10% relative uncertainty

    def test_material_database_creation(self) -> None:
        """Test creating MaterialDatabase from fuel."""
        db = create_fuel_database("gasoline")
        assert "density_gasoline" in db.list_properties()
        assert "cp_gasoline" in db.list_properties()

        # Test property evaluation
        density = db.get("density_gasoline", 300.0)
        assert 700 < density < 800  # kg/m³


class TestGasProperties:
    """Tests for gas property correlations (NASA polynomials)."""

    def test_list_gases(self) -> None:
        """Test available gas species."""
        gases = list_gases()
        assert "N2" in gases
        assert "O2" in gases
        assert "CO2" in gases
        assert "H2O" in gases
        assert "air" in gases

    def test_molecular_weights(self) -> None:
        """Test molecular weight values."""
        assert get_molecular_weight("N2") == pytest.approx(0.028, rel=0.01)
        assert get_molecular_weight("O2") == pytest.approx(0.032, rel=0.01)
        assert get_molecular_weight("H2") == pytest.approx(0.002, rel=0.01)
        assert get_molecular_weight("CO2") == pytest.approx(0.044, rel=0.01)

    def test_air_cp_room_temperature(self) -> None:
        """Test air Cp at room temperature (~1005 J/(kg·K))."""
        cp = get_cp("air", 300.0)
        assert cp == pytest.approx(1005, rel=0.02)

    def test_air_cp_high_temperature(self) -> None:
        """Test air Cp at combustion temperatures."""
        cp = get_cp("air", 1000.0)
        # Cp increases with temperature
        assert cp > 1100
        assert cp < 1200

    def test_air_gamma_room_temperature(self) -> None:
        """Test air gamma at room temperature (~1.4)."""
        gamma = get_gamma("air", 300.0)
        assert gamma == pytest.approx(1.4, rel=0.02)

    def test_gamma_decreases_with_temperature(self) -> None:
        """Test that gamma decreases with temperature for polyatomic gases."""
        gamma_300 = get_gamma("CO2", 300.0)
        gamma_1000 = get_gamma("CO2", 1000.0)
        assert gamma_1000 < gamma_300

    def test_h2_high_cp(self) -> None:
        """Test that H2 has very high specific heat."""
        cp_h2 = get_cp("H2", 500.0)
        cp_n2 = get_cp("N2", 500.0)
        # H2 Cp should be ~14x N2 due to low molecular weight
        assert cp_h2 > 10 * cp_n2

    def test_enthalpy_increases_with_temperature(self) -> None:
        """Test enthalpy monotonically increases with T."""
        h_300 = get_enthalpy("N2", 300.0)
        h_500 = get_enthalpy("N2", 500.0)
        h_1000 = get_enthalpy("N2", 1000.0)
        assert h_500 > h_300
        assert h_1000 > h_500

    def test_unknown_species_raises(self) -> None:
        """Test that unknown species raises KeyError."""
        with pytest.raises(KeyError):
            get_cp("XenonTrioxide", 300.0)


class TestThermodynamicConsistency:
    """Tests for thermodynamic consistency between modules."""

    def test_gas_constant_consistency(self) -> None:
        """Test R_universal matches between units and gases modules."""
        assert R_UNIVERSAL.value == R_GAS_FLOAT

    def test_stoich_afr_consistency(self) -> None:
        """Test AFR matches between units and fuels modules."""
        units_afr = get_stoich_afr("gasoline").value

        fuel = get_fuel("gasoline")
        assert units_afr == pytest.approx(fuel.stoich_afr, rel=0.01)

    def test_cv_plus_r_equals_cp(self) -> None:
        """Test thermodynamic identity Cp = Cv + R/M for ideal gases."""
        species = "N2"
        t = 500.0

        cp = get_cp(species, t)
        gamma = get_gamma(species, t)
        mw = get_molecular_weight(species)

        # Cv = Cp - R/M (for ideal gas)
        r_specific = R_UNIVERSAL.value / mw
        cv = cp - r_specific

        # gamma = Cp/Cv
        gamma_calc = cp / cv
        assert gamma == pytest.approx(gamma_calc, rel=1e-6)


# =============================================================================
# Property-Based Tests (Hypothesis)
# =============================================================================


class TestPropertyBasedThermodynamics:
    """Property-based tests for thermodynamic functions using Hypothesis.

    These tests verify invariants that should hold across wide input ranges.
    """

    @pytest.mark.parametrize("species", ["air", "N2", "O2", "CO2", "H2O"])
    def test_cp_positive_all_temperatures(self, species: str) -> None:
        """Cp must be positive at all valid temperatures."""

        @given(T=st.floats(min_value=200.0, max_value=3000.0))
        @settings(max_examples=50)
        def check(T: float) -> None:
            cp = get_cp(species, T)
            assert cp > 0, f"Cp must be positive, got {cp} at T={T}"

        check()

    @pytest.mark.parametrize("species", ["air", "N2", "O2", "CO2"])
    def test_gamma_in_physical_range(self, species: str) -> None:
        """Gamma must be between 1.0 and 1.67 (monatomic limit)."""

        @given(T=st.floats(min_value=200.0, max_value=3000.0))
        @settings(max_examples=50)
        def check(T: float) -> None:
            gamma = get_gamma(species, T)
            assert 1.0 < gamma <= 1.67, f"Gamma out of range: {gamma} at T={T}"

        check()

    def test_enthalpy_monotonic_with_temperature(self) -> None:
        """Enthalpy must increase monotonically with temperature."""

        @given(
            T1=st.floats(min_value=200.0, max_value=2000.0),
            delta=st.floats(min_value=1.0, max_value=500.0),
        )
        @settings(max_examples=50)
        def check(T1: float, delta: float) -> None:
            T2 = T1 + delta
            h1 = get_enthalpy("N2", T1)
            h2 = get_enthalpy("N2", T2)
            assert h2 > h1, f"Enthalpy must increase: h({T1})={h1}, h({T2})={h2}"

        check()

    @pytest.mark.parametrize("species", ["N2", "O2", "CO2", "H2O", "H2", "CO", "CH4"])
    def test_polynomial_continuity(self, species: str) -> None:
        """Test continuity of NASA polynomials at the transition temperature."""
        coeffs = GAS_COEFFS[species]
        t_mid = coeffs.t_mid
        eps = 1e-6

        # Test just below and just above T_mid
        t_low = t_mid - eps
        t_high = t_mid + eps

        # Cp/R
        cp_low = coeffs.cp_over_r(t_low)
        cp_high = coeffs.cp_over_r(t_high)
        # Using 1% tolerance for continuity
        assert cp_low == pytest.approx(cp_high, rel=0.01), (
            f"Cp discontinuity at {t_mid}K for {species}"
        )

        # H/RT
        h_low = coeffs.h_over_rt(t_low)
        h_high = coeffs.h_over_rt(t_high)
        assert h_low == pytest.approx(h_high, rel=0.01), (
            f"Enthalpy discontinuity at {t_mid}K for {species}"
        )


class TestPropertyBasedPhysicalConstant:
    """Property-based tests for PhysicalConstant algebra."""

    def test_multiplication_commutativity(self) -> None:
        """a * b == b * a for PhysicalConstant."""

        @given(
            v1=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False),
            v2=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def check(v1: float, v2: float) -> None:
            pc1 = PhysicalConstant(value=v1, unit="unit1", source="test")
            pc2 = PhysicalConstant(value=v2, unit="unit2", source="test")
            assert pc1 * pc2 == pytest.approx(pc2 * pc1)

        check()

    def test_uncertainty_propagation_positive(self) -> None:
        """Uncertainty must always be non-negative."""

        @given(
            value=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
            unc=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def check(value: float, unc: float) -> None:
            pc = PhysicalConstant(value=value, unit="m", source="test", uncertainty=unc)
            _, u = pc.with_uncertainty()
            assert u >= 0

        check()

    def test_bounds_symmetric_around_value(self) -> None:
        """Uncertainty bounds should be symmetric around the value."""

        @given(
            value=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False),
            unc=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        )
        @settings(max_examples=50)
        def check(value: float, unc: float) -> None:
            pc = PhysicalConstant(value=value, unit="Pa", source="test", uncertainty=unc)
            low, high = pc.bounds()
            # Check symmetry
            assert high - value == pytest.approx(value - low, rel=1e-10)

        check()
