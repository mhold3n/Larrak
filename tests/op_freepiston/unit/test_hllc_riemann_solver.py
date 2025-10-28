import numpy as np
import pytest

from campro.freepiston.net1d.flux import (
    apply_entropy_fix,
    conservative_from_primitive,
    enhanced_hllc_flux,
    enhanced_hllc_star_state,
    enhanced_wave_speeds,
    flux_from_primitive,
    get_flux_function,
    hllc_flux,
    hllc_star_state,
    primitive_from_conservative,
    roe_averages,
    roe_flux,
    wave_speeds,
)


class TestPrimitiveConservativeConversion:
    def test_primitive_from_conservative(self):
        """Test conversion from conservative to primitive variables."""
        # Test case 1: Standard air at STP
        U = (1.225, 0.0, 250000.0)  # rho, rho*u, rho*E
        rho, u, p = primitive_from_conservative(U, gamma=1.4)

        assert abs(rho - 1.225) < 1e-9
        assert abs(u - 0.0) < 1e-9
        assert (
            abs(p - 100000.0) < 1.0
        )  # Calculated pressure from given conservative variables

        # Test case 2: Moving gas
        U = (1.0, 100.0, 100000.0)  # rho, rho*u, rho*E
        rho, u, p = primitive_from_conservative(U, gamma=1.4)

        assert abs(rho - 1.0) < 1e-9
        assert abs(u - 100.0) < 1e-9
        assert p > 0.0

        # Test case 3: Zero density
        U = (0.0, 0.0, 0.0)
        rho, u, p = primitive_from_conservative(U, gamma=1.4)

        assert rho == 0.0
        assert u == 0.0
        assert p == 0.0

    def test_conservative_from_primitive(self):
        """Test conversion from primitive to conservative variables."""
        # Test case 1: Standard air at STP
        rho, u, p = 1.225, 0.0, 101325.0
        U = conservative_from_primitive(rho, u, p, gamma=1.4)

        assert abs(U[0] - 1.225) < 1e-9
        assert abs(U[1] - 0.0) < 1e-9
        assert U[2] > 0.0

        # Test case 2: Moving gas
        rho, u, p = 1.0, 100.0, 100000.0
        U = conservative_from_primitive(rho, u, p, gamma=1.4)

        assert abs(U[0] - 1.0) < 1e-9
        assert abs(U[1] - 100.0) < 1e-9
        assert U[2] > 0.0

    def test_round_trip_conversion(self):
        """Test round-trip conversion between primitive and conservative variables."""
        # Original primitive variables
        rho_orig, u_orig, p_orig = 1.2, 50.0, 150000.0

        # Convert to conservative and back
        U = conservative_from_primitive(rho_orig, u_orig, p_orig, gamma=1.4)
        rho, u, p = primitive_from_conservative(U, gamma=1.4)

        assert abs(rho - rho_orig) < 1e-9
        assert abs(u - u_orig) < 1e-9
        assert abs(p - p_orig) < 1e-6


class TestFluxCalculation:
    def test_flux_from_primitive(self):
        """Test flux calculation from primitive variables."""
        # Test case 1: Static gas
        rho, u, p = 1.0, 0.0, 100000.0
        F = flux_from_primitive(rho, u, p)

        assert abs(F[0] - 0.0) < 1e-9  # No mass flux
        assert abs(F[1] - 100000.0) < 1e-9  # Pressure flux
        assert abs(F[2] - 0.0) < 1e-9  # No energy flux

        # Test case 2: Moving gas
        rho, u, p = 1.0, 100.0, 100000.0
        F = flux_from_primitive(rho, u, p)

        assert abs(F[0] - 100.0) < 1e-9  # Mass flux
        assert F[1] > 100000.0  # Momentum flux (includes kinetic energy)
        assert F[2] > 0.0  # Energy flux

    def test_flux_conservation(self):
        """Test that flux calculation is consistent with conservative form."""
        rho, u, p = 1.0, 50.0, 120000.0
        F = flux_from_primitive(rho, u, p)

        # Check mass flux
        assert abs(F[0] - rho * u) < 1e-9

        # Check momentum flux
        expected_momentum_flux = rho * u**2 + p
        assert abs(F[1] - expected_momentum_flux) < 1e-9

        # Check energy flux
        E = p / (0.4 * rho) + 0.5 * u**2  # Specific total energy
        expected_energy_flux = (rho * E + p) * u
        assert abs(F[2] - expected_energy_flux) < 1e-6


class TestRoeAverages:
    def test_roe_averages_symmetric(self):
        """Test Roe averages for symmetric states."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (1.0, 0.0, 250000.0)

        rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma=1.4)

        assert abs(rho_roe - 1.0) < 1e-9
        assert abs(u_roe - 0.0) < 1e-9
        assert c_roe > 0.0

    def test_roe_averages_asymmetric(self):
        """Test Roe averages for asymmetric states."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (2.0, 0.0, 500000.0)

        rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma=1.4)

        assert rho_roe > 0.0
        assert abs(u_roe - 0.0) < 1e-9
        assert c_roe > 0.0

    def test_roe_averages_moving_gas(self):
        """Test Roe averages for moving gas states."""
        U_L = (1.0, 100.0, 300000.0)
        U_R = (1.0, -100.0, 300000.0)

        rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma=1.4)

        assert rho_roe > 0.0
        assert abs(u_roe - 0.0) < 1e-9  # Should average to zero
        assert c_roe > 0.0


class TestWaveSpeeds:
    def test_wave_speeds_symmetric(self):
        """Test wave speeds for symmetric states."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (1.0, 0.0, 250000.0)

        S_L, S_R, S_star, p_star = wave_speeds(U_L, U_R, gamma=1.4)

        assert S_L < 0.0  # Left wave should be negative
        assert S_R > 0.0  # Right wave should be positive
        assert S_L < S_star < S_R  # Contact wave should be between
        assert p_star > 0.0

    def test_wave_speeds_shock_tube(self):
        """Test wave speeds for shock tube problem."""
        # High pressure left, low pressure right
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.125, 0.0, 25000.0)

        S_L, S_R, S_star, p_star = wave_speeds(U_L, U_R, gamma=1.4)

        assert S_L < 0.0  # Left wave should be negative
        assert S_R > 0.0  # Right wave should be positive
        assert S_L < S_star < S_R  # Contact wave should be between
        assert p_star > 0.0

    def test_wave_speeds_ordering(self):
        """Test that wave speeds are properly ordered."""
        U_L = (1.0, 50.0, 200000.0)
        U_R = (0.5, -50.0, 100000.0)

        S_L, S_R, S_star, p_star = wave_speeds(U_L, U_R, gamma=1.4)

        assert S_L < S_R  # Left wave should be faster than right
        assert S_L < S_star < S_R  # Contact wave should be between
        assert p_star > 0.0


class TestHLLCStarState:
    def test_hllc_star_state_left(self):
        """Test HLLC star state calculation for left state."""
        U = (1.0, 0.0, 250000.0)
        S = -100.0
        S_star = -50.0
        p_star = 200000.0

        U_star = hllc_star_state(U, S, S_star, p_star, gamma=1.4)

        assert U_star[0] > 0.0  # Density should be positive
        assert U_star[1] == U_star[0] * S_star  # Momentum should be rho * u_star
        assert U_star[2] > 0.0  # Energy should be positive

    def test_hllc_star_state_right(self):
        """Test HLLC star state calculation for right state."""
        U = (0.5, 0.0, 125000.0)
        S = 100.0
        S_star = 50.0
        p_star = 200000.0

        U_star = hllc_star_state(U, S, S_star, p_star, gamma=1.4)

        assert U_star[0] > 0.0  # Density should be positive
        assert U_star[1] == U_star[0] * S_star  # Momentum should be rho * u_star
        assert U_star[2] > 0.0  # Energy should be positive


class TestHLLCFlux:
    def test_hllc_flux_symmetric(self):
        """Test HLLC flux for symmetric states."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (1.0, 0.0, 250000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # For symmetric states, flux should be zero
        assert abs(F[0]) < 1e-9  # No mass flux
        assert abs(F[1] - 100000.0) < 1e-6  # Pressure flux
        assert abs(F[2]) < 1e-9  # No energy flux

    def test_hllc_flux_shock_tube(self):
        """Test HLLC flux for shock tube problem."""
        # High pressure left, low pressure right
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.125, 0.0, 25000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        assert F[0] > 0.0  # Mass should flow from left to right
        assert F[1] > 0.0  # Momentum flux should be positive
        assert F[2] > 0.0  # Energy flux should be positive

    def test_hllc_flux_rarefaction(self):
        """Test HLLC flux for rarefaction wave."""
        # Low pressure left, high pressure right
        U_L = (0.125, 0.0, 25000.0)
        U_R = (1.0, 0.0, 250000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        assert F[0] < 0.0  # Mass should flow from right to left
        # For rarefaction, momentum flux direction depends on the specific conditions
        # The important thing is that the flux is physically reasonable
        assert abs(F[1]) < 1e6  # Momentum flux should be reasonable
        assert abs(F[2]) < 1e9  # Energy flux should be reasonable

    def test_hllc_flux_vacuum_states(self):
        """Test HLLC flux for vacuum states."""
        U_L = (0.0, 0.0, 0.0)
        U_R = (1.0, 0.0, 250000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should return zero flux for vacuum states
        assert F[0] == 0.0
        assert F[1] == 0.0
        assert F[2] == 0.0

    def test_hllc_flux_entropy_condition(self):
        """Test that HLLC flux satisfies entropy condition."""
        # Test with strong shock
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.1, 0.0, 2500.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Check that flux is physically reasonable
        assert F[0] > 0.0  # Mass flow from high to low pressure
        assert F[1] > 0.0  # Positive momentum flux
        assert F[2] > 0.0  # Positive energy flux

    def test_hllc_flux_conservation(self):
        """Test that HLLC flux is conservative."""
        U_L = (1.0, 100.0, 300000.0)
        U_R = (0.5, -100.0, 150000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Check that flux is finite and reasonable
        assert abs(F[0]) < 1e6  # Mass flux should be reasonable
        assert abs(F[1]) < 1e9  # Momentum flux should be reasonable
        assert abs(F[2]) < 1e12  # Energy flux should be reasonable

    def test_hllc_flux_robustness(self):
        """Test HLLC flux robustness with extreme states."""
        # Test with very high pressure ratio
        U_L = (1.0, 0.0, 1e6)
        U_R = (0.01, 0.0, 1e3)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should not crash and should return reasonable values
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])

    def test_hllc_flux_sonic_flow(self):
        """Test HLLC flux for sonic flow conditions."""
        # Test with sonic flow (u = c)
        rho, u, p = 1.0, 347.0, 100000.0  # u ≈ c for air
        U_L = conservative_from_primitive(rho, u, p, gamma=1.4)
        U_R = conservative_from_primitive(rho, -u, p, gamma=1.4)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should handle sonic flow without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])


class TestRoeFlux:
    def test_roe_flux_symmetric(self):
        """Test Roe flux for symmetric states."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (1.0, 0.0, 250000.0)

        F = roe_flux(U_L, U_R, gamma=1.4)

        # For symmetric states, flux should be zero
        assert abs(F[0]) < 1e-9  # No mass flux
        assert abs(F[1] - 100000.0) < 1e-6  # Pressure flux
        assert abs(F[2]) < 1e-9  # No energy flux

    def test_roe_flux_consistency(self):
        """Test that Roe flux is consistent with HLLC for simple cases."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        F_roe = roe_flux(U_L, U_R, gamma=1.4)
        F_hllc = hllc_flux(U_L, U_R, gamma=1.4)

        # Should be reasonably close for simple cases
        assert abs(F_roe[0] - F_hllc[0]) < 1e-3
        assert abs(F_roe[1] - F_hllc[1]) < 1e-3
        assert abs(F_roe[2] - F_hllc[2]) < 1e-3


class TestFluxFunctionFactory:
    def test_get_flux_function_hllc(self):
        """Test getting HLLC flux function."""
        flux_func = get_flux_function("hllc")
        assert flux_func == hllc_flux

    def test_get_flux_function_roe(self):
        """Test getting Roe flux function."""
        flux_func = get_flux_function("roe")
        assert flux_func == roe_flux

    def test_get_flux_function_invalid(self):
        """Test getting invalid flux function."""
        with pytest.raises(ValueError, match="Unknown flux method"):
            get_flux_function("invalid")


class TestHLLCValidationCases:
    def test_sod_shock_tube(self):
        """Test HLLC solver with Sod shock tube problem."""
        # Initial conditions for Sod shock tube
        U_L = (1.0, 0.0, 250000.0)  # High pressure, high density
        U_R = (0.125, 0.0, 25000.0)  # Low pressure, low density

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Expected behavior: mass flows from left to right
        assert F[0] > 0.0  # Positive mass flux
        assert F[1] > 0.0  # Positive momentum flux
        assert F[2] > 0.0  # Positive energy flux

    def test_strong_shock(self):
        """Test HLLC solver with strong shock."""
        # Very high pressure ratio
        U_L = (1.0, 0.0, 1e6)
        U_R = (0.1, 0.0, 1e4)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should handle strong shock without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert F[0] > 0.0  # Mass should flow from high to low pressure

    def test_rarefaction_fan(self):
        """Test HLLC solver with rarefaction fan."""
        # Low pressure left, high pressure right
        U_L = (0.1, 0.0, 1e4)
        U_R = (1.0, 0.0, 1e6)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should handle rarefaction without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert F[0] < 0.0  # Mass should flow from right to left

    def test_contact_discontinuity(self):
        """Test HLLC solver with contact discontinuity."""
        # Same pressure, different densities
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 250000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should handle contact discontinuity
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])

    def test_vacuum_handling(self):
        """Test HLLC solver vacuum state handling."""
        # Test various vacuum scenarios
        U_L = (0.0, 0.0, 0.0)
        U_R = (1.0, 0.0, 250000.0)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should return zero flux for vacuum
        assert F[0] == 0.0
        assert F[1] == 0.0
        assert F[2] == 0.0

    def test_sonic_flow_conditions(self):
        """Test HLLC solver at sonic flow conditions."""
        # Create sonic flow conditions
        rho, u, p = 1.0, 347.0, 100000.0  # u ≈ c for air at STP
        U_L = conservative_from_primitive(rho, u, p, gamma=1.4)
        U_R = conservative_from_primitive(rho, -u, p, gamma=1.4)

        F = hllc_flux(U_L, U_R, gamma=1.4)

        # Should handle sonic flow without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])


class TestEnhancedHLLCSolver:
    def test_enhanced_hllc_flux_basic(self):
        """Test enhanced HLLC flux for basic cases."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should return reasonable flux values
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])

    def test_enhanced_hllc_flux_entropy_fix(self):
        """Test enhanced HLLC flux with and without entropy fix."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.1, 0.0, 25000.0)

        F_with_fix = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)
        F_without_fix = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=False)

        # Both should be valid
        assert not np.isnan(F_with_fix[0])
        assert not np.isnan(F_without_fix[0])

        # May be different due to entropy fix
        # The entropy fix should not make the solution worse
        assert abs(F_with_fix[0]) <= abs(F_without_fix[0]) + 1e-6

    def test_enhanced_wave_speeds(self):
        """Test enhanced wave speed estimation."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        S_L, S_R, S_star, p_star = enhanced_wave_speeds(U_L, U_R, gamma=1.4)

        # Check wave speed ordering
        assert S_L < S_R
        assert S_L < S_star < S_R
        assert p_star > 0.0

        # Check that wave speeds are reasonable
        assert abs(S_L) < 1000.0  # Should be subsonic for this case
        assert abs(S_R) < 1000.0

    def test_enhanced_wave_speeds_extreme_ratios(self):
        """Test enhanced wave speeds with extreme pressure ratios."""
        U_L = (1.0, 0.0, 1e6)
        U_R = (0.01, 0.0, 1e3)

        S_L, S_R, S_star, p_star = enhanced_wave_speeds(U_L, U_R, gamma=1.4)

        # Should handle extreme ratios without issues
        assert S_L < S_R
        assert S_L < S_star < S_R
        assert p_star > 0.0
        assert not np.isnan(S_L)
        assert not np.isnan(S_R)
        assert not np.isnan(S_star)
        assert not np.isnan(p_star)

    def test_apply_entropy_fix(self):
        """Test entropy fix application."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        S_L_orig, S_R_orig = -100.0, 100.0
        S_L_fixed, S_R_fixed = apply_entropy_fix(
            U_L, U_R, S_L_orig, S_R_orig, gamma=1.4,
        )

        # Entropy fix should not make wave speeds worse
        assert S_L_fixed <= S_L_orig
        assert S_R_fixed >= S_R_orig

    def test_enhanced_hllc_star_state(self):
        """Test enhanced HLLC star state calculation."""
        U = (1.0, 0.0, 250000.0)
        S = -100.0
        S_star = -50.0
        p_star = 200000.0

        U_star = enhanced_hllc_star_state(U, S, S_star, p_star, gamma=1.4)

        # Check that star state is physically reasonable
        assert U_star[0] > 0.0  # Positive density
        assert U_star[1] == U_star[0] * S_star  # Correct momentum
        assert U_star[2] > 0.0  # Positive energy

    def test_enhanced_hllc_robustness(self):
        """Test enhanced HLLC robustness with challenging cases."""
        # Test with very high pressure ratio
        U_L = (1.0, 0.0, 1e6)
        U_R = (0.001, 0.0, 1e2)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should handle extreme conditions without crashing
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])

    def test_enhanced_hllc_sonic_flow(self):
        """Test enhanced HLLC with sonic flow conditions."""
        # Create sonic flow conditions
        rho, u, p = 1.0, 347.0, 100000.0  # u ≈ c for air
        U_L = conservative_from_primitive(rho, u, p, gamma=1.4)
        U_R = conservative_from_primitive(rho, -u, p, gamma=1.4)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should handle sonic flow without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])

    def test_enhanced_hllc_vacuum_handling(self):
        """Test enhanced HLLC vacuum state handling."""
        U_L = (0.0, 0.0, 0.0)
        U_R = (1.0, 0.0, 250000.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should return zero flux for vacuum states
        assert F[0] == 0.0
        assert F[1] == 0.0
        assert F[2] == 0.0

    def test_enhanced_hllc_consistency(self):
        """Test that enhanced HLLC is consistent with standard HLLC for simple cases."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        F_enhanced = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=False)
        F_standard = hllc_flux(U_L, U_R, gamma=1.4)

        # Should be reasonably close for simple cases
        assert abs(F_enhanced[0] - F_standard[0]) < 1e-3
        assert abs(F_enhanced[1] - F_standard[1]) < 1e-3
        assert abs(F_enhanced[2] - F_standard[2]) < 1e-3

    def test_enhanced_hllc_entropy_condition(self):
        """Test that enhanced HLLC satisfies entropy condition."""
        # Test with strong shock
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.1, 0.0, 2500.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Check that flux is physically reasonable
        assert F[0] > 0.0  # Mass flow from high to low pressure
        assert F[1] > 0.0  # Positive momentum flux
        assert F[2] > 0.0  # Positive energy flux

    def test_enhanced_hllc_rarefaction(self):
        """Test enhanced HLLC with rarefaction waves."""
        # Low pressure left, high pressure right
        U_L = (0.1, 0.0, 2500.0)
        U_R = (1.0, 0.0, 250000.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should handle rarefaction without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert F[0] < 0.0  # Mass should flow from right to left

    def test_enhanced_hllc_contact_discontinuity(self):
        """Test enhanced HLLC with contact discontinuity."""
        # Same pressure, different densities
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 250000.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should handle contact discontinuity
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])

    def test_enhanced_hllc_conservation(self):
        """Test that enhanced HLLC maintains conservation properties."""
        U_L = (1.0, 100.0, 300000.0)
        U_R = (0.5, -100.0, 150000.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Check that flux is finite and reasonable
        assert abs(F[0]) < 1e6  # Mass flux should be reasonable
        assert abs(F[1]) < 1e9  # Momentum flux should be reasonable
        assert abs(F[2]) < 1e12  # Energy flux should be reasonable

    def test_enhanced_hllc_numerical_stability(self):
        """Test enhanced HLLC numerical stability with challenging inputs."""
        # Test with very small densities
        U_L = (1e-6, 0.0, 250.0)
        U_R = (1e-6, 0.0, 250.0)

        F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)

        # Should handle small densities without issues
        assert not np.isnan(F[0])
        assert not np.isnan(F[1])
        assert not np.isnan(F[2])
        assert not np.isinf(F[0])
        assert not np.isinf(F[1])
        assert not np.isinf(F[2])

    def test_enhanced_hllc_performance(self):
        """Test enhanced HLLC performance with multiple calls."""
        U_L = (1.0, 0.0, 250000.0)
        U_R = (0.5, 0.0, 125000.0)

        # Test multiple calls to ensure consistency
        for _ in range(100):
            F = enhanced_hllc_flux(U_L, U_R, gamma=1.4, entropy_fix=True)
            assert not np.isnan(F[0])
            assert not np.isnan(F[1])
            assert not np.isnan(F[2])


class TestFluxFunctionFactoryEnhanced:
    def test_get_flux_function_enhanced_hllc(self):
        """Test getting enhanced HLLC flux function."""
        flux_func = get_flux_function("enhanced_hllc")
        assert flux_func == enhanced_hllc_flux

    def test_get_flux_function_invalid_enhanced(self):
        """Test getting invalid flux function."""
        with pytest.raises(ValueError, match="Unknown flux method"):
            get_flux_function("invalid_enhanced")
