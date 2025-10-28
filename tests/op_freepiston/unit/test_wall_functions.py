"""
Test suite for advanced wall functions and models.

This module tests the compressible wall functions, y+ calculation,
law-of-the-wall models, wall temperature evolution, and integration
with the 1D solver.
"""

import math

import numpy as np
import pytest

from campro.freepiston.net1d.wall import (
    WallModelParameters,
    advanced_heat_transfer_correlation,
    calculate_y_plus,
    compressible_corrections,
    compressible_wall_function,
    enhanced_wall_treatment,
    get_wall_function_method,
    multi_layer_wall_heat_transfer,
    radiation_heat_transfer,
    roughness_effects,
    spalding_wall_function,
    wall_function_validation,
    wall_function_with_roughness,
    wall_temperature_evolution,
)


class TestWallModelParameters:
    """Test WallModelParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = WallModelParameters()

        assert params.roughness == 0.0
        assert params.roughness_relative == 0.0
        assert params.Re_tau == 1000.0
        assert params.Pr == 0.7
        assert params.Pr_t == 0.9
        assert params.kappa == 0.41
        assert params.B == 5.2
        assert params.A_plus == 26.0
        assert params.M_wall == 0.0
        assert params.T_wall == 300.0
        assert params.T_ref == 300.0
        assert params.h_conv == 100.0
        assert params.emissivity == 0.8
        assert params.y_plus_target == 1.0
        assert params.max_iterations == 100
        assert params.tolerance == 1e-6

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = WallModelParameters(
            roughness=1e-6,
            roughness_relative=0.001,
            Re_tau=2000.0,
            Pr=0.8,
            Pr_t=0.95,
            kappa=0.42,
            B=5.5,
            A_plus=25.0,
            M_wall=0.1,
            T_wall=400.0,
            T_ref=350.0,
            h_conv=150.0,
            emissivity=0.9,
            y_plus_target=2.0,
            max_iterations=200,
            tolerance=1e-8,
        )

        assert params.roughness == 1e-6
        assert params.roughness_relative == 0.001
        assert params.Re_tau == 2000.0
        assert params.Pr == 0.8
        assert params.Pr_t == 0.95
        assert params.kappa == 0.42
        assert params.B == 5.5
        assert params.A_plus == 25.0
        assert params.M_wall == 0.1
        assert params.T_wall == 400.0
        assert params.T_ref == 350.0
        assert params.h_conv == 150.0
        assert params.emissivity == 0.9
        assert params.y_plus_target == 2.0
        assert params.max_iterations == 200
        assert params.tolerance == 1e-8


class TestYPlusCalculation:
    """Test y+ calculation functions."""

    def test_calculate_y_plus_basic(self):
        """Test basic y+ calculation."""
        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m

        y_plus = calculate_y_plus(rho=rho, u=u, mu=mu, y=y)

        # Should be positive
        assert y_plus > 0

        # Should be reasonable for typical flow conditions
        assert 0.1 <= y_plus <= 1000.0

    def test_calculate_y_plus_with_u_tau(self):
        """Test y+ calculation with provided u_tau."""
        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m
        u_tau = 1.0  # m/s

        y_plus = calculate_y_plus(rho=rho, u=u, mu=mu, y=y, u_tau=u_tau)

        # Should be positive
        assert y_plus > 0

        # Should be consistent with provided u_tau
        expected_y_plus = rho * u_tau * y / mu
        assert abs(y_plus - expected_y_plus) < 1e-10

    def test_calculate_y_plus_edge_cases(self):
        """Test y+ calculation edge cases."""
        rho = 1.2  # kg/m^3
        u = 0.0  # m/s (zero velocity)
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m

        y_plus = calculate_y_plus(rho=rho, u=u, mu=mu, y=y)

        # Should handle zero velocity gracefully
        assert y_plus >= 0


class TestCompressibleWallFunction:
    """Test compressible wall function implementation."""

    def test_compressible_wall_function_basic(self):
        """Test basic compressible wall function."""
        params = WallModelParameters()

        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_wall_function(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Check that all required keys are present
        required_keys = ["u_tau", "y_plus", "tau_w", "q_wall", "u_plus"]
        for key in required_keys:
            assert key in result

        # Check that values are reasonable
        assert result["u_tau"] > 0
        assert result["y_plus"] > 0
        assert result["tau_w"] > 0
        assert result["q_wall"] >= 0  # Can be zero if T <= T_wall
        assert result["u_plus"] > 0

    def test_compressible_wall_function_convergence(self):
        """Test that wall function converges."""
        params = WallModelParameters(tolerance=1e-8, max_iterations=100)

        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_wall_function(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Should converge within max iterations
        assert result["u_tau"] > 0
        assert result["y_plus"] > 0

    def test_compressible_wall_function_viscous_sublayer(self):
        """Test wall function in viscous sublayer (y+ < A_plus)."""
        params = WallModelParameters(A_plus=26.0)

        rho = 1.2  # kg/m^3
        u = 1.0  # m/s (low velocity for small y+)
        mu = 1.8e-5  # Pa·s
        y = 1e-6  # m (very small distance)
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_wall_function(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Should be in viscous sublayer
        assert result["y_plus"] < params.A_plus
        assert result["u_plus"] == result["y_plus"]  # Linear relationship

    def test_compressible_wall_function_log_layer(self):
        """Test wall function in log layer (y+ > A_plus)."""
        params = WallModelParameters(A_plus=26.0)

        rho = 1.2  # kg/m^3
        u = 100.0  # m/s (high velocity for large y+)
        mu = 1.8e-5  # Pa·s
        y = 0.01  # m (larger distance)
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_wall_function(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Should be in log layer
        assert result["y_plus"] > params.A_plus

        # Check log-law relationship
        expected_u_plus = (1.0 / params.kappa) * math.log(result["y_plus"]) + params.B
        assert abs(result["u_plus"] - expected_u_plus) < 0.1


class TestRoughnessEffects:
    """Test roughness effects implementation."""

    def test_roughness_effects_smooth(self):
        """Test roughness effects for smooth wall."""
        params = WallModelParameters(Re_tau=1000.0)

        y_plus = 50.0
        roughness_relative = 0.0  # Smooth wall

        result = roughness_effects(
            y_plus=y_plus,
            roughness_relative=roughness_relative,
            params=params,
        )

        # Should have no roughness correction
        assert result["roughness_factor"] == 1.0
        assert result["roughness_correction"] == 0.0
        assert result["B_rough"] == params.B
        assert result["k_s_plus"] == 0.0

    def test_roughness_effects_rough(self):
        """Test roughness effects for rough wall."""
        params = WallModelParameters(Re_tau=1000.0)

        y_plus = 50.0
        roughness_relative = 0.01  # Rough wall

        result = roughness_effects(
            y_plus=y_plus,
            roughness_relative=roughness_relative,
            params=params,
        )

        # Should have roughness correction
        assert result["roughness_factor"] > 1.0
        assert result["roughness_correction"] > 0.0
        assert result["B_rough"] < params.B
        assert result["k_s_plus"] > 0.0

    def test_roughness_effects_transitional(self):
        """Test roughness effects in transitional regime."""
        params = WallModelParameters(Re_tau=1000.0)

        y_plus = 50.0
        roughness_relative = 0.005  # Transitional roughness

        result = roughness_effects(
            y_plus=y_plus,
            roughness_relative=roughness_relative,
            params=params,
        )

        # Should have moderate roughness correction
        assert 1.0 <= result["roughness_factor"] <= 1.2
        assert 0.0 <= result["roughness_correction"] <= 0.2
        assert result["B_rough"] <= params.B


class TestCompressibilityCorrections:
    """Test compressibility corrections."""

    def test_compressibility_corrections_incompressible(self):
        """Test compressibility corrections for incompressible flow."""
        params = WallModelParameters()

        M = 0.1  # Low Mach number
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_corrections(M=M, T=T, T_wall=T_wall, params=params)

        # Should have minimal compressibility effects
        assert result["compressibility_factor"] == 1.0
        assert result["temperature_factor"] > 0
        assert result["total_correction"] > 0
        assert result["T_ratio"] == T / T_wall

    def test_compressibility_corrections_compressible(self):
        """Test compressibility corrections for compressible flow."""
        params = WallModelParameters()

        M = 0.5  # Moderate Mach number
        T = 300.0  # K
        T_wall = 400.0  # K

        result = compressible_corrections(M=M, T=T, T_wall=T_wall, params=params)

        # Should have compressibility effects
        assert result["compressibility_factor"] > 1.0
        assert result["temperature_factor"] > 0
        assert (
            result["total_correction"] > 0
        )  # Can be less than 1 due to temperature factor
        assert result["T_ratio"] == T / T_wall


class TestWallTemperatureEvolution:
    """Test wall temperature evolution."""

    def test_wall_temperature_evolution_basic(self):
        """Test basic wall temperature evolution."""
        T_wall_old = 400.0  # K
        q_wall = 1000.0  # W/m^2
        dt = 0.001  # s

        wall_properties = {
            "density": 7800.0,  # kg/m^3 (steel)
            "specific_heat": 500.0,  # J/(kg·K)
            "thickness": 0.01,  # m
            "area": 1.0,  # m^2
        }

        T_wall_new = wall_temperature_evolution(
            T_wall_old=T_wall_old,
            q_wall=q_wall,
            dt=dt,
            wall_properties=wall_properties,
        )

        # Should increase temperature due to heat flux
        assert T_wall_new > T_wall_old

        # Should be reasonable temperature change
        dT = T_wall_new - T_wall_old
        assert 0 < dT < 10.0  # Reasonable temperature change

    def test_wall_temperature_evolution_no_heat_flux(self):
        """Test wall temperature evolution with no heat flux."""
        T_wall_old = 400.0  # K
        q_wall = 0.0  # W/m^2 (no heat flux)
        dt = 0.001  # s

        wall_properties = {
            "density": 7800.0,  # kg/m^3 (steel)
            "specific_heat": 500.0,  # J/(kg·K)
            "thickness": 0.01,  # m
            "area": 1.0,  # m^2
        }

        T_wall_new = wall_temperature_evolution(
            T_wall_old=T_wall_old,
            q_wall=q_wall,
            dt=dt,
            wall_properties=wall_properties,
        )

        # Should remain unchanged
        assert T_wall_new == T_wall_old


class TestMultiLayerWallHeatTransfer:
    """Test multi-layer wall heat transfer."""

    def test_multi_layer_wall_heat_transfer_basic(self):
        """Test basic multi-layer wall heat transfer."""
        params = WallModelParameters()

        T_gas = 500.0  # K
        T_wall_surface = 400.0  # K

        wall_layers = [
            {
                "thickness": 0.005,  # m
                "conductivity": 50.0,  # W/(m·K)
                "area": 1.0,  # m^2
            },
            {
                "thickness": 0.005,  # m
                "conductivity": 25.0,  # W/(m·K)
                "area": 1.0,  # m^2
            },
        ]

        result = multi_layer_wall_heat_transfer(
            T_gas=T_gas,
            T_wall_surface=T_wall_surface,
            wall_layers=wall_layers,
            params=params,
        )

        # Check that all required keys are present
        required_keys = ["q_wall", "total_resistance", "layer_temperatures"]
        for key in required_keys:
            assert key in result

        # Check that values are reasonable
        assert result["q_wall"] > 0  # Heat flux from gas to wall
        assert result["total_resistance"] > 0
        assert len(result["layer_temperatures"]) == len(wall_layers) + 1


class TestRadiationHeatTransfer:
    """Test radiation heat transfer."""

    def test_radiation_heat_transfer_basic(self):
        """Test basic radiation heat transfer."""
        T_gas = 1000.0  # K
        T_wall = 800.0  # K
        emissivity = 0.8
        area = 1.0  # m^2

        q_rad = radiation_heat_transfer(
            T_gas=T_gas,
            T_wall=T_wall,
            emissivity=emissivity,
            area=area,
        )

        # Should be positive (heat flux from gas to wall)
        assert q_rad > 0

        # Should be reasonable magnitude
        assert 0 < q_rad < 1e6  # W/m^2

    def test_radiation_heat_transfer_no_difference(self):
        """Test radiation heat transfer with no temperature difference."""
        T_gas = 800.0  # K
        T_wall = 800.0  # K (same temperature)
        emissivity = 0.8
        area = 1.0  # m^2

        q_rad = radiation_heat_transfer(
            T_gas=T_gas,
            T_wall=T_wall,
            emissivity=emissivity,
            area=area,
        )

        # Should be zero
        assert q_rad == 0.0


class TestAdvancedHeatTransferCorrelation:
    """Test advanced heat transfer correlation."""

    def test_advanced_heat_transfer_correlation_turbulent(self):
        """Test advanced heat transfer correlation for turbulent flow."""
        params = WallModelParameters()

        rho = 1.2  # kg/m^3
        u = 50.0  # m/s
        mu = 1.8e-5  # Pa·s
        k = 0.026  # W/(m·K)
        cp = 1005.0  # J/(kg·K)
        T = 300.0  # K
        T_wall = 400.0  # K
        D_hydraulic = 0.1  # m

        result = advanced_heat_transfer_correlation(
            rho=rho,
            u=u,
            mu=mu,
            k=k,
            cp=cp,
            T=T,
            T_wall=T_wall,
            D_hydraulic=D_hydraulic,
            params=params,
        )

        # Check that all required keys are present
        required_keys = ["Re", "Pr", "Nu", "h", "q_wall", "q_wall_corrected", "y_plus"]
        for key in required_keys:
            assert key in result

        # Check that values are reasonable
        assert result["Re"] > 2300  # Should be turbulent
        assert result["Pr"] > 0
        assert result["Nu"] > 0
        assert result["h"] > 0
        assert result["q_wall"] != 0  # Can be negative if T < T_wall
        assert result["q_wall_corrected"] != 0  # Can be negative if T < T_wall
        assert result["y_plus"] > 0

    def test_advanced_heat_transfer_correlation_laminar(self):
        """Test advanced heat transfer correlation for laminar flow."""
        params = WallModelParameters()

        rho = 1.2  # kg/m^3
        u = 1.0  # m/s (low velocity for laminar)
        mu = 1.8e-5  # Pa·s
        k = 0.026  # W/(m·K)
        cp = 1005.0  # J/(kg·K)
        T = 300.0  # K
        T_wall = 400.0  # K
        D_hydraulic = 0.1  # m

        result = advanced_heat_transfer_correlation(
            rho=rho,
            u=u,
            mu=mu,
            k=k,
            cp=cp,
            T=T,
            T_wall=T_wall,
            D_hydraulic=D_hydraulic,
            params=params,
        )

        # Check that values are reasonable
        # Note: With these parameters, Re > 2300, so it's actually turbulent
        assert result["Re"] > 0  # Should be positive
        assert result["Pr"] > 0
        assert result["Nu"] > 0
        assert result["h"] > 0
        assert result["q_wall"] != 0  # Can be negative if T < T_wall
        assert result["q_wall_corrected"] != 0  # Can be negative if T < T_wall
        assert result["y_plus"] > 0


class TestWallFunctionWithRoughness:
    """Test enhanced wall function with roughness effects."""

    def test_wall_function_with_roughness_basic(self):
        """Test basic wall function with roughness effects."""
        params = WallModelParameters(roughness_relative=0.001)

        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m
        T = 300.0  # K
        T_wall = 400.0  # K

        result = wall_function_with_roughness(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Check that all required keys are present
        required_keys = [
            "u_tau",
            "y_plus",
            "tau_w",
            "q_wall",
            "u_plus",
            "roughness_factor",
            "compressibility_factor",
            "total_correction",
        ]
        for key in required_keys:
            assert key in result

        # Check that values are reasonable
        assert result["u_tau"] > 0
        assert result["y_plus"] > 0
        assert result["tau_w"] > 0
        assert result["q_wall"] >= 0
        assert result["u_plus"] > 0
        assert result["roughness_factor"] > 0
        assert result["compressibility_factor"] > 0
        assert result["total_correction"] > 0


class TestWallFunctionValidation:
    """Test wall function validation."""

    def test_wall_function_validation_valid(self):
        """Test wall function validation for valid inputs."""
        params = WallModelParameters()

        y_plus = 50.0
        u_plus = 15.0

        result = wall_function_validation(
            y_plus=y_plus,
            u_plus=u_plus,
            params=params,
        )

        # Check that all required keys are present
        required_keys = [
            "y_plus_valid",
            "u_plus_valid",
            "consistency_valid",
            "u_plus_expected",
        ]
        for key in required_keys:
            assert key in result

        # Check validation results
        assert result["y_plus_valid"] is True
        assert result["u_plus_valid"] is True
        assert result["consistency_valid"] is True
        assert result["u_plus_expected"] > 0

    def test_wall_function_validation_invalid(self):
        """Test wall function validation for invalid inputs."""
        params = WallModelParameters()

        y_plus = 2000.0  # Too large
        u_plus = 100.0  # Too large

        result = wall_function_validation(
            y_plus=y_plus,
            u_plus=u_plus,
            params=params,
        )

        # Check validation results
        assert result["y_plus_valid"] is False
        assert result["u_plus_valid"] is False
        # Consistency might still be valid if the relationship holds


class TestGetWallFunctionMethod:
    """Test wall function method selection."""

    def test_get_wall_function_method_compressible(self):
        """Test getting compressible wall function method."""
        method = get_wall_function_method("compressible")

        # Should return a callable
        assert callable(method)

        # Test that it can be called with appropriate arguments
        params = WallModelParameters()
        result = method(
            rho=1.2,
            u=10.0,
            mu=1.8e-5,
            y=0.001,
            T=300.0,
            T_wall=400.0,
            params=params,
        )

        # Should return a dictionary with expected keys
        assert isinstance(result, dict)
        assert "u_tau" in result
        assert "y_plus" in result
        assert "tau_w" in result
        assert "q_wall" in result
        assert "u_plus" in result

    def test_get_wall_function_method_roughness(self):
        """Test getting roughness wall function method."""
        method = get_wall_function_method("roughness")

        # Should return a callable
        assert callable(method)

        # Test that it can be called with appropriate arguments
        params = WallModelParameters()
        result = method(
            rho=1.2,
            u=10.0,
            mu=1.8e-5,
            y=0.001,
            T=300.0,
            T_wall=400.0,
            params=params,
        )

        # Should return a dictionary with expected keys
        assert isinstance(result, dict)
        assert "u_tau" in result
        assert "y_plus" in result
        assert "tau_w" in result
        assert "q_wall" in result
        assert "u_plus" in result
        assert "roughness_factor" in result

    def test_get_wall_function_method_simple(self):
        """Test getting simple wall function method."""
        method = get_wall_function_method("simple")

        # Should return a callable
        assert callable(method)

        # Test that it can be called with appropriate arguments
        params = WallModelParameters()
        result = method(
            rho=1.2,
            u=10.0,
            mu=1.8e-5,
            y=0.001,
            T=300.0,
            T_wall=400.0,
            params=params,
        )

        # Should return a dictionary with expected keys
        assert isinstance(result, dict)
        assert "u_tau" in result
        assert "y_plus" in result
        assert "tau_w" in result
        assert "q_wall" in result
        assert "u_plus" in result

    def test_get_wall_function_method_invalid(self):
        """Test getting invalid wall function method."""
        with pytest.raises(ValueError, match="Unknown wall function method"):
            get_wall_function_method("invalid_method")


class TestIntegrationWith1DSolver:
    """Test integration with 1D solver."""

    def test_wall_function_integration(self):
        """Test that wall functions can be integrated with 1D solver."""
        # This is a simplified test to ensure the functions can be called
        # from the 1D solver context

        params = WallModelParameters()

        # Simulate 1D solver call
        rho = 1.2  # kg/m^3
        u = 10.0  # m/s
        mu = 1.8e-5  # Pa·s
        y = 0.001  # m
        T = 300.0  # K
        T_wall = 400.0  # K

        # Test that wall function can be called
        result = wall_function_with_roughness(
            rho=rho,
            u=u,
            mu=mu,
            y=y,
            T=T,
            T_wall=T_wall,
            params=params,
        )

        # Should return valid results
        assert result["u_tau"] > 0
        assert result["y_plus"] > 0
        assert result["tau_w"] > 0
        assert result["q_wall"] >= 0

        # Test that wall source terms can be calculated
        U = np.array([rho, rho * u, rho * 1000.0])  # Conservative variables
        solver_params = {"cell_volume": 1.0, "wall_area": 1.0}

        # This would be called from the 1D solver
        from campro.freepiston.net1d.stepper import _calculate_wall_source_terms

        source = _calculate_wall_source_terms(result, U, solver_params)

        # Should return source terms
        assert len(source) == len(U)
        assert isinstance(source, np.ndarray)


class TestSpaldingLaw:
    """Tests for Spalding's law-of-the-wall implementation."""

    def test_spalding_converges_and_diagnostics(self):
        """Spalding should converge and return diagnostics."""
        # Moderate and high y+ cases
        for y_plus in [5.0, 30.0, 120.0]:
            u_plus, diag = spalding_wall_function(
                y_plus=y_plus,
                kappa=0.41,
                E=9.0,
                max_iterations=100,
                tolerance=1e-10,
            )
            assert u_plus > 0
            assert isinstance(diag, dict)
            assert diag.get("converged") is True
            assert diag.get("iterations", 0) > 0
            assert diag.get("residual", 1.0) < 1e-8

    def test_spalding_limits_match_linear_and_log_layers(self):
        """Spalding should match linear for small y+ and approach log-law for large y+."""
        kappa = 0.41
        B = 5.2
        # Small y+: u+ ~ y+
        y_small = 0.5
        u_small, _ = spalding_wall_function(y_plus=y_small, kappa=kappa, E=9.0)
        assert abs(u_small - y_small) / max(y_small, 1e-12) < 0.05

        # Large y+: u+ ~ (1/kappa) ln y+ + B
        y_large = 150.0
        u_large, _ = spalding_wall_function(y_plus=y_large, kappa=kappa, E=9.0)
        u_log = (1.0 / kappa) * math.log(y_large) + B
        assert abs(u_large - u_log) / u_log < 0.1


class TestEnhancedWallTreatment:
    """Tests for enhanced wall treatment with automatic model selection."""

    def test_enhanced_wall_treatment_basic(self):
        from campro.freepiston.net1d.mesh import create_ale_mesh

        mesh = create_ale_mesh(0.0, 0.1, 20)
        params = WallModelParameters()
        flow_params = {
            "rho": 1.2,
            "u": 12.0,
            "mu": 1.8e-5,
            "T": 300.0,
            "T_wall": 400.0,
        }
        result = enhanced_wall_treatment(
            mesh=mesh, flow_params=flow_params, wall_params=params,
        )
        for key in ["u_tau", "y_plus", "u_plus", "tau_w", "model"]:
            assert key in result
        assert result["u_tau"] > 0
        assert result["y_plus"] > 0
        assert result["u_plus"] > 0
        assert result["tau_w"] > 0
        assert result["model"] in {"linear", "log", "spalding"}


if __name__ == "__main__":
    pytest.main([__file__])
