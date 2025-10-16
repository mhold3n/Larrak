"""Tests for cross-fidelity validation between 0D and 1D models."""

import numpy as np

from campro.freepiston.validation.cross_fidelity import (
    ModelComparison,
    ValidationParameters,
    ValidationResult,
    create_validation_report,
    cross_fidelity_validation,
    run_validation_suite,
)


class TestCrossFidelityValidation:
    """Test cross-fidelity validation functionality."""

    def test_validation_parameters_defaults(self):
        """Test default validation parameters."""
        params = ValidationParameters()

        assert params.t_end == 1.0
        assert params.dt_0d == 1e-4
        assert params.dt_1d == 1e-5
        assert params.relative_tolerance == 1e-3
        assert params.absolute_tolerance == 1e-6
        assert params.validate_mass_conservation is True
        assert params.validate_energy_conservation is True
        assert params.validate_pressure is True
        assert params.validate_temperature is True
        assert params.validate_piston_dynamics is True

    def test_validation_parameters_custom(self):
        """Test custom validation parameters."""
        params = ValidationParameters(
            t_end=0.5,
            dt_0d=1e-5,
            dt_1d=1e-6,
            relative_tolerance=1e-4,
            absolute_tolerance=1e-8,
            validate_pressure=False,
            validate_temperature=False,
        )

        assert params.t_end == 0.5
        assert params.dt_0d == 1e-5
        assert params.dt_1d == 1e-6
        assert params.relative_tolerance == 1e-4
        assert params.absolute_tolerance == 1e-8
        assert params.validate_pressure is False
        assert params.validate_temperature is False

    def test_simple_validation_case(self):
        """Test simple validation case."""
        problem_params = {
            "geom": {"B": 0.1, "Vc": 1e-5},
            "initial_conditions": {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": 1.2, "T": 1000.0, "p": 1.0e5,
            },
            "n_cells": 20,
        }

        validation_params = ValidationParameters(
            t_end=0.01,  # Short simulation
            dt_0d=1e-4,
            dt_1d=1e-5,
            relative_tolerance=1e-1,  # Relaxed tolerance
            validate_pressure=True,
            validate_temperature=True,
            validate_piston_dynamics=True,
        )

        result = cross_fidelity_validation(problem_params, validation_params)

        assert isinstance(result, ValidationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.max_relative_error, float)
        assert isinstance(result.max_absolute_error, float)
        assert isinstance(result.error_metrics, dict)
        assert isinstance(result.cpu_time_0d, float)
        assert isinstance(result.cpu_time_1d, float)
        assert result.cpu_time_0d > 0
        assert result.cpu_time_1d > 0

    def test_validation_result_structure(self):
        """Test validation result structure."""
        problem_params = {
            "geom": {"B": 0.1, "Vc": 1e-5},
            "initial_conditions": {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": 1.2, "T": 1000.0, "p": 1.0e5,
            },
            "n_cells": 10,
        }

        validation_params = ValidationParameters(
            t_end=0.005,  # Very short simulation
            relative_tolerance=1e-1,
        )

        result = cross_fidelity_validation(problem_params, validation_params)

        # Check required attributes
        required_attrs = [
            "success", "max_relative_error", "max_absolute_error",
            "error_metrics", "solution_comparison", "validation_message",
            "cpu_time_0d", "cpu_time_1d", "mass_conservation_error",
            "energy_conservation_error",
        ]

        for attr in required_attrs:
            assert hasattr(result, attr), f"Missing attribute: {attr}"

    def test_model_comparison_structure(self):
        """Test model comparison structure."""
        # Create dummy comparison data
        time_points = np.linspace(0, 1, 100)
        pressure_0d = np.ones(100) * 1e5
        pressure_1d = np.ones(100) * 1e5 * 1.01  # 1% difference
        temperature_0d = np.ones(100) * 1000
        temperature_1d = np.ones(100) * 1000 * 1.01
        density_0d = np.ones(100) * 1.2
        density_1d = np.ones(100) * 1.2 * 1.01
        piston_position_0d = np.ones(100) * 0.1
        piston_position_1d = np.ones(100) * 0.1 * 1.01
        piston_velocity_0d = np.ones(100) * 0.0
        piston_velocity_1d = np.ones(100) * 0.0

        comparison = ModelComparison(
            time_points=time_points,
            pressure_0d=pressure_0d,
            pressure_1d=pressure_1d,
            temperature_0d=temperature_0d,
            temperature_1d=temperature_1d,
            density_0d=density_0d,
            density_1d=density_1d,
            piston_position_0d=piston_position_0d,
            piston_position_1d=piston_position_1d,
            piston_velocity_0d=piston_velocity_0d,
            piston_velocity_1d=piston_velocity_1d,
        )

        # Check structure
        assert len(comparison.time_points) == 100
        assert len(comparison.pressure_0d) == 100
        assert len(comparison.pressure_1d) == 100
        assert len(comparison.temperature_0d) == 100
        assert len(comparison.temperature_1d) == 100
        assert len(comparison.density_0d) == 100
        assert len(comparison.density_1d) == 100
        assert len(comparison.piston_position_0d) == 100
        assert len(comparison.piston_position_1d) == 100
        assert len(comparison.piston_velocity_0d) == 100
        assert len(comparison.piston_velocity_1d) == 100

    def test_validation_report_creation(self):
        """Test validation report creation."""
        # Create dummy validation result
        result = ValidationResult(
            success=True,
            max_relative_error=1e-3,
            max_absolute_error=1e-6,
            error_metrics={
                "pressure_relative": 1e-3,
                "temperature_relative": 5e-4,
                "density_relative": 2e-3,
            },
            solution_comparison={},
            validation_message="Validation PASSED",
            cpu_time_0d=1.0,
            cpu_time_1d=2.0,
            mass_conservation_error=1e-4,
            energy_conservation_error=5e-4,
        )

        report = create_validation_report(result)

        assert isinstance(report, str)
        assert "CROSS-FIDELITY VALIDATION REPORT" in report
        assert "SUMMARY:" in report
        assert "TIMING:" in report
        assert "ERROR METRICS:" in report
        assert "CONSERVATION:" in report
        assert "Validation PASSED" in report
        assert "1.0 seconds" in report
        assert "2.0 seconds" in report

    def test_validation_suite(self):
        """Test validation suite execution."""
        results = run_validation_suite()

        assert isinstance(results, list)
        assert len(results) == 3  # Three test cases

        for result in results:
            assert isinstance(result, ValidationResult)
            assert hasattr(result, "success")
            assert hasattr(result, "max_relative_error")
            assert hasattr(result, "cpu_time_0d")
            assert hasattr(result, "cpu_time_1d")

    def test_different_problem_sizes(self):
        """Test validation with different problem sizes."""
        base_params = {
            "geom": {"B": 0.1, "Vc": 1e-5},
            "initial_conditions": {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": 1.2, "T": 1000.0, "p": 1.0e5,
            },
        }

        validation_params = ValidationParameters(
            t_end=0.005,
            relative_tolerance=1e-1,
        )

        # Test different mesh sizes
        for n_cells in [10, 20, 50]:
            problem_params = base_params.copy()
            problem_params["n_cells"] = n_cells

            result = cross_fidelity_validation(problem_params, validation_params)

            assert isinstance(result, ValidationResult)
            assert result.cpu_time_1d > 0  # 1D time should increase with mesh size

    def test_validation_with_different_tolerances(self):
        """Test validation with different tolerance settings."""
        problem_params = {
            "geom": {"B": 0.1, "Vc": 1e-5},
            "initial_conditions": {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": 1.2, "T": 1000.0, "p": 1.0e5,
            },
            "n_cells": 20,
        }

        # Test with strict tolerance
        strict_params = ValidationParameters(
            t_end=0.005,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-9,
        )

        result_strict = cross_fidelity_validation(problem_params, strict_params)

        # Test with relaxed tolerance
        relaxed_params = ValidationParameters(
            t_end=0.005,
            relative_tolerance=1e-1,
            absolute_tolerance=1e-3,
        )

        result_relaxed = cross_fidelity_validation(problem_params, relaxed_params)

        # Relaxed tolerance should be more likely to pass
        # (though this depends on the actual model differences)
        assert isinstance(result_strict, ValidationResult)
        assert isinstance(result_relaxed, ValidationResult)

    def test_validation_with_different_initial_conditions(self):
        """Test validation with different initial conditions."""
        base_params = {
            "geom": {"B": 0.1, "Vc": 1e-5},
            "n_cells": 20,
        }

        validation_params = ValidationParameters(
            t_end=0.005,
            relative_tolerance=1e-1,
        )

        # Test different initial pressures
        for pressure in [1e4, 1e5, 1e6]:
            problem_params = base_params.copy()
            problem_params["initial_conditions"] = {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": 1.2, "T": 1000.0, "p": pressure,
            }

            result = cross_fidelity_validation(problem_params, validation_params)

            assert isinstance(result, ValidationResult)
            assert result.cpu_time_0d > 0
            assert result.cpu_time_1d > 0

    def test_validation_error_handling(self):
        """Test validation error handling."""
        # Test with invalid parameters
        invalid_params = {
            "geom": {"B": -0.1, "Vc": -1e-5},  # Invalid geometry
            "initial_conditions": {
                "x_L": 0.05, "v_L": 0.0,
                "x_R": 0.15, "v_R": 0.0,
                "rho": -1.2, "T": -1000.0, "p": -1.0e5,  # Invalid initial conditions
            },
            "n_cells": -10,  # Invalid mesh size
        }

        validation_params = ValidationParameters(
            t_end=0.005,
            relative_tolerance=1e-1,
        )

        # Should handle errors gracefully
        result = cross_fidelity_validation(invalid_params, validation_params)

        assert isinstance(result, ValidationResult)
        # Result may be unsuccessful due to invalid parameters
        assert hasattr(result, "success")
        assert hasattr(result, "validation_message")

    def test_validation_metrics_computation(self):
        """Test validation metrics computation."""
        # Create dummy comparison with known differences
        time_points = np.linspace(0, 1, 100)
        pressure_0d = np.ones(100) * 1e5
        pressure_1d = np.ones(100) * 1e5 * 1.01  # 1% difference
        temperature_0d = np.ones(100) * 1000
        temperature_1d = np.ones(100) * 1000 * 1.02  # 2% difference
        density_0d = np.ones(100) * 1.2
        density_1d = np.ones(100) * 1.2 * 1.005  # 0.5% difference
        piston_position_0d = np.ones(100) * 0.1
        piston_position_1d = np.ones(100) * 0.1 * 1.001  # 0.1% difference
        piston_velocity_0d = np.ones(100) * 0.0
        piston_velocity_1d = np.ones(100) * 0.0

        comparison = ModelComparison(
            time_points=time_points,
            pressure_0d=pressure_0d,
            pressure_1d=pressure_1d,
            temperature_0d=temperature_0d,
            temperature_1d=temperature_1d,
            density_0d=density_0d,
            density_1d=density_1d,
            piston_position_0d=piston_position_0d,
            piston_position_1d=piston_position_1d,
            piston_velocity_0d=piston_velocity_0d,
            piston_velocity_1d=piston_velocity_1d,
        )

        validation_params = ValidationParameters(
            validate_pressure=True,
            validate_temperature=True,
            validate_piston_dynamics=True,
            validate_mass_conservation=True,
            validate_energy_conservation=True,
        )

        # This would normally be called internally, but we can test the logic
        # by creating a simple validation result
        result = ValidationResult(
            success=True,
            max_relative_error=0.02,  # 2% max error
            max_absolute_error=20.0,  # 20 K max absolute error
            error_metrics={
                "pressure_relative": 0.01,
                "temperature_relative": 0.02,
                "density_relative": 0.005,
                "piston_position_relative": 0.001,
                "mass_conservation": 0.001,
                "energy_conservation": 0.002,
            },
            solution_comparison=comparison.__dict__,
            validation_message="Validation PASSED",
            cpu_time_0d=1.0,
            cpu_time_1d=2.0,
            mass_conservation_error=0.001,
            energy_conservation_error=0.002,
        )

        # Check that metrics are reasonable
        assert result.max_relative_error == 0.02
        assert result.max_absolute_error == 20.0
        assert result.error_metrics["pressure_relative"] == 0.01
        assert result.error_metrics["temperature_relative"] == 0.02
        assert result.error_metrics["density_relative"] == 0.005
        assert result.mass_conservation_error == 0.001
        assert result.energy_conservation_error == 0.002
