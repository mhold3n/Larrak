"""
Integration tests for thermal efficiency optimization.

Tests the integration between the existing motion law system and the
complex gas optimizer for thermal efficiency optimization.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from campro.logging import get_logger
from campro.optimization.base import OptimizationStatus
from campro.optimization.motion_law import MotionLawConstraints, MotionType
from campro.optimization.motion_law_optimizer import MotionLawOptimizer
from campro.optimization.thermal_efficiency_adapter import (
    ThermalEfficiencyAdapter,
    ThermalEfficiencyConfig,
    get_default_thermal_efficiency_config,
    validate_thermal_efficiency_config,
)
from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)

log = get_logger(__name__)


class TestThermalEfficiencyAdapter:
    """Test thermal efficiency adapter functionality."""

    def test_adapter_creation_default_config(self):
        """Test thermal efficiency adapter creation with default config."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter is not None
        assert adapter.config.bore == 0.082
        assert adapter.config.thermal_efficiency_weight == 1.0
        assert adapter.config.use_1d_gas_model is True

    def test_adapter_creation_custom_config(self):
        """Test thermal efficiency adapter creation with custom config."""
        config = ThermalEfficiencyConfig(
            bore=0.1,
            stroke=0.2,
            thermal_efficiency_weight=2.0,
            use_1d_gas_model=False,
        )
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter.config.bore == 0.1
        assert adapter.config.stroke == 0.2
        assert adapter.config.thermal_efficiency_weight == 2.0
        assert adapter.config.use_1d_gas_model is False

    def test_adapter_creation_no_config(self):
        """Test thermal efficiency adapter creation without config."""
        adapter = ThermalEfficiencyAdapter()

        assert adapter is not None
        assert adapter.config is not None
        assert isinstance(adapter.config, ThermalEfficiencyConfig)

    @patch("campro.optimization.thermal_efficiency_adapter.ComplexMotionLawOptimizer")
    def test_complex_optimizer_setup_success(self, mock_optimizer_class):
        """Test successful complex optimizer setup."""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer

        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter.complex_optimizer is not None
        mock_optimizer_class.assert_called_once()

    @patch("campro.optimization.thermal_efficiency_adapter.ComplexMotionLawOptimizer")
    def test_complex_optimizer_setup_failure(self, mock_optimizer_class):
        """Test complex optimizer setup failure."""
        mock_optimizer_class.side_effect = ImportError("Complex optimizer not available")

        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter.complex_optimizer is None

    def test_fallback_optimization(self):
        """Test fallback optimization when complex optimizer is not available."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = adapter.optimize(None, constraints)

        assert result is not None
        assert result.status == OptimizationStatus.CONVERGED
        assert result.solution is not None
        assert "theta" in result.solution
        assert "x" in result.solution
        assert "v" in result.solution
        assert "a" in result.solution
        assert "j" in result.solution

    def test_solve_motion_law_fallback(self):
        """Test solve_motion_law with fallback optimization."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = adapter.solve_motion_law(constraints, MotionType.MINIMUM_JERK)

        assert result is not None
        assert result.convergence_status == "converged"
        assert len(result.theta) == 360
        assert len(result.x) == 360
        assert len(result.v) == 360
        assert len(result.a) == 360
        assert len(result.j) == 360
        assert result.constraints == constraints
        assert result.motion_type == MotionType.MINIMUM_JERK

    def test_configure_adapter(self):
        """Test adapter configuration."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        # Test configuration update
        adapter.configure(
            thermal_efficiency_weight=2.0,
            use_1d_gas_model=False,
            n_cells=100,
        )

        assert adapter.config.thermal_efficiency_weight == 2.0
        assert adapter.config.use_1d_gas_model is False
        assert adapter.config.n_cells == 100


class TestMotionLawOptimizerIntegration:
    """Test motion law optimizer integration with thermal efficiency."""

    def test_optimizer_creation_without_thermal_efficiency(self):
        """Test motion law optimizer creation without thermal efficiency."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=False)

        assert optimizer.use_thermal_efficiency is False
        assert optimizer.thermal_adapter is None

    def test_optimizer_creation_with_thermal_efficiency(self):
        """Test motion law optimizer creation with thermal efficiency."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

        assert optimizer.use_thermal_efficiency is True
        # Note: thermal_adapter might be None if complex optimizer is not available

    def test_enable_thermal_efficiency(self):
        """Test enabling thermal efficiency optimization."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=False)

        optimizer.enable_thermal_efficiency()

        assert optimizer.use_thermal_efficiency is True
        # Note: thermal_adapter might be None if complex optimizer is not available

    def test_disable_thermal_efficiency(self):
        """Test disabling thermal efficiency optimization."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

        optimizer.disable_thermal_efficiency()

        assert optimizer.use_thermal_efficiency is False
        assert optimizer.thermal_adapter is None

    def test_configure_with_thermal_efficiency(self):
        """Test configuring optimizer with thermal efficiency settings."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=False)

        optimizer.configure(
            use_thermal_efficiency=True,
            n_points=50,
            tolerance=1e-8,
        )

        assert optimizer.use_thermal_efficiency is True
        assert optimizer.n_points == 50
        assert optimizer.tolerance == 1e-8

    def test_solve_motion_law_simple_optimization(self):
        """Test solve_motion_law with simple optimization."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=False)

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_JERK)

        assert result is not None
        assert result.convergence_status == "converged"
        assert len(result.theta) == 360
        assert len(result.x) == 360
        assert len(result.v) == 360
        assert len(result.a) == 360
        assert len(result.j) == 360

    def test_solve_motion_law_thermal_efficiency_fallback(self):
        """Test solve_motion_law with thermal efficiency fallback."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)
        optimizer.thermal_adapter = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_JERK)

        assert result is not None
        assert result.convergence_status == "converged"
        assert len(result.theta) == 360
        assert len(result.x) == 360
        assert len(result.v) == 360
        assert len(result.a) == 360
        assert len(result.j) == 360


class TestUnifiedFrameworkIntegration:
    """Test unified framework integration with thermal efficiency."""

    def test_framework_creation_without_thermal_efficiency(self):
        """Test unified framework creation without thermal efficiency."""
        settings = UnifiedOptimizationSettings(use_thermal_efficiency=False)
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        assert framework.settings.use_thermal_efficiency is False

    def test_framework_creation_with_thermal_efficiency(self):
        """Test unified framework creation with thermal efficiency."""
        settings = UnifiedOptimizationSettings(
            use_thermal_efficiency=True,
            thermal_efficiency_config={
                "thermal_efficiency_weight": 2.0,
                "use_1d_gas_model": True,
                "n_cells": 50,
            },
        )
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        assert framework.settings.use_thermal_efficiency is True
        assert framework.settings.thermal_efficiency_config is not None
        assert framework.settings.thermal_efficiency_config["thermal_efficiency_weight"] == 2.0

    def test_primary_optimization_with_thermal_efficiency_fallback(self):
        """Test primary optimization with thermal efficiency fallback."""
        settings = UnifiedOptimizationSettings(use_thermal_efficiency=True)
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        # Set up test data
        framework.data.stroke = 20.0
        framework.data.cycle_time = 1.0
        framework.data.upstroke_duration_percent = 60.0
        framework.data.zero_accel_duration_percent = 0.0
        framework.data.motion_type = "minimum_jerk"

        # Mock constraints
        framework.constraints.max_velocity = 100.0
        framework.constraints.max_acceleration = 1000.0
        framework.constraints.max_jerk = 10000.0

        # Run primary optimization (will use fallback since complex optimizer not available)
        result = framework._optimize_primary()

        assert result is not None
        assert result.status == OptimizationStatus.CONVERGED
        assert result.solution is not None

    def test_primary_optimization_without_thermal_efficiency(self):
        """Test primary optimization without thermal efficiency."""
        settings = UnifiedOptimizationSettings(use_thermal_efficiency=False)
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        # Set up test data
        framework.data.stroke = 20.0
        framework.data.cycle_time = 1.0
        framework.data.upstroke_duration_percent = 60.0
        framework.data.zero_accel_duration_percent = 0.0
        framework.data.motion_type = "minimum_jerk"

        # Mock constraints
        framework.constraints.max_velocity = 100.0
        framework.constraints.max_acceleration = 1000.0
        framework.constraints.max_jerk = 10000.0

        # Run primary optimization
        result = framework._optimize_primary()

        assert result is not None
        assert result.status == OptimizationStatus.CONVERGED
        assert result.solution is not None


class TestConfigurationManagement:
    """Test configuration management functionality."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_thermal_efficiency_config()

        assert config is not None
        assert isinstance(config, ThermalEfficiencyConfig)
        assert config.bore == 0.082
        assert config.thermal_efficiency_weight == 1.0

    def test_config_validation_valid(self):
        """Test configuration validation with valid config."""
        config = ThermalEfficiencyConfig(
            bore=0.1,
            stroke=0.2,
            compression_ratio=12.0,
            gamma=1.4,
            R=287.0,
            cp=1005.0,
            collocation_points=20,
            max_iterations=500,
            thermal_efficiency_weight=1.0,
        )

        assert validate_thermal_efficiency_config(config) is True

    def test_config_validation_invalid_geometry(self):
        """Test configuration validation with invalid geometry."""
        config = ThermalEfficiencyConfig(
            bore=-0.1,  # Invalid: negative bore
            stroke=0.2,
            compression_ratio=12.0,
        )

        assert validate_thermal_efficiency_config(config) is False

    def test_config_validation_invalid_thermodynamics(self):
        """Test configuration validation with invalid thermodynamics."""
        config = ThermalEfficiencyConfig(
            bore=0.1,
            stroke=0.2,
            gamma=0.5,  # Invalid: gamma <= 1.0
            R=287.0,
            cp=1005.0,
        )

        assert validate_thermal_efficiency_config(config) is False

    def test_config_validation_invalid_optimization(self):
        """Test configuration validation with invalid optimization parameters."""
        config = ThermalEfficiencyConfig(
            bore=0.1,
            stroke=0.2,
            collocation_points=2,  # Invalid: too few points
            max_iterations=50,  # Invalid: too few iterations
        )

        assert validate_thermal_efficiency_config(config) is False

    def test_config_validation_invalid_weights(self):
        """Test configuration validation with invalid weights."""
        config = ThermalEfficiencyConfig(
            bore=0.1,
            stroke=0.2,
            thermal_efficiency_weight=-1.0,  # Invalid: negative weight
        )

        assert validate_thermal_efficiency_config(config) is False


class TestDataConversion:
    """Test data conversion between systems."""

    def test_motion_law_data_extraction(self):
        """Test motion law data extraction from complex result."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        # Create mock complex result
        mock_complex_result = Mock()
        mock_complex_result.performance_metrics = {
            "thermal_efficiency": 0.45,
            "indicated_work": 1500.0,
            "max_pressure": 8e6,
            "max_temperature": 2000.0,
            "min_piston_gap": 0.001,
        }

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        # Test data extraction
        motion_law_data = adapter._extract_motion_law_data(mock_complex_result, constraints)

        assert motion_law_data is not None
        assert "theta" in motion_law_data
        assert "x" in motion_law_data
        assert "v" in motion_law_data
        assert "a" in motion_law_data
        assert "j" in motion_law_data
        assert "constraints" in motion_law_data
        assert "optimization_type" in motion_law_data
        assert "thermal_efficiency" in motion_law_data

        assert len(motion_law_data["theta"]) == 360
        assert len(motion_law_data["x"]) == 360
        assert len(motion_law_data["v"]) == 360
        assert len(motion_law_data["a"]) == 360
        assert len(motion_law_data["j"]) == 360
        assert motion_law_data["optimization_type"] == "thermal_efficiency"
        assert motion_law_data["thermal_efficiency"] == 0.45

    def test_fallback_motion_law_generation(self):
        """Test fallback motion law generation."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        # Test fallback motion law generation
        x, v, a, j = adapter._generate_fallback_motion_law(constraints)

        assert len(x) == 360
        assert len(v) == 360
        assert len(a) == 360
        assert len(j) == 360

        # Check that motion law is reasonable
        assert np.max(x) > 0  # Should have positive displacement
        assert np.min(x) >= 0  # Should not have negative displacement


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_adapter_creation_with_invalid_config(self):
        """Test adapter creation with invalid configuration."""
        # This should not raise an exception, but should handle gracefully
        config = ThermalEfficiencyConfig()
        config.bore = -0.1  # Invalid configuration

        adapter = ThermalEfficiencyAdapter(config)
        assert adapter is not None
        assert adapter.config.bore == -0.1  # Config is stored as-is

    def test_optimization_with_invalid_constraints(self):
        """Test optimization with invalid constraints."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Use fallback

        # Create invalid constraints
        constraints = MotionLawConstraints(
            stroke=-10.0,  # Invalid: negative stroke
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        # This should handle gracefully and return a result
        result = adapter.optimize(None, constraints)

        assert result is not None
        assert result.status in [OptimizationStatus.CONVERGED, OptimizationStatus.FAILED]

    def test_optimization_with_missing_constraints(self):
        """Test optimization with missing constraint attributes."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Use fallback

        # Create constraints with missing attributes
        class IncompleteConstraints:
            stroke = 20.0
            # Missing upstroke_duration_percent

        constraints = IncompleteConstraints()

        # This should handle gracefully
        try:
            result = adapter.optimize(None, constraints)
            # If it doesn't raise an exception, check the result
            assert result is not None
        except AttributeError:
            # This is also acceptable behavior
            pass

    def test_complex_optimizer_failure_handling(self):
        """Test handling of complex optimizer failures."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        # Mock complex optimizer that fails
        mock_optimizer = Mock()
        mock_optimizer.optimize_with_validation.side_effect = Exception("Complex optimizer failed")
        adapter.complex_optimizer = mock_optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        # This should handle the failure gracefully
        result = adapter.optimize(None, constraints)

        assert result is not None
        assert result.status == OptimizationStatus.FAILED
        assert "error" in result.metadata


class TestPerformanceAndConvergence:
    """Test performance and convergence characteristics."""

    def test_optimization_timing(self):
        """Test optimization timing."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Use fallback for consistent timing

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        import time
        start_time = time.time()
        result = adapter.optimize(None, constraints)
        end_time = time.time()

        assert result is not None
        assert result.solve_time >= 0
        assert (end_time - start_time) >= 0

    def test_convergence_consistency(self):
        """Test convergence consistency across multiple runs."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Use fallback for consistency

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        # Run optimization multiple times
        results = []
        for _ in range(3):
            result = adapter.optimize(None, constraints)
            results.append(result)

        # All results should be successful
        for result in results:
            assert result.status == OptimizationStatus.CONVERGED
            assert result.solution is not None

    def test_memory_usage(self):
        """Test memory usage during optimization."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Use fallback

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = adapter.optimize(None, constraints)
        final_memory = process.memory_info().rss

        assert result is not None
        # Memory usage should be reasonable (less than 100MB increase)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100 * 1024 * 1024  # 100MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_objective_from_thermal_efficiency():
    """Objective should reflect thermal efficiency when complex optimizer succeeds."""
    config = ThermalEfficiencyConfig()
    adapter = ThermalEfficiencyAdapter(config)

    # Build a mock complex result with required attributes
    class MockSolution:
        def __init__(self):
            self.data = {
                "states": {
                    "x_L": [0.0, 0.01, 0.02],
                    "x_R": [0.1, 0.11, 0.12],
                    "v_L": [0.0, 0.1, 0.2],
                    "v_R": [0.0, -0.1, -0.2],
                },
            }

    class MockComplexResult:
        def __init__(self, eta):
            self.success = True
            self.solution = MockSolution()
            self.performance_metrics = {
                "thermal_efficiency": eta,
                "indicated_work": 1000.0,
                "max_pressure": 5e6,
                "max_temperature": 1500.0,
                "min_piston_gap": 0.002,
            }
            self.objective_value = 123.456  # Should be ignored by adapter objective computation
            self.iterations = 10
            self.cpu_time = 0.1

    # Patch the adapter to use our mock complex optimizer
    mock_optimizer = MagicMock()
    mock_optimizer.optimize_with_validation.return_value = MockComplexResult(eta=0.5)
    adapter.complex_optimizer = mock_optimizer

    constraints = MotionLawConstraints(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        zero_accel_duration_percent=0.0,
    )

    result = adapter.optimize(objective=None, constraints=constraints)

    # Expect objective to be 1 - eta_th
    assert result.objective_value == pytest.approx(0.5, rel=1e-6)
    assert result.metadata.get("thermal_efficiency") == pytest.approx(0.5, rel=1e-12)
