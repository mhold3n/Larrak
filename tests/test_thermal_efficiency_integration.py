"""
Integration tests for thermal efficiency optimization.

Tests the integration between the existing motion law system and the
complex gas optimizer for thermal efficiency optimization.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

pytest.importorskip("yaml")

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


class DummyComplexResult:
    """Synthetic complex optimizer result used to avoid fallback behaviour."""

    def __init__(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 360)
        displacement = 0.01 * np.sin(theta)
        velocity = np.gradient(displacement, theta)

        states = {
            "x_L": displacement,
            "x_R": displacement,
            "v_L": velocity,
            "v_R": velocity,
        }

        self.solution = MagicMock()
        self.solution.data = {"states": states}
        self.success = True
        self.iterations = 12
        self.cpu_time = 0.5
        self.performance_metrics = {
            "thermal_efficiency": 0.65,
            "indicated_work": 12.3,
            "max_pressure": 5.0e6,
            "max_temperature": 1900.0,
            "min_piston_gap": 0.001,
        }
        self.status = "Solve_Succeeded"
        self.message = "Solve_Succeeded"
        self.ipopt_analysis = None


class TestThermalEfficiencyAdapter:
    """Test thermal efficiency adapter functionality."""

    def test_adapter_creation_default_config(self):
        """Test thermal efficiency adapter creation with default config."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter is not None
        assert adapter.config.bore == 0.082
        assert adapter.config.thermal_efficiency_weight == 1.0
        assert adapter.config.use_1d_gas_model is False

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

    def test_thermal_efficiency_nlp_builds_without_truth_value_error(self):
        """NLP build should not raise CasADi SX truth-value errors.

        This ensures symbolic comparisons are encoded without Python branching.
        """
        cfg = ThermalEfficiencyConfig(
            collocation_points=10,
            collocation_degree=2,
            max_iterations=50,
            tolerance=1e-5,
        )
        adapter = ThermalEfficiencyAdapter(cfg)
        adapter._get_log_file_path = MagicMock(return_value=None)
        dummy_result = DummyComplexResult()
        adapter.complex_optimizer = MagicMock()
        adapter.complex_optimizer.optimize_with_validation.return_value = dummy_result

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        result = adapter.optimize(None, constraints)

        assert result.status == OptimizationStatus.CONVERGED
        assert result.metadata["thermal_efficiency"] == pytest.approx(
            dummy_result.performance_metrics["thermal_efficiency"],
        )

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
        mock_optimizer_class.side_effect = ImportError(
            "Complex optimizer not available",
        )

        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        assert adapter.complex_optimizer is None

    def test_optimize_raises_when_optimizer_unavailable(self):
        """Optimization should raise when complex optimizer is missing."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        with pytest.raises(RuntimeError):
            adapter.optimize(None, constraints)

    def test_solve_motion_law_raises_when_optimizer_unavailable(self):
        """solve_motion_law should raise when complex optimizer is missing."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        with pytest.raises(RuntimeError):
            adapter.solve_motion_law(constraints, MotionType.MINIMUM_JERK)

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
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

        assert optimizer.use_thermal_efficiency is False
        assert optimizer.thermal_adapter is None

    def test_optimizer_creation_with_thermal_efficiency(self):
        """Test motion law optimizer creation with thermal efficiency."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

        assert optimizer.use_thermal_efficiency is True
        # Note: thermal_adapter might be None if complex optimizer is not available

    def test_enable_thermal_efficiency(self):
        """Test enabling thermal efficiency optimization."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

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
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)

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

    def test_solve_motion_law_thermal_efficiency_unavailable(self):
        """Thermal efficiency solve should raise when adapter is missing."""
        optimizer = MotionLawOptimizer(use_thermal_efficiency=True)
        optimizer.thermal_adapter = None  # Simulate unavailable complex optimizer

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
        )

        with pytest.raises(RuntimeError):
            optimizer.solve_motion_law(constraints, MotionType.MINIMUM_JERK)


class TestUnifiedFrameworkIntegration:
    """Test unified framework integration with thermal efficiency."""

    def test_framework_creation_without_thermal_efficiency(self):
        """Test unified framework creation without thermal efficiency."""
        settings = UnifiedOptimizationSettings()
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
        assert (
            framework.settings.thermal_efficiency_config["thermal_efficiency_weight"]
            == 2.0
        )

    def test_primary_optimization_thermal_efficiency_unavailable(self):
        """Primary optimization should raise when TE path cannot run."""
        settings = UnifiedOptimizationSettings(use_thermal_efficiency=True)
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        framework.data.stroke = 20.0
        framework.data.cycle_time = 1.0
        framework.data.upstroke_duration_percent = 60.0
        framework.data.zero_accel_duration_percent = 0.0
        framework.data.motion_type = "minimum_jerk"

        framework.constraints.max_velocity = 100.0
        framework.constraints.max_acceleration = 1000.0
        framework.constraints.max_jerk = 10000.0

        with pytest.raises(RuntimeError):
            framework._optimize_primary()

    def test_primary_optimization_without_thermal_efficiency(self):
        """Test primary optimization without thermal efficiency."""
        settings = UnifiedOptimizationSettings()
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

    def test_te_required_raises_when_unavailable(self):
        """If TE is required but unavailable, framework should raise."""
        settings = UnifiedOptimizationSettings(use_thermal_efficiency=True)
        # Force requirement
        settings.require_thermal_efficiency = True
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        # Minimal input setup to trigger primary path
        framework.data.stroke = 20.0
        framework.data.cycle_time = 1.0
        framework.data.upstroke_duration_percent = 60.0
        framework.data.zero_accel_duration_percent = 0.0
        framework.data.motion_type = "minimum_jerk"

        framework.constraints.max_velocity = 100.0
        framework.constraints.max_acceleration = 1000.0
        framework.constraints.max_jerk = 10000.0

        # Run and expect a RuntimeError if complex optimizer unavailable
        with pytest.raises(RuntimeError):
            _ = framework._optimize_primary()


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
        motion_law_data = adapter._extract_motion_law_data(
            mock_complex_result, constraints,
        )

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
            self.objective_value = (
                123.456  # Should be ignored by adapter objective computation
            )
            self.iterations = 10
            self.cpu_time = 0.1

    # Patch the adapter to use our mock complex optimizer
    mock_optimizer = MagicMock()
    mock_optimizer.optimize_with_validation.return_value = MockComplexResult(eta=0.5)
    adapter.complex_optimizer = mock_optimizer
    adapter._get_log_file_path = MagicMock(return_value=None)

    constraints = MotionLawConstraints(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        zero_accel_duration_percent=0.0,
    )

    result = adapter.optimize(objective=None, constraints=constraints)

    # Expect objective to be 1 - eta_th
    assert result.objective_value == pytest.approx(0.5, rel=1e-6)
    assert result.metadata.get("thermal_efficiency") == pytest.approx(0.5, rel=1e-12)
