"""
Tests for thermal efficiency adapter analysis integration.

This module tests the extraction and integration of Ipopt analysis
from the thermal efficiency adapter into the unified framework.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from campro.optimization.base import OptimizationStatus
from campro.optimization.solver_analysis import MA57ReadinessReport
from campro.optimization.thermal_efficiency_adapter import (
    ThermalEfficiencyAdapter,
    ThermalEfficiencyConfig,
)
from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)


class TestThermalEfficiencyAnalysis:
    """Test thermal efficiency adapter analysis extraction."""

    def test_thermal_efficiency_adapter_provides_analysis(self):
        """Test that thermal efficiency adapter extracts and provides Ipopt analysis."""
        # Create mock complex optimizer result with analysis
        mock_analysis = MA57ReadinessReport(
            grade="medium",
            reasons=["High iteration count (1500)"],
            suggested_action="Consider MA57 if available",
            stats={
                "success": True,
                "iter_count": 1500,
                "ls_time_ratio": 0.3,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            },
        )

        mock_complex_result = Mock()
        mock_complex_result.success = True
        mock_complex_result.iterations = 1500
        mock_complex_result.cpu_time = 2.5
        mock_complex_result.ipopt_analysis = mock_analysis
        mock_complex_result.performance_metrics = {
            "thermal_efficiency": 0.85,
            "indicated_work": 1000.0,
            "max_pressure": 8e6,
            "max_temperature": 2000.0,
            "min_piston_gap": 0.001,
        }
        mock_complex_result.solution = Mock()
        mock_complex_result.solution.data = {
            "states": {
                "x_L": [0.1, 0.2, 0.3],
                "x_R": [0.1, 0.2, 0.3],
                "v_L": [0.01, 0.02, 0.03],
                "v_R": [0.01, 0.02, 0.03],
            },
        }

        # Create adapter with mock complex optimizer
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = Mock()
        adapter.complex_optimizer.optimize_with_validation.return_value = (
            mock_complex_result
        )

        # Create motion law constraints
        from campro.optimization.motion_law import MotionLawConstraints

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
            max_velocity=100.0,
            max_acceleration=1000.0,
            max_jerk=10000.0,
        )

        # Run optimization
        result = adapter.optimize(None, constraints)

        # Verify analysis is included in metadata
        assert result.status == OptimizationStatus.CONVERGED
        assert "ipopt_analysis" in result.metadata
        assert result.metadata["ipopt_analysis"] == mock_analysis
        assert result.metadata["ipopt_analysis"].grade == "medium"

    def test_thermal_efficiency_analysis_generation_from_stats(self):
        """Test analysis generation when complex optimizer doesn't provide analysis."""
        # Create mock complex optimizer result without analysis
        mock_complex_result = Mock()
        mock_complex_result.success = True
        mock_complex_result.iterations = 2000
        mock_complex_result.cpu_time = 3.0
        mock_complex_result.ipopt_analysis = None  # No analysis provided
        mock_complex_result.primal_inf = 1e-5
        mock_complex_result.dual_inf = 1e-5
        mock_complex_result.status = "Solve_Succeeded"
        mock_complex_result.performance_metrics = {
            "thermal_efficiency": 0.82,
            "indicated_work": 950.0,
            "max_pressure": 7.5e6,
            "max_temperature": 1950.0,
            "min_piston_gap": 0.0012,
        }
        mock_complex_result.solution = Mock()
        mock_complex_result.solution.data = {
            "states": {
                "x_L": [0.1, 0.2, 0.3],
                "x_R": [0.1, 0.2, 0.3],
                "v_L": [0.01, 0.02, 0.03],
                "v_R": [0.01, 0.02, 0.03],
            },
        }

        # Create adapter with mock complex optimizer
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = Mock()
        adapter.complex_optimizer.optimize_with_validation.return_value = (
            mock_complex_result
        )

        # Create motion law constraints
        from campro.optimization.motion_law import MotionLawConstraints

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
            max_velocity=100.0,
            max_acceleration=1000.0,
            max_jerk=10000.0,
        )

        # Run optimization
        result = adapter.optimize(None, constraints)

        # Verify analysis is generated and included
        assert result.status == OptimizationStatus.CONVERGED
        assert "ipopt_analysis" in result.metadata
        analysis = result.metadata["ipopt_analysis"]
        assert isinstance(analysis, MA57ReadinessReport)
        assert analysis.stats["iter_count"] == 2000

    def test_thermal_efficiency_analysis_with_log_file(self):
        """Test analysis extraction from log file when available."""
        # Create temporary log file
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "ipopt_20240101_120000.log"
            log_content = """
            Total CPU secs in linear solver = 1.5
            Total CPU secs in IPOPT = 3.0
            Restoration phase activated
            """
            log_file.write_text(log_content)

            # Mock the IPOPT_LOG_DIR to point to our temp directory
            with patch("campro.constants.IPOPT_LOG_DIR", temp_dir):
                # Create mock complex optimizer result
                mock_complex_result = Mock()
                mock_complex_result.success = True
                mock_complex_result.iterations = 1500
                mock_complex_result.cpu_time = 3.0
                mock_complex_result.ipopt_analysis = None
                mock_complex_result.primal_inf = 1e-6
                mock_complex_result.dual_inf = 1e-6
                mock_complex_result.status = "Solve_Succeeded"
                mock_complex_result.performance_metrics = {
                    "thermal_efficiency": 0.88,
                    "indicated_work": 1100.0,
                    "max_pressure": 9e6,
                    "max_temperature": 2100.0,
                    "min_piston_gap": 0.0008,
                }
                mock_complex_result.solution = Mock()
                mock_complex_result.solution.data = {
                    "states": {
                        "x_L": [0.1, 0.2, 0.3],
                        "x_R": [0.1, 0.2, 0.3],
                        "v_L": [0.01, 0.02, 0.03],
                        "v_R": [0.01, 0.02, 0.03],
                    },
                }

                # Create adapter
                config = ThermalEfficiencyConfig()
                adapter = ThermalEfficiencyAdapter(config)
                adapter.complex_optimizer = Mock()
                adapter.complex_optimizer.optimize_with_validation.return_value = (
                    mock_complex_result
                )

                # Create motion law constraints
                from campro.optimization.motion_law import MotionLawConstraints

                constraints = MotionLawConstraints(
                    stroke=20.0,
                    upstroke_duration_percent=60.0,
                    zero_accel_duration_percent=0.0,
                    max_velocity=100.0,
                    max_acceleration=1000.0,
                    max_jerk=10000.0,
                )

                # Run optimization
                result = adapter.optimize(None, constraints)

                # Verify analysis includes log file data
                assert result.status == OptimizationStatus.CONVERGED
                assert "ipopt_analysis" in result.metadata
                analysis = result.metadata["ipopt_analysis"]
                assert analysis.stats["ls_time_ratio"] == 0.5  # 1.5/3.0
                assert "Restoration phase encountered" in analysis.reasons

    def test_thermal_efficiency_analysis_in_unified_framework(self):
        """Test that unified framework stores primary phase analysis."""
        # Create mock analysis
        mock_analysis = MA57ReadinessReport(
            grade="low",
            reasons=["No adverse indicators detected"],
            suggested_action="Stick with MA27; no strong indicators for MA57.",
            stats={
                "success": True,
                "iter_count": 500,
                "ls_time_ratio": 0.2,
                "primal_inf": 1e-8,
                "dual_inf": 1e-8,
            },
        )

        # Create mock adapter result
        mock_adapter_result = Mock()
        mock_adapter_result.convergence_status = "converged"
        mock_adapter_result.objective_value = 0.15
        mock_adapter_result.iterations = 500
        mock_adapter_result.solve_time = 1.5
        mock_adapter_result.to_dict.return_value = {
            "cam_angle": [0, 1, 2],
            "position": [0, 10, 20],
            "velocity": [0, 5, 0],
            "acceleration": [0, 0, 0],
            "jerk": [0, 0, 0],
            "ipopt_analysis": mock_analysis,
        }

        # Create unified framework with thermal efficiency enabled
        settings = UnifiedOptimizationSettings(
            use_thermal_efficiency=True,
            enable_ipopt_analysis=True,
        )
        framework = UnifiedOptimizationFramework(settings=settings)

        # Mock the thermal efficiency adapter
        with patch(
            "campro.optimization.thermal_efficiency_adapter.ThermalEfficiencyAdapter",
        ) as mock_adapter_class:
            mock_adapter = Mock()
            mock_adapter.solve_motion_law.return_value = mock_adapter_result
            mock_adapter_class.return_value = mock_adapter

            # Test primary optimization directly
            result = framework._optimize_primary()

            # Verify analysis is stored in framework data
            assert result.metadata["ipopt_analysis"] == mock_analysis
            assert framework.data.primary_ipopt_analysis == mock_analysis
            assert framework.data.primary_ipopt_analysis.grade == "low"

    def test_thermal_efficiency_adapter_log_file_path_handling(self):
        """Test log file path handling when no log files exist."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)

        # Test with non-existent directory
        with patch("campro.constants.IPOPT_LOG_DIR", "/non/existent/path"):
            log_path = adapter._get_log_file_path()
            assert log_path is None

        # Test with empty directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("campro.constants.IPOPT_LOG_DIR", temp_dir):
                log_path = adapter._get_log_file_path()
                assert log_path is None

    def test_thermal_efficiency_adapter_fallback_motion_law(self):
        """Test fallback motion law generation when complex optimizer fails."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        adapter.complex_optimizer = None  # Simulate unavailable complex optimizer

        # Create motion law constraints
        from campro.optimization.motion_law import MotionLawConstraints

        constraints = MotionLawConstraints(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0,
            max_velocity=100.0,
            max_acceleration=1000.0,
            max_jerk=10000.0,
        )

        # Should raise RuntimeError when complex optimizer is unavailable
        with pytest.raises(RuntimeError, match="Complex gas optimizer.*not available"):
            adapter.optimize(None, constraints)
