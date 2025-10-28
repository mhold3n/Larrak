"""
Test suite for validation statistics collection framework.

This module tests the validation statistics collection and analysis
capabilities for CasADi physics validation mode.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from campro.optimization.validation_statistics import (
    ValidationMetrics,
    ValidationStatisticsCollector,
)


class TestValidationStatisticsCollection:
    """Test validation statistics collection framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.collector = ValidationStatisticsCollector(output_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validation_metrics_creation(self):
        """Test creation of validation metrics."""
        metrics = ValidationMetrics(
            problem_id="test_001",
            timestamp=datetime.now().isoformat(),
            problem_type="crank_center",
            stroke=20.0,
            cycle_time=1.0,
            crank_radius_range=(20.0, 100.0),
            rod_length_range=(100.0, 300.0),
            python_torque_avg=150.0,
            casadi_torque_avg=150.1,
            python_torque_ripple=5.0,
            casadi_torque_ripple=5.05,
            python_side_load_penalty=100.0,
            casadi_side_load_penalty=100.2,
            python_litvin_objective=0.0,
            casadi_litvin_objective=0.01,
            python_litvin_closure=0.0,
            casadi_litvin_closure=0.005,
            torque_avg_diff=0.1,
            torque_ripple_diff=0.05,
            side_load_penalty_diff=0.2,
            litvin_objective_diff=0.01,
            litvin_closure_diff=0.005,
            torque_avg_rel_diff=0.00067,
            torque_ripple_rel_diff=0.01,
            side_load_penalty_rel_diff=0.002,
            litvin_objective_rel_diff=0.0,  # Division by zero handled
            litvin_closure_rel_diff=0.0,  # Division by zero handled
            python_evaluation_time_ms=2.5,
            casadi_evaluation_time_ms=0.8,
            python_gradient_time_ms=5.0,
            casadi_gradient_time_ms=1.2,
            evaluation_speedup=3.125,
            gradient_speedup=4.167,
            python_iterations=50,
            casadi_iterations=45,
            python_converged=True,
            casadi_converged=True,
            within_tolerance=True,
            tolerance_threshold=1e-4,
            casadi_version="3.6.0",
            python_version="3.11.5",
            hardware_info={"cpu": "test", "memory": "8GB"},
        )

        assert metrics.problem_id == "test_001"
        assert metrics.within_tolerance is True
        assert metrics.evaluation_speedup > 1.0

    def test_collector_initialization(self):
        """Test validation statistics collector initialization."""
        assert self.collector.output_dir == self.temp_dir
        assert len(self.collector.metrics) == 0
        assert self.collector.current_session_id is not None

    def test_add_validation_metrics(self):
        """Test adding validation metrics to collector."""
        metrics = self._create_sample_metrics("test_001")

        self.collector.add_validation_metrics(metrics)

        assert len(self.collector.metrics) == 1
        assert self.collector.metrics[0].problem_id == "test_001"

    def test_compute_statistics_single_metric(self):
        """Test computing statistics from a single validation metric."""
        metrics = self._create_sample_metrics("test_001")
        self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()

        assert statistics.total_validations == 1
        assert statistics.successful_validations == 1
        assert statistics.within_tolerance_count == 1
        assert statistics.tolerance_success_rate == 1.0

    def test_compute_statistics_multiple_metrics(self):
        """Test computing statistics from multiple validation metrics."""
        # Add multiple metrics with different outcomes
        metrics1 = self._create_sample_metrics("test_001", within_tolerance=True)
        metrics2 = self._create_sample_metrics("test_002", within_tolerance=True)
        metrics3 = self._create_sample_metrics("test_003", within_tolerance=False)

        self.collector.add_validation_metrics(metrics1)
        self.collector.add_validation_metrics(metrics2)
        self.collector.add_validation_metrics(metrics3)

        statistics = self.collector.compute_statistics()

        assert statistics.total_validations == 3
        assert statistics.within_tolerance_count == 2
        assert statistics.tolerance_exceeded_count == 1
        assert statistics.tolerance_success_rate == 2 / 3

    def test_statistics_parity_analysis(self):
        """Test parity analysis in statistics computation."""
        # Create metrics with known differences
        metrics1 = self._create_sample_metrics("test_001", torque_avg_diff=0.1)
        metrics2 = self._create_sample_metrics("test_002", torque_avg_diff=0.2)
        metrics3 = self._create_sample_metrics("test_003", torque_avg_diff=0.05)

        self.collector.add_validation_metrics(metrics1)
        self.collector.add_validation_metrics(metrics2)
        self.collector.add_validation_metrics(metrics3)

        statistics = self.collector.compute_statistics()

        # Check torque average statistics
        assert statistics.torque_avg_stats["min"] == 0.05
        assert statistics.torque_avg_stats["max"] == 0.2
        assert abs(statistics.torque_avg_stats["mean"] - 0.1167) < 0.01
        assert statistics.torque_avg_stats["std"] > 0

    def test_statistics_performance_analysis(self):
        """Test performance analysis in statistics computation."""
        # Create metrics with different speedups
        metrics1 = self._create_sample_metrics(
            "test_001", evaluation_speedup=2.0, gradient_speedup=3.0,
        )
        metrics2 = self._create_sample_metrics(
            "test_002", evaluation_speedup=4.0, gradient_speedup=5.0,
        )
        metrics3 = self._create_sample_metrics(
            "test_003", evaluation_speedup=1.5, gradient_speedup=2.0,
        )

        self.collector.add_validation_metrics(metrics1)
        self.collector.add_validation_metrics(metrics2)
        self.collector.add_validation_metrics(metrics3)

        statistics = self.collector.compute_statistics()

        # Check performance statistics
        assert statistics.evaluation_speedup_stats["min"] == 1.5
        assert statistics.evaluation_speedup_stats["max"] == 4.0
        assert abs(statistics.evaluation_speedup_stats["mean"] - 2.5) < 0.1

        assert statistics.gradient_speedup_stats["min"] == 2.0
        assert statistics.gradient_speedup_stats["max"] == 5.0
        assert abs(statistics.gradient_speedup_stats["mean"] - 3.33) < 0.1

    def test_statistics_problem_type_breakdown(self):
        """Test problem type breakdown in statistics computation."""
        # Create metrics with different problem types
        metrics1 = self._create_sample_metrics(
            "test_001", problem_type="crank_center", within_tolerance=True,
        )
        metrics2 = self._create_sample_metrics(
            "test_002", problem_type="crank_center", within_tolerance=True,
        )
        metrics3 = self._create_sample_metrics(
            "test_003", problem_type="litvin", within_tolerance=False,
        )
        metrics4 = self._create_sample_metrics(
            "test_004", problem_type="litvin", within_tolerance=True,
        )

        self.collector.add_validation_metrics(metrics1)
        self.collector.add_validation_metrics(metrics2)
        self.collector.add_validation_metrics(metrics3)
        self.collector.add_validation_metrics(metrics4)

        statistics = self.collector.compute_statistics()

        # Check problem type counts
        assert statistics.problem_type_counts["crank_center"] == 2
        assert statistics.problem_type_counts["litvin"] == 2

        # Check success rates
        assert statistics.problem_type_success_rates["crank_center"] == 1.0  # 2/2
        assert statistics.problem_type_success_rates["litvin"] == 0.5  # 1/2

    def test_save_and_load_statistics(self):
        """Test saving and loading statistics."""
        metrics = self._create_sample_metrics("test_001")
        self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()
        filepath = self.collector.save_statistics(statistics)

        # Verify file was created
        assert filepath.exists()

        # Verify file content
        import json

        with open(filepath) as f:
            data = json.load(f)

        assert data["total_validations"] == 1
        assert data["tolerance_success_rate"] == 1.0

    def test_generate_report(self):
        """Test generating human-readable validation report."""
        metrics = self._create_sample_metrics("test_001")
        self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()
        report = self.collector.generate_report(statistics)

        # Check that report contains expected sections
        assert "# CasADi Physics Validation Report" in report
        assert "## Summary" in report
        assert "## Parity Analysis" in report
        assert "## Performance Analysis" in report
        assert "## Recommendation" in report

        # Check that recommendation is positive for good results
        assert "✅ **RECOMMEND ENABLING**" in report

    def test_generate_report_conditional_recommendation(self):
        """Test generating report with conditional recommendation."""
        # Create metrics with 90% success rate (conditional)
        for i in range(10):
            within_tolerance = i < 9  # 9 out of 10 within tolerance
            metrics = self._create_sample_metrics(
                f"test_{i:03d}", within_tolerance=within_tolerance,
            )
            self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()
        report = self.collector.generate_report(statistics)

        assert "⚠️ **CONDITIONAL ENABLING**" in report

    def test_generate_report_negative_recommendation(self):
        """Test generating report with negative recommendation."""
        # Create metrics with 80% success rate (below threshold)
        for i in range(10):
            within_tolerance = i < 8  # 8 out of 10 within tolerance
            metrics = self._create_sample_metrics(
                f"test_{i:03d}", within_tolerance=within_tolerance,
            )
            self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()
        report = self.collector.generate_report(statistics)

        assert "❌ **DO NOT ENABLE**" in report

    def test_save_report(self):
        """Test saving validation report to file."""
        metrics = self._create_sample_metrics("test_001")
        self.collector.add_validation_metrics(metrics)

        statistics = self.collector.compute_statistics()
        report = self.collector.generate_report(statistics)
        filepath = self.collector.save_report(report)

        # Verify file was created
        assert filepath.exists()
        assert filepath.suffix == ".md"

        # Verify file content
        with open(filepath) as f:
            content = f.read()

        assert "# CasADi Physics Validation Report" in content

    def test_clear_metrics(self):
        """Test clearing collected metrics."""
        metrics = self._create_sample_metrics("test_001")
        self.collector.add_validation_metrics(metrics)

        assert len(self.collector.metrics) == 1

        self.collector.clear_metrics()

        assert len(self.collector.metrics) == 0

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        # Empty collector
        summary = self.collector.get_metrics_summary()
        assert summary["total_metrics"] == 0

        # Add some metrics
        metrics1 = self._create_sample_metrics("test_001", problem_type="crank_center")
        metrics2 = self._create_sample_metrics("test_002", problem_type="litvin")

        self.collector.add_validation_metrics(metrics1)
        self.collector.add_validation_metrics(metrics2)

        summary = self.collector.get_metrics_summary()

        assert summary["total_metrics"] == 2
        assert set(summary["problem_types"]) == {"crank_center", "litvin"}
        assert "date_range" in summary
        assert "tolerance_success_rate" in summary

    def test_compute_stats_dict_edge_cases(self):
        """Test statistics computation with edge cases."""
        # Test with empty list
        stats = self.collector._compute_stats_dict([])
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["median"] == 0.0

        # Test with single value
        stats = self.collector._compute_stats_dict([5.0])
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0
        assert stats["median"] == 5.0

    def _create_sample_metrics(self, problem_id: str, **kwargs) -> ValidationMetrics:
        """Create sample validation metrics for testing."""
        defaults = {
            "timestamp": datetime.now().isoformat(),
            "problem_type": "crank_center",
            "stroke": 20.0,
            "cycle_time": 1.0,
            "crank_radius_range": (20.0, 100.0),
            "rod_length_range": (100.0, 300.0),
            "python_torque_avg": 150.0,
            "casadi_torque_avg": 150.1,
            "python_torque_ripple": 5.0,
            "casadi_torque_ripple": 5.05,
            "python_side_load_penalty": 100.0,
            "casadi_side_load_penalty": 100.2,
            "python_litvin_objective": 0.0,
            "casadi_litvin_objective": 0.01,
            "python_litvin_closure": 0.0,
            "casadi_litvin_closure": 0.005,
            "torque_avg_diff": 0.1,
            "torque_ripple_diff": 0.05,
            "side_load_penalty_diff": 0.2,
            "litvin_objective_diff": 0.01,
            "litvin_closure_diff": 0.005,
            "torque_avg_rel_diff": 0.00067,
            "torque_ripple_rel_diff": 0.01,
            "side_load_penalty_rel_diff": 0.002,
            "litvin_objective_rel_diff": 0.0,
            "litvin_closure_rel_diff": 0.0,
            "python_evaluation_time_ms": 2.5,
            "casadi_evaluation_time_ms": 0.8,
            "python_gradient_time_ms": 5.0,
            "casadi_gradient_time_ms": 1.2,
            "evaluation_speedup": 3.125,
            "gradient_speedup": 4.167,
            "python_iterations": 50,
            "casadi_iterations": 45,
            "python_converged": True,
            "casadi_converged": True,
            "within_tolerance": True,
            "tolerance_threshold": 1e-4,
            "casadi_version": "3.6.0",
            "python_version": "3.11.5",
            "hardware_info": {"cpu": "test", "memory": "8GB"},
        }

        # Update defaults with any provided kwargs
        defaults.update(kwargs)
        defaults["problem_id"] = problem_id

        return ValidationMetrics(**defaults)
