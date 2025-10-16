"""
Tests for MA57 migration analyzer.

This module tests the MA57 migration analyzer functionality including
data collection, analysis, and migration planning.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from campro.optimization.ma57_migration_analyzer import (
    MA57MigrationAnalyzer,
    MigrationDataPoint,
    MigrationAnalysis,
)
from campro.optimization.solver_analysis import MA57ReadinessReport


class TestMA57MigrationAnalyzer:
    """Test MA57 migration analyzer functionality."""

    def _create_temp_analyzer(self):
        """Create analyzer with temporary file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        return MA57MigrationAnalyzer(temp_file), temp_file

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            analyzer = MA57MigrationAnalyzer(temp_file)
            
            assert analyzer.data_file == temp_file
            assert analyzer.data_points == []
            assert isinstance(analyzer, MA57MigrationAnalyzer)
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_analyzer_initialization_with_custom_file(self):
        """Test analyzer initialization with custom data file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            custom_file = f.name
        
        try:
            analyzer = MA57MigrationAnalyzer(custom_file)
            
            assert analyzer.data_file == custom_file
            assert analyzer.data_points == []
        finally:
            Path(custom_file).unlink(missing_ok=True)

    def test_add_ma27_run(self):
        """Test adding MA27 run data."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            analyzer = MA57MigrationAnalyzer(temp_file)
            
            # Create mock MA27 report
            ma27_report = MA57ReadinessReport(
                grade="medium",
                reasons=["High iteration count (1500)"],
                suggested_action="Consider MA57 if available",
                stats={
                    "success": True,
                    "iter_count": 1500,
                    "ls_time_ratio": 0.4,
                    "primal_inf": 1e-6,
                    "dual_inf": 1e-6,
                }
            )
            
            # Add MA27 run
            data_point = analyzer.add_ma27_run(
                phase="secondary",
                problem_size=(200, 100),
                ma27_report=ma27_report,
                metadata={"stroke": 50.0, "cycle_time": 0.1}
            )
            
            # Verify data point
            assert isinstance(data_point, MigrationDataPoint)
            assert data_point.phase == "secondary"
            assert data_point.problem_size == (200, 100)
            assert data_point.ma27_report == ma27_report
            assert data_point.ma57_report is None
            assert data_point.metadata["stroke"] == 50.0
            
            # Verify it was added to analyzer
            assert len(analyzer.data_points) == 1
            assert analyzer.data_points[0] == data_point
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_update_with_ma57_run(self):
        """Test updating data point with MA57 run results."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Create initial MA27 run
            ma27_report = MA57ReadinessReport(
                grade="high",
                reasons=["High iteration count (2000)"],
                suggested_action="Strongly consider MA57",
                stats={
                    "success": True,
                    "iter_count": 2000,
                    "ls_time_ratio": 0.6,
                    "primal_inf": 1e-6,
                    "dual_inf": 1e-6,
                }
            )
            
            data_point = analyzer.add_ma27_run(
                phase="primary",
                problem_size=(500, 200),
                ma27_report=ma27_report
            )
            
            # Create MA57 report
            ma57_report = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={
                    "success": True,
                    "iter_count": 800,
                    "ls_time_ratio": 0.2,
                    "primal_inf": 1e-6,
                    "dual_inf": 1e-6,
                }
            )
            
            # Update with MA57 results
            analyzer.update_with_ma57_run(
                data_point,
                ma57_report,
                performance_improvement=2.5,
                convergence_improvement=True
            )
            
            # Verify update
            assert data_point.ma57_report == ma57_report
            assert data_point.performance_improvement == 2.5
            assert data_point.convergence_improvement is True
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_analyze_migration_readiness_no_data(self):
        """Test migration readiness analysis with no data."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            analysis = analyzer.analyze_migration_readiness()
            
            assert analysis.total_runs == 0
            assert analysis.ma57_beneficial_runs == 0
            assert analysis.average_speedup is None
            assert analysis.convergence_improvements == 0
            assert analysis.migration_priority == "low"
            assert analysis.estimated_effort == "low"
            assert "No data available for analysis" in analysis.recommendations
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_analyze_migration_readiness_with_data(self):
        """Test migration readiness analysis with data."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Add some test data
            ma27_report1 = MA57ReadinessReport(
                grade="high",
                reasons=["High iteration count (2000)"],
                suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            ma27_report2 = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 500, "ls_time_ratio": 0.2}
            )
            
            # Add MA27 runs
            dp1 = analyzer.add_ma27_run("primary", (500, 200), ma27_report1)
            dp2 = analyzer.add_ma27_run("secondary", (100, 50), ma27_report2)
            
            # Add MA57 results
            ma57_report1 = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
            )
            
            ma57_report2 = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 400, "ls_time_ratio": 0.1}
            )
            
            analyzer.update_with_ma57_run(dp1, ma57_report1, 2.5, True)
            analyzer.update_with_ma57_run(dp2, ma57_report2, 1.25, False)
            
            # Analyze
            analysis = analyzer.analyze_migration_readiness()
            
            assert analysis.total_runs == 2
            assert analysis.ma57_beneficial_runs == 2  # Both are beneficial (speedup > 1.2)
            assert analysis.average_speedup == 1.875  # (2.5 + 1.25) / 2
            assert analysis.convergence_improvements == 1
            assert analysis.migration_priority == "high"  # 100% beneficial
            assert analysis.estimated_effort == "medium"
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_problem_size_analysis(self):
        """Test problem size analysis."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Add problems of different sizes
            small_problem = MA57ReadinessReport(
                grade="low", reasons=[], suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 100, "ls_time_ratio": 0.1}
            )
            
            large_problem = MA57ReadinessReport(
                grade="high", reasons=["High iteration count"], suggested_action="Consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.7}
            )
            
            analyzer.add_ma27_run("primary", (50, 25), small_problem)
            analyzer.add_ma27_run("secondary", (1000, 500), large_problem)
            
            analysis = analyzer.analyze_migration_readiness()
            size_analysis = analysis.problem_size_analysis
            
            assert "small_problems" in size_analysis
            assert "medium_problems" in size_analysis
            assert "large_problems" in size_analysis
            assert size_analysis["small_problems"]["count"] == 1
            assert size_analysis["large_problems"]["count"] == 1
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_phase_analysis(self):
        """Test phase analysis."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Add problems from different phases
            primary_report = MA57ReadinessReport(
                grade="medium", reasons=["Moderate issues"], suggested_action="Consider MA57",
                stats={"success": True, "iter_count": 1000, "ls_time_ratio": 0.4}
            )
            
            secondary_report = MA57ReadinessReport(
                grade="high", reasons=["High iteration count"], suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            analyzer.add_ma27_run("primary", (200, 100), primary_report)
            analyzer.add_ma27_run("secondary", (300, 150), secondary_report)
            
            analysis = analyzer.analyze_migration_readiness()
            phase_analysis = analysis.phase_analysis
            
            assert "primary" in phase_analysis
            assert "secondary" in phase_analysis
            assert "tertiary" in phase_analysis
            assert phase_analysis["primary"]["count"] == 1
            assert phase_analysis["secondary"]["count"] == 1
            assert phase_analysis["tertiary"]["count"] == 0
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_migration_plan_generation(self):
        """Test migration plan generation."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Add some test data
            ma27_report = MA57ReadinessReport(
                grade="high",
                reasons=["High iteration count (2000)"],
                suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            dp = analyzer.add_ma27_run("primary", (500, 200), ma27_report)
            
            ma57_report = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
            )
            
            analyzer.update_with_ma57_run(dp, ma57_report, 2.5, True)
            
            plan = analyzer.get_migration_plan()
            
            assert "analysis_summary" in plan
            assert "recommendations" in plan
            assert "phase_priorities" in plan
            assert "implementation_steps" in plan
            assert "success_metrics" in plan
            assert "rollback_plan" in plan
            
            assert plan["analysis_summary"]["total_runs"] == 1
            assert plan["analysis_summary"]["ma57_beneficial_runs"] == 1
            assert plan["analysis_summary"]["migration_priority"] == "high"
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_data_persistence(self):
        """Test data persistence to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Create analyzer with temp file
            analyzer = MA57MigrationAnalyzer(temp_file)
            
            # Add some data
            ma27_report = MA57ReadinessReport(
                grade="medium",
                reasons=["Moderate issues"],
                suggested_action="Consider MA57",
                stats={"success": True, "iter_count": 1000, "ls_time_ratio": 0.4}
            )
            
            analyzer.add_ma27_run("primary", (200, 100), ma27_report)
            
            # Create new analyzer with same file
            analyzer2 = MA57MigrationAnalyzer(temp_file)
            
            # Verify data was loaded
            assert len(analyzer2.data_points) == 1
            assert analyzer2.data_points[0].phase == "primary"
            assert analyzer2.data_points[0].problem_size == (200, 100)
            
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)

    def test_export_analysis_report(self):
        """Test export of analysis report."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Add some data
            ma27_report = MA57ReadinessReport(
                grade="high",
                reasons=["High iteration count (2000)"],
                suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            dp = analyzer.add_ma27_run("primary", (500, 200), ma27_report)
            
            ma57_report = MA57ReadinessReport(
                grade="low",
                reasons=["No adverse indicators detected"],
                suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
            )
            
            analyzer.update_with_ma57_run(dp, ma57_report, 2.5, True)
            
            # Export report
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                export_file = f.name
            
            analyzer.export_analysis_report(export_file)
            
            # Verify file was created and contains expected data
            assert Path(export_file).exists()
            
            with open(export_file, 'r') as f:
                report = json.load(f)
            
            assert "timestamp" in report
            assert "migration_analysis" in report
            assert "recommendations" in report
            assert "migration_plan" in report
            assert "data_points" in report
            
            assert report["migration_analysis"]["total_runs"] == 1
            assert report["migration_analysis"]["ma57_beneficial_runs"] == 1
            
            # Clean up export file
            Path(export_file).unlink(missing_ok=True)
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_beneficial_run_detection(self):
        """Test detection of MA57 beneficial runs."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Test case 1: Performance improvement
            ma27_report1 = MA57ReadinessReport(
                grade="medium", reasons=["Moderate issues"], suggested_action="Consider MA57",
                stats={"success": True, "iter_count": 1000, "ls_time_ratio": 0.4}
            )
            
            ma57_report1 = MA57ReadinessReport(
                grade="low", reasons=["No issues"], suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
            )
            
            dp1 = analyzer.add_ma27_run("primary", (200, 100), ma27_report1)
            analyzer.update_with_ma57_run(dp1, ma57_report1, 1.5, False)  # 1.5x speedup
            
            # Test case 2: Convergence improvement
            ma27_report2 = MA57ReadinessReport(
                grade="high", reasons=["High iteration count"], suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            ma57_report2 = MA57ReadinessReport(
                grade="medium", reasons=["Moderate issues"], suggested_action="Consider MA57",
                stats={"success": True, "iter_count": 1500, "ls_time_ratio": 0.4}
            )
            
            dp2 = analyzer.add_ma27_run("secondary", (300, 150), ma27_report2)
            analyzer.update_with_ma57_run(dp2, ma57_report2, 1.1, True)  # Convergence improvement
            
            # Test case 3: Grade improvement
            ma27_report3 = MA57ReadinessReport(
                grade="high", reasons=["High iteration count"], suggested_action="Strongly consider MA57",
                stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
            )
            
            ma57_report3 = MA57ReadinessReport(
                grade="low", reasons=["No issues"], suggested_action="Stick with MA27",
                stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
            )
            
            dp3 = analyzer.add_ma27_run("tertiary", (4, 8), ma27_report3)
            analyzer.update_with_ma57_run(dp3, ma57_report3, 1.1, False)  # Grade improvement
            
            # Analyze
            analysis = analyzer.analyze_migration_readiness()
            
            # All three should be beneficial
            assert analysis.ma57_beneficial_runs == 3
            assert analysis.migration_priority == "medium"  # 100% beneficial but avg speedup < 1.3
        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_migration_priority_calculation(self):
        """Test migration priority calculation logic."""
        analyzer, temp_file = self._create_temp_analyzer()
        
        try:
            # Test high priority case
            for i in range(8):  # 8 out of 10 beneficial
                ma27_report = MA57ReadinessReport(
                    grade="high", reasons=["High iteration count"], suggested_action="Strongly consider MA57",
                    stats={"success": True, "iter_count": 2000, "ls_time_ratio": 0.6}
                )
                
                ma57_report = MA57ReadinessReport(
                    grade="low", reasons=["No issues"], suggested_action="Stick with MA27",
                    stats={"success": True, "iter_count": 800, "ls_time_ratio": 0.2}
                )
                
                dp = analyzer.add_ma27_run(f"phase_{i}", (200, 100), ma27_report)
                analyzer.update_with_ma57_run(dp, ma57_report, 2.0, True)
            
            # Add 2 non-beneficial runs
            for i in range(2):
                ma27_report = MA57ReadinessReport(
                    grade="low", reasons=["No issues"], suggested_action="Stick with MA27",
                    stats={"success": True, "iter_count": 500, "ls_time_ratio": 0.2}
                )
                
                ma57_report = MA57ReadinessReport(
                    grade="low", reasons=["No issues"], suggested_action="Stick with MA27",
                    stats={"success": True, "iter_count": 400, "ls_time_ratio": 0.1}
                )
                
                dp = analyzer.add_ma27_run(f"phase_{i+8}", (100, 50), ma27_report)
                analyzer.update_with_ma57_run(dp, ma57_report, 1.1, False)
            
            analysis = analyzer.analyze_migration_readiness()
            
            # Should be high priority (80% beneficial, good speedup, convergence improvements)
            assert analysis.migration_priority == "high"
            assert analysis.ma57_beneficial_runs == 8
            assert analysis.total_runs == 10
        finally:
            Path(temp_file).unlink(missing_ok=True)
