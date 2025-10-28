"""
Validation statistics collection for CasADi physics validation mode.

This module provides comprehensive statistics collection and analysis
for comparing Python and CasADi physics implementations during validation mode.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class ValidationMetrics:
    """Individual validation comparison metrics."""

    # Problem identification
    problem_id: str
    timestamp: str
    problem_type: str  # "crank_center", "litvin", "thermal_efficiency"

    # Parameter ranges
    stroke: float
    cycle_time: float
    crank_radius_range: tuple[float, float]
    rod_length_range: tuple[float, float]

    # Results comparison
    python_torque_avg: float
    casadi_torque_avg: float
    python_torque_ripple: float
    casadi_torque_ripple: float
    python_side_load_penalty: float
    casadi_side_load_penalty: float
    python_litvin_objective: float
    casadi_litvin_objective: float
    python_litvin_closure: float
    casadi_litvin_closure: float

    # Differences (CasADi - Python)
    torque_avg_diff: float
    torque_ripple_diff: float
    side_load_penalty_diff: float
    litvin_objective_diff: float
    litvin_closure_diff: float

    # Relative differences (|diff| / |python_value|)
    torque_avg_rel_diff: float
    torque_ripple_rel_diff: float
    side_load_penalty_rel_diff: float
    litvin_objective_rel_diff: float
    litvin_closure_rel_diff: float

    # Performance metrics
    python_evaluation_time_ms: float
    casadi_evaluation_time_ms: float
    python_gradient_time_ms: float
    casadi_gradient_time_ms: float

    # Performance ratios (CasADi / Python)
    evaluation_speedup: float
    gradient_speedup: float

    # Convergence metrics
    python_iterations: int
    casadi_iterations: int
    python_converged: bool
    casadi_converged: bool

    # Tolerance check
    within_tolerance: bool
    tolerance_threshold: float

    # Additional metadata
    casadi_version: str
    python_version: str
    hardware_info: dict[str, Any]


@dataclass
class ValidationStatistics:
    """Aggregated validation statistics."""

    # Summary statistics
    total_validations: int
    successful_validations: int
    failed_validations: int
    within_tolerance_count: int
    tolerance_exceeded_count: int

    # Parity statistics
    torque_avg_stats: dict[str, float]  # min, max, mean, std of differences
    torque_ripple_stats: dict[str, float]
    side_load_penalty_stats: dict[str, float]
    litvin_objective_stats: dict[str, float]
    litvin_closure_stats: dict[str, float]

    # Performance statistics
    evaluation_speedup_stats: dict[str, float]
    gradient_speedup_stats: dict[str, float]

    # Convergence statistics
    python_convergence_rate: float
    casadi_convergence_rate: float

    # Tolerance analysis
    tolerance_success_rate: float
    tolerance_threshold: float

    # Problem type breakdown
    problem_type_counts: dict[str, int]
    problem_type_success_rates: dict[str, float]

    # Time period
    collection_start: str
    collection_end: str
    collection_duration_days: float


class ValidationStatisticsCollector:
    """
    Collects and analyzes validation statistics for CasADi physics validation mode.

    This class provides comprehensive statistics collection, analysis, and reporting
    for comparing Python and CasADi physics implementations.
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize the validation statistics collector.

        Parameters
        ----------
        output_dir : Path, optional
            Directory to save statistics files. Defaults to logs/validation_stats/
        """
        self.output_dir = output_dir or Path("logs/validation_stats")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: list[ValidationMetrics] = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        log.info(
            f"Validation statistics collector initialized. Output directory: {self.output_dir}",
        )

    def add_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """
        Add validation metrics to the collection.

        Parameters
        ----------
        metrics : ValidationMetrics
            Individual validation comparison metrics
        """
        self.metrics.append(metrics)
        log.debug(f"Added validation metrics for problem {metrics.problem_id}")

        # Save individual metrics to file
        self._save_individual_metrics(metrics)

    def _save_individual_metrics(self, metrics: ValidationMetrics) -> None:
        """Save individual metrics to a JSON file."""
        filename = f"validation_{metrics.problem_id}_{self.current_session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    def compute_statistics(self) -> ValidationStatistics:
        """
        Compute aggregated statistics from collected metrics.

        Returns
        -------
        ValidationStatistics
            Aggregated validation statistics
        """
        if not self.metrics:
            raise ValueError("No validation metrics collected yet")

        # Basic counts
        total_validations = len(self.metrics)
        successful_validations = sum(
            1 for m in self.metrics if m.python_converged and m.casadi_converged
        )
        failed_validations = total_validations - successful_validations
        within_tolerance_count = sum(1 for m in self.metrics if m.within_tolerance)
        tolerance_exceeded_count = total_validations - within_tolerance_count

        # Parity statistics
        torque_avg_diffs = [m.torque_avg_diff for m in self.metrics]
        torque_ripple_diffs = [m.torque_ripple_diff for m in self.metrics]
        side_load_penalty_diffs = [m.side_load_penalty_diff for m in self.metrics]
        litvin_objective_diffs = [m.litvin_objective_diff for m in self.metrics]
        litvin_closure_diffs = [m.litvin_closure_diff for m in self.metrics]

        # Performance statistics
        evaluation_speedups = [
            m.evaluation_speedup for m in self.metrics if m.evaluation_speedup > 0
        ]
        gradient_speedups = [
            m.gradient_speedup for m in self.metrics if m.gradient_speedup > 0
        ]

        # Convergence statistics
        python_converged_count = sum(1 for m in self.metrics if m.python_converged)
        casadi_converged_count = sum(1 for m in self.metrics if m.casadi_converged)

        # Problem type analysis
        problem_types = {}
        problem_type_success = {}
        for metrics in self.metrics:
            problem_type = metrics.problem_type
            if problem_type not in problem_types:
                problem_types[problem_type] = 0
                problem_type_success[problem_type] = 0
            problem_types[problem_type] += 1
            if metrics.within_tolerance:
                problem_type_success[problem_type] += 1

        problem_type_success_rates = {
            ptype: success_count / total_count
            for ptype, (total_count, success_count) in zip(
                problem_types.keys(),
                zip(problem_types.values(), problem_type_success.values()),
            )
        }

        # Time period
        timestamps = [datetime.fromisoformat(m.timestamp) for m in self.metrics]
        collection_start = min(timestamps).isoformat()
        collection_end = max(timestamps).isoformat()
        collection_duration = (max(timestamps) - min(timestamps)).total_seconds() / (
            24 * 3600
        )

        return ValidationStatistics(
            total_validations=total_validations,
            successful_validations=successful_validations,
            failed_validations=failed_validations,
            within_tolerance_count=within_tolerance_count,
            tolerance_exceeded_count=tolerance_exceeded_count,
            torque_avg_stats=self._compute_stats_dict(torque_avg_diffs),
            torque_ripple_stats=self._compute_stats_dict(torque_ripple_diffs),
            side_load_penalty_stats=self._compute_stats_dict(side_load_penalty_diffs),
            litvin_objective_stats=self._compute_stats_dict(litvin_objective_diffs),
            litvin_closure_stats=self._compute_stats_dict(litvin_closure_diffs),
            evaluation_speedup_stats=self._compute_stats_dict(evaluation_speedups),
            gradient_speedup_stats=self._compute_stats_dict(gradient_speedups),
            python_convergence_rate=python_converged_count / total_validations,
            casadi_convergence_rate=casadi_converged_count / total_validations,
            tolerance_success_rate=within_tolerance_count / total_validations,
            tolerance_threshold=self.metrics[0].tolerance_threshold,
            problem_type_counts=problem_types,
            problem_type_success_rates=problem_type_success_rates,
            collection_start=collection_start,
            collection_end=collection_end,
            collection_duration_days=collection_duration,
        )

    def _compute_stats_dict(self, values: list[float]) -> dict[str, float]:
        """Compute statistics dictionary for a list of values."""
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}

        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    def save_statistics(self, statistics: ValidationStatistics) -> Path:
        """
        Save aggregated statistics to a JSON file.

        Parameters
        ----------
        statistics : ValidationStatistics
            Aggregated validation statistics

        Returns
        -------
        Path
            Path to the saved statistics file
        """
        filename = f"validation_statistics_{self.current_session_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(asdict(statistics), f, indent=2)

        log.info(f"Validation statistics saved to {filepath}")
        return filepath

    def generate_report(self, statistics: ValidationStatistics) -> str:
        """
        Generate a human-readable validation report.

        Parameters
        ----------
        statistics : ValidationStatistics
            Aggregated validation statistics

        Returns
        -------
        str
            Human-readable validation report
        """
        report = f"""
# CasADi Physics Validation Report

## Summary
- **Total Validations**: {statistics.total_validations}
- **Successful Validations**: {statistics.successful_validations} ({statistics.successful_validations / statistics.total_validations * 100:.1f}%)
- **Within Tolerance**: {statistics.within_tolerance_count} ({statistics.tolerance_success_rate * 100:.1f}%)
- **Collection Period**: {statistics.collection_start} to {statistics.collection_end}
- **Duration**: {statistics.collection_duration_days:.1f} days

## Parity Analysis
### Torque Average Differences
- **Mean**: {statistics.torque_avg_stats["mean"]:.2e}
- **Std Dev**: {statistics.torque_avg_stats["std"]:.2e}
- **Range**: [{statistics.torque_avg_stats["min"]:.2e}, {statistics.torque_avg_stats["max"]:.2e}]

### Torque Ripple Differences
- **Mean**: {statistics.torque_ripple_stats["mean"]:.2e}
- **Std Dev**: {statistics.torque_ripple_stats["std"]:.2e}
- **Range**: [{statistics.torque_ripple_stats["min"]:.2e}, {statistics.torque_ripple_stats["max"]:.2e}]

### Side Load Penalty Differences
- **Mean**: {statistics.side_load_penalty_stats["mean"]:.2e}
- **Std Dev**: {statistics.side_load_penalty_stats["std"]:.2e}
- **Range**: [{statistics.side_load_penalty_stats["min"]:.2e}, {statistics.side_load_penalty_stats["max"]:.2e}]

## Performance Analysis
### Evaluation Speedup
- **Mean**: {statistics.evaluation_speedup_stats["mean"]:.2f}x
- **Median**: {statistics.evaluation_speedup_stats["median"]:.2f}x
- **Range**: [{statistics.evaluation_speedup_stats["min"]:.2f}x, {statistics.evaluation_speedup_stats["max"]:.2f}x]

### Gradient Speedup
- **Mean**: {statistics.gradient_speedup_stats["mean"]:.2f}x
- **Median**: {statistics.gradient_speedup_stats["median"]:.2f}x
- **Range**: [{statistics.gradient_speedup_stats["min"]:.2f}x, {statistics.gradient_speedup_stats["max"]:.2f}x]

## Convergence Analysis
- **Python Convergence Rate**: {statistics.python_convergence_rate * 100:.1f}%
- **CasADi Convergence Rate**: {statistics.casadi_convergence_rate * 100:.1f}%

## Problem Type Breakdown
"""

        for problem_type, count in statistics.problem_type_counts.items():
            success_rate = statistics.problem_type_success_rates[problem_type]
            report += f"- **{problem_type}**: {count} problems, {success_rate * 100:.1f}% success rate\n"

        report += f"""
## Decision Criteria Analysis
- **Tolerance Threshold**: {statistics.tolerance_threshold:.2e}
- **Success Rate**: {statistics.tolerance_success_rate * 100:.1f}%
- **Target Success Rate**: 95.0%

## Recommendation
"""

        if statistics.tolerance_success_rate >= 0.95:
            report += "✅ **RECOMMEND ENABLING**: CasADi physics meets all validation criteria.\n"
        elif statistics.tolerance_success_rate >= 0.90:
            report += "⚠️ **CONDITIONAL ENABLING**: CasADi physics mostly meets criteria but needs monitoring.\n"
        else:
            report += "❌ **DO NOT ENABLE**: CasADi physics does not meet validation criteria.\n"

        return report

    def save_report(self, report: str) -> Path:
        """
        Save the validation report to a markdown file.

        Parameters
        ----------
        report : str
            Human-readable validation report

        Returns
        -------
        Path
            Path to the saved report file
        """
        filename = f"validation_report_{self.current_session_id}.md"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(report)

        log.info(f"Validation report saved to {filepath}")
        return filepath

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        log.info("Validation metrics cleared")

    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get a summary of collected metrics.

        Returns
        -------
        Dict[str, Any]
            Summary of collected metrics
        """
        if not self.metrics:
            return {"total_metrics": 0}

        return {
            "total_metrics": len(self.metrics),
            "problem_types": list(set(m.problem_type for m in self.metrics)),
            "date_range": {
                "start": min(m.timestamp for m in self.metrics),
                "end": max(m.timestamp for m in self.metrics),
            },
            "tolerance_success_rate": sum(1 for m in self.metrics if m.within_tolerance)
            / len(self.metrics),
        }
