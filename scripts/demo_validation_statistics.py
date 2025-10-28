#!/usr/bin/env python3
"""
Demonstration script for CasADi physics validation statistics collection.

This script demonstrates how the validation statistics collection framework
would work in practice during the 2-4 week validation period.
"""

import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
from campro.optimization.validation_statistics import (  # noqa: E402
    ValidationMetrics,
    ValidationStatisticsCollector,
)

log = get_logger(__name__)


def generate_realistic_validation_metrics(
    problem_id: str, problem_type: str, stroke: float, cycle_time: float,
) -> ValidationMetrics:
    """
    Generate realistic validation metrics for demonstration.

    This simulates what would be collected during actual validation runs.
    """
    # Base values based on problem parameters
    base_torque = stroke * 10.0  # Rough scaling
    base_ripple = stroke * 0.5
    base_side_load = stroke * 5.0

    # Add some realistic variation (much smaller for validation mode)
    torque_variation = random.uniform(-0.0001, 0.0001)  # ±0.01% variation
    ripple_variation = random.uniform(-0.0005, 0.0005)  # ±0.05% variation
    side_load_variation = random.uniform(-0.0001, 0.0001)  # ±0.01% variation

    # Python results (baseline)
    python_torque_avg = base_torque * (1 + random.uniform(-0.01, 0.01))
    python_torque_ripple = base_ripple * (1 + random.uniform(-0.02, 0.02))
    python_side_load_penalty = base_side_load * (1 + random.uniform(-0.01, 0.01))

    # CasADi results (with small differences from Python)
    casadi_torque_avg = python_torque_avg * (1 + torque_variation)
    casadi_torque_ripple = python_torque_ripple * (1 + ripple_variation)
    casadi_side_load_penalty = python_side_load_penalty * (1 + side_load_variation)

    # Litvin metrics (usually small values)
    python_litvin_objective = random.uniform(0.0, 0.1)
    casadi_litvin_objective = python_litvin_objective + random.uniform(-0.0001, 0.0001)
    python_litvin_closure = random.uniform(0.0, 0.05)
    casadi_litvin_closure = python_litvin_closure + random.uniform(-0.00005, 0.00005)

    # Calculate differences
    torque_avg_diff = casadi_torque_avg - python_torque_avg
    torque_ripple_diff = casadi_torque_ripple - python_torque_ripple
    side_load_penalty_diff = casadi_side_load_penalty - python_side_load_penalty
    litvin_objective_diff = casadi_litvin_objective - python_litvin_objective
    litvin_closure_diff = casadi_litvin_closure - python_litvin_closure

    # Calculate relative differences (as percentages)
    torque_avg_rel_diff = (
        abs(torque_avg_diff) / abs(python_torque_avg) if python_torque_avg != 0 else 0
    )
    torque_ripple_rel_diff = (
        abs(torque_ripple_diff) / abs(python_torque_ripple)
        if python_torque_ripple != 0
        else 0
    )
    side_load_penalty_rel_diff = (
        abs(side_load_penalty_diff) / abs(python_side_load_penalty)
        if python_side_load_penalty != 0
        else 0
    )
    litvin_objective_rel_diff = (
        abs(litvin_objective_diff) / abs(python_litvin_objective)
        if python_litvin_objective != 0
        else 0
    )
    litvin_closure_rel_diff = (
        abs(litvin_closure_diff) / abs(python_litvin_closure)
        if python_litvin_closure != 0
        else 0
    )

    # Performance metrics (CasADi typically faster)
    python_eval_time = random.uniform(2.0, 5.0)  # 2-5ms
    casadi_eval_time = random.uniform(0.5, 1.5)  # 0.5-1.5ms
    python_grad_time = random.uniform(4.0, 8.0)  # 4-8ms
    casadi_grad_time = random.uniform(1.0, 2.5)  # 1-2.5ms

    evaluation_speedup = python_eval_time / casadi_eval_time
    gradient_speedup = python_grad_time / casadi_grad_time

    # Convergence metrics
    python_iterations = random.randint(30, 80)
    casadi_iterations = random.randint(25, 75)
    python_converged = random.random() > 0.05  # 95% convergence rate
    casadi_converged = random.random() > 0.03  # 97% convergence rate

    # Tolerance check (95% within tolerance)
    max_rel_diff = max(
        torque_avg_rel_diff,
        torque_ripple_rel_diff,
        side_load_penalty_rel_diff,
        litvin_objective_rel_diff,
        litvin_closure_rel_diff,
    )
    within_tolerance = (
        max_rel_diff < 1e-4
    )  # Should be within tolerance with small variations

    # Force 95% to be within tolerance for realistic demo
    if random.random() < 0.95:
        within_tolerance = True

    return ValidationMetrics(
        problem_id=problem_id,
        timestamp=datetime.now().isoformat(),
        problem_type=problem_type,
        stroke=stroke,
        cycle_time=cycle_time,
        crank_radius_range=(stroke * 0.8, stroke * 2.0),
        rod_length_range=(stroke * 3.0, stroke * 8.0),
        python_torque_avg=python_torque_avg,
        casadi_torque_avg=casadi_torque_avg,
        python_torque_ripple=python_torque_ripple,
        casadi_torque_ripple=casadi_torque_ripple,
        python_side_load_penalty=python_side_load_penalty,
        casadi_side_load_penalty=casadi_side_load_penalty,
        python_litvin_objective=python_litvin_objective,
        casadi_litvin_objective=casadi_litvin_objective,
        python_litvin_closure=python_litvin_closure,
        casadi_litvin_closure=casadi_litvin_closure,
        torque_avg_diff=torque_avg_diff,
        torque_ripple_diff=torque_ripple_diff,
        side_load_penalty_diff=side_load_penalty_diff,
        litvin_objective_diff=litvin_objective_diff,
        litvin_closure_diff=litvin_closure_diff,
        torque_avg_rel_diff=torque_avg_rel_diff,
        torque_ripple_rel_diff=torque_ripple_rel_diff,
        side_load_penalty_rel_diff=side_load_penalty_rel_diff,
        litvin_objective_rel_diff=litvin_objective_rel_diff,
        litvin_closure_rel_diff=litvin_closure_rel_diff,
        python_evaluation_time_ms=python_eval_time,
        casadi_evaluation_time_ms=casadi_eval_time,
        python_gradient_time_ms=python_grad_time,
        casadi_gradient_time_ms=casadi_grad_time,
        evaluation_speedup=evaluation_speedup,
        gradient_speedup=gradient_speedup,
        python_iterations=python_iterations,
        casadi_iterations=casadi_iterations,
        python_converged=python_converged,
        casadi_converged=casadi_converged,
        within_tolerance=within_tolerance,
        tolerance_threshold=1e-4,
        casadi_version="3.6.0",
        python_version="3.11.5",
        hardware_info={"cpu": "Apple M2", "memory": "16GB", "os": "macOS 14.6"},
    )


def run_validation_demo():
    """Run a demonstration of validation statistics collection."""
    print("🚀 CasADi Physics Validation Statistics Collection Demo")
    print("=" * 60)

    # Create collector
    output_dir = Path("logs/validation_demo")
    collector = ValidationStatisticsCollector(output_dir=output_dir)

    # Problem types and parameters
    problem_types = ["crank_center", "litvin", "thermal_efficiency"]
    strokes = [15.0, 20.0, 25.0, 30.0, 35.0]
    cycle_times = [0.8, 1.0, 1.2, 1.5, 2.0]

    print("📊 Collecting validation metrics...")
    print(f"📁 Output directory: {output_dir}")

    # Simulate 2 weeks of validation runs (100 problems)
    total_problems = 100
    for i in range(total_problems):
        # Select random problem parameters
        problem_type = random.choice(problem_types)
        stroke = random.choice(strokes)
        cycle_time = random.choice(cycle_times)

        # Generate problem ID
        problem_id = f"val_{i + 1:03d}_{problem_type}"

        # Generate realistic metrics
        metrics = generate_realistic_validation_metrics(
            problem_id,
            problem_type,
            stroke,
            cycle_time,
        )

        # Add to collector
        collector.add_validation_metrics(metrics)

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{total_problems} problems...")

    print(f"✅ Collected {total_problems} validation metrics")

    # Compute statistics
    print("\n📈 Computing validation statistics...")
    statistics = collector.compute_statistics()

    # Generate and save report
    print("📝 Generating validation report...")
    report = collector.generate_report(statistics)
    report_path = collector.save_report(report)

    # Save statistics
    stats_path = collector.save_statistics(statistics)

    # Display summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total Validations: {statistics.total_validations}")
    print(
        f"Successful Validations: {statistics.successful_validations} ({statistics.successful_validations / statistics.total_validations * 100:.1f}%)",
    )
    print(
        f"Within Tolerance: {statistics.within_tolerance_count} ({statistics.tolerance_success_rate * 100:.1f}%)",
    )
    print(f"Collection Period: {statistics.collection_duration_days:.1f} days")

    print("\n🎯 PARITY ANALYSIS")
    print(f"Torque Avg - Mean Diff: {statistics.torque_avg_stats['mean']:.2e}")
    print(f"Torque Avg - Std Dev: {statistics.torque_avg_stats['std']:.2e}")
    print(f"Torque Ripple - Mean Diff: {statistics.torque_ripple_stats['mean']:.2e}")
    print(f"Side Load - Mean Diff: {statistics.side_load_penalty_stats['mean']:.2e}")

    print("\n⚡ PERFORMANCE ANALYSIS")
    print(
        f"Evaluation Speedup - Mean: {statistics.evaluation_speedup_stats['mean']:.2f}x",
    )
    print(
        f"Evaluation Speedup - Median: {statistics.evaluation_speedup_stats['median']:.2f}x",
    )
    print(f"Gradient Speedup - Mean: {statistics.gradient_speedup_stats['mean']:.2f}x")
    print(
        f"Gradient Speedup - Median: {statistics.gradient_speedup_stats['median']:.2f}x",
    )

    print("\n🔄 CONVERGENCE ANALYSIS")
    print(f"Python Convergence Rate: {statistics.python_convergence_rate * 100:.1f}%")
    print(f"CasADi Convergence Rate: {statistics.casadi_convergence_rate * 100:.1f}%")

    print("\n📋 PROBLEM TYPE BREAKDOWN")
    for problem_type, count in statistics.problem_type_counts.items():
        success_rate = statistics.problem_type_success_rates[problem_type]
        print(
            f"{problem_type}: {count} problems, {success_rate * 100:.1f}% success rate",
        )

    # Recommendation
    print("\n🎯 RECOMMENDATION")
    if statistics.tolerance_success_rate >= 0.95:
        print("✅ RECOMMEND ENABLING: CasADi physics meets all validation criteria!")
    elif statistics.tolerance_success_rate >= 0.90:
        print(
            "⚠️  CONDITIONAL ENABLING: CasADi physics mostly meets criteria but needs monitoring.",
        )
    else:
        print("❌ DO NOT ENABLE: CasADi physics does not meet validation criteria.")

    print("\n📁 Files saved:")
    print(f"   Report: {report_path}")
    print(f"   Statistics: {stats_path}")

    return statistics, report_path


def main():
    """Main function."""
    try:
        statistics, report_path = run_validation_demo()

        print("\n🎉 Validation statistics collection demo completed successfully!")
        print(f"📖 View the full report at: {report_path}")

        return 0

    except Exception as e:
        print(f"❌ Error during validation demo: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
