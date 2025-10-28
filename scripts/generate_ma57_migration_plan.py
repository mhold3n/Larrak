#!/usr/bin/env python3
"""
MA57 Migration Plan Generator

This script generates a comprehensive migration plan for transitioning from MA27 to MA57
linear solver based on collected optimization data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from campro.logging import get_logger
from campro.optimization.ma57_migration_analyzer import MA57MigrationAnalyzer

log = get_logger(__name__)


def generate_migration_plan(
    data_file: str,
    output_file: str,
    include_detailed_analysis: bool = True,
) -> dict[str, Any]:
    """
    Generate a comprehensive MA57 migration plan.

    Args:
        data_file: Path to migration data file
        output_file: Path to output plan file
        include_detailed_analysis: Whether to include detailed analysis

    Returns:
        Migration plan dictionary
    """
    log.info(f"Generating MA57 migration plan from {data_file}")

    # Initialize analyzer
    analyzer = MA57MigrationAnalyzer(data_file)

    # Get migration analysis
    analysis = analyzer.analyze_migration_readiness()
    plan = analyzer.get_migration_plan()

    # Create comprehensive plan
    migration_plan = {
        "executive_summary": {
            "migration_priority": analysis.migration_priority,
            "estimated_effort": analysis.estimated_effort,
            "total_optimization_runs": analysis.total_runs,
            "ma57_beneficial_runs": analysis.ma57_beneficial_runs,
            "beneficial_percentage": analysis.ma57_beneficial_runs / analysis.total_runs
            if analysis.total_runs > 0
            else 0,
            "average_speedup": analysis.average_speedup,
            "convergence_improvements": analysis.convergence_improvements,
        },
        "recommendations": analysis.recommendations,
        "implementation_roadmap": {
            "phase_priorities": plan["phase_priorities"],
            "implementation_steps": plan["implementation_steps"],
            "success_metrics": plan["success_metrics"],
            "rollback_plan": plan["rollback_plan"],
        },
    }

    if include_detailed_analysis:
        migration_plan["detailed_analysis"] = {
            "problem_size_analysis": analysis.problem_size_analysis,
            "phase_analysis": analysis.phase_analysis,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat(),
                    "phase": dp.phase,
                    "problem_size": list(dp.problem_size),
                    "ma27_grade": dp.ma27_report.grade,
                    "ma57_grade": dp.ma57_report.grade if dp.ma57_report else None,
                    "performance_improvement": dp.performance_improvement,
                    "convergence_improvement": dp.convergence_improvement,
                }
                for dp in analyzer.data_points
            ],
        }

    # Add implementation timeline
    migration_plan["implementation_timeline"] = generate_implementation_timeline(
        analysis.migration_priority,
    )

    # Add risk assessment
    migration_plan["risk_assessment"] = generate_risk_assessment(analysis)

    # Add cost-benefit analysis
    migration_plan["cost_benefit_analysis"] = generate_cost_benefit_analysis(analysis)

    # Save plan
    with open(output_file, "w") as f:
        json.dump(migration_plan, f, indent=2)

    log.info(f"Migration plan saved to {output_file}")
    return migration_plan


def generate_implementation_timeline(priority: str) -> dict[str, Any]:
    """Generate implementation timeline based on priority."""
    if priority == "high":
        return {
            "week_1_2": "Install MA57 and run initial comparison tests",
            "week_3_4": "Implement selective MA57 usage for high-benefit phases",
            "week_5_6": "Monitor performance and expand usage gradually",
            "week_7_8": "Full migration with fallback mechanisms",
            "week_9_10": "Performance optimization and documentation",
            "week_11_12": "Training and knowledge transfer",
        }
    if priority == "medium":
        return {
            "week_1_2": "Evaluate MA57 availability and licensing",
            "week_3_4": "Run targeted comparison tests",
            "week_5_6": "Implement selective usage for specific use cases",
            "week_7_8": "Monitor results and expand gradually",
            "week_9_10": "Document lessons learned",
        }
    return {
        "month_1_3": "Continue data collection and monitoring",
        "month_4_6": "Re-evaluate migration decision based on new data",
        "month_7_12": "Consider migration if evidence improves",
    }


def generate_risk_assessment(analysis) -> dict[str, Any]:
    """Generate risk assessment for migration."""
    risks = []
    mitigations = []

    # Technical risks
    if analysis.total_runs < 10:
        risks.append("Limited data for statistical confidence")
        mitigations.append("Continue collecting optimization data before migration")

    if analysis.average_speedup and analysis.average_speedup < 1.2:
        risks.append("Limited performance improvement expected")
        mitigations.append("Focus on convergence improvements rather than speed")

    # Implementation risks
    risks.append("Potential compatibility issues with existing systems")
    mitigations.append("Maintain MA27 as fallback option")

    risks.append("Learning curve for team members")
    mitigations.append("Provide training and documentation")

    # Business risks
    risks.append("Potential disruption to optimization workflows")
    mitigations.append("Implement gradual rollout with monitoring")

    return {
        "risk_level": "low"
        if analysis.migration_priority == "low"
        else "medium"
        if analysis.migration_priority == "medium"
        else "high",
        "risks": risks,
        "mitigations": mitigations,
    }


def generate_cost_benefit_analysis(analysis) -> dict[str, Any]:
    """Generate cost-benefit analysis for migration."""
    costs = []
    benefits = []

    # Costs
    costs.append("Development time for implementation and testing")
    costs.append("Potential licensing costs for MA57")
    costs.append("Training and documentation effort")
    costs.append("Risk of temporary performance degradation")

    # Benefits
    if analysis.average_speedup and analysis.average_speedup > 1.2:
        benefits.append(
            f"Performance improvement of {analysis.average_speedup:.1f}x on average",
        )

    if analysis.convergence_improvements > 0:
        benefits.append(
            f"Improved convergence in {analysis.convergence_improvements} cases",
        )

    benefits.append("Better numerical stability for ill-conditioned problems")
    benefits.append("Future-proofing for larger optimization problems")

    # ROI calculation
    roi_estimate = (
        "positive" if analysis.migration_priority in ["high", "medium"] else "neutral"
    )

    return {
        "costs": costs,
        "benefits": benefits,
        "roi_estimate": roi_estimate,
        "payback_period": "3-6 months"
        if analysis.migration_priority == "high"
        else "6-12 months"
        if analysis.migration_priority == "medium"
        else "12+ months",
    }


def print_plan_summary(plan: dict[str, Any]):
    """Print a summary of the migration plan."""
    summary = plan["executive_summary"]

    print("\n" + "=" * 60)
    print("MA57 MIGRATION PLAN SUMMARY")
    print("=" * 60)
    print(f"Migration Priority: {summary['migration_priority'].upper()}")
    print(f"Estimated Effort: {summary['estimated_effort'].upper()}")
    print(f"Total Optimization Runs: {summary['total_optimization_runs']}")
    print(f"MA57 Beneficial Runs: {summary['ma57_beneficial_runs']}")
    print(f"Beneficial Percentage: {summary['beneficial_percentage']:.1%}")

    if summary["average_speedup"]:
        print(f"Average Speedup: {summary['average_speedup']:.1f}x")

    print(f"Convergence Improvements: {summary['convergence_improvements']}")

    print("\nKey Recommendations:")
    for i, rec in enumerate(plan["recommendations"][:3], 1):
        print(f"  {i}. {rec}")

    print("\nImplementation Steps:")
    for i, step in enumerate(
        plan["implementation_roadmap"]["implementation_steps"][:3], 1,
    ):
        print(f"  {i}. {step}")

    print("=" * 60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate MA57 migration plan")
    parser.add_argument(
        "--data-file",
        default="ma57_migration_data.json",
        help="Path to migration data file (default: ma57_migration_data.json)",
    )
    parser.add_argument(
        "--output-file",
        default="ma57_migration_plan.json",
        help="Path to output plan file (default: ma57_migration_plan.json)",
    )
    parser.add_argument(
        "--no-detailed-analysis",
        action="store_true",
        help="Exclude detailed analysis from plan",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print plan summary to console",
    )

    args = parser.parse_args()

    try:
        # Generate migration plan
        plan = generate_migration_plan(
            data_file=args.data_file,
            output_file=args.output_file,
            include_detailed_analysis=not args.no_detailed_analysis,
        )

        if args.print_summary:
            print_plan_summary(plan)

        print(f"\nMigration plan generated successfully: {args.output_file}")

    except Exception as e:
        log.error(f"Failed to generate migration plan: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
