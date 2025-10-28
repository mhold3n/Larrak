#!/usr/bin/env python3
"""
Demonstration of the new library-based architecture.

This script shows how to use the new modular libraries for constraint
definition, optimization, and result analysis.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import new library components
from campro.constraints import CamMotionConstraints
from campro.logging import get_logger
from campro.optimization import CollocationSettings, MotionOptimizer
from campro.utils.plotting import plot_solution

log = get_logger(__name__)


def demo_constraint_system():
    """Demonstrate the new constraint system."""
    print("\n=== Constraint System Demo ===")

    # Create cam constraints
    cam_constraints = CamMotionConstraints(
        stroke=25.0,  # 25mm stroke
        upstroke_duration_percent=55.0,  # 55% upstroke
        zero_accel_duration_percent=20.0,  # 20% zero acceleration
        max_velocity=100.0,  # 100 mm/s max velocity
        max_acceleration=50.0,  # 50 mm/s² max acceleration
        max_jerk=10.0,  # 10 mm/s³ max jerk
        dwell_at_tdc=True,
        dwell_at_bdc=True,
    )

    print("Created cam constraints:")
    print(f"  - Stroke: {cam_constraints.stroke} mm")
    print(f"  - Upstroke duration: {cam_constraints.upstroke_duration_percent}%")
    print(
        f"  - Zero acceleration duration: {cam_constraints.zero_accel_duration_percent}%",
    )
    print(f"  - Max velocity: {cam_constraints.max_velocity} mm/s")

    # Validate constraints
    is_valid = cam_constraints.validate()
    print(f"  - Validation: {'PASSED' if is_valid else 'FAILED'}")

    if cam_constraints.has_violations():
        print("  - Violations found:")
        for violation in cam_constraints.get_violations():
            print(f"    * {violation.message}")

    # Convert to motion constraints
    motion_constraints = cam_constraints.to_motion_constraints(cycle_time=1.0)
    print(
        f"  - Converted to motion constraints with {len(motion_constraints.list_constraints())} constraints",
    )

    return cam_constraints, motion_constraints


def demo_optimization_system():
    """Demonstrate the new optimization system."""
    print("\n=== Optimization System Demo ===")

    # Create optimizer with custom settings
    settings = CollocationSettings(
        degree=3,
        method="legendre",
        tolerance=1e-6,
        max_iterations=1000,
        verbose=False,
    )

    optimizer = MotionOptimizer(settings)
    print(f"Created motion optimizer: {optimizer.name}")
    print(f"  - Configured: {optimizer.is_configured()}")
    print(f"  - Collocation method: {settings.method}")
    print(f"  - Collocation degree: {settings.degree}")

    # Get optimizer info
    info = optimizer.get_optimizer_info()
    print(f"  - Optimizer info: {info}")

    return optimizer


def demo_motion_law_solving():
    """Demonstrate motion law solving with new libraries."""
    print("\n=== Motion Law Solving Demo ===")

    # Create constraints and optimizer
    cam_constraints, motion_constraints = demo_constraint_system()
    optimizer = demo_optimization_system()

    # Solve different motion law types
    motion_types = ["minimum_jerk", "minimum_energy", "minimum_time"]
    results = {}

    for motion_type in motion_types:
        print(f"\nSolving {motion_type} motion law...")

        try:
            result = optimizer.solve_cam_motion_law(
                cam_constraints=cam_constraints,
                motion_type=motion_type,
                cycle_time=1.0,
            )

            results[motion_type] = result

            print(f"  - Status: {result.status.value}")
            print(f"  - Successful: {result.is_successful()}")
            print(f"  - Has solution: {result.has_solution()}")

            if result.solve_time:
                print(f"  - Solve time: {result.solve_time:.3f} seconds")

            if result.objective_value is not None:
                print(f"  - Objective value: {result.objective_value:.6f}")

            # Check constraint violations
            violations = cam_constraints.check_violations(result.solution)
            print(f"  - Constraint violations: {len(violations)}")

            if violations:
                for violation in violations[:3]:  # Show first 3 violations
                    print(f"    * {violation.message}")
                if len(violations) > 3:
                    print(f"    * ... and {len(violations) - 3} more")

        except Exception as e:
            print(f"  - Error: {e}")
            results[motion_type] = None

    return results


def demo_plotting_system():
    """Demonstrate the new plotting system."""
    print("\n=== Plotting System Demo ===")

    # Get results from previous demo
    results = demo_motion_law_solving()

    # Create plots for each motion type
    for motion_type, result in results.items():
        if result and result.is_successful():
            print(f"\nCreating plot for {motion_type}...")

            try:
                # Create plot using new plotting utilities
                fig = plot_solution(
                    solution=result.solution,
                    title=f"{motion_type.replace('_', ' ').title()} Motion Law",
                    use_cam_angle=True,
                )

                # Save plot
                output_dir = Path("plots")
                output_dir.mkdir(exist_ok=True)
                plot_path = output_dir / f"{motion_type}_motion_law.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")

                print(f"  - Plot saved to: {plot_path}")

                # Get solution summary
                summary = result.get_solution_summary()
                if summary:
                    print("  - Solution summary:")
                    for key, stats in summary.items():
                        print(
                            f"    * {key}: {stats['shape']} points, range [{stats['min']:.2f}, {stats['max']:.2f}]",
                        )

            except Exception as e:
                print(f"  - Plotting error: {e}")


def demo_performance_tracking():
    """Demonstrate performance tracking capabilities."""
    print("\n=== Performance Tracking Demo ===")

    optimizer = demo_optimization_system()

    # Run multiple optimizations to build history
    cam_constraints = CamMotionConstraints(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        max_velocity=80.0,
        max_acceleration=40.0,
        max_jerk=8.0,
    )

    print("Running multiple optimizations for performance tracking...")

    for i in range(3):
        print(f"  - Optimization {i + 1}/3")
        result = optimizer.solve_cam_motion_law(
            cam_constraints=cam_constraints,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )
        print(f"    Status: {result.status.value}, Time: {result.solve_time:.3f}s")

    # Get performance summary
    performance = optimizer.get_performance_summary()
    print("\nPerformance Summary:")
    print(f"  - Total optimizations: {performance.get('total_optimizations', 0)}")
    print(
        f"  - Successful optimizations: {performance.get('successful_optimizations', 0)}",
    )
    print(f"  - Success rate: {performance.get('success_rate', 0):.1%}")

    if "avg_solve_time" in performance:
        print(f"  - Average solve time: {performance['avg_solve_time']:.3f} seconds")
        print(f"  - Min solve time: {performance['min_solve_time']:.3f} seconds")
        print(f"  - Max solve time: {performance['max_solve_time']:.3f} seconds")


def main():
    """Run all demonstrations."""
    print("Larrak Library Architecture Demonstration")
    print("=" * 50)

    try:
        # Run all demos
        demo_constraint_system()
        demo_optimization_system()
        demo_motion_law_solving()
        demo_plotting_system()
        demo_performance_tracking()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nThe new library architecture provides:")
        print("  + Modular constraint system")
        print("  + Flexible optimization framework")
        print("  + Advanced plotting capabilities")
        print("  + Performance tracking")
        print("  + Extensible design for future physics simulation")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
