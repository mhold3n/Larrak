#!/usr/bin/env python3
"""
Demonstration of cascaded optimization system.

This script shows how the primary motion law optimizer and secondary
collocation optimizer work together, with the secondary optimizer
accessing and using results from the primary optimizer.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import new library components
from campro.constraints import CamMotionConstraints
from campro.logging import get_logger
from campro.optimization import CollocationSettings, MotionOptimizer, SecondaryOptimizer
from campro.storage import OptimizationRegistry
from campro.utils.plotting import plot_solution

log = get_logger(__name__)


def demo_cascaded_optimization():
    """Demonstrate cascaded optimization with result sharing."""
    print("\n=== Cascaded Optimization Demo ===")

    # Create shared registry for result storage
    registry = OptimizationRegistry()

    # Register optimizers in the same chain
    registry.register_optimizer("motion_optimizer", "cam_optimization_chain")
    registry.register_optimizer("secondary_optimizer", "cam_optimization_chain")

    print("Created optimization registry with chain:")
    print("  - motion_optimizer (primary)")
    print("  - secondary_optimizer (secondary)")

    # Create cam constraints
    cam_constraints = CamMotionConstraints(
        stroke=30.0,  # 30mm stroke
        upstroke_duration_percent=65.0,  # 65% upstroke
        max_velocity=120.0,  # 120 mm/s max velocity
        max_acceleration=60.0,  # 60 mm/s² max acceleration
        max_jerk=15.0,  # 15 mm/s³ max jerk
        dwell_at_tdc=True,
        dwell_at_bdc=True,
    )

    print("\nCam constraints:")
    print(f"  - Stroke: {cam_constraints.stroke} mm")
    print(f"  - Upstroke duration: {cam_constraints.upstroke_duration_percent}%")
    print(f"  - Max velocity: {cam_constraints.max_velocity} mm/s")

    # Create primary motion optimizer
    primary_settings = CollocationSettings(
        degree=3,
        method="legendre",
        tolerance=1e-6,
        max_iterations=1000,
        verbose=False,
    )

    primary_optimizer = MotionOptimizer(primary_settings)
    primary_optimizer.registry = registry  # Connect to registry

    print("\nCreated primary motion optimizer:")
    print(f"  - Method: {primary_settings.method}")
    print(f"  - Degree: {primary_settings.degree}")

    # Create secondary optimizer
    secondary_settings = CollocationSettings(
        degree=4,  # Higher degree for refinement
        method="radau",  # Different method
        tolerance=1e-8,  # Tighter tolerance
        max_iterations=1500,
        verbose=False,
    )

    secondary_optimizer = SecondaryOptimizer(
        name="secondary_optimizer",
        registry=registry,
        settings=secondary_settings,
    )

    print("\nCreated secondary optimizer:")
    print(f"  - Method: {secondary_settings.method}")
    print(f"  - Degree: {secondary_settings.degree}")
    print(f"  - Tighter tolerance: {secondary_settings.tolerance}")

    return registry, cam_constraints, primary_optimizer, secondary_optimizer


def run_primary_optimization(cam_constraints, primary_optimizer):
    """Run primary motion law optimization."""
    print("\n=== Primary Optimization ===")

    # Solve primary motion law
    print("Running primary motion law optimization...")
    primary_result = primary_optimizer.solve_cam_motion_law(
        cam_constraints=cam_constraints,
        motion_type="minimum_jerk",
        cycle_time=1.0,
    )

    print("Primary optimization results:")
    print(f"  - Status: {primary_result.status.value}")
    print(f"  - Successful: {primary_result.is_successful()}")
    print(f"  - Solve time: {primary_result.solve_time:.3f} seconds")

    if primary_result.objective_value is not None:
        print(f"  - Objective value: {primary_result.objective_value:.6f}")

    # Store result in registry
    if primary_result.is_successful():
        storage_result = primary_optimizer.registry.store_result(
            optimizer_id="motion_optimizer",
            result_data=primary_result.solution,
            metadata={
                "objective_value": primary_result.objective_value,
                "solve_time": primary_result.solve_time,
                "motion_type": "minimum_jerk",
                "constraints": cam_constraints.to_dict(),
            },
            expires_in=3600,  # Expire in 1 hour
        )
        print(f"  - Stored in registry with key: {storage_result.storage_id}")

    return primary_result


def run_secondary_optimization(secondary_optimizer):
    """Run secondary optimization using primary results."""
    print("\n=== Secondary Optimization ===")

    # Check available primary results
    available_results = secondary_optimizer.get_available_primary_results()
    print(f"Available primary results: {list(available_results.keys())}")

    if not available_results:
        print("No primary results available for secondary optimization")
        return None

    # Run different types of secondary optimization
    secondary_results = {}

    # 1. Motion law refinement
    print("\n1. Motion Law Refinement (Smoothness)")
    try:
        refinement_result = secondary_optimizer.refine_motion_law(
            primary_optimizer_id="motion_optimizer",
            refinement_type="smoothness",
            refinement_factor=0.2,
        )

        print(f"  - Status: {refinement_result.status.value}")
        print(f"  - Solve time: {refinement_result.solve_time:.3f} seconds")
        if refinement_result.objective_value is not None:
            print(f"  - Objective value: {refinement_result.objective_value:.6f}")

        secondary_results["refinement"] = refinement_result

        # Store refined result
        secondary_optimizer.registry.store_result(
            optimizer_id="secondary_optimizer",
            result_data=refinement_result.solution,
            metadata={
                "objective_value": refinement_result.objective_value,
                "solve_time": refinement_result.solve_time,
                "optimization_type": "refinement",
                "refinement_type": "smoothness",
            },
        )

    except Exception as e:
        print(f"  - Error: {e}")

    # 2. Multi-objective optimization
    print("\n2. Multi-Objective Optimization")
    try:
        multi_obj_result = secondary_optimizer.multi_objective_optimization(
            primary_optimizer_id="motion_optimizer",
            objectives=[("smoothness", 0.4), ("efficiency", 0.3), ("accuracy", 0.3)],
        )

        print(f"  - Status: {multi_obj_result.status.value}")
        print(f"  - Solve time: {multi_obj_result.solve_time:.3f} seconds")
        if multi_obj_result.objective_value is not None:
            print(f"  - Objective value: {multi_obj_result.objective_value:.6f}")

        secondary_results["multi_objective"] = multi_obj_result

    except Exception as e:
        print(f"  - Error: {e}")

    # 3. Constraint tightening
    print("\n3. Constraint Tightening")
    try:
        tightening_result = secondary_optimizer.constraint_tightening(
            primary_optimizer_id="motion_optimizer",
            tightening_factor=0.1,
        )

        print(f"  - Status: {tightening_result.status.value}")
        print(f"  - Solve time: {tightening_result.solve_time:.3f} seconds")
        if tightening_result.objective_value is not None:
            print(f"  - Objective value: {tightening_result.objective_value:.6f}")

        secondary_results["constraint_tightening"] = tightening_result

    except Exception as e:
        print(f"  - Error: {e}")

    return secondary_results


def compare_results(primary_result, secondary_results):
    """Compare primary and secondary optimization results."""
    print("\n=== Result Comparison ===")

    if not primary_result.is_successful():
        print("Primary optimization failed, cannot compare results")
        return

    # Get primary solution summary
    primary_summary = primary_result.get_solution_summary()
    print("Primary optimization solution:")
    for key, stats in primary_summary.items():
        print(
            f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}",
        )

    # Compare with secondary results
    for result_type, result in secondary_results.items():
        if result and result.is_successful():
            print(f"\n{result_type.replace('_', ' ').title()} solution:")
            secondary_summary = result.get_solution_summary()
            for key, stats in secondary_summary.items():
                if key in primary_summary:
                    primary_stats = primary_summary[key]
                    print(
                        f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}",
                    )

                    # Compare with primary
                    range_change = (
                        (
                            (stats["max"] - stats["min"])
                            - (primary_stats["max"] - primary_stats["min"])
                        )
                        / (primary_stats["max"] - primary_stats["min"])
                        * 100
                    )
                    mean_change = (
                        (stats["mean"] - primary_stats["mean"])
                        / abs(primary_stats["mean"])
                        * 100
                        if primary_stats["mean"] != 0
                        else 0
                    )

                    print(
                        f"    Range change: {range_change:+.1f}%, Mean change: {mean_change:+.1f}%",
                    )


def create_comparison_plots(primary_result, secondary_results):
    """Create comparison plots of primary and secondary results."""
    print("\n=== Creating Comparison Plots ===")

    if not primary_result.is_successful():
        print("Primary optimization failed, cannot create plots")
        return

    # Create output directory
    output_dir = Path("plots/cascaded_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot primary result
    try:
        fig = plot_solution(
            solution=primary_result.solution,
            title="Primary Motion Law Optimization",
            use_cam_angle=True,
        )
        primary_plot_path = output_dir / "primary_optimization.png"
        fig.savefig(primary_plot_path, dpi=300, bbox_inches="tight")
        print(f"Primary plot saved to: {primary_plot_path}")
    except Exception as e:
        print(f"Error creating primary plot: {e}")

    # Plot secondary results
    for result_type, result in secondary_results.items():
        if result and result.is_successful():
            try:
                fig = plot_solution(
                    solution=result.solution,
                    title=f"Secondary Optimization: {result_type.replace('_', ' ').title()}",
                    use_cam_angle=True,
                )
                secondary_plot_path = output_dir / f"secondary_{result_type}.png"
                fig.savefig(secondary_plot_path, dpi=300, bbox_inches="tight")
                print(f"Secondary plot saved to: {secondary_plot_path}")
            except Exception as e:
                print(f"Error creating {result_type} plot: {e}")


def demo_registry_functionality(registry):
    """Demonstrate registry functionality."""
    print("\n=== Registry Functionality Demo ===")

    # Get registry statistics
    stats = registry.get_registry_stats()
    print("Registry statistics:")
    print(f"  - Total chains: {stats['total_chains']}")
    print(f"  - Total optimizers: {stats['total_optimizers']}")
    print(f"  - Storage entries: {stats['storage_stats']['total_entries']}")
    print(f"  - Accessible entries: {stats['storage_stats']['accessible_entries']}")

    # List all stored keys
    all_keys = registry.storage.list_keys()
    print(f"  - All stored keys: {all_keys}")

    # List accessible keys
    accessible_keys = registry.storage.list_accessible_keys()
    print(f"  - Accessible keys: {accessible_keys}")

    # Get chain results
    chain_results = registry.get_chain_results("cam_optimization_chain")
    print(f"  - Chain results: {list(chain_results.keys())}")


def main():
    """Run the cascaded optimization demonstration."""
    print("Cascaded Optimization System Demonstration")
    print("=" * 50)

    try:
        # Setup cascaded optimization system
        registry, cam_constraints, primary_optimizer, secondary_optimizer = (
            demo_cascaded_optimization()
        )

        # Run primary optimization
        primary_result = run_primary_optimization(cam_constraints, primary_optimizer)

        # Run secondary optimization
        secondary_results = run_secondary_optimization(secondary_optimizer)

        # Compare results
        compare_results(primary_result, secondary_results)

        # Create comparison plots
        create_comparison_plots(primary_result, secondary_results)

        # Demonstrate registry functionality
        demo_registry_functionality(registry)

        print("\n" + "=" * 50)
        print("Cascaded optimization demonstration completed successfully!")
        print("\nThe cascaded optimization system provides:")
        print("  + Result sharing between optimizers")
        print("  + Secondary optimization using primary results")
        print(
            "  + Multiple optimization strategies (refinement, multi-objective, constraint tightening)",
        )
        print("  + Centralized result storage and management")
        print("  + Performance tracking across optimization chains")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
