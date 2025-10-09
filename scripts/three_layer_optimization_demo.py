#!/usr/bin/env python3
"""
Demonstration of three-layer optimization system with full context visibility.

This script shows how the primary, secondary, and tertiary optimizers work together
with complete visibility into constraints, rules, and results from all previous layers.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import new library components
from campro.constraints import CamMotionConstraints
from campro.logging import get_logger
from campro.optimization import (
    CollocationSettings,
    LinkageParameters,
    MotionOptimizer,
    SecondaryOptimizer,
    TertiaryOptimizer,
)
from campro.storage import OptimizationRegistry
from campro.utils.plotting import plot_solution

log = get_logger(__name__)


def demo_three_layer_optimization():
    """Demonstrate three-layer optimization with full context visibility."""
    print("\n=== Three-Layer Optimization Demo ===")

    # Create shared registry for result storage
    registry = OptimizationRegistry()

    # Register all three optimizers in the same chain
    registry.register_optimizer("motion_optimizer", "three_layer_chain")
    registry.register_optimizer("secondary_optimizer", "three_layer_chain")
    registry.register_optimizer("tertiary_optimizer", "three_layer_chain")

    print("Created optimization registry with three-layer chain:")
    print("  - motion_optimizer (primary)")
    print("  - secondary_optimizer (secondary)")
    print("  - tertiary_optimizer (tertiary)")

    # Create cam constraints
    cam_constraints = CamMotionConstraints(
        stroke=35.0,  # 35mm stroke
        upstroke_duration_percent=70.0,  # 70% upstroke
        max_velocity=150.0,  # 150 mm/s max velocity
        max_acceleration=75.0,  # 75 mm/s² max acceleration
        max_jerk=20.0,  # 20 mm/s³ max jerk
        dwell_at_tdc=True,
        dwell_at_bdc=True,
    )

    print("\nCam constraints:")
    print(f"  - Stroke: {cam_constraints.stroke} mm")
    print(f"  - Upstroke duration: {cam_constraints.upstroke_duration_percent}%")
    print(f"  - Max velocity: {cam_constraints.max_velocity} mm/s")
    print(f"  - Max acceleration: {cam_constraints.max_acceleration} mm/s²")
    print(f"  - Max jerk: {cam_constraints.max_jerk} mm/s³")

    # Create primary motion optimizer
    primary_settings = CollocationSettings(
        degree=3,
        method="legendre",
        tolerance=1e-6,
        max_iterations=1000,
        verbose=False,
    )

    primary_optimizer = MotionOptimizer(primary_settings, registry)

    print("\nCreated primary motion optimizer:")
    print(f"  - Method: {primary_settings.method}")
    print(f"  - Degree: {primary_settings.degree}")
    print(f"  - Tolerance: {primary_settings.tolerance}")

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

    # Create tertiary optimizer
    tertiary_settings = CollocationSettings(
        degree=5,  # Highest degree for final optimization
        method="lobatto",  # Different method
        tolerance=1e-10,  # Tightest tolerance
        max_iterations=2000,
        verbose=False,
    )

    # Create initial linkage parameters
    initial_linkage = LinkageParameters(
        cam_center_x=0.0,
        cam_center_y=0.0,
        follower_center_x=0.0,
        follower_center_y=50.0,
        linkage_length=50.0,
        linkage_angle=0.0,
        follower_radius=5.0,
        follower_offset=0.0,
    )

    tertiary_optimizer = TertiaryOptimizer(
        name="tertiary_optimizer",
        registry=registry,
        settings=tertiary_settings,
    )
    tertiary_optimizer.configure(linkage_parameters=initial_linkage)

    print("\nCreated tertiary optimizer:")
    print(f"  - Method: {tertiary_settings.method}")
    print(f"  - Degree: {tertiary_settings.degree}")
    print(f"  - Tightest tolerance: {tertiary_settings.tolerance}")
    print(f"  - Initial linkage length: {initial_linkage.linkage_length} mm")
    print(f"  - Initial linkage angle: {initial_linkage.linkage_angle}°")

    return registry, cam_constraints, primary_optimizer, secondary_optimizer, tertiary_optimizer


def run_primary_optimization(cam_constraints, primary_optimizer):
    """Run primary motion law optimization with complete context storage."""
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

    # Store result with complete context
    if primary_result.is_successful():
        primary_optimizer.store_result(
            result=primary_result,
            optimizer_id="motion_optimizer",
            constraints=cam_constraints.to_dict(),
            optimization_rules={
                "motion_type": "minimum_jerk",
                "cycle_time": 1.0,
                "stroke": cam_constraints.stroke,
                "upstroke_duration_percent": cam_constraints.upstroke_duration_percent,
            },
        )
        print("  - Stored with complete context (constraints, rules, settings)")

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

    # Run motion law refinement
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

        # Store secondary result
        secondary_optimizer.registry.store_result(
            optimizer_id="secondary_optimizer",
            result_data=refinement_result.solution,
            metadata={
                "objective_value": refinement_result.objective_value,
                "solve_time": refinement_result.solve_time,
                "optimization_type": "refinement",
                "refinement_type": "smoothness",
            },
            constraints=None,  # Uses primary constraints
            optimization_rules={
                "refinement_type": "smoothness",
                "refinement_factor": 0.2,
                "primary_optimizer_id": "motion_optimizer",
            },
            solver_settings={
                "collocation_method": "radau",
                "collocation_degree": 4,
                "tolerance": 1e-8,
            },
        )

        return refinement_result

    except Exception as e:
        print(f"  - Error: {e}")
        return None


def run_tertiary_optimization(tertiary_optimizer):
    """Run tertiary optimization with full context visibility."""
    print("\n=== Tertiary Optimization ===")

    # Get complete optimization context
    try:
        context = tertiary_optimizer.get_complete_context(
            primary_optimizer_id="motion_optimizer",
            secondary_optimizer_id="secondary_optimizer",
        )

        print("Complete optimization context available:")
        print(f"  - Primary results: {list(context['primary_results'].keys())}")
        print(f"  - Secondary results: {list(context['secondary_results'].keys())}")
        print(f"  - Primary constraints: {'Available' if context['primary_constraints'] else 'None'}")
        print(f"  - Primary rules: {'Available' if context['primary_rules'] else 'None'}")
        print(f"  - Primary settings: {'Available' if context['primary_settings'] else 'None'}")
        print(f"  - Secondary constraints: {'Available' if context['secondary_constraints'] else 'None'}")
        print(f"  - Secondary rules: {'Available' if context['secondary_rules'] else 'None'}")
        print(f"  - Secondary settings: {'Available' if context['secondary_settings'] else 'None'}")
        print(f"  - Linkage parameters: {context['linkage_parameters']}")

    except Exception as e:
        print(f"Error getting context: {e}")
        return None

    # Run different types of tertiary optimization
    tertiary_results = {}

    # 1. Motion Law Tuning
    print("\n1. Motion Law Tuning")
    try:
        tuning_result = tertiary_optimizer.tune_motion_law(
            primary_optimizer_id="motion_optimizer",
            secondary_optimizer_id="secondary_optimizer",
            tuning_parameters={
                "smoothness": 0.4,
                "efficiency": 0.3,
                "accuracy": 0.3,
                "blend_factor": 0.6,
            },
        )

        print(f"  - Status: {tuning_result.status.value}")
        print(f"  - Solve time: {tuning_result.solve_time:.3f} seconds")
        if tuning_result.objective_value is not None:
            print(f"  - Objective value: {tuning_result.objective_value:.6f}")

        tertiary_results["motion_law_tuning"] = tuning_result

    except Exception as e:
        print(f"  - Error: {e}")

    # 2. Linkage Placement Optimization
    print("\n2. Linkage Placement Optimization")
    try:
        linkage_result = tertiary_optimizer.optimize_linkage_placement(
            primary_optimizer_id="motion_optimizer",
            secondary_optimizer_id="secondary_optimizer",
            linkage_bounds={
                "linkage_length": (30.0, 100.0),
                "linkage_angle": (-45.0, 45.0),
            },
        )

        print(f"  - Status: {linkage_result.status.value}")
        print(f"  - Solve time: {linkage_result.solve_time:.3f} seconds")
        if linkage_result.objective_value is not None:
            print(f"  - Objective value: {linkage_result.objective_value:.6f}")

        # Show optimized linkage parameters
        if "linkage_parameters" in linkage_result.solution:
            linkage_params = linkage_result.solution["linkage_parameters"]
            print(f"  - Optimized linkage length: {linkage_params['linkage_length']:.2f} mm")
            print(f"  - Optimized linkage angle: {linkage_params['linkage_angle']:.2f}°")
            print(f"  - Follower center position: ({linkage_params['follower_center_x']:.2f}, {linkage_params['follower_center_y']:.2f})")

        tertiary_results["linkage_placement"] = linkage_result

    except Exception as e:
        print(f"  - Error: {e}")

    # 3. Combined Optimization
    print("\n3. Combined Motion Law and Linkage Optimization")
    try:
        combined_result = tertiary_optimizer.combined_optimization(
            primary_optimizer_id="motion_optimizer",
            secondary_optimizer_id="secondary_optimizer",
            optimization_weights={
                "motion": 0.6,
                "linkage": 0.4,
            },
        )

        print(f"  - Status: {combined_result.status.value}")
        print(f"  - Solve time: {combined_result.solve_time:.3f} seconds")
        if combined_result.objective_value is not None:
            print(f"  - Objective value: {combined_result.objective_value:.6f}")

        tertiary_results["combined_optimization"] = combined_result

    except Exception as e:
        print(f"  - Error: {e}")

    return tertiary_results


def compare_three_layer_results(primary_result, secondary_result, tertiary_results):
    """Compare results across all three optimization layers."""
    print("\n=== Three-Layer Result Comparison ===")

    if not primary_result.is_successful():
        print("Primary optimization failed, cannot compare results")
        return

    # Get primary solution summary
    primary_summary = primary_result.get_solution_summary()
    print("Primary optimization solution:")
    for key, stats in primary_summary.items():
        print(f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}")

    # Compare with secondary results
    if secondary_result and secondary_result.is_successful():
        print("\nSecondary optimization solution:")
        secondary_summary = secondary_result.get_solution_summary()
        for key, stats in secondary_summary.items():
            if key in primary_summary:
                primary_stats = primary_summary[key]
                print(f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}")

                # Compare with primary
                range_change = ((stats["max"] - stats["min"]) - (primary_stats["max"] - primary_stats["min"])) / (primary_stats["max"] - primary_stats["min"]) * 100
                mean_change = (stats["mean"] - primary_stats["mean"]) / abs(primary_stats["mean"]) * 100 if primary_stats["mean"] != 0 else 0

                print(f"    Range change: {range_change:+.1f}%, Mean change: {mean_change:+.1f}%")

    # Compare with tertiary results
    for result_type, result in tertiary_results.items():
        if result and result.is_successful():
            print(f"\nTertiary {result_type.replace('_', ' ').title()} solution:")
            tertiary_summary = result.get_solution_summary()
            for key, stats in tertiary_summary.items():
                if key in primary_summary:
                    primary_stats = primary_summary[key]
                    print(f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}")

                    # Compare with primary
                    range_change = ((stats["max"] - stats["min"]) - (primary_stats["max"] - primary_stats["min"])) / (primary_stats["max"] - primary_stats["min"]) * 100
                    mean_change = (stats["mean"] - primary_stats["mean"]) / abs(primary_stats["mean"]) * 100 if primary_stats["mean"] != 0 else 0

                    print(f"    Range change: {range_change:+.1f}%, Mean change: {mean_change:+.1f}%")


def create_three_layer_plots(primary_result, secondary_result, tertiary_results):
    """Create comparison plots for all three optimization layers."""
    print("\n=== Creating Three-Layer Comparison Plots ===")

    if not primary_result.is_successful():
        print("Primary optimization failed, cannot create plots")
        return

    # Create output directory
    output_dir = Path("plots/three_layer_optimization")
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

    # Plot secondary result
    if secondary_result and secondary_result.is_successful():
        try:
            fig = plot_solution(
                solution=secondary_result.solution,
                title="Secondary Motion Law Refinement",
                use_cam_angle=True,
            )
            secondary_plot_path = output_dir / "secondary_refinement.png"
            fig.savefig(secondary_plot_path, dpi=300, bbox_inches="tight")
            print(f"Secondary plot saved to: {secondary_plot_path}")
        except Exception as e:
            print(f"Error creating secondary plot: {e}")

    # Plot tertiary results
    for result_type, result in tertiary_results.items():
        if result and result.is_successful():
            try:
                fig = plot_solution(
                    solution=result.solution,
                    title=f"Tertiary Optimization: {result_type.replace('_', ' ').title()}",
                    use_cam_angle=True,
                )
                tertiary_plot_path = output_dir / f"tertiary_{result_type}.png"
                fig.savefig(tertiary_plot_path, dpi=300, bbox_inches="tight")
                print(f"Tertiary plot saved to: {tertiary_plot_path}")
            except Exception as e:
                print(f"Error creating {result_type} plot: {e}")


def demo_registry_functionality(registry):
    """Demonstrate registry functionality with three layers."""
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
    chain_results = registry.get_chain_results("three_layer_chain")
    print(f"  - Chain results: {list(chain_results.keys())}")

    # Show context visibility for each optimizer
    print("\nContext visibility:")
    for optimizer_id in ["motion_optimizer", "secondary_optimizer", "tertiary_optimizer"]:
        available_results = registry.get_available_results(optimizer_id)
        print(f"  - {optimizer_id}: can access {list(available_results.keys())}")


def main():
    """Run the three-layer optimization demonstration."""
    print("Three-Layer Optimization System Demonstration")
    print("=" * 60)

    try:
        # Setup three-layer optimization system
        registry, cam_constraints, primary_optimizer, secondary_optimizer, tertiary_optimizer = demo_three_layer_optimization()

        # Run primary optimization
        primary_result = run_primary_optimization(cam_constraints, primary_optimizer)

        # Run secondary optimization
        secondary_result = run_secondary_optimization(secondary_optimizer)

        # Run tertiary optimization
        tertiary_results = run_tertiary_optimization(tertiary_optimizer)

        # Compare results
        compare_three_layer_results(primary_result, secondary_result, tertiary_results)

        # Create comparison plots
        create_three_layer_plots(primary_result, secondary_result, tertiary_results)

        # Demonstrate registry functionality
        demo_registry_functionality(registry)

        print("\n" + "=" * 60)
        print("Three-layer optimization demonstration completed successfully!")
        print("\nThe three-layer optimization system provides:")
        print("  + Complete context visibility (results, constraints, rules)")
        print("  + Motion law tuning with full optimization history")
        print("  + Linkage placement optimization")
        print("  + Combined motion law and linkage optimization")
        print("  + Robust cascaded optimization with full picture access")
        print("  + Centralized result storage and management")
        print("  + Performance tracking across all optimization layers")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


