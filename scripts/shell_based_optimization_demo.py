#!/usr/bin/env python3
"""
Demonstration of shell-based optimization system.

This script shows how the secondary and tertiary optimizers work as generic shells
that receive their specific constraints, relationships, and optimization targets
from external sources rather than having hardcoded implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import new library components
from campro.constraints import CamMotionConstraints
from campro.optimization import (
    MotionOptimizer, SecondaryOptimizer, TertiaryOptimizer, 
    CollocationSettings, LinkageParameters
)
from campro.storage import OptimizationRegistry
from campro.utils.plotting import plot_solution

from campro.logging import get_logger

log = get_logger(__name__)


def demo_shell_based_optimization():
    """Demonstrate shell-based optimization with external specifications."""
    print("\n=== Shell-Based Optimization Demo ===")
    
    # Create shared registry for result storage
    registry = OptimizationRegistry()
    
    # Register all three optimizers in the same chain
    registry.register_optimizer("motion_optimizer", "shell_optimization_chain")
    registry.register_optimizer("secondary_optimizer", "shell_optimization_chain")
    registry.register_optimizer("tertiary_optimizer", "shell_optimization_chain")
    
    print("Created optimization registry with shell-based chain:")
    print("  - motion_optimizer (primary - has implementation)")
    print("  - secondary_optimizer (shell - receives external specifications)")
    print("  - tertiary_optimizer (shell - receives external specifications)")
    
    # Create cam constraints
    cam_constraints = CamMotionConstraints(
        stroke=40.0,  # 40mm stroke
        upstroke_duration_percent=75.0,  # 75% upstroke
        max_velocity=180.0,  # 180 mm/s max velocity
        max_acceleration=90.0,  # 90 mm/s² max acceleration
        max_jerk=25.0,  # 25 mm/s³ max jerk
        dwell_at_tdc=True,
        dwell_at_bdc=True
    )
    
    print(f"\nCam constraints:")
    print(f"  - Stroke: {cam_constraints.stroke} mm")
    print(f"  - Upstroke duration: {cam_constraints.upstroke_duration_percent}%")
    print(f"  - Max velocity: {cam_constraints.max_velocity} mm/s")
    
    # Create primary motion optimizer (has implementation)
    primary_settings = CollocationSettings(
        degree=3,
        method="legendre",
        tolerance=1e-6,
        max_iterations=1000,
        verbose=False
    )
    
    primary_optimizer = MotionOptimizer(primary_settings, registry)
    
    print(f"\nCreated primary motion optimizer (with implementation):")
    print(f"  - Method: {primary_settings.method}")
    print(f"  - Degree: {primary_settings.degree}")
    
    # Create secondary optimizer (shell)
    secondary_settings = CollocationSettings(
        degree=4,
        method="radau",
        tolerance=1e-8,
        max_iterations=1500,
        verbose=False
    )
    
    secondary_optimizer = SecondaryOptimizer(
        name="secondary_optimizer",
        registry=registry,
        settings=secondary_settings
    )
    
    print(f"\nCreated secondary optimizer (shell):")
    print(f"  - Method: {secondary_settings.method}")
    print(f"  - Degree: {secondary_settings.degree}")
    print(f"  - No hardcoded implementations - receives external specifications")
    
    # Create tertiary optimizer (shell)
    tertiary_settings = CollocationSettings(
        degree=5,
        method="lobatto",
        tolerance=1e-10,
        max_iterations=2000,
        verbose=False
    )
    
    tertiary_optimizer = TertiaryOptimizer(
        name="tertiary_optimizer",
        registry=registry,
        settings=tertiary_settings
    )
    
    print(f"\nCreated tertiary optimizer (shell):")
    print(f"  - Method: {tertiary_settings.method}")
    print(f"  - Degree: {tertiary_settings.degree}")
    print(f"  - No hardcoded implementations - receives external specifications")
    
    return registry, cam_constraints, primary_optimizer, secondary_optimizer, tertiary_optimizer


def run_primary_optimization(cam_constraints, primary_optimizer):
    """Run primary motion law optimization (has implementation)."""
    print("\n=== Primary Optimization (Has Implementation) ===")
    
    # Solve primary motion law
    print("Running primary motion law optimization...")
    primary_result = primary_optimizer.solve_cam_motion_law(
        cam_constraints=cam_constraints,
        motion_type="minimum_jerk",
        cycle_time=1.0
    )
    
    print(f"Primary optimization results:")
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
                'motion_type': 'minimum_jerk',
                'cycle_time': 1.0,
                'stroke': cam_constraints.stroke,
                'upstroke_duration_percent': cam_constraints.upstroke_duration_percent
            }
        )
        print(f"  - Stored with complete context (constraints, rules, settings)")
    
    return primary_result


def run_secondary_optimization_shell(secondary_optimizer):
    """Run secondary optimization using shell with external specifications."""
    print("\n=== Secondary Optimization (Shell with External Specifications) ===")
    
    # Define external specifications for secondary optimization
    secondary_constraints = {
        'refinement_type': 'smoothness',
        'refinement_factor': 0.2,
        'max_jerk_reduction': 0.5
    }
    
    secondary_relationships = {
        'primary_dependency': 'motion_optimizer',
        'processing_type': 'refinement',
        'blend_factor': 0.6
    }
    
    optimization_targets = {
        'target_jerk_reduction': 0.3,
        'target_smoothness_improvement': 0.4,
        'maintain_stroke': True
    }
    
    # Define processing function (external specification)
    def smoothness_refinement_processor(primary_solution, constraints, relationships, targets, **kwargs):
        """External processing function for smoothness refinement."""
        print(f"  - Processing with external function: smoothness_refinement_processor")
        print(f"  - Constraints: {constraints}")
        print(f"  - Relationships: {relationships}")
        print(f"  - Targets: {targets}")
        
        # Apply smoothing based on external specifications
        processed_solution = primary_solution.copy()
        
        if 'acceleration' in primary_solution:
            # Apply smoothing based on refinement factor
            refinement_factor = constraints.get('refinement_factor', 0.2)
            
            # Simple smoothing (in real implementation, this would be more sophisticated)
            acceleration = primary_solution['acceleration']
            smoothed_acceleration = acceleration * (1 - refinement_factor) + np.roll(acceleration, 1) * refinement_factor
            processed_solution['acceleration'] = smoothed_acceleration
            
            # Recalculate jerk from smoothed acceleration
            if 'time' in primary_solution:
                processed_solution['control'] = np.gradient(smoothed_acceleration, primary_solution['time'])
        
        return processed_solution
    
    # Define objective function (external specification)
    def smoothness_objective(t, x, v, a, u):
        """External objective function for smoothness optimization."""
        return np.trapz(u**2, t)  # Minimize jerk
    
    print("External specifications for secondary optimization:")
    print(f"  - Secondary constraints: {secondary_constraints}")
    print(f"  - Secondary relationships: {secondary_relationships}")
    print(f"  - Optimization targets: {optimization_targets}")
    print(f"  - Processing function: smoothness_refinement_processor")
    print(f"  - Objective function: smoothness_objective")
    
    # Run secondary optimization with external specifications
    try:
        secondary_result = secondary_optimizer.process_primary_result(
            primary_optimizer_id="motion_optimizer",
            secondary_constraints=secondary_constraints,
            secondary_relationships=secondary_relationships,
            optimization_targets=optimization_targets,
            processing_function=smoothness_refinement_processor,
            objective_function=smoothness_objective
        )
        
        print(f"\nSecondary optimization results:")
        print(f"  - Status: {secondary_result.status.value}")
        print(f"  - Successful: {secondary_result.is_successful()}")
        print(f"  - Solve time: {secondary_result.solve_time:.3f} seconds")
        if secondary_result.objective_value is not None:
            print(f"  - Objective value: {secondary_result.objective_value:.6f}")
        
        # Store secondary result
        secondary_optimizer.registry.store_result(
            optimizer_id="secondary_optimizer",
            result_data=secondary_result.solution,
            metadata={
                'objective_value': secondary_result.objective_value,
                'solve_time': secondary_result.solve_time,
                'optimization_type': 'smoothness_refinement'
            },
            constraints=secondary_constraints,
            optimization_rules=secondary_relationships,
            solver_settings={
                'collocation_method': 'radau',
                'collocation_degree': 4,
                'tolerance': 1e-8
            }
        )
        
        return secondary_result
        
    except Exception as e:
        print(f"  - Error: {e}")
        return None


def run_tertiary_optimization_shell(tertiary_optimizer):
    """Run tertiary optimization using shell with external specifications."""
    print("\n=== Tertiary Optimization (Shell with External Specifications) ===")
    
    # Define external specifications for tertiary optimization
    tertiary_constraints = {
        'optimization_type': 'motion_law_tuning',
        'linkage_optimization': True,
        'max_linkage_length': 100.0,
        'min_linkage_length': 30.0
    }
    
    tertiary_relationships = {
        'primary_dependency': 'motion_optimizer',
        'secondary_dependency': 'secondary_optimizer',
        'processing_type': 'combined_optimization',
        'linkage_aware': True
    }
    
    optimization_targets = {
        'target_efficiency_improvement': 0.2,
        'target_linkage_optimization': True,
        'maintain_motion_quality': True
    }
    
    # Define processing function (external specification)
    def combined_optimization_processor(optimization_context, constraints, relationships, targets, **kwargs):
        """External processing function for combined optimization."""
        print(f"  - Processing with external function: combined_optimization_processor")
        print(f"  - Constraints: {constraints}")
        print(f"  - Relationships: {relationships}")
        print(f"  - Targets: {targets}")
        
        # Get primary and secondary results
        primary_results = optimization_context['primary_results']
        secondary_results = optimization_context['secondary_results']
        
        # Start with primary solution
        primary_result = list(primary_results.values())[0]
        primary_solution = primary_result.data
        
        # Blend with secondary solution if available
        if secondary_results:
            secondary_result = list(secondary_results.values())[0]
            secondary_solution = secondary_result.data
            
            # Blend based on external specifications
            blend_factor = relationships.get('blend_factor', 0.5)
            processed_solution = {}
            
            for key in ['position', 'velocity', 'acceleration', 'control']:
                if key in primary_solution and key in secondary_solution:
                    processed_solution[key] = (blend_factor * primary_solution[key] + 
                                             (1 - blend_factor) * secondary_solution[key])
                elif key in primary_solution:
                    processed_solution[key] = primary_solution[key]
        else:
            processed_solution = primary_solution.copy()
        
        # Add linkage optimization if specified
        if constraints.get('linkage_optimization', False):
            # Simple linkage optimization based on motion characteristics
            max_velocity = np.max(np.abs(processed_solution.get('velocity', [0])))
            optimal_linkage_length = 50.0 + (max_velocity / 100.0) * 10.0
            
            # Apply bounds
            optimal_linkage_length = np.clip(optimal_linkage_length, 
                                           constraints['min_linkage_length'], 
                                           constraints['max_linkage_length'])
            
            processed_solution['linkage_parameters'] = {
                'linkage_length': optimal_linkage_length,
                'linkage_angle': 0.0,
                'follower_center_x': 0.0,
                'follower_center_y': optimal_linkage_length,
                'cam_center_x': 0.0,
                'cam_center_y': 0.0,
                'follower_radius': 5.0,
                'follower_offset': 0.0
            }
        
        return processed_solution
    
    # Define objective function (external specification)
    def combined_objective(t, x, v, a, u):
        """External objective function for combined optimization."""
        # Combine smoothness and efficiency objectives
        smoothness_obj = np.trapz(u**2, t)  # Minimize jerk
        efficiency_obj = np.trapz(a**2, t)  # Minimize acceleration
        return 0.6 * smoothness_obj + 0.4 * efficiency_obj
    
    print("External specifications for tertiary optimization:")
    print(f"  - Tertiary constraints: {tertiary_constraints}")
    print(f"  - Tertiary relationships: {tertiary_relationships}")
    print(f"  - Optimization targets: {optimization_targets}")
    print(f"  - Processing function: combined_optimization_processor")
    print(f"  - Objective function: combined_objective")
    
    # Run tertiary optimization with external specifications
    try:
        tertiary_result = tertiary_optimizer.process_optimization_context(
            primary_optimizer_id="motion_optimizer",
            secondary_optimizer_id="secondary_optimizer",
            tertiary_constraints=tertiary_constraints,
            tertiary_relationships=tertiary_relationships,
            optimization_targets=optimization_targets,
            processing_function=combined_optimization_processor,
            objective_function=combined_objective
        )
        
        print(f"\nTertiary optimization results:")
        print(f"  - Status: {tertiary_result.status.value}")
        print(f"  - Successful: {tertiary_result.is_successful()}")
        print(f"  - Solve time: {tertiary_result.solve_time:.3f} seconds")
        if tertiary_result.objective_value is not None:
            print(f"  - Objective value: {tertiary_result.objective_value:.6f}")
        
        # Show linkage optimization results
        if 'linkage_parameters' in tertiary_result.solution:
            linkage_params = tertiary_result.solution['linkage_parameters']
            print(f"  - Optimized linkage length: {linkage_params['linkage_length']:.2f} mm")
            print(f"  - Optimized linkage angle: {linkage_params['linkage_angle']:.2f}°")
        
        return tertiary_result
        
    except Exception as e:
        print(f"  - Error: {e}")
        return None


def compare_shell_results(primary_result, secondary_result, tertiary_result):
    """Compare results from shell-based optimization."""
    print("\n=== Shell-Based Result Comparison ===")
    
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
        print(f"\nSecondary optimization solution (external processing):")
        secondary_summary = secondary_result.get_solution_summary()
        for key, stats in secondary_summary.items():
            if key in primary_summary:
                primary_stats = primary_summary[key]
                print(f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}")
                
                # Compare with primary
                range_change = ((stats['max'] - stats['min']) - (primary_stats['max'] - primary_stats['min'])) / (primary_stats['max'] - primary_stats['min']) * 100
                mean_change = (stats['mean'] - primary_stats['mean']) / abs(primary_stats['mean']) * 100 if primary_stats['mean'] != 0 else 0
                
                print(f"    Range change: {range_change:+.1f}%, Mean change: {mean_change:+.1f}%")
    
    # Compare with tertiary results
    if tertiary_result and tertiary_result.is_successful():
        print(f"\nTertiary optimization solution (external processing):")
        tertiary_summary = tertiary_result.get_solution_summary()
        for key, stats in tertiary_summary.items():
            if key in primary_summary:
                primary_stats = primary_summary[key]
                print(f"  - {key}: range [{stats['min']:.2f}, {stats['max']:.2f}], mean {stats['mean']:.2f}")
                
                # Compare with primary
                range_change = ((stats['max'] - stats['min']) - (primary_stats['max'] - primary_stats['min'])) / (primary_stats['max'] - primary_stats['min']) * 100
                mean_change = (stats['mean'] - primary_stats['mean']) / abs(primary_stats['mean']) * 100 if primary_stats['mean'] != 0 else 0
                
                print(f"    Range change: {range_change:+.1f}%, Mean change: {mean_change:+.1f}%")


def create_shell_plots(primary_result, secondary_result, tertiary_result):
    """Create comparison plots for shell-based optimization."""
    print("\n=== Creating Shell-Based Comparison Plots ===")
    
    if not primary_result.is_successful():
        print("Primary optimization failed, cannot create plots")
        return
    
    # Create output directory
    output_dir = Path("plots/shell_based_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot primary result
    try:
        fig = plot_solution(
            solution=primary_result.solution,
            title="Primary Motion Law Optimization (Implementation)",
            use_cam_angle=True
        )
        primary_plot_path = output_dir / "primary_implementation.png"
        fig.savefig(primary_plot_path, dpi=300, bbox_inches='tight')
        print(f"Primary plot saved to: {primary_plot_path}")
    except Exception as e:
        print(f"Error creating primary plot: {e}")
    
    # Plot secondary result
    if secondary_result and secondary_result.is_successful():
        try:
            fig = plot_solution(
                solution=secondary_result.solution,
                title="Secondary Optimization (Shell with External Specifications)",
                use_cam_angle=True
            )
            secondary_plot_path = output_dir / "secondary_shell.png"
            fig.savefig(secondary_plot_path, dpi=300, bbox_inches='tight')
            print(f"Secondary plot saved to: {secondary_plot_path}")
        except Exception as e:
            print(f"Error creating secondary plot: {e}")
    
    # Plot tertiary result
    if tertiary_result and tertiary_result.is_successful():
        try:
            fig = plot_solution(
                solution=tertiary_result.solution,
                title="Tertiary Optimization (Shell with External Specifications)",
                use_cam_angle=True
            )
            tertiary_plot_path = output_dir / "tertiary_shell.png"
            fig.savefig(tertiary_plot_path, dpi=300, bbox_inches='tight')
            print(f"Tertiary plot saved to: {tertiary_plot_path}")
        except Exception as e:
            print(f"Error creating tertiary plot: {e}")


def main():
    """Run the shell-based optimization demonstration."""
    print("Shell-Based Optimization System Demonstration")
    print("=" * 60)
    
    try:
        # Setup shell-based optimization system
        registry, cam_constraints, primary_optimizer, secondary_optimizer, tertiary_optimizer = demo_shell_based_optimization()
        
        # Run primary optimization (has implementation)
        primary_result = run_primary_optimization(cam_constraints, primary_optimizer)
        
        # Run secondary optimization (shell with external specifications)
        secondary_result = run_secondary_optimization_shell(secondary_optimizer)
        
        # Run tertiary optimization (shell with external specifications)
        tertiary_result = run_tertiary_optimization_shell(tertiary_optimizer)
        
        # Compare results
        compare_shell_results(primary_result, secondary_result, tertiary_result)
        
        # Create comparison plots
        create_shell_plots(primary_result, secondary_result, tertiary_result)
        
        print("\n" + "=" * 60)
        print("Shell-based optimization demonstration completed successfully!")
        print("\nThe shell-based optimization system provides:")
        print("  + Primary optimizer with implementation")
        print("  + Secondary optimizer as generic shell")
        print("  + Tertiary optimizer as generic shell")
        print("  + External specifications for constraints, relationships, and targets")
        print("  + External processing functions for optimization logic")
        print("  + External objective functions for optimization goals")
        print("  + Complete context visibility for all optimization layers")
        print("  + Modular design ready for future specific implementations")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


