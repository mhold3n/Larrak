"""
Usage Examples for Motion Law Optimization Library

This module provides comprehensive examples of how to use the motion law
optimization library for different scenarios and use cases.
"""

from __future__ import annotations

import time
from pathlib import Path

from campro.freepiston.opt.config_factory import (
    ConfigFactory,
    create_engine_config,
    create_optimization_scenario,
)
from campro.freepiston.opt.optimization_lib import (
    create_adaptive_optimizer,
    create_robust_optimizer,
    create_standard_optimizer,
    quick_optimize,
)
from campro.logging import get_logger

log = get_logger(__name__)


def example_basic_optimization():
    """Example 1: Basic optimization with default settings."""
    print("=" * 60)
    print("Example 1: Basic Optimization")
    print("=" * 60)

    # Create default configuration
    config = ConfigFactory.create_default_config()

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Run optimization
    start_time = time.time()
    result = optimizer.optimize()
    elapsed_time = time.time() - start_time

    # Print results
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Iterations: {result.iterations}")
    print(f"CPU time: {result.cpu_time:.2f}s")

    if result.warnings:
        print(f"Warnings: {result.warnings}")

    if result.errors:
        print(f"Errors: {result.errors}")

    return result


def example_custom_configuration():
    """Example 2: Custom configuration with specific parameters."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    config = ConfigFactory.create_custom_config(
        geometry={
            "bore": 0.12,  # Larger bore
            "stroke": 0.08,  # Shorter stroke
            "compression_ratio": 15.0,  # Higher compression
            "mass": 1.5,  # Heavier piston
        },
        bounds={
            "xL_min": -0.05, "xL_max": 0.05,
            "xR_min": 0.05, "xR_max": 0.15,
            "v_max": 40.0,  # Lower max velocity
        },
        num={"K": 30, "C": 4},  # Higher resolution
        objective={
            "method": "thermal_efficiency",
            "w": {
                "smooth": 0.02,
                "short_circuit": 1.5,
                "eta_th": 0.8,
            },
        },
    )

    # Create optimizer with custom config
    optimizer = create_standard_optimizer(config)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Performance metrics: {result.performance_metrics}")

    return result


def example_1d_gas_model():
    """Example 3: 1D gas model optimization."""
    print("=" * 60)
    print("Example 3: 1D Gas Model Optimization")
    print("=" * 60)

    # Create 1D configuration
    config = ConfigFactory.create_1d_config(n_cells=30)

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print(f"Success: {result.success}")
    print(f"Model type: {config.model_type}")
    print(f"Number of cells: {config.n_cells}")
    print(f"Objective value: {result.objective_value:.6e}")

    return result


def example_robust_optimization():
    """Example 4: Robust optimization with conservative settings."""
    print("=" * 60)
    print("Example 4: Robust Optimization")
    print("=" * 60)

    # Create robust configuration
    config = ConfigFactory.create_robust_config()

    # Create robust optimizer
    optimizer = create_robust_optimizer(config)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print(f"Success: {result.success}")
    print(f"Refinement strategy: {config.refinement_strategy}")
    print(f"Max refinements: {config.max_refinements}")
    print(f"Objective value: {result.objective_value:.6e}")

    return result


def example_adaptive_optimization():
    """Example 5: Adaptive optimization with refinement."""
    print("=" * 60)
    print("Example 5: Adaptive Optimization")
    print("=" * 60)

    # Create default configuration
    config = ConfigFactory.create_default_config()

    # Create adaptive optimizer
    optimizer = create_adaptive_optimizer(config, max_refinements=2)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Validation metrics: {result.validation_metrics}")

    return result


def example_engine_specific():
    """Example 6: Engine-specific optimization."""
    print("=" * 60)
    print("Example 6: Engine-Specific Optimization")
    print("=" * 60)

    # Create opposed piston engine configuration
    config = create_engine_config("opposed_piston")

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print("Engine type: Opposed Piston")
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Geometry: {config.geometry}")

    return result


def example_scenario_optimization():
    """Example 7: Scenario-based optimization."""
    print("=" * 60)
    print("Example 7: Scenario-Based Optimization")
    print("=" * 60)

    # Create efficiency-focused configuration
    config = create_optimization_scenario("efficiency")

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Run optimization
    result = optimizer.optimize()

    # Print results
    print("Scenario: Efficiency")
    print(f"Success: {result.success}")
    print(f"Objective method: {config.objective['method']}")
    print(f"Objective value: {result.objective_value:.6e}")

    return result


def example_quick_optimization():
    """Example 8: Quick optimization for testing."""
    print("=" * 60)
    print("Example 8: Quick Optimization")
    print("=" * 60)

    # Create quick test configuration
    config = ConfigFactory.create_quick_test_config()

    # Use quick optimize function
    result = quick_optimize(config, backend="standard")

    # Print results
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"CPU time: {result.cpu_time:.2f}s")

    return result


def example_problem_builder():
    """Example 9: Using the problem builder for configuration."""
    print("=" * 60)
    print("Example 9: Problem Builder")
    print("=" * 60)

    # Create default configuration
    config = ConfigFactory.create_default_config()

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Use problem builder to modify configuration
    builder = optimizer.get_problem_builder()
    builder.with_geometry({
        "bore": 0.15,
        "stroke": 0.1,
        "compression_ratio": 12.0,
    }).with_bounds({
        "v_max": 30.0,
        "a_max": 800.0,
    }).with_objective({
        "method": "indicated_work",
        "w": {
            "smooth": 0.01,
            "short_circuit": 1.0,
        },
    })

    # Run optimization with modified configuration
    result = optimizer.optimize()

    # Print results
    print(f"Success: {result.success}")
    print(f"Modified geometry: {config.geometry}")
    print(f"Objective value: {result.objective_value:.6e}")

    return result


def example_validation_and_metrics():
    """Example 10: Optimization with validation and metrics."""
    print("=" * 60)
    print("Example 10: Validation and Metrics")
    print("=" * 60)

    # Create configuration with validation enabled
    config = ConfigFactory.create_default_config()
    config.validation = {
        "check_convergence": True,
        "check_physics": True,
        "check_constraints": True,
    }

    # Create optimizer
    optimizer = create_standard_optimizer(config)

    # Run optimization with validation
    result = optimizer.optimize_with_validation(validate=True)

    # Print results
    print(f"Success: {result.success}")
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Validation metrics: {result.validation_metrics}")
    print(f"Physics validation: {result.physics_validation}")
    print(f"Performance metrics: {result.performance_metrics}")

    if result.warnings:
        print(f"Warnings: {result.warnings}")

    return result


def example_comparison_study():
    """Example 11: Comparison study with different configurations."""
    print("=" * 60)
    print("Example 11: Comparison Study")
    print("=" * 60)

    # Define configurations to compare
    configs = {
        "default": ConfigFactory.create_default_config(),
        "high_performance": ConfigFactory.create_high_performance_config(),
        "quick_test": ConfigFactory.create_quick_test_config(),
    }

    results = {}

    # Run optimization for each configuration
    for name, config in configs.items():
        print(f"\nRunning {name} configuration...")

        optimizer = create_standard_optimizer(config)
        result = optimizer.optimize()
        results[name] = result

        print(f"  Success: {result.success}")
        print(f"  Objective: {result.objective_value:.6e}")
        print(f"  CPU time: {result.cpu_time:.2f}s")
        print(f"  Iterations: {result.iterations}")

    # Print comparison summary
    print("\n" + "=" * 40)
    print("COMPARISON SUMMARY")
    print("=" * 40)

    for name, result in results.items():
        print(f"{name:15} | Success: {result.success:5} | "
              f"Objective: {result.objective_value:12.6e} | "
              f"Time: {result.cpu_time:6.2f}s")

    return results


def example_save_and_load_config():
    """Example 12: Save and load configuration."""
    print("=" * 60)
    print("Example 12: Save and Load Configuration")
    print("=" * 60)

    # Create custom configuration
    config = ConfigFactory.create_custom_config(
        geometry={"bore": 0.12, "stroke": 0.08},
        num={"K": 25, "C": 3},
    )

    # Save configuration
    config_file = Path("example_config.yaml")
    ConfigFactory.to_yaml(config, config_file)
    print(f"Configuration saved to {config_file}")

    # Load configuration
    loaded_config = ConfigFactory.from_yaml(config_file)
    print(f"Configuration loaded from {config_file}")

    # Verify they are the same
    print(f"Original bore: {config.geometry['bore']}")
    print(f"Loaded bore: {loaded_config.geometry['bore']}")

    # Clean up
    config_file.unlink()
    print("Temporary file cleaned up")

    return loaded_config


def run_all_examples():
    """Run all examples."""
    print("Motion Law Optimization Library - Examples")
    print("=" * 80)

    examples = [
        example_basic_optimization,
        example_custom_configuration,
        example_1d_gas_model,
        example_robust_optimization,
        example_adaptive_optimization,
        example_engine_specific,
        example_scenario_optimization,
        example_quick_optimization,
        example_problem_builder,
        example_validation_and_metrics,
        example_comparison_study,
        example_save_and_load_config,
    ]

    results = {}

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\nRunning Example {i}: {example_func.__name__}")
            result = example_func()
            results[example_func.__name__] = result
            print(f"Example {i} completed successfully")
        except Exception as e:
            print(f"Example {i} failed: {e}")
            results[example_func.__name__] = None

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)

    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    print(f"Successful: {successful}/{total}")

    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
