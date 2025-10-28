#!/usr/bin/env python3
"""
Debug Linear Solver Issue

This script traces where an unexpected linear solver is being set
instead of the configured HSL solver (MA27/MA57).
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
from campro.optimization.ipopt_factory import (  # noqa: E402
    create_ipopt_solver,
    is_linear_solver_initialized,
    reset_linear_solver_flag,
)
from campro.optimization.unified_framework import (  # noqa: E402
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)

log = get_logger(__name__)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def test_linear_solver_factory():
    """Test the linear solver factory directly."""
    print_section("Testing Linear Solver Factory")

    import casadi as ca

    # Reset the global flag
    reset_linear_solver_flag()
    print(f"Linear solver initialized: {is_linear_solver_initialized()}")

    # Create a simple test problem
    x = ca.MX.sym("x")
    f = x**2
    g = x - 1
    nlp = {"x": x, "f": f, "g": g}

    # Test 1: First solver creation
    print("\n--- Test 1: First Solver Creation ---")
    solver1 = create_ipopt_solver("test1", nlp, force_linear_solver=True)
    print(f"Linear solver initialized after first: {is_linear_solver_initialized()}")

    # Test 2: Second solver creation (should not set linear solver)
    print("\n--- Test 2: Second Solver Creation ---")
    solver2 = create_ipopt_solver("test2", nlp, force_linear_solver=True)
    print(f"Linear solver initialized after second: {is_linear_solver_initialized()}")

    # Test 3: Check solver options
    print("\n--- Test 3: Solver Options ---")
    # Note: CasADi doesn't expose solver options directly, but we can test solving
    try:
        result1 = solver1(x0=0, lbg=0, ubg=0)
        print(f"Solver 1 result: {result1['x']}")
    except Exception as e:
        print(f"Solver 1 error: {e}")

    try:
        result2 = solver2(x0=0, lbg=0, ubg=0)
        print(f"Solver 2 result: {result2['x']}")
    except Exception as e:
        print(f"Solver 2 error: {e}")


def test_validation_process():
    """Test the validation process to see where solver is initialized."""
    print_section("Testing Validation Process")

    from campro.environment.validator import validate_environment

    # Reset the global flag
    reset_linear_solver_flag()
    print(
        f"Linear solver initialized before validation: {is_linear_solver_initialized()}",
    )

    try:
        # Run validation
        print("\nRunning environment validation...")
        results = validate_environment()

        print(
            f"Linear solver initialized after validation: {is_linear_solver_initialized()}",
        )

        # Print validation results
        print("\nValidation Results:")
        for result in results:
            print(f"  {result.status.value.upper()}: {result.message}")
            if result.details:
                print(f"    Details: {result.details}")

    except Exception as e:
        print(f"Validation error: {e}")
        import traceback

        traceback.print_exc()


def test_unified_framework_initialization():
    """Test the unified framework initialization."""
    print_section("Testing Unified Framework Initialization")

    # Reset the global flag
    reset_linear_solver_flag()
    print(
        f"Linear solver initialized before framework: {is_linear_solver_initialized()}",
    )

    try:
        # Create framework
        settings = UnifiedOptimizationSettings()
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        print(
            f"Linear solver initialized after framework: {is_linear_solver_initialized()}",
        )

        # Check if framework components are using the right solver
        print("\nFramework components:")
        print(f"  Primary optimizer: {type(framework.primary_optimizer)}")
        print(f"  Secondary optimizer: {type(framework.secondary_optimizer)}")
        print(f"  Tertiary optimizer: {type(framework.tertiary_optimizer)}")

    except Exception as e:
        print(f"Framework initialization error: {e}")
        import traceback

        traceback.print_exc()


def test_optimization_with_tracing():
    """Test optimization with linear solver tracing."""
    print_section("Testing Optimization with Linear Solver Tracing")

    # Reset the global flag
    reset_linear_solver_flag()
    print(
        f"Linear solver initialized before optimization: {is_linear_solver_initialized()}",
    )

    try:
        # Create framework
        settings = UnifiedOptimizationSettings()
        settings.use_thermal_efficiency = (
            True  # Use thermal efficiency to trigger the issue
        )
        framework = UnifiedOptimizationFramework("TestFramework", settings)

        print(
            f"Linear solver initialized after framework creation: {is_linear_solver_initialized()}",
        )

        # Create test input data
        input_data = {
            "motion_law": {
                "stroke": 20.0,
                "upstroke_duration_percent": 60.0,
                "zero_accel_duration_percent": 0.0,
            },
        }

        print("\nRunning primary optimization...")
        framework._update_data_from_input(input_data)

        print(
            f"Linear solver initialized before primary optimization: {is_linear_solver_initialized()}",
        )

        # Run primary optimization
        primary_result = framework._optimize_primary()

        print(
            f"Linear solver initialized after primary optimization: {is_linear_solver_initialized()}",
        )
        print(f"Primary optimization status: {primary_result.status}")
        print(f"Primary optimization iterations: {primary_result.iterations}")
        print(f"Primary optimization solve time: {primary_result.solve_time:.3f}s")

    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all debug tests."""
    print_section("Linear Solver Debug Script")
    print(
        "This script traces where an unexpected linear solver is set instead of MA27/MA57",
    )
    print("and identifies potential misconfiguration sources.")

    # Test 1: Factory directly
    test_linear_solver_factory()

    # Test 2: Validation process
    test_validation_process()

    # Test 3: Framework initialization
    test_unified_framework_initialization()

    # Test 4: Optimization with tracing
    test_optimization_with_tracing()

    print_section("Debug Summary")
    print("Check the output above to identify where the linear solver is initialized")
    print("and why the factory options may not be applied.")


if __name__ == "__main__":
    main()
