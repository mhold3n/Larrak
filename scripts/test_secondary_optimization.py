#!/usr/bin/env python3
"""
Secondary Optimization Test Script

This script tests the secondary optimization (cam-ring optimization) in isolation
to diagnose why it's not completing properly in the cascaded optimization flow.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
from campro.optimization.cam_ring_optimizer import CamRingOptimizer  # noqa: E402
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


def print_result_details(result, test_name):
    """Print detailed information about an optimization result."""
    print(f"\n--- {test_name} Results ---")
    print(f"Status: {result.status}")
    print(f"Success: {result.is_successful()}")
    print(f"Objective Value: {result.objective_value}")
    print(f"Iterations: {result.iterations}")
    print(f"Solve Time: {result.solve_time:.3f} seconds")

    if hasattr(result, "error_message") and result.error_message:
        print(f"Error Message: {result.error_message}")

    if hasattr(result, "solution") and result.solution:
        print(f"Solution Keys: {list(result.solution.keys())}")

        # Check for specific solution data
        if "optimized_parameters" in result.solution:
            opt_params = result.solution["optimized_parameters"]
            print(f"Optimized Parameters: {opt_params}")

        if "gear_geometry" in result.solution:
            gear_geom = result.solution["gear_geometry"]
            print(f"Gear Geometry: {gear_geom}")

    if hasattr(result, "metadata") and result.metadata:
        print(f"Metadata Keys: {list(result.metadata.keys())}")
        if "optimization_method" in result.metadata:
            print(f"Optimization Method: {result.metadata['optimization_method']}")
        if "order_results" in result.metadata:
            order_results = result.metadata["order_results"]
            print(f"Order Results: {order_results}")
        if "ipopt_analysis" in result.metadata:
            print("IPOPT Analysis: Available")
        else:
            print("IPOPT Analysis: Not available")


def create_mock_primary_data():
    """Create mock primary optimization data for testing."""
    # Create realistic motion law data
    n_points = 360
    theta = np.linspace(0, 2 * np.pi, n_points)

    # Simple sinusoidal motion law for testing
    stroke = 20.0  # mm
    position = stroke * 0.5 * (1 + np.sin(theta))
    velocity = stroke * 0.5 * np.cos(theta)
    acceleration = -stroke * 0.5 * np.sin(theta)

    primary_data = {
        "cam_angle": theta,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "time": np.linspace(0, 1.0, n_points),  # 1 second cycle
    }

    return primary_data


def test_secondary_optimization_isolated():
    """Test secondary optimization in complete isolation."""
    print_section("Testing Secondary Optimization - Isolated")

    # Create mock primary data
    primary_data = create_mock_primary_data()

    print("Mock Primary Data:")
    print(f"  - cam_angle shape: {primary_data['cam_angle'].shape}")
    print(f"  - position shape: {primary_data['position'].shape}")
    print(f"  - velocity shape: {primary_data['velocity'].shape}")
    print(f"  - acceleration shape: {primary_data['acceleration'].shape}")
    print(f"  - time shape: {primary_data['time'].shape}")
    print(
        f"  - position range: [{primary_data['position'].min():.2f}, {primary_data['position'].max():.2f}] mm",
    )
    print(
        f"  - velocity range: [{primary_data['velocity'].min():.2f}, {primary_data['velocity'].max():.2f}] mm/rad",
    )

    # Create initial guess
    initial_guess = {
        "base_radius": 20.0,  # mm
    }

    print("\nInitial Guess:")
    print(f"  - base_radius: {initial_guess['base_radius']} mm")

    # Create and configure secondary optimizer
    secondary_optimizer = CamRingOptimizer(name="TestCamRingOptimizer")

    try:
        print("\nRunning secondary optimization...")
        start_time = time.time()
        result = secondary_optimizer.optimize(
            primary_data=primary_data,
            initial_guess=initial_guess,
        )
        end_time = time.time()

        print(
            f"Secondary optimization completed in {end_time - start_time:.3f} seconds",
        )

        # Print result details
        print_result_details(result, "Isolated Secondary")

        return result

    except Exception as e:
        print(f"❌ Error in isolated secondary optimization: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_secondary_optimization_with_framework():
    """Test secondary optimization using the unified framework setup."""
    print_section("Testing Secondary Optimization - With Framework")

    # Configure test settings
    settings = UnifiedOptimizationSettings()
    settings.use_thermal_efficiency = False  # Use simple path for primary
    settings.verbose = True
    settings.enable_ipopt_analysis = False  # Disable for cleaner output

    print("Settings:")
    print(f"  - use_thermal_efficiency: {settings.use_thermal_efficiency}")
    print(f"  - method: {settings.method}")
    print(f"  - enable_ipopt_analysis: {settings.enable_ipopt_analysis}")

    # Create test input data
    input_data = {
        "motion_law": {
            "stroke": 20.0,
            "upstroke_duration_percent": 60.0,
            "zero_accel_duration_percent": 0.0,
        },
    }

    print("\nInput Data:")
    print(f"  - stroke: {input_data['motion_law']['stroke']} mm")
    print(
        f"  - upstroke_duration_percent: {input_data['motion_law']['upstroke_duration_percent']}%",
    )
    print(
        f"  - zero_accel_duration_percent: {input_data['motion_law']['zero_accel_duration_percent']}%",
    )

    # Initialize framework
    framework = UnifiedOptimizationFramework("TestFramework", settings)

    try:
        # Step 1: Update data from input
        print("\nStep 1: Updating data from input...")
        framework._update_data_from_input(input_data)

        # Step 2: Run primary optimization
        print("\nStep 2: Running primary optimization...")
        primary_result = framework._optimize_primary()
        print(f"Primary optimization: {primary_result.status}")

        # Step 3: Update data from primary result
        print("\nStep 3: Updating data from primary result...")
        framework._update_data_from_primary(primary_result)

        print("Data after primary update:")
        print(f"  - primary_theta: {framework.data.primary_theta is not None}")
        if framework.data.primary_theta is not None:
            print(f"    Shape: {framework.data.primary_theta.shape}")
        print(f"  - primary_position: {framework.data.primary_position is not None}")
        if framework.data.primary_position is not None:
            print(f"    Shape: {framework.data.primary_position.shape}")

        # Step 4: Run secondary optimization
        print("\nStep 4: Running secondary optimization...")
        start_time = time.time()
        secondary_result = framework._optimize_secondary()
        end_time = time.time()

        print(
            f"Secondary optimization completed in {end_time - start_time:.3f} seconds",
        )

        # Print result details
        print_result_details(secondary_result, "Framework Secondary")

        # Step 5: Update data from secondary result
        print("\nStep 5: Updating data from secondary result...")
        framework._update_data_from_secondary(secondary_result)

        print("Data after secondary update:")
        print(f"  - secondary_base_radius: {framework.data.secondary_base_radius}")
        print(
            f"  - secondary_cam_curves: {framework.data.secondary_cam_curves is not None}",
        )
        print(f"  - secondary_psi: {framework.data.secondary_psi is not None}")
        print(f"  - secondary_R_psi: {framework.data.secondary_R_psi is not None}")
        print(
            f"  - secondary_gear_geometry: {framework.data.secondary_gear_geometry is not None}",
        )

        return secondary_result, framework

    except Exception as e:
        print(f"❌ Error in framework secondary optimization: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_secondary_optimization_direct_call():
    """Test secondary optimization by directly calling the framework's _optimize_secondary method."""
    print_section("Testing Secondary Optimization - Direct Framework Call")

    # Configure test settings
    settings = UnifiedOptimizationSettings()
    settings.use_thermal_efficiency = False
    settings.verbose = True
    settings.enable_ipopt_analysis = False

    # Create test input data
    input_data = {
        "motion_law": {
            "stroke": 20.0,
            "upstroke_duration_percent": 60.0,
            "zero_accel_duration_percent": 0.0,
        },
    }

    # Initialize framework
    framework = UnifiedOptimizationFramework("TestFramework", settings)

    try:
        # Set up the framework data manually
        print("\nSetting up framework data manually...")
        framework._update_data_from_input(input_data)

        # Run primary optimization and update data
        primary_result = framework._optimize_primary()
        framework._update_data_from_primary(primary_result)

        print("Framework data after primary optimization:")
        print(f"  - primary_theta: {framework.data.primary_theta is not None}")
        print(f"  - primary_position: {framework.data.primary_position is not None}")
        print(f"  - stroke: {framework.data.stroke}")

        # Now call secondary optimization directly
        print("\nCalling _optimize_secondary() directly...")
        start_time = time.time()
        secondary_result = framework._optimize_secondary()
        end_time = time.time()

        print(
            f"Secondary optimization completed in {end_time - start_time:.3f} seconds",
        )

        # Print result details
        print_result_details(secondary_result, "Direct Framework Call")

        return secondary_result

    except Exception as e:
        print(f"❌ Error in direct framework call: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all secondary optimization tests."""
    print_section("Secondary Optimization Test Script")
    print("This script tests the secondary optimization in isolation to diagnose")
    print("why it's not completing properly in the cascaded optimization flow.")

    # Test 1: Isolated secondary optimization
    isolated_result = test_secondary_optimization_isolated()

    # Test 2: Secondary optimization with framework setup
    framework_result, framework = test_secondary_optimization_with_framework()

    # Test 3: Direct framework call
    direct_result = test_secondary_optimization_direct_call()

    # Summary
    print_section("Test Summary")

    print("Isolated Secondary Optimization Test:")
    if isolated_result:
        print(f"  ✅ Completed successfully: {isolated_result.is_successful()}")
        print(f"  Status: {isolated_result.status}")
        print(f"  Iterations: {isolated_result.iterations}")
        print(f"  Solve Time: {isolated_result.solve_time:.3f}s")
    else:
        print("  ❌ Failed to complete")

    print("\nFramework Secondary Optimization Test:")
    if framework_result:
        print(f"  ✅ Completed successfully: {framework_result.is_successful()}")
        print(f"  Status: {framework_result.status}")
        print(f"  Iterations: {framework_result.iterations}")
        print(f"  Solve Time: {framework_result.solve_time:.3f}s")
    else:
        print("  ❌ Failed to complete")

    print("\nDirect Framework Call Test:")
    if direct_result:
        print(f"  ✅ Completed successfully: {direct_result.is_successful()}")
        print(f"  Status: {direct_result.status}")
        print(f"  Iterations: {direct_result.iterations}")
        print(f"  Solve Time: {direct_result.solve_time:.3f}s")
    else:
        print("  ❌ Failed to complete")

    # Analysis
    print_section("Analysis")

    success_count = sum(
        [
            isolated_result and isolated_result.is_successful(),
            framework_result and framework_result.is_successful(),
            direct_result and direct_result.is_successful(),
        ],
    )

    if success_count == 3:
        print("✅ All secondary optimization tests completed successfully")
        print("🔍 Secondary optimization works in isolation")
        print("🔍 Issue likely in cascaded optimization flow orchestration")
    elif success_count == 2:
        print("⚠️  Most secondary optimization tests completed successfully")
        print("🔍 Secondary optimization generally works")
        print("🔍 Issue may be in specific test conditions or data setup")
    elif success_count == 1:
        print("⚠️  Some secondary optimization tests completed successfully")
        print("🔍 Secondary optimization has conditional issues")
        print("🔍 Issue may be in data preparation or framework setup")
    else:
        print("❌ All secondary optimization tests failed")
        print("🔍 Issue likely in secondary optimization implementation")
        print("🔍 Check CamRingOptimizer and Litvin optimization")

    # Check if secondary_base_radius is being set
    if framework and framework.data.secondary_base_radius is not None:
        print(
            f"\n✅ Secondary base radius is set: {framework.data.secondary_base_radius}",
        )
        print("🔍 Secondary optimization is properly updating framework data")
    else:
        print(
            f"\n❌ Secondary base radius is not set: {framework.data.secondary_base_radius if framework else 'N/A'}",
        )
        print("🔍 Secondary optimization is not properly updating framework data")
        print("🔍 This explains why tertiary optimization fails")

    print("\nTest completed. Check the output above for detailed diagnostics.")


if __name__ == "__main__":
    main()
