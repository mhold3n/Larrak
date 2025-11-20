#!/usr/bin/env python3
"""
Primary Optimization Test Script

This script tests the primary optimization (motion law optimization) in isolation
to diagnose why it's not completing properly in the cascaded optimization flow.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
from campro.optimization.unified_framework import (  # noqa: E402
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)
from campro.utils import format_duration  # noqa: E402

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

        # Check for motion law data
        if "cam_angle" in result.solution:
            cam_angle = result.solution["cam_angle"]
            print(
                f"Cam Angle Shape: {cam_angle.shape if hasattr(cam_angle, 'shape') else len(cam_angle)}",
            )

        if "position" in result.solution:
            position = result.solution["position"]
            print(
                f"Position Shape: {position.shape if hasattr(position, 'shape') else len(position)}",
            )

        if "velocity" in result.solution:
            velocity = result.solution["velocity"]
            print(
                f"Velocity Shape: {velocity.shape if hasattr(velocity, 'shape') else len(velocity)}",
            )

    if hasattr(result, "metadata") and result.metadata:
        print(f"Metadata Keys: {list(result.metadata.keys())}")
        if "ipopt_analysis" in result.metadata:
            print("IPOPT Analysis: Available")
        else:
            print("IPOPT Analysis: Not available")


def test_primary_optimization_simple():
    """Test primary optimization with simple (non-thermal efficiency) path."""
    print_section("Testing Primary Optimization - Simple Path")

    # Configure test settings
    settings = UnifiedOptimizationSettings()
    settings.use_thermal_efficiency = False
    settings.verbose = True
    settings.enable_ipopt_analysis = False  # Disable for cleaner output

    print("Settings:")
    print(f"  - use_thermal_efficiency: {settings.use_thermal_efficiency}")
    print(f"  - method: {settings.method}")
    print(f"  - collocation_degree: {settings.collocation_degree}")
    print(f"  - tolerance: {settings.tolerance}")

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
        # Update data from input
        print("\nUpdating data from input...")
        framework._update_data_from_input(input_data)

        print("Data after input update:")
        print(f"  - stroke: {framework.data.stroke}")
        print(
            f"  - upstroke_duration_percent: {framework.data.upstroke_duration_percent}",
        )
        print(
            f"  - zero_accel_duration_percent: {framework.data.zero_accel_duration_percent}",
        )
        print(f"  - motion_type: {framework.data.motion_type}")

        # Run primary optimization
        print("\nRunning primary optimization...")
        start_time = time.time()
        result = framework._optimize_primary()
        end_time = time.time()

        print(f"Primary optimization completed in {format_duration(end_time - start_time)}")

        # Print result details
        print_result_details(result, "Simple Path")

        # Update data from primary result
        print("\nUpdating data from primary result...")
        framework._update_data_from_primary(result)

        print("Data after primary update:")
        print(f"  - primary_theta: {framework.data.primary_theta is not None}")
        if framework.data.primary_theta is not None:
            print(f"    Shape: {framework.data.primary_theta.shape}")
        print(f"  - primary_position: {framework.data.primary_position is not None}")
        if framework.data.primary_position is not None:
            print(f"    Shape: {framework.data.primary_position.shape}")
        print(f"  - primary_velocity: {framework.data.primary_velocity is not None}")
        if framework.data.primary_velocity is not None:
            print(f"    Shape: {framework.data.primary_velocity.shape}")
        print(
            f"  - primary_acceleration: {framework.data.primary_acceleration is not None}",
        )
        if framework.data.primary_acceleration is not None:
            print(f"    Shape: {framework.data.primary_acceleration.shape}")

        return result, framework

    except Exception as e:
        print(f"❌ Error in simple path test: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_primary_optimization_thermal_efficiency():
    """Test primary optimization with thermal efficiency path."""
    print_section("Testing Primary Optimization - Thermal Efficiency Path")

    # Configure test settings
    settings = UnifiedOptimizationSettings()
    settings.use_thermal_efficiency = True
    settings.verbose = True
    settings.enable_ipopt_analysis = False  # Disable for cleaner output

    print("Settings:")
    print(f"  - use_thermal_efficiency: {settings.use_thermal_efficiency}")
    print(f"  - method: {settings.method}")
    print(f"  - constant_temperature_K: {settings.constant_temperature_K}")
    print(f"  - constant_load_value: {settings.constant_load_value}")

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
        # Update data from input
        print("\nUpdating data from input...")
        framework._update_data_from_input(input_data)

        print("Data after input update:")
        print(f"  - stroke: {framework.data.stroke}")
        print(
            f"  - upstroke_duration_percent: {framework.data.upstroke_duration_percent}",
        )
        print(
            f"  - zero_accel_duration_percent: {framework.data.zero_accel_duration_percent}",
        )
        print(f"  - motion_type: {framework.data.motion_type}")

        # Run primary optimization
        print("\nRunning primary optimization...")
        start_time = time.time()
        result = framework._optimize_primary()
        end_time = time.time()

        print(f"Primary optimization completed in {format_duration(end_time - start_time)}")

        # Print result details
        print_result_details(result, "Thermal Efficiency Path")

        # Update data from primary result
        print("\nUpdating data from primary result...")
        framework._update_data_from_primary(result)

        print("Data after primary update:")
        print(f"  - primary_theta: {framework.data.primary_theta is not None}")
        if framework.data.primary_theta is not None:
            print(f"    Shape: {framework.data.primary_theta.shape}")
        print(f"  - primary_position: {framework.data.primary_position is not None}")
        if framework.data.primary_position is not None:
            print(f"    Shape: {framework.data.primary_position.shape}")
        print(f"  - primary_velocity: {framework.data.primary_velocity is not None}")
        if framework.data.primary_velocity is not None:
            print(f"    Shape: {framework.data.primary_velocity.shape}")
        print(
            f"  - primary_acceleration: {framework.data.primary_acceleration is not None}",
        )
        if framework.data.primary_acceleration is not None:
            print(f"    Shape: {framework.data.primary_acceleration.shape}")

        return result, framework

    except Exception as e:
        print(f"❌ Error in thermal efficiency path test: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def main():
    """Run all primary optimization tests."""
    print_section("Primary Optimization Test Script")
    print("This script tests the primary optimization in isolation to diagnose")
    print("why it's not completing properly in the cascaded optimization flow.")

    # Test 1: Simple path
    simple_result, simple_framework = test_primary_optimization_simple()

    # Test 2: Thermal efficiency path
    thermal_result, thermal_framework = test_primary_optimization_thermal_efficiency()

    # Summary
    print_section("Test Summary")

    print("Simple Path Test:")
    if simple_result:
        print(f"  ✅ Completed successfully: {simple_result.is_successful()}")
        print(f"  Status: {simple_result.status}")
        print(f"  Iterations: {simple_result.iterations}")
        print(f"  Solve Time: {simple_result.solve_time:.3f}s")
    else:
        print("  ❌ Failed to complete")

    print("\nThermal Efficiency Path Test:")
    if thermal_result:
        print(f"  ✅ Completed successfully: {thermal_result.is_successful()}")
        print(f"  Status: {thermal_result.status}")
        print(f"  Iterations: {thermal_result.iterations}")
        print(f"  Solve Time: {thermal_result.solve_time:.3f}s")
    else:
        print("  ❌ Failed to complete")

    # Analysis
    print_section("Analysis")

    if simple_result and thermal_result:
        print("Both optimization paths completed. Comparing results:")
        print(f"  Simple path success: {simple_result.is_successful()}")
        print(f"  Thermal efficiency path success: {thermal_result.is_successful()}")

        if simple_result.is_successful() and thermal_result.is_successful():
            print("  ✅ Both paths work in isolation")
            print(
                "  🔍 Issue likely in cascaded optimization flow, not primary optimization",
            )
        elif simple_result.is_successful() and not thermal_result.is_successful():
            print("  ⚠️  Simple path works, thermal efficiency path fails")
            print("  🔍 Issue likely in thermal efficiency optimization")
        elif not simple_result.is_successful() and thermal_result.is_successful():
            print("  ⚠️  Thermal efficiency path works, simple path fails")
            print("  🔍 Issue likely in simple motion law optimization")
        else:
            print("  ❌ Both paths fail")
            print("  🔍 Issue likely in primary optimization setup")

    elif simple_result or thermal_result:
        print("One optimization path completed successfully:")
        if simple_result:
            print("  ✅ Simple path works in isolation")
        if thermal_result:
            print("  ✅ Thermal efficiency path works in isolation")
        print("  🔍 Issue likely in cascaded optimization flow")

    else:
        print("❌ Both optimization paths failed")
        print("🔍 Issue likely in primary optimization setup or dependencies")

    print("\nTest completed. Check the output above for detailed diagnostics.")


if __name__ == "__main__":
    main()
