#!/usr/bin/env python3
"""
Debug Secondary Data Extraction

This script tests the data extraction from secondary optimization results
to identify why secondary_base_radius is not being set.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
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


def debug_secondary_data_extraction():
    """Debug the secondary data extraction process."""
    print_section("Debug Secondary Data Extraction")

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
        # Step 1: Run primary optimization
        print("Step 1: Running primary optimization...")
        framework._update_data_from_input(input_data)
        primary_result = framework._optimize_primary()
        framework._update_data_from_primary(primary_result)

        print(f"Primary optimization completed: {primary_result.status}")
        print(
            f"Primary data set: theta={framework.data.primary_theta is not None}, position={framework.data.primary_position is not None}",
        )

        # Step 2: Run secondary optimization
        print("\nStep 2: Running secondary optimization...")
        secondary_result = framework._optimize_secondary()

        print(f"Secondary optimization completed: {secondary_result.status}")
        print(f"Secondary result type: {type(secondary_result)}")
        print(f"Secondary result attributes: {dir(secondary_result)}")

        # Debug the result structure
        print("\n--- Secondary Result Structure ---")
        print(f"Status: {secondary_result.status}")
        print(f"Success: {secondary_result.is_successful()}")
        print(f"Objective Value: {secondary_result.objective_value}")
        print(f"Iterations: {secondary_result.iterations}")
        print(f"Solve Time: {secondary_result.solve_time}")

        if hasattr(secondary_result, "solution") and secondary_result.solution:
            print("\n--- Solution Structure ---")
            print(f"Solution type: {type(secondary_result.solution)}")
            print(f"Solution keys: {list(secondary_result.solution.keys())}")

            if "optimized_parameters" in secondary_result.solution:
                opt_params = secondary_result.solution["optimized_parameters"]
                print(f"Optimized parameters: {opt_params}")
                print(f"Optimized parameters type: {type(opt_params)}")
                if isinstance(opt_params, dict):
                    print(f"Optimized parameters keys: {list(opt_params.keys())}")
                    if "base_radius" in opt_params:
                        print(f"Base radius value: {opt_params['base_radius']}")
                        print(f"Base radius type: {type(opt_params['base_radius'])}")
                    else:
                        print("❌ 'base_radius' not found in optimized_parameters")
                else:
                    print(f"❌ Optimized parameters is not a dict: {type(opt_params)}")
            else:
                print("❌ 'optimized_parameters' not found in solution")

        if hasattr(secondary_result, "metadata") and secondary_result.metadata:
            print("\n--- Metadata Structure ---")
            print(f"Metadata keys: {list(secondary_result.metadata.keys())}")

            if "optimized_gear_config" in secondary_result.metadata:
                gear_config = secondary_result.metadata["optimized_gear_config"]
                print(f"Optimized gear config: {gear_config}")
                if (
                    isinstance(gear_config, dict)
                    and "base_center_radius" in gear_config
                ):
                    print(f"Base center radius: {gear_config['base_center_radius']}")

        # Step 3: Test data extraction manually
        print("\n--- Manual Data Extraction Test ---")
        if secondary_result.status.value == "CONVERGED":
            solution = secondary_result.solution
            optimized_params = solution.get("optimized_parameters", {})

            print(f"Solution: {solution}")
            print(f"Optimized params: {optimized_params}")

            base_radius = optimized_params.get("base_radius")
            print(f"Extracted base_radius: {base_radius}")
            print(f"Base radius type: {type(base_radius)}")

            if base_radius is not None:
                print(f"✅ Base radius successfully extracted: {base_radius}")
            else:
                print("❌ Base radius extraction failed")

                # Try alternative extraction methods
                print("\n--- Alternative Extraction Methods ---")

                # Method 1: Check metadata
                if hasattr(secondary_result, "metadata") and secondary_result.metadata:
                    metadata = secondary_result.metadata
                    if "optimized_gear_config" in metadata:
                        gear_config = metadata["optimized_gear_config"]
                        if isinstance(gear_config, dict):
                            alt_base_radius = gear_config.get("base_center_radius")
                            print(
                                f"Alternative base_radius from metadata: {alt_base_radius}",
                            )

                # Method 2: Check solution directly
                if "base_radius" in solution:
                    direct_base_radius = solution["base_radius"]
                    print(f"Direct base_radius from solution: {direct_base_radius}")

        # Step 4: Test the actual _update_data_from_secondary method
        print("\n--- Testing _update_data_from_secondary Method ---")
        print(
            f"Before update - secondary_base_radius: {framework.data.secondary_base_radius}",
        )

        framework._update_data_from_secondary(secondary_result)

        print(
            f"After update - secondary_base_radius: {framework.data.secondary_base_radius}",
        )

        if framework.data.secondary_base_radius is not None:
            print("✅ _update_data_from_secondary worked correctly")
        else:
            print("❌ _update_data_from_secondary failed to set secondary_base_radius")

            # Debug why it failed
            print("\n--- Debugging _update_data_from_secondary Failure ---")
            if secondary_result.status.value == "CONVERGED":
                print(f"Status check passed: {secondary_result.status.value}")
                solution = secondary_result.solution
                print(f"Solution exists: {solution is not None}")
                if solution:
                    optimized_params = solution.get("optimized_parameters", {})
                    print(f"Optimized params: {optimized_params}")
                    base_radius = optimized_params.get("base_radius")
                    print(f"Base radius from optimized_params: {base_radius}")
            else:
                print(f"Status check failed: {secondary_result.status.value}")

        return secondary_result, framework

    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def main():
    """Run the debug script."""
    print_section("Debug Secondary Data Extraction Script")
    print("This script debugs why secondary_base_radius is not being set")
    print("in the cascaded optimization flow.")

    secondary_result, framework = debug_secondary_data_extraction()

    print_section("Debug Summary")

    if secondary_result and framework:
        print("✅ Debug completed successfully")
        print(f"Secondary base radius: {framework.data.secondary_base_radius}")

        if framework.data.secondary_base_radius is not None:
            print("✅ Secondary base radius is properly set")
            print("🔍 The issue may be elsewhere in the cascaded flow")
        else:
            print("❌ Secondary base radius is still not set")
            print("🔍 The issue is in the data extraction/update process")
    else:
        print("❌ Debug failed")
        print("🔍 Check the error messages above")


if __name__ == "__main__":
    main()
