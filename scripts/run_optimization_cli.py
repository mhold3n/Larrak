#!/usr/bin/env python3
"""
Command-line optimization runner that emulates the GUI optimization button press.

This script simulates the GUI optimization process without requiring the GUI interface,
allowing for quick testing and debugging of the optimization flow.

IMPORTANT: This script must be run in the 'larrak' conda environment where CasADi is installed.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


# Check if we're in the correct conda environment
def check_environment():
    """Check if we're running in the correct conda environment (local or global)."""
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Try to import env manager - if it fails, we'll do basic checks
    try:
        from campro.environment.env_manager import (  # noqa: E402
            ensure_local_env_activated,
            get_active_conda_env_path,
        )
        from campro.environment.platform_detector import (  # noqa: E402
            get_local_conda_env_path,
            is_local_conda_env_present,
        )
        
        # Check for local environment first
        if is_local_conda_env_present(project_root):
            local_env_path = get_local_conda_env_path(project_root)
            active_env_path = get_active_conda_env_path(project_root)
            
            if active_env_path and str(active_env_path) == str(local_env_path):
                print(f"✅ Running in local conda environment: {local_env_path}")
                return
            elif active_env_path:
                print(
                    f"⚠️  WARNING: Local environment exists at '{local_env_path}' "
                    f"but active environment is '{active_env_path}'",
                )
                print(f"   Consider activating: conda activate {local_env_path}")
                print()
            else:
                print(f"⚠️  WARNING: Local environment exists at '{local_env_path}' but is not active")
                print(f"   Please activate: conda activate {local_env_path}")
                print()
        
        # Check global environment as fallback
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        if conda_env == "larrak":
            print(f"✅ Running in global conda environment: {conda_env}")
            return
        
        # Neither local nor global 'larrak' environment is active
        if is_local_conda_env_present(project_root):
            local_env_path = get_local_conda_env_path(project_root)
            print(
                f"⚠️  WARNING: Running in conda environment '{conda_env}' instead of local 'larrak'",
            )
            print(f"   Local environment available at: {local_env_path}")
            print(f"   Please run: conda activate {local_env_path}")
        else:
        print(
            f"⚠️  WARNING: Running in conda environment '{conda_env}' instead of 'larrak'",
        )
        print("   This may cause performance issues or optimization failures.")
        print("   Please run: conda activate larrak")
        print()

        # Ask user if they want to continue
        try:
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                print("Exiting. Please activate the correct environment first.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(1)
    except ImportError:
        # Fallback to simple check if imports fail
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
        if conda_env != "larrak":
            print(
                f"⚠️  WARNING: Running in conda environment '{conda_env}' instead of 'larrak'",
            )
            print("   This may cause performance issues or optimization failures.")
            print("   Please run: conda activate larrak")
            print()
            
            try:
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ["y", "yes"]:
                    print("Exiting. Please activate the 'larrak' environment first.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(1)
    else:
            print(f"✅ Running in conda environment: {conda_env}")


# Check environment before importing anything else
check_environment()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.logging import get_logger  # noqa: E402
from campro.optimization.unified_framework import (  # noqa: E402
    OptimizationMethod,
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)

log = get_logger(__name__)


def create_default_input_data() -> dict[str, Any]:
    """Create default input data matching GUI defaults."""
    return {
        "stroke": 20.0,  # mm
        "cycle_time": 1.0,  # s
        "upstroke_duration_percent": 60.0,  # %
        "zero_accel_duration_percent": 0.0,  # %
        "motion_type": "minimum_jerk",  # motion law type
    }


def create_optimization_settings(
    use_thermal_efficiency: bool = True,
    method: str = "legendre_collocation",
    enable_analysis: bool = True,
    verbose: bool = True,
) -> UnifiedOptimizationSettings:
    """Create optimization settings matching GUI configuration."""

    # Map method string to enum (non-collocation modes were removed; they were legacy no-ops)
    method_map = {
        "legendre_collocation": OptimizationMethod.LEGENDRE_COLLOCATION,
        "radau_collocation": OptimizationMethod.RADAU_COLLOCATION,
        "hermite_collocation": OptimizationMethod.HERMITE_COLLOCATION,
    }

    optimization_method = method_map.get(method, OptimizationMethod.LEGENDRE_COLLOCATION)

    settings = UnifiedOptimizationSettings(
        use_thermal_efficiency=use_thermal_efficiency,
        method=optimization_method,
        collocation_degree=3,
        tolerance=1e-6,
        max_iterations=5000,
        enable_ipopt_analysis=enable_analysis,
        verbose=verbose,
        save_intermediate_results=True,
    )

    return settings


def run_optimization(
    input_data: dict[str, Any],
    settings: UnifiedOptimizationSettings,
    show_progress: bool = True,
) -> Any:
    """Run the optimization process."""

    if show_progress:
        print("=" * 60)
        print(" CAM MOTION LAW OPTIMIZATION")
        print("=" * 60)
        print("Input Parameters:")
        for key, value in input_data.items():
            unit = ""
            if key == "stroke":
                unit = " mm"
            elif key == "cycle_time":
                unit = " s"
            elif "percent" in key:
                unit = " %"
            print(f"  - {key}: {value}{unit}")
        print()

    # Create framework
    framework = UnifiedOptimizationFramework("CLIOptimizer", settings)

    if show_progress:
        print("Starting optimization...")
        print(f"Method: {settings.method.value}")
        print(
            f"Thermal Efficiency: {'Enabled' if settings.use_thermal_efficiency else 'Disabled'}",
        )
        print(
            f"IPOPT Analysis: {'Enabled' if settings.enable_ipopt_analysis else 'Disabled'}",
        )
        print()

    # Run optimization
    start_time = time.time()
    result = framework.optimize_cascaded(input_data)
    end_time = time.time()

    if show_progress:
        print("=" * 60)
        print(" OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"Total Solve Time: {result.total_solve_time:.3f} seconds")
        print(f"Actual Runtime: {end_time - start_time:.3f} seconds")
        print(f"Optimization Method: {result.optimization_method}")
        print()

        # Show key results
        print("Key Results:")
        print(f"  - Secondary Base Radius: {result.secondary_base_radius} mm")
        print(f"  - Tertiary Crank Center X: {result.tertiary_crank_center_x} mm")
        print(f"  - Tertiary Crank Center Y: {result.tertiary_crank_center_y} mm")
        print(f"  - Tertiary Crank Radius: {result.tertiary_crank_radius} mm")
        print(f"  - Tertiary Rod Length: {result.tertiary_rod_length} mm")
        print()

        # Pressure invariance diagnostics (if available)
        pi = result.convergence_info.get("primary", {}) if hasattr(result, "convergence_info") else {}
        md = getattr(result, "metadata", {})
        pi_meta = md.get("pressure_invariance") if isinstance(md, dict) else None
        if pi_meta:
            print("Phase 1 (Pressure Invariance):")
            # Handle None values before formatting
            loss_p_mean = pi_meta.get('loss_p_mean')
            if loss_p_mean is None:
                loss_p_mean = 0.0
            imep_avg = pi_meta.get('imep_avg')
            if imep_avg is None:
                imep_avg = 0.0
            print(f"  - Loss_p_mean: {loss_p_mean:.3e}")
            print(f"  - iMEP_avg: {imep_avg:.2f} kPa")
            print(f"  - Fuel sweep: {pi_meta.get('fuel_sweep')}")
            print(f"  - Load sweep: {pi_meta.get('load_sweep')}")
            print()

        # Show convergence info
        if hasattr(result, "convergence_info") and result.convergence_info:
            print("Convergence Information:")
            for phase, info in result.convergence_info.items():
                print(f"  - {phase.title()} Phase:")
                print(f"    Status: {info.get('status', 'Unknown')}")
                print(f"    Iterations: {info.get('iterations', 'Unknown')}")
                # Handle None values before formatting
                solve_time = info.get('solve_time')
                if solve_time is None:
                    solve_time = 0.0
                print(f"    Solve Time: {solve_time:.3f}s")
            print()

        # Show IPOPT analysis status
        print("IPOPT Analysis Status:")
        print(
            f"  - Primary Analysis: {'Available' if result.primary_ipopt_analysis else 'Not Available'}",
        )
        print(
            f"  - Secondary Analysis: {'Available' if result.secondary_ipopt_analysis else 'Not Available'}",
        )
        print(
            f"  - Tertiary Analysis: {'Available' if result.tertiary_ipopt_analysis else 'Not Available'}",
        )
        print()

    return result


def show_detailed_analysis(result: Any):
    """Show detailed analysis similar to GUI."""
    print("=" * 60)
    print(" DETAILED OPTIMIZATION ANALYSIS")
    print("=" * 60)

    # Phase 1 Analysis
    p1_analysis = getattr(result, "primary_ipopt_analysis", None)
    print("\nPhase 1 (Thermal Efficiency):")
    if p1_analysis:
        print(f"  MA57 Readiness: {p1_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p1_analysis.reasons)}")
        print(f"  Suggested Action: {p1_analysis.suggested_action}")
        if "iterations" in p1_analysis.stats:
            print(f"  Iterations: {p1_analysis.stats['iterations']}")
        if "solve_time" in p1_analysis.stats:
            # Handle None values before formatting
            solve_time = p1_analysis.stats['solve_time']
            if solve_time is None:
                solve_time = 0.0
            print(f"  Solve Time: {solve_time:.3f}s")
    else:
        print("  Analysis: Not available (thermal efficiency adapter)")

    # Phase 2 Analysis
    p2_analysis = getattr(result, "secondary_ipopt_analysis", None)
    print("\nPhase 2 (Litvin/Cam-Ring):")
    if p2_analysis:
        print(f"  MA57 Readiness: {p2_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p2_analysis.reasons)}")
        print(f"  Suggested Action: {p2_analysis.suggested_action}")
        if "iterations" in p2_analysis.stats:
            print(f"  Iterations: {p2_analysis.stats['iterations']}")
        if "solve_time" in p2_analysis.stats:
            # Handle None values before formatting
            solve_time = p2_analysis.stats['solve_time']
            if solve_time is None:
                solve_time = 0.0
            print(f"  Solve Time: {solve_time:.3f}s")
    else:
        print("  Analysis: Not available")

    # Phase 3 Analysis
    p3_analysis = getattr(result, "tertiary_ipopt_analysis", None)
    print("\nPhase 3 (Crank Center):")
    if p3_analysis:
        print(f"  MA57 Readiness: {p3_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p3_analysis.reasons)}")
        print(f"  Suggested Action: {p3_analysis.suggested_action}")
        if "iterations" in p3_analysis.stats:
            print(f"  Iterations: {p3_analysis.stats['iterations']}")
        if "solve_time" in p3_analysis.stats:
            # Handle None values before formatting
            solve_time = p3_analysis.stats['solve_time']
            if solve_time is None:
                solve_time = 0.0
            print(f"  Solve Time: {solve_time:.3f}s")
    else:
        print("  Analysis: Not available")

    # Overall Summary
    print("\nOverall Summary:")
    print(f"  Total Solve Time: {result.total_solve_time:.3f}s")
    print(f"  Optimization Method: {result.optimization_method}")

    # Count phases with analysis
    phases_with_analysis = sum(
        [
            result.primary_ipopt_analysis is not None,
            result.secondary_ipopt_analysis is not None,
            result.tertiary_ipopt_analysis is not None,
        ],
    )
    print(f"  Phases with Analysis: {phases_with_analysis}/3")
    print("=" * 60)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run Cam Motion Law optimization from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python scripts/run_optimization_cli.py

  # Run with custom stroke
  python scripts/run_optimization_cli.py --stroke 25.0

  # Run without thermal efficiency
  python scripts/run_optimization_cli.py --no-thermal-efficiency

  # Run with different method
  python scripts/run_optimization_cli.py --method radau_collocation

  # Run quietly (minimal output)
  python scripts/run_optimization_cli.py --quiet
        """,
    )

    # Input parameters
    parser.add_argument(
        "--stroke", type=float, default=20.0, help="Stroke length in mm (default: 20.0)",
    )
    parser.add_argument(
        "--cycle-time",
        type=float,
        default=1.0,
        help="Cycle time in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--upstroke-duration",
        type=float,
        default=60.0,
        help="Upstroke duration percentage (default: 60.0)",
    )
    parser.add_argument(
        "--zero-accel-duration",
        type=float,
        default=0.0,
        help="Zero acceleration duration percentage (default: 0.0)",
    )
    parser.add_argument(
        "--motion-type",
        type=str,
        default="minimum_jerk",
        choices=["minimum_jerk", "cycloidal", "harmonic"],
        help="Motion law type (default: minimum_jerk)",
    )

    # Optimization settings
    parser.add_argument(
        "--method",
        type=str,
        default="legendre_collocation",
        choices=[
            "legendre_collocation",
            "radau_collocation",
            "hermite_collocation",
        ],
        help="Optimization method (default: legendre_collocation)",
    )
    parser.add_argument(
        "--no-thermal-efficiency",
        action="store_true",
        help="Disable thermal efficiency optimization",
    )
    parser.add_argument(
        "--fuel-sweep",
        type=str,
        default="0.7,1.0,1.3",
        help="Comma-separated fuel multipliers (e.g., 0.7,1.0,1.3)",
    )
    parser.add_argument(
        "--load-sweep",
        type=str,
        default="50,100,150",
        help="Comma-separated load damping values (N·s/m equivalent)",
    )
    parser.add_argument(
        "--dpdt-weight",
        type=float,
        default=1.0,
        help="Weight for pressure-slope invariance term",
    )
    parser.add_argument(
        "--jerk-weight",
        type=float,
        default=1.0,
        help="Weight for jerk term",
    )
    parser.add_argument(
        "--imep-weight",
        type=float,
        default=0.2,
        help="Weight for average iMEP reward (subtracted)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA alpha for updating reference slope",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=2,
        help="Number of outer iterations for EMA update",
    )
    parser.add_argument(
        "--bounce-alpha",
        type=float,
        default=30.0,
        help="Fuel→base-pressure slope (kPa per fuel-mult)",
    )
    parser.add_argument(
        "--bounce-beta",
        type=float,
        default=100.0,
        help="Base pressure intercept (kPa)",
    )
    parser.add_argument(
        "--piston-area-mm2",
        type=float,
        default=1.0,
        help="Piston crown area [mm^2] for iMEP scaling",
    )
    parser.add_argument(
        "--clearance-volume-mm3",
        type=float,
        default=1.0,
        help:"Clearance volume [mm^3] for iMEP scaling",
    )
    # Guardrail parameters for Stage B (TE refine)
    parser.add_argument(
        "--pressure-guard-eps",
        type=float,
        default=1e-3,
        help="Guardrail epsilon for invariance loss in TE stage",
    )
    parser.add_argument(
        "--pressure-guard-lambda",
        type=float,
        default=1e4,
        help="Quadratic penalty lambda for guardrail in TE stage",
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Disable IPOPT analysis collection",
    )

    # Output options
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output (no progress indicators)",
    )
    parser.add_argument(
        "--no-detailed-analysis",
        action="store_true",
        help="Skip detailed analysis output",
    )

    args = parser.parse_args()

    # Create input data
    input_data = {
        "stroke": args.stroke,
        "cycle_time": args.cycle_time,
        "upstroke_duration_percent": args.upstroke_duration,
        "zero_accel_duration_percent": args.zero_accel_duration,
        "motion_type": args.motion_type,
    }

    # Create settings
    settings = create_optimization_settings(
        use_thermal_efficiency=not args.no_thermal_efficiency,
        method=args.method,
        enable_analysis=not args.no_analysis,
        verbose=not args.quiet,
    )

    # Parse sweeps
    try:
        settings.fuel_sweep = [float(x) for x in args.fuel_sweep.split(",") if x]
    except Exception:
        settings.fuel_sweep = [0.7, 1.0, 1.3]
    try:
        settings.load_sweep = [float(x) for x in args.load_sweep.split(",") if x]
    except Exception:
        settings.load_sweep = [50.0, 100.0, 150.0]
    settings.dpdt_weight = float(args.dpdt_weight)
    settings.jerk_weight = float(args.jerk_weight)
    settings.imep_weight = float(args.imep_weight)
    settings.ema_alpha = float(args.ema_alpha)
    settings.outer_iterations = int(max(1, args.outer_iters))
    settings.bounce_alpha = float(args.bounce_alpha)
    settings.bounce_beta = float(args.bounce_beta)
    settings.piston_area_mm2 = float(args.piston_area_mm2)
    settings.clearance_volume_mm3 = float(args.clearance_volume_mm3)
    settings.pressure_guard_epsilon = float(args.pressure_guard_eps)
    settings.pressure_guard_lambda = float(args.pressure_guard_lambda)

    try:
        # Run optimization
        result = run_optimization(
            input_data=input_data,
            settings=settings,
            show_progress=not args.quiet,
        )

        # Show detailed analysis unless disabled
        if not args.no_detailed_analysis and not args.quiet:
            show_detailed_analysis(result)

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nOptimization failed: {e}")
        if not args.quiet:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
