#!/usr/bin/env python3
"""
Command-line optimization runner that emulates the GUI optimization button press.

This script simulates the GUI optimization process without requiring the GUI interface,
allowing for quick testing and debugging of the optimization flow.

IMPORTANT: This script must be run in the 'larrak' conda environment where CasADi is installed.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Check if we're in the correct conda environment
def check_environment():
    """Check if we're running in the correct conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    if conda_env != 'larrak':
        print(f"⚠️  WARNING: Running in conda environment '{conda_env}' instead of 'larrak'")
        print("   This may cause performance issues or optimization failures.")
        print("   Please run: conda activate larrak")
        print("   Then run this script again.")
        print()
        
        # Ask user if they want to continue
        try:
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Exiting. Please activate the 'larrak' environment first.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(1)
    else:
        print(f"✅ Running in correct conda environment: {conda_env}")

# Check environment before importing anything else
check_environment()

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework, 
    UnifiedOptimizationSettings,
    OptimizationMethod
)
from campro.logging import get_logger

log = get_logger(__name__)


def create_default_input_data() -> Dict[str, Any]:
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
    verbose: bool = True
) -> UnifiedOptimizationSettings:
    """Create optimization settings matching GUI configuration."""
    
    # Map method string to enum
    method_map = {
        "legendre_collocation": OptimizationMethod.LEGENDRE_COLLOCATION,
        "radau_collocation": OptimizationMethod.RADAU_COLLOCATION,
        "hermite_collocation": OptimizationMethod.HERMITE_COLLOCATION,
        "slsqp": OptimizationMethod.SLSQP,
        "l_bfgs_b": OptimizationMethod.L_BFGS_B,
        "tnc": OptimizationMethod.TNC,
        "cobyla": OptimizationMethod.COBYLA,
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
    input_data: Dict[str, Any],
    settings: UnifiedOptimizationSettings,
    show_progress: bool = True
) -> Any:
    """Run the optimization process."""
    
    if show_progress:
        print("=" * 60)
        print(" CAM MOTION LAW OPTIMIZATION")
        print("=" * 60)
        print(f"Input Parameters:")
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
    framework = UnifiedOptimizationFramework('CLIOptimizer', settings)
    
    if show_progress:
        print("Starting optimization...")
        print(f"Method: {settings.method.value}")
        print(f"Thermal Efficiency: {'Enabled' if settings.use_thermal_efficiency else 'Disabled'}")
        print(f"IPOPT Analysis: {'Enabled' if settings.enable_ipopt_analysis else 'Disabled'}")
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
        
        # Show convergence info
        if hasattr(result, 'convergence_info') and result.convergence_info:
            print("Convergence Information:")
            for phase, info in result.convergence_info.items():
                print(f"  - {phase.title()} Phase:")
                print(f"    Status: {info.get('status', 'Unknown')}")
                print(f"    Iterations: {info.get('iterations', 'Unknown')}")
                print(f"    Solve Time: {info.get('solve_time', 0):.3f}s")
            print()
        
        # Show IPOPT analysis status
        print("IPOPT Analysis Status:")
        print(f"  - Primary Analysis: {'Available' if result.primary_ipopt_analysis else 'Not Available'}")
        print(f"  - Secondary Analysis: {'Available' if result.secondary_ipopt_analysis else 'Not Available'}")
        print(f"  - Tertiary Analysis: {'Available' if result.tertiary_ipopt_analysis else 'Not Available'}")
        print()
    
    return result


def show_detailed_analysis(result: Any):
    """Show detailed analysis similar to GUI."""
    print("=" * 60)
    print(" DETAILED OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Phase 1 Analysis
    p1_analysis = getattr(result, 'primary_ipopt_analysis', None)
    print(f"\nPhase 1 (Thermal Efficiency):")
    if p1_analysis:
        print(f"  MA57 Readiness: {p1_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p1_analysis.reasons)}")
        print(f"  Suggested Action: {p1_analysis.suggested_action}")
        if 'iterations' in p1_analysis.stats:
            print(f"  Iterations: {p1_analysis.stats['iterations']}")
        if 'solve_time' in p1_analysis.stats:
            print(f"  Solve Time: {p1_analysis.stats['solve_time']:.3f}s")
    else:
        print("  Analysis: Not available (thermal efficiency adapter)")
    
    # Phase 2 Analysis
    p2_analysis = getattr(result, 'secondary_ipopt_analysis', None)
    print(f"\nPhase 2 (Litvin/Cam-Ring):")
    if p2_analysis:
        print(f"  MA57 Readiness: {p2_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p2_analysis.reasons)}")
        print(f"  Suggested Action: {p2_analysis.suggested_action}")
        if 'iterations' in p2_analysis.stats:
            print(f"  Iterations: {p2_analysis.stats['iterations']}")
        if 'solve_time' in p2_analysis.stats:
            print(f"  Solve Time: {p2_analysis.stats['solve_time']:.3f}s")
    else:
        print("  Analysis: Not available")
    
    # Phase 3 Analysis
    p3_analysis = getattr(result, 'tertiary_ipopt_analysis', None)
    print(f"\nPhase 3 (Crank Center):")
    if p3_analysis:
        print(f"  MA57 Readiness: {p3_analysis.grade.upper()}")
        print(f"  Reasons: {', '.join(p3_analysis.reasons)}")
        print(f"  Suggested Action: {p3_analysis.suggested_action}")
        if 'iterations' in p3_analysis.stats:
            print(f"  Iterations: {p3_analysis.stats['iterations']}")
        if 'solve_time' in p3_analysis.stats:
            print(f"  Solve Time: {p3_analysis.stats['solve_time']:.3f}s")
    else:
        print("  Analysis: Not available")
    
    # Overall Summary
    print(f"\nOverall Summary:")
    print(f"  Total Solve Time: {result.total_solve_time:.3f}s")
    print(f"  Optimization Method: {result.optimization_method}")
    
    # Count phases with analysis
    phases_with_analysis = sum([
        result.primary_ipopt_analysis is not None,
        result.secondary_ipopt_analysis is not None,
        result.tertiary_ipopt_analysis is not None
    ])
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
        """
    )
    
    # Input parameters
    parser.add_argument('--stroke', type=float, default=20.0,
                       help='Stroke length in mm (default: 20.0)')
    parser.add_argument('--cycle-time', type=float, default=1.0,
                       help='Cycle time in seconds (default: 1.0)')
    parser.add_argument('--upstroke-duration', type=float, default=60.0,
                       help='Upstroke duration percentage (default: 60.0)')
    parser.add_argument('--zero-accel-duration', type=float, default=0.0,
                       help='Zero acceleration duration percentage (default: 0.0)')
    parser.add_argument('--motion-type', type=str, default='minimum_jerk',
                       choices=['minimum_jerk', 'cycloidal', 'harmonic'],
                       help='Motion law type (default: minimum_jerk)')
    
    # Optimization settings
    parser.add_argument('--method', type=str, default='legendre_collocation',
                       choices=['legendre_collocation', 'radau_collocation', 'hermite_collocation', 'slsqp', 'l_bfgs_b', 'tnc', 'cobyla'],
                       help='Optimization method (default: legendre_collocation)')
    parser.add_argument('--no-thermal-efficiency', action='store_true',
                       help='Disable thermal efficiency optimization')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Disable IPOPT analysis collection')
    
    # Output options
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output (no progress indicators)')
    parser.add_argument('--no-detailed-analysis', action='store_true',
                       help='Skip detailed analysis output')
    
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
        verbose=not args.quiet
    )
    
    try:
        # Run optimization
        result = run_optimization(
            input_data=input_data,
            settings=settings,
            show_progress=not args.quiet
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
