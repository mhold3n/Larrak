import logging
import os
import sys

import numpy as np

print("[DEBUG] Started run_planet_ring_opt.py", flush=True)

# [FIX] Critical: Setup CasADi PATH before imports
from scripts.setup import env_setup 
print("[DEBUG] env_setup imported", flush=True)

# Configure Parallelism for MA86
# Default to 12 threads on this 16-core machine for better MA86 performance
os.environ.setdefault("OMP_NUM_THREADS", "12")

# Add root to path
sys.path.append(os.getcwd())


def run_optimization():
    print("[DEBUG] Entering run_optimization...", flush=True)
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("run_opt")
    print("[DEBUG] Logging initialized", flush=True)

    # --- Automated Archiving (Pre-Run) ---
    from pathlib import Path

    try:
        print("[DEBUG] Importing ArchiveManager...", flush=True)
        from campro.utils.archiving import ArchiveManager
        print("[DEBUG] ArchiveManager imported", flush=True)


        project_root = Path(os.getcwd())
        archive_root = project_root / "PERMANENT_ARCHIVE"

        log.info("Running Automated Archiving (Pre-Run)...")
        archiver = ArchiveManager(
            root_dir=project_root, archive_root=archive_root, retention_count=5
        )
        archiver.run_default_cleanup()
        log.info("Archiving complete.")

    except ImportError:
        log.warning("Could not import ArchiveManager. Skipping archiving.")
    except Exception as e:
        log.error(f"Archiving routine failed: {e}")
    # -------------------------------------

    # Parameters from User:
    # - Mean planet diameter = 20mm -> Stroke = 0.02m
    # - Outer surface diameter = 42mm -> OD = 0.042m
    # - Angle between TDC and BDC = 180 -> theta_bdc = pi
    # - Angle between midpoints = 90 -> defaults (0.5 split) are correct.
    # - Ratios: Inf1=6:1 (6.0), Inf2=1:3 (0.333)

    # Scale Assumptions:
    # Scaling from 100mm to 20mm (1/5 scale).
    # Mass ~ scale^3. Old mass 5.0kg. New mass ~ 5.0 / 125 = 0.04kg (40g).
    # Bore: Keep aspect ratio? Old: 80mm bore, 100mm stroke (0.8 ratio).
    # New bore: 0.8 * 20mm = 16mm = 0.016m.
    # Cycle Time: Smaller scale -> faster.
    # Assume 3000 RPM equivalent -> 50 Hz -> 0.02s.

    # --- Dashbaord Param Injection ---
    # We read env vars set by the Dashboard Runner: LARRAK_COLLOCATION_POINTS, etc.
    K_points = int(os.environ.get("LARRAK_COLLOCATION_POINTS", 40))
    mass_kg = float(os.environ.get("LARRAK_MASS_KG", 0.04))
    r_planet_max = float(os.environ.get("LARRAK_R_PLANET_MAX", 0.050))
    R_ring_max = float(os.environ.get("LARRAK_R_RING_MAX", 0.100))

    P = {
        "problem_type": "kinematic",  # Explicit flag
        "planet_ring": {
            "stroke": 0.02,
            "outer_diameter": 0.042,
            "theta_bdc": np.pi,
            # User Ratios: 6.0 (Fast) and 0.333 (Slow/Flat)
            # Interpretation: Multipliers of the Mean Slope (stroke / pi)
            # Mean Slope ~ 0.02 / 3.14 ~ 0.00637 m/rad
            # R1 = 6.0 * Mean ~ 0.038 (Steep)
            # R2 = 0.33 * Mean ~ 0.002 (Flat)
            "ratios": (6.0 * (0.02 / np.pi), 0.3333 * (0.02 / np.pi)),
            # Hypocycloid Golden Configuration (Enabled for analytical optimization)
            # We use 'hypocycloid' for smooth, efficient C2+ geometry.
            # "gen_mode": "hypocycloid",
            # Litvin Mode:    # --- Run Configuration ---
            # use "hypocycloid", "spline", "litvin", or "litvin_debug_geometry"
            "gen_mode": "litvin_debug_geometry",
            "mean_ratio": 2.0,  # Target closure ratio (2:1 => 2 turns per cycle)
            # r_drive determines the stroke. Stroke = 2 * r_drive.
            # For 20mm stroke, r_drive should be 10mm = 0.01m.
            # r_drive determines the stroke. Stroke = 2 * r_drive.
            # For 20mm stroke, r_drive should be 10mm = 0.01m.
            "r_drive": 0.01,
            # User proposed setting this to 0.0 to mitigate stroke variation
            "attachment_radius": 0.0,
            # Removed: p0_guess, T0_guess, use_load_model
        },
        # Removed: combustion, thermo, flow sections
        "geometry": {
            "bore": 0.016,  # 16mm
            "stroke": 0.02,
            "mass": mass_kg,  # Configurable Mass
            # Removed: clearance_volume, compression_ratio, c_load
        },
        "bounds": {  # Bounds for Kinematic Optimization
            "r_planet_min": 0.005,
            "r_planet_max": r_planet_max, # Configurable
            "R_ring_min": 0.010,
            "R_ring_max": R_ring_max, # Configurable
            # Bounds for Piston (0 to Stroke)
            "xL_min": -0.01,  # Buffer
            "xL_max": 0.03,  # Buffer
        },
        "constraints": {
            # Ensure we produce some work or at least function
        },
        "num": {
            "K": K_points,  # Configurable Collocation Points
            "C": 3,  # Degree
            "cycle_time": 0.02,  # Not really used in angle domain except for omega calc
        },
        "solver": {
            "ipopt": {
                "print_level": 5,
                "linear_solver": "ma57",  # MA57 is more reliable for this problem
            },
        },
        "obj": {
            "type": "kinematic_tracking",  # maximize work output
            "w": {
                "ratio_tracking": 10.0,  # Target the mean_ratio profile
            },
        },
        "auto_plot": True,
    }

    # --- Argument Parsing for Isolation Mode ---
    import argparse
    import copy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", type=str, help="Specific shape file to run (e.g. circle.json)"
    )
    args = parser.parse_args()

    # --- Mode 1: Supervisor (No Shape arg) ---
    # Scans files and launches subprocesses
    if not args.shape:
        import glob
        import subprocess

        # Find all json files
        shape_files = glob.glob("shapes/*.json")
        shapes_to_test = [os.path.basename(f) for f in shape_files]
        shapes_to_test.sort()

        log.info(
            f"Supervisor: Found {len(shapes_to_test)} shapes to processing in isolation."
        )

        failed = []
        for shape in shapes_to_test:
            log.info(f"--- Launching process for {shape} ---")
            try:
                # Run this script recursively with --shape arg
                # Check=False allows us to handle exit code 139 (Segfault) gracefully
                result = subprocess.run(
                    [sys.executable, __file__, "--shape", shape], check=False
                )

                # Check return code
                if result.returncode == 0:
                    log.info(f"Process for {shape} finished successfully.")
                elif result.returncode == -11 or result.returncode == 139:
                    log.warning(
                        f"Process for {shape} finished with Segmentation Fault (Expected on macOS). detailed results should be saved."
                    )
                else:
                    log.error(
                        f"Process for {shape} failed with exit code {result.returncode}"
                    )
                    failed.append(shape)

            except Exception as e:
                log.error(f"Failed to launch process for {shape}: {e}")
                failed.append(shape)

        if failed:
            log.error(f"Supervisor: Batch completed with failures in: {failed}")
            sys.exit(1)
        else:
            log.info(
                "Supervisor: Batch completed successfully (all subprocesses finished)."
            )
            return

    # --- Mode 2: Worker (Specific Shape) ---
    # Runs the optimization for the single specified shape
    shape_name = args.shape
    log.info(f"Worker: Processing single shape: {shape_name}")

    shapes_to_test = [shape_name]  # List of one

    from campro.optimization.driver import solve_cycle

    had_failure = False

    for shape_name in shapes_to_test:
        print(f"\n{'=' * 40}")
        print(f"Running Optimization for Shape: {shape_name}")
        print(f"{'=' * 40}\n")

        # Deep copy P to ensure clean run each time
        P_run = copy.deepcopy(P)

        # Update config for this shape
        shape_slug = shape_name.replace(".json", "")
        P_run["shape_file"] = shape_name

        # Load JSON to merge config
        import json

        try:
            with open(f"shapes/{shape_name}") as f:
                shape_data = json.load(f)
        except Exception as e:
            log.error(f"Could not load shape file {shape_name}: {e}")
            sys.exit(1)

        # Merge Constraints
        if "constraints" in shape_data:
            P_run["planet_ring"].update(shape_data["constraints"])
            P_run["constraints"].update(shape_data["constraints"])  # Critical Fix
            P_run["bounds"]["xL_min"] = shape_data["constraints"]["x_limits"][0]
            P_run["bounds"]["xL_max"] = shape_data["constraints"]["x_limits"][1]

        # Merge Geometry
        if "geometry" in shape_data:
            P_run["geometry"].update(shape_data["geometry"])

        # Merge Operating Point
        if "operating_point" in shape_data:
            # P_run["combustion"]["cycle_time_s"] = ... # Removed
            P_run["num"]["cycle_time"] = shape_data["operating_point"]["cycle_time"]

        # Create distinct plot directories
        base_plot_dir = f"out/plots/{shape_slug}"
        # Only use the base directory + cam_ring_mapping for specific kinematic plots if needed by the plotter logic
        # But based on user request to "clean up", let's try to keep it minimal.
        # Ideally just base_plot_dir and maybe ONE subdir if the plotter strictly requires it.
        # Looking at previous file listings, `cam_ring_mapping` seemed to be the only one with unique content?
        # No, all of them had the same content.
        # So we just provide the base dir.
        # Wait, if I provide just the base dir, the plotter might expect a list.
        P_run["plot_dirs"] = [
            f"{base_plot_dir}/cam_ring_mapping"  # concise subfolder
        ]

        try:
            # 4. Run Optimization
            # Ensure P_run is passed, not P
            final_solution = solve_cycle(P_run)

            if final_solution and final_solution.success:
                print(f"\n[SUCCESS] {shape_name}: Optimization converged.")
                if hasattr(final_solution, "objective_value"):
                    print(f"Objective: {final_solution.objective_value}")
            else:
                status = "Unknown"
                if (
                    final_solution
                    and final_solution.meta
                    and "optimization" in final_solution.meta
                ):
                    status = final_solution.meta["optimization"].get(
                        "status", "Unknown"
                    )
                print(
                    f"\n[FAILURE] {shape_name}: Optimization failed with status: {status}"
                )
                had_failure = True
        except Exception as e:
            log.error(f"Run crashed for {shape_name}: {e}", exc_info=True)
            had_failure = True

    log.info("Batch run completed.")

    exit_code = 1 if had_failure else 0

    # Use platform-aware exit to bypass IPOPT/CasADi teardown segfault on macOS
    from campro.environment.resolve import exit_safely

    exit_safely(exit_code)


if __name__ == "__main__":
    run_optimization()
