from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from campro.logging import get_logger
from campro.optimization.driver import solve_cycle

log = get_logger(__name__)


import glob
import json


def visualize_breathing_gears():
    """
    Run Phase 3 optimization and generate diagnostic plots for each shape file.
    """
    # Find all shape files
    shape_files = sorted(glob.glob("shapes/*.json"))

    if not shape_files:
        log.warning("No shape files found in shapes/*.json")
        return

    for shape_file in shape_files:
        shape_name = Path(shape_file).stem
        log.info(f"Processing shape: {shape_name}")

        try:
            with open(shape_file) as f:
                shape_data = json.load(f)

            target_profile = shape_data.get("ratio_profile")
            if not target_profile:
                log.warning(f"Skipping {shape_name}: No 'ratio_profile' found")
                continue

            # Ensure it's a list
            if not isinstance(target_profile, list):
                target_profile = list(target_profile)

            # 1. Setup Problem
            track_weight = 100.0
            P = {
                "problem_type": "phase3_mechanical",
                "target_ratio_profile": target_profile,
                "design": {
                    "r_planet_min": 0.02,
                    "r_planet_max": 0.05,
                    "R_ring_min": 0.10,
                    "R_ring_max": 0.15,
                    "b": 0.02,
                },
                "num": {"K": 20, "C": 3},  # Match test config
                "solver": {"ipopt": {"ipopt.max_iter": 500, "ipopt.print_level": 5}},
                "weights": {"tracking": track_weight, "efficiency": 0.1, "stress": 0.1},
                "load_profile": {
                    "angle": np.linspace(0, 2 * np.pi, 100).tolist(),
                    "F_gas": (1000 * np.sin(np.linspace(0, 2 * np.pi, 100)) ** 2).tolist(),
                },
            }

            # 2. Solve
            log.info(f"Running optimization for {shape_name}...")
            res = solve_cycle(P)

            # Check success
            is_success = False
            if hasattr(res, "meta") and isinstance(res.meta, dict):
                opt_res = res.meta.get("optimization", {})
                if isinstance(opt_res, dict):
                    is_success = opt_res.get("success", False)

            if not is_success:
                log.error(f"Optimization failed for {shape_name}! Plotting available results...")

            # 3. Extract Results
            opt_res = res.meta.get("optimization", {})
            mech_data = opt_res.get("mechanical", {})

            if not mech_data:
                log.error(f"No mechanical data found for {shape_name}")
                continue

            # Extract data (lists -> arrays)
            psi = np.array(mech_data.get("psi", []))
            r_p = np.array(mech_data.get("r", []))
            R_r = np.array(mech_data.get("R", []))
            r_j = np.zeros_like(r_p)  # Placeholder
            phi = np.array(mech_data.get("phi", []))

            # 4. Create Output Directory
            out_dir = Path(f"out/plots/breathing_gears/{shape_name}")
            out_dir.mkdir(parents=True, exist_ok=True)

            # 5. Plot 1: Geometric Profiles
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(phi, r_p * 1000, label="$r_{planet}$ (mm)", linewidth=2)
            ax.plot(phi, R_r * 1000, label="$R_{ring}$ (mm)", linewidth=2)
            ax.plot(phi, r_j * 1000, label="$r_{journal}$ (mm)", linewidth=2, linestyle="--")

            ax.set_xlabel("Cam Angle $\phi$ (rad)")
            ax.set_ylabel("Radius (mm)")
            ax.set_title(f"Breathing Gear Geometric Profiles ({shape_name})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            save_path = out_dir / "geom_profiles.png"
            fig.savefig(save_path, dpi=300)
            log.info(f"Saved {save_path}")
            plt.close(fig)

            # 6. Plot 2: Gear Ratio Tracking
            ratio_actual = R_r / r_planet_safe(r_p)

            # Interpolate target profile to result grid for comparison
            # Target profile is defined on [0, 2pi] usually.
            # Assuming target_profile corresponds to linspace(0, 2pi, len(target_profile))
            target_phi = np.linspace(0, 2 * np.pi, len(target_profile))
            ratio_target_interp = np.interp(phi, target_phi, target_profile)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(phi, ratio_actual, label="Actual Ratio", linewidth=2)
            ax.plot(phi, ratio_target_interp, label="Target Ratio", linestyle="--", alpha=0.7)

            ax.set_xlabel("Cam Angle $\phi$ (rad)")
            ax.set_ylabel("Gear Ratio (-)")
            ax.set_title(f"Gear Ratio Tracking ({shape_name})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            save_path = out_dir / "ratio_tracking.png"
            fig.savefig(save_path, dpi=300)
            log.info(f"Saved {save_path}")
            plt.close(fig)

            # 7. Plot 3: Polar Shapes
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
            ax.plot(phi, R_r, label="Ring Profile")
            ax.plot(phi, r_p + 0.05, label="Planet Profile (Offset)")

            ax.set_title(f"Polar Profiles ({shape_name})")
            ax.legend()

            save_path = out_dir / "polar_shapes.png"
            fig.savefig(save_path, dpi=300)
            log.info(f"Saved {save_path}")
            plt.close(fig)

        except Exception as e:
            log.error(f"Failed to process {shape_name}: {e}")
            import traceback

            traceback.print_exc()


def r_planet_safe(r_p):
    return np.maximum(r_p, 1e-6)


if __name__ == "__main__":
    visualize_breathing_gears()
