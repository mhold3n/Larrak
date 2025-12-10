
import numpy as np

from campro.optimization.driver import solve_cycle

# Base Parameters
P = {
    "planet_ring": {
        "stroke": 0.02,
        "outer_diameter": 0.042,
        "theta_bdc": np.pi,
        "gen_mode": "litvin_debug_geometry",
        "use_load_model": True,
    },
    "combustion": {
        "cycle_time_s": 0.02,
        "use_integrated_model": True,
        "ign_duration": 0.002,
        "Q_total": 500.0,
        "fuel_mass": 2e-6,
    },
    "geometry": {
        "bore": 0.016,
        "stroke": 0.02,
        "mass": 0.04,
        "c_load": 1.25,  # Tuned for 20mm stroke at 2:1?
    },
    "flow": {"mesh_cells": 10},
    "bounds": {
        "xL_min": -0.05,
        "xL_max": 0.05,
        "vL_min": -50,
        "vL_max": 50,
        "rho_min": 0.1,
        "rho_max": 100,
        "T_min": 200,
        "T_max": 3000,
        "p_max": 200e5,
        "Ain_max": 5e-4,
        "Aex_max": 5e-4,
    },
    "num": {
        "K": 20,  # Reduced for speed in sweep
        "C": 3,
    },
    "solver": {
        "ipopt": {
            "max_iter": 500,
            "tol": 1e-4,
            "print_level": 3,
            "linear_solver": "ma57",
        },
        "plateau_check_enabled": False,
    },
    "obj": {
        "type": "work",
        "w": {"work": 1.0, "mass_penalty": 1e-4},
    },
    "auto_plot": True,
    "plot_dirs": ["plots"],
}


def test_ratio(target_mean):
    print(f"--- Testing Target Ratio ~ {target_mean} ---")
    # Tighten bounds around target to force exploration of this regime
    delta = 0.5
    P["bounds"]["i_ratio_min"] = max(1.1, target_mean - delta)
    P["bounds"]["i_ratio_max"] = target_mean + delta

    # Set initial guess to ensure feasibility at start
    P["bounds"]["i_ratio_init"] = target_mean

    # Try to set initial guess if possible?
    # Not easily accessible via P dict without modifying nlp.py defaults.
    # But nlp.py uses init=2.0. If we target 3.0, 2.0 is feasible?
    # If 2.0 is out of bounds [2.5, 3.5], initialization will be clamped or infeasible.

    # Relax tolerances
    P["solver"]["ipopt"]["tol"] = 1e-3
    P["solver"]["ipopt"]["max_iter"] = 1000

    # Update load? Higher ratio might mean different stroke/velocity mapping?
    # For now, keep physics same.

    try:
        sol = solve_cycle(P)
        success = sol.success
        obj = sol.objective_value

        # Extract profile stats
        profs = sol.meta["optimization"].get("profiles", {})
        i_stats = np.array(profs.get("i", []))

        print(f"Success: {success}")
        print(f"Objective: {obj}")
        if len(i_stats) > 0:
            print(
                f"Result i_ratio: Mean={i_stats.mean():.4f}, Range=[{i_stats.min():.4f}, {i_stats.max():.4f}]"
            )
            return sol, i_stats
        else:
            print("No profile data.")
            return sol, None

    except Exception as e:
        print(f"Run failed: {e}")
        return None, None


def main():
    ratios = [2.0, 2.5, 3.0]
    results = {}

    for r in ratios:
        sol, i_data = test_ratio(r)
        if sol and sol.success:
            results[r] = i_data

            # Save a plot for this ratio?
            # We can reuse the plotting script logic or just rely on the main script's output if we ran it.
            # But here we are running `solve_cycle` directly.
            pass


if __name__ == "__main__":
    main()
