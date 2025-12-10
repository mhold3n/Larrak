import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from campro.optimization.driver import solve_cycle

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("verify_kinematics")


def verify():
    log.info("Starting Kinematic Verification (Pitch Curve Consistency)...")

    # 1. Load Shape
    shape_file = "shapes/sine.json"
    with open(shape_file) as f:
        shape_data = json.load(f)

    # 2. Setup Config
    P = {
        "problem_type": "kinematic",
        "planet_ring": {
            "stroke": 0.02,
            "outer_diameter": 0.042,
            "theta_bdc": np.pi,
            "ratios": (6.0 * (0.02 / np.pi), 0.3333 * (0.02 / np.pi)),
            "gen_mode": "litvin_debug_geometry",
            "mean_ratio": 2.0,
            "r_drive": 0.01,
            "attachment_radius": 0.0,
        },
        "geometry": {
            "bore": 0.016,
            "stroke": 0.02,
            "mass": 0.04,
        },
        "bounds": {
            "r_planet_min": 0.005,
            "r_planet_max": 0.050,
            "R_ring_min": 0.010,
            "R_ring_max": 0.100,
            "xL_min": -0.01,
            "xL_max": 0.03,
        },
        "constraints": {},
        "num": {
            "K": 40,
            "C": 3,
            "cycle_time": 0.02,
        },
        "solver": {
            "ipopt": {
                "print_level": 0,
                "linear_solver": "ma57",
            },
        },
        "obj": {
            "type": "kinematic_tracking",
            "w": {"ratio_tracking": 10.0},
        },
        "auto_plot": False,
        "shape_file": shape_file,
    }

    if "constraints" in shape_data:
        P["planet_ring"].update(shape_data["constraints"])
        P["constraints"].update(shape_data["constraints"])  # Critical Fix
        P["bounds"]["xL_min"] = shape_data["constraints"]["x_limits"][0]
        P["bounds"]["xL_max"] = shape_data["constraints"]["x_limits"][1]
    if "geometry" in shape_data:
        P["geometry"].update(shape_data["geometry"])
    if "operating_point" in shape_data:
        P["num"]["cycle_time"] = shape_data["operating_point"]["cycle_time"]

    # 3. Run Optimization
    log.info("Running Kinematic Optimization...")
    solution = solve_cycle(P)

    if not solution.success:
        log.error("Optimization failed!")
        return

    # 4. Extract Results
    meta = solution.meta["meta"]
    x_opt = solution.data["x"]

    def get_var(name):
        indices = meta["variables_detailed"].get(name, [])
        return x_opt[indices]

    r_planet = get_var("r_planet")
    R_ring = get_var("R_ring")
    psi = get_var("psi")

    # --- Broadcasting Fix ---
    def match_shape(ctrl, target_arr):
        if len(ctrl) == len(target_arr):
            return ctrl
        if len(ctrl) < len(target_arr):
            matched = []
            K_ctrl = len(ctrl)
            N_target = len(target_arr)
            # Typically N = K*(C+1) + 1. Or similar.
            # We assume piecewise constant controls (1 per interval).
            # Each interval has (N-1)//K points usually.
            c_plus_1 = (N_target - 1) // K_ctrl if K_ctrl > 0 else 0
            c_plus_1 = max(c_plus_1, 1)  # Fallback

            for k in range(K_ctrl):
                val = ctrl[k]
                matched.extend([val] * c_plus_1)

            # Fill remainder
            while len(matched) < N_target:
                matched.append(ctrl[-1])

            return np.array(matched[:N_target])
        return ctrl

    r_planet = match_shape(r_planet, psi)
    R_ring = match_shape(R_ring, psi)
    # ------------------------

    N = len(r_planet)
    phi_grid = np.linspace(0, 2 * np.pi, N)  # Approximate grid for plotting

    log.info(f"Extracted profiles. N={N}")

    # 5. Verify Kinematics
    # Check 1: Closure (PSI should advance by 2*pi * (MeanRatio - 1)?)
    # Wait, the periodicity constraint in NLP enforces psi_end - psi_0 = 2*pi*(i-1).
    psi_start = psi[0]
    psi_end = psi[-1]
    delta_psi = psi_end - psi_start
    # Note: Depending on collocation, the last point might be the start of next cycle or end of this one.
    # Usually endpoint is constrained.
    # Expected advance for ratio 2.0: (2-1)*360 = 360 deg = 2pi.

    log.info(f"Psi Delta: {delta_psi:.4f} rad")
    log.info(f"Expected : {2.0 * np.pi:.4f} rad (for Ratio 2:1)")

    # Check 2: Physical Realism
    # xL = (R - r) * cos(psi)
    # This is the derived piston position.
    xL_derived = (R_ring - r_planet) * np.cos(psi)
    x_min_obs = np.min(xL_derived)
    x_max_obs = np.max(xL_derived)
    stroke_obs = x_max_obs - x_min_obs

    log.info(f"Observed Stroke: {stroke_obs * 1000:.2f} mm")
    log.info(f"Target Stroke  : {P['planet_ring']['stroke'] * 1000:.2f} mm")

    if abs(stroke_obs - P["planet_ring"]["stroke"]) < 1e-3:
        log.info("[SUCCESS] Stroke matches target within tolerance.")
    else:
        log.warning("[WARNING] Stroke deviation detected.")

    # Check 3: Constraints
    # Ensure r and R are always positive
    if np.all(r_planet > 0) and np.all(R_ring > 0):
        log.info("[SUCCESS] Radii are physically positive.")
    else:
        log.error("[FAILURE] Negative radii detected!")

    # Plot Pitch Curves
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(psi, r_planet, "b-", label="Planet Pitch (r)", linewidth=2)
    # For Ring, it's stationary in one frame, but relative in another.
    # Usually we plot them both?
    # Just plotting r_planet vs psi gives the planet shape.
    ax.set_title(f"Optimized Planet Pitch Curve\nStroke: {stroke_obs * 1000:.1f}mm")
    ax.legend()
    plt.savefig("plots/verification_pitch_curve_kinematics.png")
    log.info("Saved pitch curve plot to plots/verification_pitch_curve_kinematics.png")


if __name__ == "__main__":
    verify()
