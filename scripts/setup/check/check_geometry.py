import logging
import os
import sys

import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from campro.optimization.core.polar_geometry import PolarCamGeometry


def check_geometry():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("check_geo")

    # User Params
    stroke = 0.02
    od = 0.042
    theta_bdc = np.pi
    # Scaled ratios from run_planet_ring_opt.py
    ratios = (6.0 * 1e-3, 0.3333 * 1e-3)

    log.info(f"Params: Stroke={stroke}, OD={od}, Ratios={ratios}")

    geo = PolarCamGeometry(stroke, od, theta_bdc, ratios)

    # Evaluate grid
    theta = np.linspace(0, 2 * np.pi, 200)
    r_vals = []
    dr_vals = []

    min_r = 1e9
    max_r = -1e9

    for t in theta:
        r, dr, _ = geo.evaluate(t)
        r_vals.append(r)
        dr_vals.append(dr)

        min_r = min(min_r, r)
        max_r = max(max_r, r)

    log.info(f"Min Radius: {min_r:.6f} m")
    log.info(f"Max Radius: {max_r:.6f} m")

    theoretical_min = (od / 2) - stroke
    log.info(f"Theoretical Min: {theoretical_min:.6f} m")

    if min_r < 0:
        log.error("CRITICAL: Negative Radius detected! Pistons will collide/cross.")
    elif min_r < theoretical_min - 1e-4:
        log.warning("Spline undershoot detected below theoretical min.")

    log.info(f"Max Velocity Factor (dr/dtheta): {max(np.abs(dr_vals)):.4f}")


if __name__ == "__main__":
    check_geometry()
