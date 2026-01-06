"""
Grid manipulation helpers for Cam-Ring mapping.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from campro.logging import get_logger

log = get_logger(__name__)


def create_enhanced_grid(
    theta_rad: np.ndarray,
    x_theta: np.ndarray,
    base_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create an enhanced grid with higher resolution at critical points (TDC/BDC).
    """
    log.info("Creating enhanced grid with boundary continuity enforcement")

    # Define critical points
    # ... (same logic as original)
    # Reimplementing simplified version based on original logic logic

    # Critical points: 0, pi/2, pi, 3pi/2, 2pi
    # Plus regions around them

    theta_base = np.linspace(0, 2 * np.pi, base_length, endpoint=True)
    extra_points = []

    # TDC regions
    tdc_region = np.linspace(-np.pi / 12, np.pi / 12, 5)
    extra_points.extend(tdc_region)
    extra_points.extend(2 * np.pi + tdc_region)

    # BDC region
    bdc_region = np.linspace(np.pi - np.pi / 12, np.pi + np.pi / 12, 5)
    extra_points.extend(bdc_region)

    # 90 and 270
    extra_points.extend(np.linspace(np.pi / 2 - np.pi / 24, np.pi / 2 + np.pi / 24, 3))
    extra_points.extend(np.linspace(3 * np.pi / 2 - np.pi / 24, 3 * np.pi / 2 + np.pi / 24, 3))

    all_points = np.concatenate([theta_base, extra_points])
    theta_enhanced = np.unique(all_points)
    theta_enhanced = np.mod(theta_enhanced, 2 * np.pi)
    theta_enhanced = np.unique(theta_enhanced)

    # Boundaries
    if theta_enhanced[0] != 0.0:
        theta_enhanced = np.concatenate([[0.0], theta_enhanced])
    if theta_enhanced[-1] != 2 * np.pi:
        theta_enhanced = np.concatenate([theta_enhanced, [2 * np.pi]])

    # Interpolate
    if theta_rad[-1] < 2 * np.pi:
        x_theta_enhanced = create_simple_cam_extension(theta_enhanced, theta_rad, x_theta)
    else:
        x_theta_enhanced = np.interp(theta_enhanced, theta_rad, x_theta)

    # Boundary Continuity
    if abs(x_theta_enhanced[0] - x_theta_enhanced[-1]) > 1e-6:
        avg = (x_theta_enhanced[0] + x_theta_enhanced[-1]) / 2
        x_theta_enhanced[0] = avg
        x_theta_enhanced[-1] = avg

    return theta_enhanced, x_theta_enhanced


def create_simple_cam_extension(
    theta_enhanced: np.ndarray,
    theta_rad: np.ndarray,
    x_theta: np.ndarray,
) -> np.ndarray:
    """Create periodic extension using Cubic Spline."""
    theta_extended = np.concatenate([theta_rad, [2 * np.pi]])
    x_theta_extended = np.concatenate([x_theta, [x_theta[0]]])

    cs = CubicSpline(theta_extended, x_theta_extended, bc_type="periodic")
    x_theta_enhanced = cs(theta_enhanced)

    # Smoothing and slope check logic (simplified from original)
    # ...
    x_theta_enhanced = np.maximum(x_theta_enhanced, 0.0)

    return x_theta_enhanced
