"""
Standalone Kinematic Inversion for Phase 2.
"""

import numpy as np


def inverse_slider_crank(
    x: np.ndarray,
    v: np.ndarray,
    stroke: float,
    conrod: float,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Invert slider-crank kinematics: Find crank angle psi from piston position x.

    Args:
        x: Piston position array (distance from crank center).
        v: Piston velocity array (dx/dt or dx/dtheta_cycle). Used to resolve quadrant.
        stroke: Engine stroke (2 * radius).
        conrod: Connecting rod length.
        offset: Piston pin offset (optional).

    Returns:
        psi: Crank angle array in [0, 2pi].
    """
    r = stroke / 2.0
    l = conrod

    # x = r*cos(psi) + sqrt(l^2 - (r*sin(psi) - offset)^2)
    # This equation is hard to invert directly with offset.
    # Without offset (offset=0):
    # x = r*c + sqrt(l^2 - r^2 s^2)
    # Cos law: l^2 = r^2 + x^2 - 2rx cos(psi)  <-- This is for triangle Crank-Rod-Slider
    # Yes! Cosine rule on triangle with sides r, l, x and angle psi between r and x.
    # r^2 + x^2 - 2rx cos(psi) = l^2
    # cos(psi) = (r^2 + x^2 - l^2) / (2rx)

    if abs(offset) > 1e-9:
        raise NotImplementedError("Offset slider crank inversion not yet implemented.")

    numerator = r**2 + x**2 - l**2
    denominator = 2 * r * x

    # Clip for numerical safety
    cos_psi = np.clip(numerator / denominator, -1.0, 1.0)

    # Initial angle in [0, pi]
    psi_base = np.arccos(cos_psi)

    # Resolve full circle using velocity sign
    # v = dx/dt.
    # x approx r*cos(psi) + l ...
    # dx/dpsi approx -r*sin(psi)
    # If dx/dt < 0 (moving down/in), then -r*sin(psi) * dpsi/dt < 0.
    # Assuming dpsi/dt > 0 (normal rotation):
    # sin(psi) > 0 implies v < 0.
    # sin(psi) < 0 implies v > 0.
    # psi in [0, pi] -> sin > 0 -> v < 0 (Expansion/Intake)
    # psi in [pi, 2pi] -> sin < 0 -> v > 0 (Compression/Exhaust)

    # So if v > 0, we are in [pi, 2pi] -> psi = 2pi - psi_base
    # If v < 0, we are in [0, pi] -> psi = psi_base

    psi = np.where(v >= 0, 2 * np.pi - psi_base, psi_base)

    # Unwrap to handle multi-cycle if needed (optional)
    return np.unwrap(psi)


def calculate_ideal_ratio(
    x: np.ndarray, v: np.ndarray, stroke: float, conrod: float, dphi_dtheta: float = 1.0
) -> np.ndarray:
    """
    Calculate Ideal Gear Ratio i_ideal = dpsi/dphi + 1 (for hypocycloid/internal).

    Args:
        x: Piston position.
        v: Piston velocity (dx/dtheta_cycle).
        stroke: Stroke.
        conrod: Rod length.
        dphi_dtheta: Rate of Ring Angle change per Cycle Angle change.
                     If theta_cycle is Ring Angle, this is 1.0.

    Returns:
        i_ideal: Instantaneous gear ratio.
    """
    psi = inverse_slider_crank(x, v, stroke, conrod)

    # We need dpsi/dphi = (dx/dphi) / (dx/dpsi)
    # v = dx/dtheta_cycle.
    # dx/dphi = (dx/dtheta_cycle) * (dtheta_cycle/dphi) = v / dphi_dtheta

    dx_dphi = v / dphi_dtheta

    # Compute analytical dx/dpsi
    r = stroke / 2.0
    l = conrod
    # x = r cos psi + sqrt(l^2 - r^2 sin^2 psi)
    # dx/dpsi = -r sin psi + (1/2 * ...) * (-2 r^2 sin psi cos psi)
    # dx/dpsi = -r sin psi - (r^2 sin psi cos psi) / sqrt(l^2 - r^2 sin^2 psi)

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    root_term = np.sqrt(np.maximum(l**2 - r**2 * sin_psi**2, 1e-9))

    dx_dpsi = -r * sin_psi - (r**2 * sin_psi * cos_psi) / root_term

    # Avoid division by zero at TDC/BDC (dx_dpsi = 0)
    # At dead centers, Velocity should also be 0, so ratio is 0/0 limit.
    # Use L'Hopital or just clip?
    # Physically, dpsi/dphi is constrained by transmission.
    # Let's clip denominator magnitude.

    safe_dx_dpsi = np.where(np.abs(dx_dpsi) < 1e-4, np.sign(dx_dpsi + 1e-9) * 1e-4, dx_dpsi)

    dpsi_dphi = dx_dphi / safe_dx_dpsi

    # Definition of ratio i for internal gears:
    # dpsi/dphi = i - 1
    # i = dpsi/dphi + 1

    i_ideal = dpsi_dphi + 1.0

    return i_ideal
