"""
Geometric computations for Cam-Ring systems.
"""

import numpy as np

from campro.logging import get_logger

from .config import CamRingParameters

log = get_logger(__name__)


def compute_cam_curves(
    theta: np.ndarray,
    x_theta: np.ndarray,
    params: CamRingParameters,
) -> dict[str, np.ndarray]:
    """Compute cam curves from linear follower motion."""
    log.info(f"Computing cam curves for {len(theta)} points")
    pitch_radius = params.base_radius + x_theta
    profile_radius = pitch_radius  # Direct contact
    contact_radius = profile_radius.copy()

    return {
        "pitch_radius": pitch_radius,
        "profile_radius": profile_radius,
        "contact_radius": contact_radius,
        "theta": theta,
    }


def compute_cam_curvature(theta: np.ndarray, r_contact: np.ndarray) -> np.ndarray:
    """Compute curvature of contacting cam curve."""
    dr = np.gradient(r_contact, theta)
    d2r = np.gradient(dr, theta)

    num = r_contact**2 + 2 * dr**2 - r_contact * d2r
    den = (r_contact**2 + dr**2) ** 1.5

    kappa = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return kappa


def compute_osculating_radius(kappa: np.ndarray) -> np.ndarray:
    """Compute osculating radius."""
    rho = np.divide(1.0, kappa, out=np.full_like(kappa, np.inf), where=np.abs(kappa) > 1e-12)
    return rho


def design_ring_radius(
    psi: np.ndarray,
    design_type: str = "constant",
    **kwargs,
) -> np.ndarray:
    """Design ring instantaneous radius R(psi)."""
    if design_type == "constant":
        base = kwargs.get("base_radius", 15.0)
        return np.full_like(psi, base)

    if design_type == "linear":
        base = kwargs.get("base_radius", 15.0)
        slope = kwargs.get("slope", 0.0)
        return base + slope * psi

    if design_type == "sinusoidal":
        base = kwargs.get("base_radius", 15.0)
        amp = kwargs.get("amplitude", 2.0)
        freq = kwargs.get("frequency", 1.0)
        return base + amp * np.sin(freq * psi)

    # Custom function support via kwargs? Hard to serialize.
    # Assuming user passes function if calling directly.
    if design_type == "custom":
        func = kwargs.get("custom_function")
        if func:
            return func(psi)

    raise ValueError(f"Unknown design type: {design_type}")


def validate_design(results: dict[str, np.ndarray]) -> dict[str, bool]:
    """Validate cam-ring design."""
    validation = {}
    kappa = results["kappa_c"]
    validation["no_cusps"] = bool(np.all(np.abs(kappa) < 10.0))
    validation["positive_cam_radii"] = bool(np.all(results["cam_curves"]["profile_radius"] > 0))
    validation["positive_ring_radii"] = bool(np.all(results["R_psi"] > 0))
    validation["reasonable_curvature"] = bool(np.all(np.isfinite(kappa)))
    validation["reasonable_osculating_radius"] = bool(np.all(np.isfinite(results["rho_c"])))
    validation["smooth_angle_relationship"] = bool(np.all(np.isfinite(results["psi"])))
    return validation
