"""
Litvin assembly kinematics utilities.

This module converts phase-2 Litvin synthesis results into per-frame
assembly states for animation and analysis. The assembly is driven by
ring contact angle psi (radians) and ring radius R_psi (mm) and uses
base-circle kinematics to compute the planet (cam) orbit and spin.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class AssemblyInputs:
    """Inputs required to compute Litvin assembly state."""

    base_circle_cam: float  # planet (cam) base circle radius (mm)
    base_circle_ring: float  # ring base circle radius (mm)
    z_cam: int  # number of planet teeth
    contact_type: str  # "external" or "internal"
    psi: np.ndarray  # ring angle array [rad]
    R_psi: np.ndarray  # ring instantaneous radius [mm]
    theta_cam_rad: np.ndarray  # cam profile theta [rad], aligned with psi if available
    # Center stepping inputs (Phase 2)
    center_base_radius: Optional[float] = (
        None  # C0 [mm], user-provided initial planet center radius
    )
    motion_theta_deg: Optional[np.ndarray] = None  # θ grid (deg) of primary motion
    motion_offset_mm: Optional[np.ndarray] = None  # x(θ) in mm (≥0)


@dataclass
class AssemblyState:
    """Per-frame assembly state arrays for animation/drawing."""

    center_distance: float
    planet_center_angle: np.ndarray  # radians (same as psi)
    planet_center_radius: np.ndarray  # constant = C
    planet_spin_angle: np.ndarray  # radians
    contact_theta: np.ndarray  # same as psi
    contact_radius: np.ndarray  # same as R_psi
    z_cam: int


def _center_distance(rb_ring: float, rb_cam: float, contact_type: str) -> float:
    if contact_type == "internal":
        return float(rb_ring + rb_cam)
    return float(rb_ring - rb_cam)


def compute_assembly_state(
    inputs: AssemblyInputs, ring_omega: Optional[float] = None,
) -> AssemblyState:
    """Compute assembly kinematics from Litvin results.

    The no-slip base-circle relationship gives dphi = (rb_ring/rb_cam) dpsi
    so phi[k] = sign * (rb_ring/rb_cam) * (psi[k] - psi[0]).
    """
    psi = np.asarray(inputs.psi)
    R_psi = np.asarray(inputs.R_psi)
    rb_cam = float(inputs.base_circle_cam)
    rb_ring = float(inputs.base_circle_ring)
    z_cam = int(inputs.z_cam)
    if len(psi) == 0 or len(R_psi) == 0:
        raise ValueError("psi and R_psi must be non-empty arrays")

    # Center radius stepping synchronized to motion law: C(θ) = C0 + x(θ)
    # Resample offset onto ψ-grid if motion arrays are provided
    if inputs.motion_theta_deg is not None and inputs.motion_offset_mm is not None:
        th = np.asarray(inputs.motion_theta_deg).flatten()
        off = np.asarray(inputs.motion_offset_mm).flatten()
        if th.size != off.size or th.size == 0:
            raise ValueError(
                "motion_theta_deg and motion_offset_mm must be same non-zero length",
            )
        xsrc = np.linspace(0.0, 1.0, th.size)
        xdst = np.linspace(0.0, 1.0, psi.size)
        offset = np.interp(xdst, xsrc, off)
    else:
        offset = np.zeros_like(psi)

    # Default base center distance if not supplied
    C_default = _center_distance(rb_ring, rb_cam, inputs.contact_type)
    C0 = (
        float(inputs.center_base_radius)
        if inputs.center_base_radius is not None
        else float(C_default)
    )
    planet_center_radius = C0 + offset

    # Spin sign: internal meshes reverse spin relative to ring rotation.
    sign = -1.0 if inputs.contact_type == "internal" else 1.0
    ratio = (rb_ring / rb_cam) if rb_cam != 0 else 0.0
    phi = sign * ratio * (psi - psi[0])

    planet_center_angle = psi.copy()

    return AssemblyState(
        center_distance=float(np.mean(planet_center_radius)),
        planet_center_angle=planet_center_angle,
        planet_center_radius=planet_center_radius,
        planet_spin_angle=phi,
        contact_theta=psi,
        contact_radius=R_psi,
        z_cam=z_cam,
    )


def transform_to_world_polar(
    tooth_theta_local: np.ndarray,
    tooth_r_local: np.ndarray,
    center_distance: float,
    orbit_angle: float,
    spin_angle: float,
    tooth_index: int,
    z_cam: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a single planet tooth from local polar to world polar.

    - Rotate the tooth by total angle = spin_angle + tooth_index * 2π / z_cam
    - Translate by planet center at distance C and angle orbit_angle
    - Return world polar arrays (theta_world [rad], r_world [mm])
    """
    if z_cam <= 0:
        raise ValueError("z_cam must be positive")
    # Local rotation
    theta_rot = tooth_theta_local + spin_angle + (2.0 * np.pi * tooth_index / z_cam)
    r = tooth_r_local

    # Convert to Cartesian in planet frame
    x_local = r * np.cos(theta_rot)
    y_local = r * np.sin(theta_rot)

    # Orbit translation to world coords (ring-centered)
    cx = center_distance * np.cos(orbit_angle)
    cy = center_distance * np.sin(orbit_angle)
    x_world = x_local + cx
    y_world = y_local + cy

    # World polar
    theta_world = np.arctan2(y_world, x_world)
    r_world = np.hypot(x_world, y_world)
    return theta_world, r_world


def compute_global_rmax(state: AssemblyState) -> float:
    """Compute a stable global radial limit for fixed scaling plots."""
    max_contact = float(np.max(state.contact_radius))
    # Max distance of farthest planet point is center distance + rb_cam (approx upper bound)
    approx_planet_max = float(state.center_distance) + 1.0 * max_contact
    return 1.1 * max(max_contact, approx_planet_max)
