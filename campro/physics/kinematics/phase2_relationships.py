"""
Deterministic Phase-2 relationship builder for animation.

This module constructs the runtime assembly state from already-optimized
Phase-2 results and the primary motion law without performing any solving
or optimization. It faithfully enforces the Phase-2 relationships:

- Cam radius: r_c(θ) = r_b + x(θ)
- Center stepping: C(ψ) = C0 + x(θ) resampled onto ψ, with C0 = r_b
- Angular synchronization: planet center angle = ψ
- No-slip at base circles: φ(ψ) = sign · (rb_ring/rb_cam) · (ψ − ψ₀)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from campro.logging import get_logger
from .litvin_assembly import AssemblyInputs, AssemblyState, compute_assembly_state

log = get_logger(__name__)


@dataclass
class Phase2AnimationInputs:
    """Container for deterministic Phase-2 animation inputs."""

    theta_deg: np.ndarray            # Primary motion θ grid [deg]
    x_theta_mm: np.ndarray           # Primary motion x(θ) [mm]
    base_radius_mm: float            # Optimized base radius r_b [mm] (used as C0)
    psi_rad: np.ndarray              # Synthesized ring angle ψ [rad]
    R_psi_mm: np.ndarray             # Ring instantaneous radius R(ψ) [mm]
    gear_geometry: Dict[str, Any]    # Gear bases and optional flanks/metadata
    contact_type: str = "internal"   # "internal" or "external"
    constrain_center_to_x_axis: bool = False  # If True, fix orbit angle to 0 (horizontal axis)
    align_tdc_at_theta0: bool = False        # If True, rotate ψ so max R occurs at θ=0


def build_phase2_relationships(inputs: Phase2AnimationInputs) -> AssemblyState:
    """Build deterministic Phase-2 assembly state for animation.

    Parameters
    ----------
    inputs : Phase2AnimationInputs
        Precomputed Phase-2 and primary motion data.

    Returns
    -------
    AssemblyState
        Assembly kinematic state honoring Phase-2 relationships without solving.
    """
    # Validate arrays
    theta_deg = np.asarray(inputs.theta_deg).flatten()
    x_theta = np.asarray(inputs.x_theta_mm).flatten()
    psi = np.asarray(inputs.psi_rad).flatten()
    R_psi = np.asarray(inputs.R_psi_mm).flatten()

    if theta_deg.size == 0 or x_theta.size == 0:
        raise ValueError("Primary motion arrays (theta_deg, x_theta_mm) must be non-empty")
    if psi.size == 0 or R_psi.size == 0:
        raise ValueError("Secondary arrays (psi_rad, R_psi_mm) must be non-empty")
    if theta_deg.size != x_theta.size:
        raise ValueError("theta_deg and x_theta_mm must have equal length")

    # Optionally align TDC at θ=0 by rotating ψ so that R(ψ) is maximized at index 0
    if inputs.align_tdc_at_theta0 and R_psi.size > 0:
        k = int(np.argmax(R_psi))
        if 0 < k < R_psi.size:
            psi = np.roll(psi, -k)
            R_psi = np.roll(R_psi, -k)

    # Extract base circles from gear geometry if present, otherwise infer reasonable defaults
    gg = inputs.gear_geometry or {}
    base_circle_cam = float(gg.get("base_circle_cam", max(1e-6, inputs.base_radius_mm * 0.95)))
    base_circle_ring = float(gg.get("base_circle_ring", max(1e-6, np.mean(R_psi) * 0.95)))
    z_cam = int(gg.get("z_cam", 20))

    # Align theta domain to radians for assembly inputs (optional for downstream tools)
    theta_cam_rad = np.deg2rad(theta_deg)

    assembly_inputs = AssemblyInputs(
        base_circle_cam=base_circle_cam,
        base_circle_ring=base_circle_ring,
        z_cam=z_cam,
        contact_type=inputs.contact_type,
        psi=psi,
        R_psi=R_psi,
        theta_cam_rad=theta_cam_rad,
        center_base_radius=float(inputs.base_radius_mm),
        motion_theta_deg=theta_deg,
        motion_offset_mm=x_theta,
    )

    state = compute_assembly_state(assembly_inputs)

    # If constrained to horizontal axis, override the orbit angle to 0 for all frames
    # while preserving the per-frame center radius and spin angle (no-slip still holds).
    if inputs.constrain_center_to_x_axis:
        state.planet_center_angle = np.zeros_like(state.planet_center_angle)

    # Diagnostics (optional; INFO-level): confirm invariants at a coarse level
    try:
        sign = -1.0 if inputs.contact_type == "internal" else 1.0
        dphi = np.gradient(state.planet_spin_angle)
        dpsi = np.gradient(psi)
        # Guard dt against zeros (constant speed is evaluated by proportion)
        dphi_dpsi = np.divide(dphi, np.maximum(dpsi, 1e-9))
        ratio = float(np.mean(dphi_dpsi))
        expected = sign * (base_circle_ring / max(base_circle_cam, 1e-9))
        log.info(
            f"Phase-2 no-slip check: mean(dφ/dψ) ≈ {ratio:.6f}, expected {expected:.6f}"
        )
    except Exception:
        # Do not raise in animation path; log only
        pass

    return state


