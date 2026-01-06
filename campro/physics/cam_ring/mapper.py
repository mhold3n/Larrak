"""
Main Mapper logic for Cam-Ring system.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campro.logging import get_logger

from .config import CamRingParameters
from .geometry import (
    compute_cam_curvature,
    compute_cam_curves,
    compute_osculating_radius,
    design_ring_radius,
    validate_design,
)
from .grid import create_enhanced_grid
from .meshing import compute_time_kinematics, generate_complete_ring_profile, solve_meshing_law

log = get_logger(__name__)


class CamRingMapper:
    """
    Cam-ring-linear follower mapping system.

    Orchestrates the mapping from linear follower to ring follower design.
    Delegates core logic to submodules.
    """

    def __init__(self, parameters: CamRingParameters | None = None):
        self.parameters = parameters or CamRingParameters()
        log.info(f"Initialized CamRingMapper: {self.parameters.to_dict()}")

    # Expose helper methods as instance methods for compatibility
    def compute_cam_curves(self, theta: np.ndarray, x_theta: np.ndarray) -> dict[str, np.ndarray]:
        return compute_cam_curves(theta, x_theta, self.parameters)

    def compute_cam_curvature(self, theta: np.ndarray, r_contact: np.ndarray) -> np.ndarray:
        return compute_cam_curvature(theta, r_contact)

    def compute_osculating_radius(self, kappa_c: np.ndarray) -> np.ndarray:
        return compute_osculating_radius(kappa_c)

    def design_ring_radius(
        self, psi: np.ndarray, design_type: str = "constant", **kwargs
    ) -> np.ndarray:
        return design_ring_radius(psi, design_type, **kwargs)

    def solve_meshing_law(
        self,
        theta: np.ndarray,
        rho_c: np.ndarray,
        psi: np.ndarray,
        R_psi: np.ndarray,
    ) -> np.ndarray:
        return solve_meshing_law(theta, rho_c, psi, R_psi, self.parameters)

    def compute_time_kinematics(
        self,
        theta: np.ndarray,
        psi: np.ndarray,
        rho_c: np.ndarray,
        R_psi: np.ndarray,
        driver: str = "cam",
        omega: float | None = None,
        Omega: float | None = None,
    ) -> dict[str, np.ndarray]:
        return compute_time_kinematics(theta, psi, rho_c, R_psi, driver, omega, Omega)

    def _create_enhanced_grid(
        self, theta_rad, x_theta, base_length
    ) -> tuple[np.ndarray, np.ndarray]:
        return create_enhanced_grid(theta_rad, x_theta, base_length)

    def _generate_complete_ring_profile(self, psi, R_psi) -> tuple[np.ndarray, np.ndarray]:
        return generate_complete_ring_profile(psi, R_psi)

    def validate_design(self, results: dict[str, np.ndarray]) -> dict[str, bool]:
        return validate_design(results)

    def map_linear_to_ring_follower(
        self,
        theta: np.ndarray,
        x_theta: np.ndarray,
        ring_design: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Complete mapping pipeline."""
        log.info("Starting complete linear-to-ring follower mapping")
        theta_rad = np.deg2rad(theta)

        # 1. Cam Curves
        cam_curves = self.compute_cam_curves(theta_rad, x_theta)

        # 2. Curvature
        kappa_c = self.compute_cam_curvature(theta_rad, cam_curves["contact_radius"])
        rho_c = self.compute_osculating_radius(kappa_c)

        # 3. Initial Ring Design (Guess)
        psi_init = np.linspace(0, 2 * np.pi, len(theta))
        R_psi_init = self.design_ring_radius(psi_init, **ring_design)

        # 4. Meshing Law
        psi = self.solve_meshing_law(theta_rad, rho_c, psi_init, R_psi_init)

        # 5. Re-evaluate Ring Radius
        R_psi = self.design_ring_radius(psi, **ring_design)

        # 6. Complete Profiles (Grid Enhancement)
        theta_comp, x_comp = self._create_enhanced_grid(theta_rad, x_theta, len(theta))
        theta_deg_comp = np.degrees(theta_comp)

        # Recalculate full cam curves
        cam_curves_comp = self.compute_cam_curves(theta_comp, x_comp)
        cam_curves_comp["theta"] = theta_deg_comp  # Fix unit

        # Generate Ring Profile (reusing theta grid logic for ring? Original code reused theta_comp for psi?)
        # Original: "psi_complete = theta_complete.copy() # Use same enhanced grid for ring"
        # This assumes psi maps 1:1 to theta roughly, or just reuses the grid points?
        # Actually R(psi) depends on psi.
        # "R_psi_complete = self.design_ring_radius(psi_complete, **ring_design)"
        # So it just designs a ring profile on that grid.
        psi_comp = theta_comp.copy()
        R_psi_comp = self.design_ring_radius(psi_comp, **ring_design)

        # Kinematics
        time_kin = None
        if ring_design.get("compute_time_kinematics", False):
            driver = ring_design.get("driver", "cam")
            omega = ring_design.get("omega")
            Omega = ring_design.get("Omega")
            time_kin = self.compute_time_kinematics(
                theta_rad, psi, rho_c, R_psi, driver, omega, Omega
            )

        return {
            "theta": theta_deg_comp,
            "x_theta": x_comp,
            "cam_curves": cam_curves_comp,
            "kappa_c": kappa_c,  # On original grid? Original code returned original kappa?
            # Original code: returned kappa_c computed on theta_rad (original)
            "rho_c": rho_c,
            "psi": psi_comp,
            "R_psi": R_psi_comp,
            "time_kinematics": time_kin,
        }
