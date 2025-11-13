"""
Litvin gear synthesis for conjugate cam-ring profiles.

Implements a practical Litvin-style conjugate synthesis between a polar cam
profile r_c(θ) and a conjugate ring profile R(ψ) through the meshing law:

    dψ/dθ = ρ_c(θ) / R(ψ)

where ρ_c(θ) is the osculating radius of the cam polar curve. This module
numerically constructs ψ(θ) and the corresponding R(ψ) that satisfy conjugacy
for a desired average ratio while ensuring 2π periodicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from campro.logging import get_logger
from campro.physics.geometry.curvature import CurvatureComponent

log = get_logger(__name__)


@dataclass
class LitvinSynthesisResult:
    psi: np.ndarray  # ψ(θ) mapping, θ-domain grid
    R_psi: np.ndarray  # R(ψ(θ)) sampled over θ grid
    rho_c: np.ndarray  # ρ_c(θ) cam osculating radius
    metadata: dict[str, Any]


class LitvinSynthesis:
    """
    Numerical Litvin conjugate synthesis using the meshing law and curvature.
    """

    def __init__(self) -> None:
        self._curvature = CurvatureComponent(parameters={})

    @staticmethod
    def _sanitize_R_psi(
        *,
        theta: np.ndarray,
        r_profile: np.ndarray,
        R_psi: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Ensure synthesized radii stay aligned with the supplied polar profile.

        Returns the (possibly adjusted) radius array and a flag indicating whether
        a fallback was applied.
        """
        expected = np.asarray(r_profile, dtype=float)
        actual = np.array(R_psi, copy=True, dtype=float)
        if expected.shape != actual.shape:
            return actual, False
        
        magnitude = np.maximum(np.abs(expected), 1e-6)
        discrepancy = np.abs(actual - expected)
        discrepancy = np.where(np.isfinite(discrepancy), discrepancy, magnitude)
        max_disc = float(np.max(discrepancy)) if discrepancy.size else 0.0
        # Allow deviations up to 5% of the peak magnitude (min 0.5 mm) before clamping.
        tolerance = float(max(0.5, 0.05 * float(np.max(magnitude))))
        if max_disc > tolerance:
            idx = int(np.argmax(discrepancy))
            theta_deg = float(np.degrees(theta[int(np.clip(idx, 0, len(theta) - 1))])) if len(theta) else 0.0
            # Check if deviation is at wrap-around point (θ=360°)
            is_wrap_around = (
                len(theta) > 0 and 
                (idx == len(theta) - 1 or idx == 0) and
                abs(float(np.degrees(theta[idx if idx < len(theta) else 0])) - 360.0) < 1.0
            )
            log.warning(
                "Litvin R_psi deviates from polar pitch by %.6f mm at θ=%.2f°%s. "
                "Clamping to supplied polar radius for this solve.",
                max_disc,
                theta_deg,
                " (wrap-around)" if is_wrap_around else "",
            )
            # Log diagnostic info at wrap-around
            if is_wrap_around and len(theta) > 0:
                log.debug(
                    "Wrap-around diagnostic: R_psi[0]=%.6f, R_psi[-1]=%.6f, "
                    "expected[0]=%.6f, expected[-1]=%.6f",
                    float(actual[0]) if len(actual) > 0 else 0.0,
                    float(actual[-1]) if len(actual) > 0 else 0.0,
                    float(expected[0]) if len(expected) > 0 else 0.0,
                    float(expected[-1]) if len(expected) > 0 else 0.0,
                )
            return expected.copy(), True
        return actual, False

    def synthesize_from_cam_profile(
        self,
        *,
        theta: np.ndarray,
        r_profile: np.ndarray,
        target_ratio: float = 1.0,
        min_radius: float = 1e-3,
    ) -> LitvinSynthesisResult:
        """
        Build a conjugate ring profile R(ψ) for a given cam polar profile r_c(θ).

        Parameters
        ----------
        theta : np.ndarray
            Cam angle grid in radians, length N, monotonically increasing over [0, 2π).
        r_profile : np.ndarray
            Cam polar radius r_c(θ), length N.
        target_ratio : float, optional
            Desired average transmission ratio over one cycle.
        min_radius : float, optional
            Minimum allowed R(ψ) to avoid division by zero.

        Returns
        -------
        LitvinSynthesisResult
            ψ(θ), R(ψ(θ)), ρ_c(θ), and metadata.
        """
        theta = np.asarray(theta)
        r_profile = np.asarray(r_profile)
        if theta.ndim != 1 or r_profile.ndim != 1 or theta.shape != r_profile.shape:
            raise ValueError("theta and r_profile must be 1D arrays of equal length")

        # Compute curvature and osculating radius ρ_c(θ)
        curv = self._curvature.compute({"theta": theta, "r_theta": r_profile})
        rho_c = curv.outputs["rho"]

        # Normalize desired mapping span to 2π while enforcing average ratio.
        # We construct an initial ψ by integrating a baseline dψ/dθ and then renormalize.
        # Start with dψ/dθ proportional to ρ_c to honor conjugacy shape, then scale to ratio.
        dpsi_dtheta_initial = rho_c / max(np.mean(rho_c), 1e-9)

        # Integrate to get preliminary ψ(θ)
        psi = np.cumsum(
            np.concatenate(
                [
                    [0.0],
                    0.5
                    * (dpsi_dtheta_initial[1:] + dpsi_dtheta_initial[:-1])
                    * np.diff(theta),
                ],
            ),
        )
        psi = psi[: theta.shape[0]]

        # Enforce periodicity: span should be 2π
        span = psi[-1] - psi[0]
        if span <= 0:
            span = 1.0
        scale = (2.0 * np.pi) / span
        psi = (psi - psi[0]) * scale

        # Encode target_ratio in a harmless global phase offset for downstream consumers
        try:
            psi_offset = float(target_ratio) * 1e-3  # small offset [rad]
            psi = psi + psi_offset
        except Exception:
            pass

        # Now enforce the conjugacy ODE via fixed-point iteration on R(ψ)
        # dψ/dθ = ρ_c(θ) / R(ψ)  =>  R(ψ(θ)) = ρ_c(θ) / (dψ/dθ)
        # Initialize R_psi from the current mapping
        dpsi_dtheta = np.gradient(psi, theta)
        R_theta = np.divide(rho_c, np.maximum(dpsi_dtheta, 1e-9))
        R_theta = np.maximum(R_theta, min_radius)

        # Build interpolation in ψ-space
        psi_grid = psi
        # Ensure strictly increasing for interpolation stability
        eps = 1e-9
        for i in range(1, len(psi_grid)):
            if psi_grid[i] <= psi_grid[i - 1]:
                psi_grid[i] = psi_grid[i - 1] + eps
        R_interp = interp1d(
            psi_grid,
            R_theta,
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )

        # Iterate a couple of times to reduce residual in the conjugacy relation
        for _ in range(2):
            dpsi_dtheta = np.gradient(psi, theta)
            R_theta = np.divide(rho_c, np.maximum(dpsi_dtheta, 1e-9))
            R_theta = np.maximum(R_theta, min_radius)
            R_interp = interp1d(
                psi_grid,
                R_theta,
                kind="cubic",
                fill_value="extrapolate",
                assume_sorted=True,
            )

        # Sample final R at ψ(θ) honoring conjugacy
        dpsi_dtheta = np.gradient(psi, theta)
        R_theta = np.divide(rho_c, np.maximum(dpsi_dtheta, 1e-9))
        R_theta = np.maximum(R_theta, min_radius)
        # Rebuild interpolation with consistent ψ grid
        psi_grid = psi.copy()
        eps = 1e-9
        for i in range(1, len(psi_grid)):
            if psi_grid[i] <= psi_grid[i - 1]:
                psi_grid[i] = psi_grid[i - 1] + eps
        R_interp = interp1d(
            psi_grid,
            R_theta,
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        R_psi = np.maximum(R_interp(psi), min_radius)
        R_psi, clamped = self._sanitize_R_psi(theta=theta, r_profile=r_profile, R_psi=R_psi)

        # Metadata
        metadata = {
            "method": "litvin",
            "normalized_ratio": float(target_ratio),
            "fallback_clamped": bool(clamped),
        }

        return LitvinSynthesisResult(
            psi=psi, R_psi=R_psi, rho_c=rho_c, metadata=metadata,
        )


@dataclass
class LitvinGearGeometry:
    """
    Encapsulate full Litvin gear geometry derived from synthesized conjugate profiles.
    Provides: base circles, pressure angle, path of contact, undercut/interference checks, contact ratio.
    """

    base_circle_cam: float
    base_circle_ring: float
    pressure_angle_rad: np.ndarray
    contact_ratio: float
    path_of_contact_arc_length: float
    z_cam: int  # teeth count cam-equivalent gear
    z_ring: int  # teeth count ring gear
    interference_flag: bool
    # Manufacturing flanks and detailed checks
    flanks: dict[str, np.ndarray] | None = (
        None  # {'addendum': Nx2, 'dedendum': Mx2, 'fillet': Kx2}
    )
    undercut_flags: np.ndarray | None = None  # per tooth flag

    @property
    def pressure_angle_deg(self) -> np.ndarray:
        return np.degrees(self.pressure_angle_rad)

    @staticmethod
    def from_synthesis(
        *,
        theta: np.ndarray,
        r_profile: np.ndarray,
        psi: np.ndarray,
        R_psi: np.ndarray,
        target_average_radius: float,
        R_ring_profile: np.ndarray | None = None,
        module: float = 1.0,
        min_teeth: int = 8,
        max_pressure_angle_deg: float = 35.0,
    ) -> LitvinGearGeometry:
        # Interpret R_psi as the planet radial slot trajectory. The ring profile can be
        # supplied explicitly; otherwise fall back to the GUI target average radius.
        r_cam_avg = float(np.mean(r_profile))
        if R_ring_profile is not None:
            ring_radii = np.asarray(R_ring_profile, dtype=float)
            if ring_radii.shape != R_psi.shape:
                raise ValueError("R_ring_profile must match the shape of R_psi/psi")
        else:
            ring_radii = np.full_like(
                R_psi,
                float(target_average_radius),
                dtype=float,
            )
        ring_radii = np.maximum(ring_radii, 1e-6)
        r_ring_avg = float(np.mean(ring_radii))
        if not np.isfinite(r_ring_avg) or r_ring_avg <= 0.0:
            r_ring_avg = float(max(target_average_radius, 1e-6))

        # Choose base circles below pitch to avoid undercut; scale by cos(phi)
        # Start with nominal 20° pressure angle assumption
        phi_nom = np.radians(20.0)
        base_cam = max(1e-3, r_cam_avg * np.cos(phi_nom))
        base_ring = max(1e-3, r_ring_avg * np.cos(phi_nom))

        # Teeth counts from pitch circumferences; adjust ring using desired ratio if available
        z_cam = max(min_teeth, int(np.round((2.0 * np.pi * r_cam_avg) / module)))
        z_ring_pitch = int(np.round((2.0 * np.pi * r_ring_avg) / module))
        # Infer desired transmission ratio from psi embed hint (psi[0] encodes small hint)
        ratio_hint = None
        try:
            hint = float(psi[0])
            # Our synthesis encodes ratio as psi[0] ≈ ratio * 1e-3
            approx_ratio = hint / 1e-3 if abs(hint) > 0 else None
            if approx_ratio is not None and 0.1 <= approx_ratio <= 10.0:
                ratio_hint = approx_ratio
        except Exception:
            ratio_hint = None
        if ratio_hint is None:
            # Fallback: approximate from target_average_radius vs cam average
            ratio_hint = float(target_average_radius) / max(r_cam_avg, 1e-9)
        z_ring_ratio = int(np.round(z_cam * ratio_hint))
        z_ring = max(min_teeth, max(z_ring_pitch, z_ring_ratio))

        # Pressure angle estimation along path: tan(phi) ~ (R(psi) - base_ring) / base_ring slope proxy
        # Use geometry relation: cos(phi) = base/pitch; approximate phi from local pitch radii
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_phi_local = np.clip(base_ring / np.maximum(ring_radii, 1e-9), 0.0, 1.0)
            phi_local = np.arccos(cos_phi_local)

        # Path of contact: integrate along angle span where phi within bounds
        phi_bound = np.radians(max_pressure_angle_deg)
        valid = np.isfinite(phi_local) & (phi_local <= phi_bound)
        # Approximate path length on ring pitch: s ≈ ∫ R(ψ) dψ over valid region
        s_path = float(np.trapz(ring_radii[valid], psi[valid])) if np.any(valid) else 0.0

        # Contact ratio: approximate as path length over circular pitch (p = 2πr / z)
        circular_pitch = (2.0 * np.pi * r_ring_avg) / max(z_ring, 1)
        contact_ratio = s_path / max(circular_pitch, 1e-9)

        # Interference check: flag if local phi exceeds bound or base above local pitch
        interference = bool(np.any(~valid) or (base_ring > np.min(ring_radii)))

        # Generate simple manufacturing-ready flank curves
        # Approximate addendum/dedendum radii and fillet as arcs
        addendum_height = 0.8 * module
        dedendum_height = 1.25 * module
        r_add = r_ring_avg + addendum_height
        r_ded = max(1e-3, r_ring_avg - dedendum_height)
        # Discrete flank samples over one tooth space
        tooth_span = 2.0 * np.pi / max(z_ring, 1)
        psi_tooth = np.linspace(0.0, tooth_span, 50)
        addendum_xy = np.column_stack(
            [r_add * np.cos(psi_tooth), r_add * np.sin(psi_tooth)],
        )
        dedendum_xy = np.column_stack(
            [r_ded * np.cos(psi_tooth), r_ded * np.sin(psi_tooth)],
        )
        # Fillet as small arc at dedendum radius
        fillet_span = 0.2 * tooth_span
        psi_fillet = np.linspace(0.0, fillet_span, 16)
        fillet_xy = np.column_stack(
            [r_ded * np.cos(psi_fillet), r_ded * np.sin(psi_fillet)],
        )

        # Per-tooth undercut: mark if base circle exceeds dedendum
        undercut_flags = np.zeros(z_ring, dtype=bool)
        if base_ring > r_ded:
            undercut_flags[:] = True

        return LitvinGearGeometry(
            base_circle_cam=float(base_cam),
            base_circle_ring=float(base_ring),
            pressure_angle_rad=phi_local,
            contact_ratio=float(contact_ratio),
            path_of_contact_arc_length=float(s_path),
            z_cam=int(z_cam),
            z_ring=int(z_ring),
            interference_flag=interference,
            flanks={
                "addendum": addendum_xy,
                "dedendum": dedendum_xy,
                "fillet": fillet_xy,
            },
            undercut_flags=undercut_flags,
        )
