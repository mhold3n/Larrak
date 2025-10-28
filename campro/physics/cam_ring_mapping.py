"""
Cam-ring-linear follower mapping implementation.

This module implements the mathematical framework for relating:
- Linear follower displacement x(θ) (1st follower)
- Cam rotation θ and its contacting curve geometry
- Ring/radial follower instantaneous radius R(ψ) (2nd follower)

Based on the cam-ring-linear follower mapping document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class CamRingParameters:
    """Parameters for cam-ring system design."""

    # Cam parameters
    base_radius: float = 10.0  # r_b: base radius of cam

    # Connecting rod parameters
    connecting_rod_length: float = (
        25.0  # Length of connecting rod from cam center to linear follower
    )

    # Ring parameters
    ring_center_x: float = 0.0
    ring_center_y: float = 0.0

    # Contact parameters
    contact_type: str = "external"  # "external" or "internal"

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "base_radius": self.base_radius,
            "connecting_rod_length": self.connecting_rod_length,
            "ring_center_x": self.ring_center_x,
            "ring_center_y": self.ring_center_y,
            "contact_type": self.contact_type,
        }


class CamRingMapper:
    """
    Cam-ring-linear follower mapping system.

    Implements the mathematical framework for relating linear follower motion
    to cam geometry and ring follower design through rolling kinematics.
    """

    def __init__(self, parameters: CamRingParameters | None = None):
        """
        Initialize the cam-ring mapper.

        Parameters
        ----------
        parameters : CamRingParameters, optional
            System parameters. If None, uses default values.
        """
        self.parameters = parameters or CamRingParameters()
        log.info(
            f"Initialized CamRingMapper with parameters: {self.parameters.to_dict()}",
        )

    def compute_cam_curves(
        self, theta: np.ndarray, x_theta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Compute cam curves from linear follower motion law via connecting rod.

        Parameters
        ----------
        theta : np.ndarray
            Cam angles (radians)
        x_theta : np.ndarray
            Linear follower displacement vs cam angle

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'pitch_radius': r_p(θ) = r_b + x(θ) (connecting rod connection point)
            - 'profile_radius': r_n(θ) = r_p(θ) (cam surface, same as pitch for direct contact)
            - 'contact_radius': Contacting curve radius for ring contact
        """
        log.info(f"Computing cam curves for {len(theta)} points")

        # Cam radius at connecting rod connection point
        # The connecting rod connects the linear follower to the cam at radius r_b + x(θ)
        pitch_radius = self.parameters.base_radius + x_theta

        # Cam profile (actual cam surface that contacts the ring)
        # For direct contact with ring follower, the profile is the same as the pitch curve
        profile_radius = pitch_radius

        # Contacting curve (for ring contact)
        # Since cam directly contacts ring follower, use the profile radius
        contact_radius = profile_radius.copy()

        return {
            "pitch_radius": pitch_radius,
            "profile_radius": profile_radius,
            "contact_radius": contact_radius,
            "theta": theta,
        }

    def compute_cam_curvature(
        self, theta: np.ndarray, r_contact: np.ndarray,
    ) -> np.ndarray:
        """
        Compute curvature of the contacting cam curve.

        For a curve given in polar form r(θ), the curvature is:
        κ_c(θ) = (r² + 2(r')² - r·r'') / (r² + (r')²)^(3/2)

        Parameters
        ----------
        theta : np.ndarray
            Cam angles (radians)
        r_contact : np.ndarray
            Contacting curve radius

        Returns
        -------
        np.ndarray
            Curvature κ_c(θ)
        """
        log.info("Computing cam curvature")

        # Compute derivatives using finite differences
        dr_dtheta = np.gradient(r_contact, theta)
        d2r_dtheta2 = np.gradient(dr_dtheta, theta)

        # Curvature formula for polar curves
        numerator = r_contact**2 + 2 * dr_dtheta**2 - r_contact * d2r_dtheta2
        denominator = (r_contact**2 + dr_dtheta**2) ** (3 / 2)

        # Avoid division by zero
        kappa_c = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )

        return kappa_c

    def compute_osculating_radius(self, kappa_c: np.ndarray) -> np.ndarray:
        """
        Compute osculating radius from curvature.

        ρ_c(θ) = 1/κ_c(θ)

        Parameters
        ----------
        kappa_c : np.ndarray
            Curvature

        Returns
        -------
        np.ndarray
            Osculating radius ρ_c(θ)
        """
        # Avoid division by zero
        rho_c = np.divide(
            1.0, kappa_c, out=np.full_like(kappa_c, np.inf), where=abs(kappa_c) > 1e-12,
        )

        return rho_c

    def design_ring_radius(
        self, psi: np.ndarray, design_type: str = "constant", **kwargs,
    ) -> np.ndarray:
        """
        Design the ring's instantaneous radius R(ψ).

        Parameters
        ----------
        psi : np.ndarray
            Ring angles (radians)
        design_type : str
            Type of ring design: "constant", "linear", "sinusoidal", "custom"
        **kwargs
            Additional parameters for specific design types

        Returns
        -------
        np.ndarray
            Ring instantaneous radius R(ψ)
        """
        log.info(f"Designing ring radius with type: {design_type}")

        if design_type == "constant":
            base_radius = kwargs.get("base_radius", 15.0)
            return np.full_like(psi, base_radius)

        if design_type == "linear":
            base_radius = kwargs.get("base_radius", 15.0)
            slope = kwargs.get("slope", 0.0)
            return base_radius + slope * psi

        if design_type == "sinusoidal":
            base_radius = kwargs.get("base_radius", 15.0)
            amplitude = kwargs.get("amplitude", 2.0)
            frequency = kwargs.get("frequency", 1.0)
            return base_radius + amplitude * np.sin(frequency * psi)

        if design_type == "custom":
            custom_function = kwargs.get("custom_function")
            if custom_function is None:
                raise ValueError("custom_function must be provided for custom design")
            return custom_function(psi)

        raise ValueError(f"Unknown design type: {design_type}")

    def solve_meshing_law(
        self, theta: np.ndarray, rho_c: np.ndarray, psi: np.ndarray, R_psi: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the pitch-curve meshing law to relate cam and ring angles.

        The meshing law is: ρ_c(θ) dθ = R(ψ) dψ

        This function integrates to find ψ(θ) given the relationship.

        Parameters
        ----------
        theta : np.ndarray
            Cam angles
        rho_c : np.ndarray
            Cam osculating radius
        psi : np.ndarray
            Ring angles (initial guess)
        R_psi : np.ndarray
            Ring instantaneous radius

        Returns
        -------
        np.ndarray
            Ring angles ψ(θ) that satisfy the meshing law
        """
        log.info("Solving pitch-curve meshing law")

        # Create interpolation functions
        rho_c_interp = interp1d(
            theta, rho_c, kind="cubic", bounds_error=False, fill_value="extrapolate",
        )
        R_psi_interp = interp1d(
            psi, R_psi, kind="cubic", bounds_error=False, fill_value="extrapolate",
        )

        # Define the ODE: dψ/dθ = ρ_c(θ) / R(ψ)
        def meshing_ode(theta_val, psi_val):
            rho_c_val = rho_c_interp(theta_val)
            R_psi_val = R_psi_interp(psi_val)

            # Avoid division by zero
            if abs(R_psi_val) < 1e-12:
                return 0.0

            # Apply sign based on contact type
            sign = -1.0 if self.parameters.contact_type == "internal" else 1.0
            return sign * rho_c_val / R_psi_val

        # Solve the ODE
        try:
            sol = solve_ivp(
                meshing_ode,
                [theta[0], theta[-1]],
                [psi[0]],
                t_eval=theta,
                method="RK45",
                rtol=1e-8,
            )

            if sol.success:
                return sol.y[0]
            log.warning("ODE solution failed, using linear approximation")
            return np.linspace(psi[0], psi[-1], len(theta))

        except Exception as e:
            log.warning(f"ODE solution failed: {e}, using linear approximation")
            return np.linspace(psi[0], psi[-1], len(theta))

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
        """
        Compute time-based kinematics for the system.

        Parameters
        ----------
        theta : np.ndarray
            Cam angles
        psi : np.ndarray
            Ring angles
        rho_c : np.ndarray
            Cam osculating radius
        R_psi : np.ndarray
            Ring instantaneous radius
        driver : str
            Driver type: "cam" (constant ω) or "ring" (constant Ω)
        omega : float, optional
            Cam angular speed (if driver="cam")
        Omega : float, optional
            Ring angular speed (if driver="ring")

        Returns
        -------
        Dict[str, np.ndarray]
            Time-based kinematics including time, positions, velocities
        """
        log.info(f"Computing time kinematics with driver: {driver}")

        if driver == "cam" and omega is not None:
            # Cam-driven: θ(t) = θ₀ + ωt, solve for ψ(t)
            t = (theta - theta[0]) / omega
            dpsi_dt = rho_c / R_psi * omega

        elif driver == "ring" and Omega is not None:
            # Ring-driven: ψ(t) = ψ₀ + Ωt, solve for θ(t)
            t = (psi - psi[0]) / Omega
            dtheta_dt = R_psi / rho_c * Omega

        else:
            raise ValueError(
                "Must specify either omega (cam-driven) or Omega (ring-driven)",
            )

        return {
            "time": t,
            "theta": theta,
            "psi": psi,
            "rho_c": rho_c,
            "R_psi": R_psi,
            "driver": driver,
        }

    def map_linear_to_ring_follower(
        self, theta: np.ndarray, x_theta: np.ndarray, ring_design: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Complete mapping from linear follower motion law to ring follower design.

        This is the main synthesis pipeline:
        1. Lift → cam curve
        2. Design ring radius R(ψ)
        3. Relate angles via meshing law
        4. Compute time kinematics

        Parameters
        ----------
        theta : np.ndarray
            Cam angles (degrees)
        x_theta : np.ndarray
            Linear follower displacement vs cam angle
        ring_design : Dict[str, Any]
            Ring design parameters

        Returns
        -------
        Dict[str, np.ndarray]
            Complete mapping results including cam curves, ring design, and kinematics
        """
        log.info("Starting complete linear-to-ring follower mapping")

        # Convert theta from degrees to radians for internal calculations
        theta_rad = np.deg2rad(theta)

        # Step 1: Build cam curves from linear follower motion
        cam_curves = self.compute_cam_curves(theta_rad, x_theta)

        # Step 2: Compute cam curvature and osculating radius
        kappa_c = self.compute_cam_curvature(theta_rad, cam_curves["contact_radius"])
        rho_c = self.compute_osculating_radius(kappa_c)

        # Step 3: Design ring radius
        psi_initial = np.linspace(0, 2 * np.pi, len(theta))  # Initial guess
        R_psi = self.design_ring_radius(psi_initial, **ring_design)

        # Always generate complete 360° profiles with proper boundary continuity
        # This ensures the optimization always works with complete profiles
        log.info(
            "Generating complete 360° profiles with boundary continuity enforcement",
        )

        # Create enhanced grid with higher resolution at critical points (TDC/BDC)
        theta_complete, x_theta_complete = self._create_enhanced_grid(
            theta_rad,
            x_theta,
            len(theta),
        )
        theta_deg_complete = np.degrees(theta_complete)

        # Generate full 360° ring profile with same enhanced grid
        psi_complete = theta_complete.copy()  # Use same enhanced grid for ring
        R_psi_complete = self.design_ring_radius(psi_complete, **ring_design)

        # Step 5: Compute time kinematics (optional)
        time_kinematics = None
        if ring_design.get("compute_time_kinematics", False):
            driver = ring_design.get("driver", "cam")
            omega = ring_design.get("omega")
            Omega = ring_design.get("Omega")
            time_kinematics = self.compute_time_kinematics(
                theta_rad,
                psi,
                rho_c,
                R_psi,
                driver,
                omega,
                Omega,
            )

        # Generate complete cam curves for the full 360° range
        cam_curves_complete = self.compute_cam_curves(theta_complete, x_theta_complete)

        # Convert cam curves theta to degrees for consistency
        cam_curves_complete["theta"] = theta_deg_complete

        # Compile results (return theta in degrees for consistency with input)
        results = {
            "theta": theta_deg_complete,  # Complete cam profile in degrees
            "x_theta": x_theta_complete,  # Complete linear follower displacement
            "cam_curves": cam_curves_complete,  # Complete cam curves (theta in degrees)
            "kappa_c": kappa_c,
            "rho_c": rho_c,
            "psi": psi_complete,  # Complete ring profile in radians
            "R_psi": R_psi_complete,  # Complete ring radius profile
            "time_kinematics": time_kinematics,
        }

        log.info("Completed linear-to-ring follower mapping")
        return results

    def _create_enhanced_grid(
        self, theta_rad: np.ndarray, x_theta: np.ndarray, base_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create an enhanced grid with higher resolution at critical points (TDC/BDC)
        and enforce proper boundary continuity between 0 and 2π.

        Parameters
        ----------
        theta_rad : np.ndarray
            Original cam angles in radians
        x_theta : np.ndarray
            Original linear follower displacement
        base_length : int
            Base number of points for the grid

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Enhanced theta grid and interpolated x_theta values
        """
        log.info("Creating enhanced grid with boundary continuity enforcement")

        # Define critical points for higher resolution (TDC, BDC, and boundary)
        critical_points = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])

        # Create base grid with endpoint=True to include 2π
        theta_base = np.linspace(0, 2 * np.pi, base_length, endpoint=True)

        # Add extra points around critical regions for higher resolution
        extra_points = []

        # Add points around TDC (0° and 360°)
        tdc_region = np.linspace(-np.pi / 12, np.pi / 12, 5)  # ±15° around TDC
        extra_points.extend(tdc_region)
        extra_points.extend(2 * np.pi + tdc_region)  # Around 360°

        # Add points around BDC (180°)
        bdc_region = np.linspace(
            np.pi - np.pi / 12, np.pi + np.pi / 12, 5,
        )  # ±15° around BDC
        extra_points.extend(bdc_region)

        # Add points around 90° and 270°
        extra_points.extend(
            np.linspace(np.pi / 2 - np.pi / 24, np.pi / 2 + np.pi / 24, 3),
        )
        extra_points.extend(
            np.linspace(3 * np.pi / 2 - np.pi / 24, 3 * np.pi / 2 + np.pi / 24, 3),
        )

        # Combine base grid with extra points
        all_points = np.concatenate([theta_base, extra_points])

        # Remove duplicates and sort
        theta_enhanced = np.unique(all_points)

        # Normalize to [0, 2π] range
        theta_enhanced = np.mod(theta_enhanced, 2 * np.pi)
        theta_enhanced = np.unique(
            theta_enhanced,
        )  # Remove duplicates after normalization

        # Ensure we have the exact boundary points
        if theta_enhanced[0] != 0.0:
            theta_enhanced = np.concatenate([[0.0], theta_enhanced])
        if theta_enhanced[-1] != 2 * np.pi:
            theta_enhanced = np.concatenate([theta_enhanced, [2 * np.pi]])

        # Interpolate x_theta to the enhanced grid
        # Simplified approach: focus only on cam-ring optimization (phase 2)
        # Linkage journal placement will be handled in phase 3
        if theta_rad[-1] < 2 * np.pi:
            # Create a simple, smooth extension for phase 2 cam-ring optimization
            # This avoids the complexity of trying to solve everything at once
            x_theta_enhanced = self._create_simple_cam_extension(
                theta_enhanced,
                theta_rad,
                x_theta,
            )
        else:
            # Input already covers full period
            x_theta_enhanced = np.interp(theta_enhanced, theta_rad, x_theta)

        # Enforce boundary continuity: ensure x_theta(0) = x_theta(2π)
        if abs(x_theta_enhanced[0] - x_theta_enhanced[-1]) > 1e-6:
            # Average the boundary values to ensure continuity
            boundary_avg = (x_theta_enhanced[0] + x_theta_enhanced[-1]) / 2
            x_theta_enhanced[0] = boundary_avg
            x_theta_enhanced[-1] = boundary_avg
            log.info(
                f"Enforced boundary continuity: x(0) = x(2pi) = {boundary_avg:.6f}",
            )

        log.info(f"Enhanced grid: {len(theta_enhanced)} points (base: {base_length})")
        log.info(f"Grid range: {theta_enhanced[0]:.3f} to {theta_enhanced[-1]:.3f} rad")
        log.info(
            f"Boundary continuity: x(0) = {x_theta_enhanced[0]:.6f}, x(2pi) = {x_theta_enhanced[-1]:.6f}",
        )

        return theta_enhanced, x_theta_enhanced

    def _create_simple_cam_extension(
        self, theta_enhanced: np.ndarray, theta_rad: np.ndarray, x_theta: np.ndarray,
    ) -> np.ndarray:
        """
        Create a simple cam extension for phase 2 cam-ring optimization.

        This simplified approach focuses only on cam-ring optimization without
        trying to solve linkage journal placement (deferred to phase 3).

        Parameters
        ----------
        theta_enhanced : np.ndarray
            Enhanced theta grid (0 to 2π)
        theta_rad : np.ndarray
            Original theta values (partial range)
        x_theta : np.ndarray
            Original motion law values

        Returns
        -------
        np.ndarray
            Simple extended motion law for phase 2 optimization
        """
        log.info("Creating simple cam extension for phase 2 cam-ring optimization")

        # Create a smooth periodic extension with proper slope continuity
        # This ensures the cam profile is continuous at 0°/360° boundary

        # Create extended arrays for periodic interpolation
        theta_extended = np.concatenate([theta_rad, [2 * np.pi]])
        x_theta_extended = np.concatenate([x_theta, [x_theta[0]]])  # Close the loop

        # Use cubic spline interpolation for smooth extension
        from scipy.interpolate import CubicSpline

        # Create cubic spline with periodic boundary conditions
        # This ensures smooth continuity at the boundary
        cs = CubicSpline(theta_extended, x_theta_extended, bc_type="periodic")

        # Evaluate on enhanced grid
        x_theta_enhanced = cs(theta_enhanced)

        # Apply additional smoothing to ensure better slope continuity
        # Use a small smoothing window around the boundary
        smooth_window = min(10, len(x_theta_enhanced) // 20)
        if smooth_window > 1:
            # Apply Gaussian-like smoothing around the boundary
            for i in range(smooth_window):
                # Weight decreases as we move away from boundary
                weight = np.exp(-((i / smooth_window) ** 2))
                # Blend with the opposite side of the boundary
                opposite_idx = len(x_theta_enhanced) - smooth_window + i
                if opposite_idx < len(x_theta_enhanced):
                    x_theta_enhanced[i] = (
                        weight * x_theta_enhanced[i]
                        + (1 - weight) * x_theta_enhanced[opposite_idx]
                    )
                    x_theta_enhanced[opposite_idx] = (
                        weight * x_theta_enhanced[opposite_idx]
                        + (1 - weight) * x_theta_enhanced[i]
                    )

        # Ensure boundary continuity for smooth cam profile
        if abs(x_theta_enhanced[0] - x_theta_enhanced[-1]) > 1e-6:
            boundary_avg = (x_theta_enhanced[0] + x_theta_enhanced[-1]) / 2
            x_theta_enhanced[0] = boundary_avg
            x_theta_enhanced[-1] = boundary_avg

        # Ensure positive displacements for physical cam profile
        x_theta_enhanced = np.maximum(x_theta_enhanced, 0.0)

        # Check slope continuity at boundary
        if len(x_theta_enhanced) > 2:
            # Compute slopes at boundary
            slope_start = (x_theta_enhanced[1] - x_theta_enhanced[0]) / (
                theta_enhanced[1] - theta_enhanced[0]
            )
            slope_end = (x_theta_enhanced[-1] - x_theta_enhanced[-2]) / (
                theta_enhanced[-1] - theta_enhanced[-2]
            )
            slope_diff = abs(slope_start - slope_end)

            if slope_diff > 0.1:  # Significant slope discontinuity
                log.warning(f"Slope discontinuity detected: {slope_diff:.3f}")
                # Apply smoothing to reduce slope discontinuity
                # Use a small smoothing window around the boundary
                smooth_window = min(5, len(x_theta_enhanced) // 10)
                if smooth_window > 1:
                    # Smooth the boundary region
                    for i in range(smooth_window):
                        weight = (smooth_window - i) / smooth_window
                        x_theta_enhanced[i] = (
                            weight * x_theta_enhanced[i]
                            + (1 - weight) * x_theta_enhanced[-smooth_window + i]
                        )
                        x_theta_enhanced[-smooth_window + i] = (
                            weight * x_theta_enhanced[-smooth_window + i]
                            + (1 - weight) * x_theta_enhanced[i]
                        )

        log.info(
            f"Created smooth cam extension: range {x_theta_enhanced[0]:.3f} to {x_theta_enhanced[-1]:.3f}",
        )
        log.info(
            f"Boundary continuity: x(0) = {x_theta_enhanced[0]:.6f}, x(2pi) = {x_theta_enhanced[-1]:.6f}",
        )

        return x_theta_enhanced

    def _generate_complete_ring_profile(
        self, psi: np.ndarray, R_psi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete 360° ring profile from the meshing law solution.

        The meshing law solution may not cover the full 360° range due to the
        specific cam geometry and motion law. This method interpolates and
        extrapolates to create a complete ring profile.

        Parameters
        ----------
        psi : np.ndarray
            Ring angles from meshing law solution (radians)
        R_psi : np.ndarray
            Ring radius values from meshing law solution

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Complete ring angles and radius values covering 360°
        """
        # Normalize psi to [0, 2π] range
        psi_norm = np.mod(psi, 2 * np.pi)

        # Sort by angle for proper interpolation
        sort_idx = np.argsort(psi_norm)
        psi_sorted = psi_norm[sort_idx]
        R_sorted = R_psi[sort_idx]

        # Create complete angle range
        psi_complete = np.linspace(0, 2 * np.pi, len(psi), endpoint=False)

        # Interpolate ring radius for complete range
        # Use periodic interpolation to handle the wrap-around
        R_complete = np.interp(psi_complete, psi_sorted, R_sorted, period=2 * np.pi)

        # For constant ring design, ensure all values are the same
        if len(np.unique(R_psi)) == 1:  # All values are the same (constant design)
            R_complete = np.full_like(psi_complete, R_psi[0])

        log.info(
            f"Generated complete ring profile: {len(psi_complete)} points covering 360°",
        )

        return psi_complete, R_complete

    def validate_design(self, results: dict[str, np.ndarray]) -> dict[str, bool]:
        """
        Validate the cam-ring design for practical constraints.

        Parameters
        ----------
        results : Dict[str, np.ndarray]
            Mapping results from map_linear_to_ring_follower

        Returns
        -------
        Dict[str, bool]
            Validation results for various design checks
        """
        log.info("Validating cam-ring design")

        validation = {}

        # Check for cusps/undercuts
        # For direct cam-ring contact (no rollers), check for excessive curvature
        kappa_c = results["kappa_c"]
        validation["no_cusps"] = bool(
            np.all(abs(kappa_c) < 10.0),
        )  # Reasonable curvature limit

        # Check for positive radii
        cam_curves = results["cam_curves"]
        validation["positive_cam_radii"] = bool(
            np.all(cam_curves["profile_radius"] > 0),
        )
        validation["positive_ring_radii"] = bool(np.all(results["R_psi"] > 0))

        # Check for reasonable curvature values
        validation["reasonable_curvature"] = bool(np.all(np.isfinite(kappa_c)))
        validation["reasonable_osculating_radius"] = bool(
            np.all(np.isfinite(results["rho_c"])),
        )

        # Check for smooth angle relationships
        psi = results["psi"]
        validation["smooth_angle_relationship"] = bool(np.all(np.isfinite(psi)))

        log.info(f"Design validation results: {validation}")
        return validation
