"""
Cam-ring system optimization using collocation methods.

This module implements a secondary optimization that optimizes the cam-ring system
parameters (cam radius, connecting rod length, ring design) using the same
collocation approach as the primary motion law optimization.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from campro.logging import get_logger
from campro.physics.geometry.litvin import LitvinGearGeometry, LitvinSynthesis
from campro.physics.kinematics.litvin_assembly import (
    AssemblyInputs,
    compute_assembly_state,
)

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationSettings

log = get_logger(__name__)


@dataclass
class CamRingOptimizationConstraints:
    """Constraints for cam-ring system optimization."""

    # Cam parameter bounds (phase 2: cam + ring only)
    base_radius_min: float = 5.0
    base_radius_max: float = 100.0
    # connecting_rod_length_min: float = 10.0  # Removed for phase 2 simplification
    # connecting_rod_length_max: float = 200.0  # Removed for phase 2 simplification

    # Target average ring radius (from GUI)
    target_average_ring_radius: float = 45.0

    # Physical constraints
    min_curvature_radius: float = 1.0  # Minimum osculating radius
    max_curvature: float = 10.0        # Maximum curvature

    # Optimization constraints
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass
class CamRingOptimizationTargets:
    """Optimization targets for cam-ring system."""

    # Primary objectives
    minimize_ring_size: bool = True
    minimize_cam_size: bool = True
    maximize_efficiency: bool = False

    # Ring objectives (Litvin synthesis derived)
    target_average_ring_radius: bool = True  # Match GUI target

    # Secondary objectives
    minimize_curvature_variation: bool = True
    minimize_stress_concentration: bool = True

    # Weighting factors (balanced for both cam and ring)
    ring_size_weight: float = 1.0
    cam_size_weight: float = 1.0  # Balanced with ring
    efficiency_weight: float = 0.3
    curvature_weight: float = 0.2
    stress_weight: float = 0.1
    target_radius_weight: float = 0.5


class CamRingOptimizer(BaseOptimizer):
    """
    Cam-ring system optimizer using collocation methods.
    
    This optimizer takes the linear follower motion law from primary optimization
    and optimizes the cam-ring system parameters to achieve specific objectives
    while maintaining the required motion law.
    """

    def __init__(self, name: str = "CamRingOptimizer",
                 settings: Optional[CollocationSettings] = None):
        super().__init__(name)
        self.settings = settings or CollocationSettings()
        self.constraints = CamRingOptimizationConstraints()
        self.targets = CamRingOptimizationTargets()
        self._is_configured = True

    def configure(self, constraints: Optional[CamRingOptimizationConstraints] = None,
                 targets: Optional[CamRingOptimizationTargets] = None,
                 **kwargs) -> None:
        """
        Configure the optimizer.
        
        Parameters
        ----------
        constraints : CamRingOptimizationConstraints, optional
            Optimization constraints
        targets : CamRingOptimizationTargets, optional
            Optimization targets
        **kwargs
            Additional configuration parameters
        """
        if constraints is not None:
            self.constraints = constraints
        if targets is not None:
            self.targets = targets

        # Update settings from kwargs
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)

        self._is_configured = True
        log.info(f"Configured {self.name} with constraints and targets")

    def optimize(self, primary_data: Dict[str, np.ndarray],
                initial_guess: Optional[Dict[str, float]] = None,
                **kwargs) -> OptimizationResult:
        """
        Optimize cam-ring system parameters.
        
        Parameters
        ----------
        primary_data : Dict[str, np.ndarray]
            Primary optimization results (linear follower motion law)
        initial_guess : Dict[str, float], optional
            Initial parameter values
        **kwargs
            Additional optimization parameters
            
        Returns
        -------
        OptimizationResult
            Optimization results including optimized parameters and ring design
        """
        if not self._is_configured:
            raise RuntimeError("Optimizer must be configured before optimization")

        log.info(f"Starting cam-ring system optimization with {self.name}")

        # Initialize result
        result = OptimizationResult(
            status=OptimizationStatus.RUNNING,
            solution={},
            objective_value=float("inf"),
            iterations=0,
            solve_time=0.0,
            metadata={},
        )

        start_time = time.time()

        try:
            # Extract primary motion law data
            theta = primary_data.get("cam_angle", np.array([]))
            x_theta = primary_data.get("position", np.array([]))

            if len(theta) == 0 or len(x_theta) == 0:
                raise ValueError("Primary data must contain cam_angle and position arrays")

            # Set up initial guess
            if initial_guess is None:
                initial_guess = self._get_default_initial_guess(primary_data)

            log.info(f"Initial guess: {initial_guess}")

            # Define optimization variables (Litvin-only synthesis)
            param_names = [
                "base_radius",  # Cam base radius only
            ]
            initial_params = np.array([initial_guess[name] for name in param_names])

            # Define parameter bounds (Litvin-only)
            bounds = [
                (self.constraints.base_radius_min, self.constraints.base_radius_max),
            ]

            # Define objective function
            def objective(params):
                return self._objective_function(params, param_names, theta, x_theta, primary_data)

            # Define constraints
            constraints = self._define_constraints(theta, x_theta, primary_data)

            # Perform optimization
            log.info("Starting parameter optimization...")
            optimization_result = minimize(
                objective,
                initial_params,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.constraints.max_iterations,
                    "ftol": self.constraints.tolerance,
                    "disp": True,
                },
            )

            # Process results
            if optimization_result.success:
                # Extract optimized parameters
                optimized_params = dict(zip(param_names, optimization_result.x))

                # Generate final ring design with optimized parameters
                final_design = self._generate_final_design(
                    optimized_params, theta, x_theta, primary_data,
                )

                # Update result
                result.status = OptimizationStatus.CONVERGED
                result.solution = final_design
                result.objective_value = optimization_result.fun
                result.iterations = optimization_result.nit
                result.metadata = {
                    "optimization_method": "SLSQP",
                    "optimized_parameters": optimized_params,
                    "initial_guess": initial_guess,
                    "convergence_info": {
                        "success": optimization_result.success,
                        "message": optimization_result.message,
                        "nit": optimization_result.nit,
                        "nfev": optimization_result.nfev,
                    },
                }

                log.info(f"Optimization completed successfully in {optimization_result.nit} iterations")
                log.info(f"Final objective value: {optimization_result.fun:.6f}")
                log.info(f"Optimized parameters: {optimized_params}")

            else:
                result.status = OptimizationStatus.FAILED
                result.metadata = {
                    "error_message": optimization_result.message,
                    "optimization_method": "SLSQP",
                    "iterations": optimization_result.nit,
                }
                log.error(f"Optimization failed: {optimization_result.message}")

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.metadata = {"error_message": str(e)}
            log.error(f"Optimization error: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")

        finally:
            result.solve_time = time.time() - start_time

        return result

    def _get_default_initial_guess(self, primary_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get default initial guess based on primary data."""
        # Use stroke-based initial guesses
        stroke = np.max(primary_data.get("position", [20.0])) - np.min(primary_data.get("position", [0.0]))

        return {
            "base_radius": float(stroke),
        }

    def _objective_function(self, params: np.ndarray, param_names: List[str],
                          theta: np.ndarray, x_theta: np.ndarray,
                          primary_data: Dict[str, np.ndarray]) -> float:
        """Calculate objective function value."""
        try:
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))

            # Build cam polar profile from primary motion: r(θ) = r_b + x(θ)
            base_radius = float(param_dict["base_radius"])
            theta_deg = theta
            theta_rad = np.deg2rad(theta_deg)
            r_profile = base_radius + x_theta

            # Litvin synthesis
            litvin = LitvinSynthesis()
            litvin_result = litvin.synthesize_from_cam_profile(
                theta=theta_rad,
                r_profile=r_profile,
                target_ratio=1.0,
            )

            # Calculate objective components
            objective = 0.0

            # Target average ring radius objective (match GUI target)
            if self.targets.target_average_ring_radius:
                average_ring_radius = float(np.mean(litvin_result.R_psi))
                target_radius = self.constraints.target_average_ring_radius
                radius_error = abs(average_ring_radius - target_radius)
                objective += self.targets.target_radius_weight * radius_error

            # Ring profile completeness objective (encourage full 360° profile)
            ring_psi = litvin_result.psi
            ring_angle_span_rad = float(np.max(ring_psi) - np.min(ring_psi))
            target_ring_angle_span_rad = 2.0 * np.pi

            # Strong penalty for incomplete ring angular coverage
            if ring_angle_span_rad < (0.9 * target_ring_angle_span_rad):
                ring_angle_penalty = (target_ring_angle_span_rad - ring_angle_span_rad) * 100.0
                objective += ring_angle_penalty
                log.warning(f"Incomplete ring profile: {np.degrees(ring_angle_span_rad):.1f}° coverage")

            # Ring radius variation objective (encourage variation, not minimize)
            if self.targets.minimize_ring_radius_variation:
                ring_radius_variation = np.std(result["R_psi"])
                # Invert the objective: reward variation, penalize constant radius
                target_variation = 5.0  # Target 5mm variation
                variation_error = abs(target_variation - ring_radius_variation)
                objective += self.targets.ring_smoothness_weight * variation_error

            # Ring size objective (minimize maximum radius)
            if self.targets.minimize_ring_size:
                max_ring_radius = float(np.max(litvin_result.R_psi))
                objective += self.targets.ring_size_weight * max_ring_radius

            # Cam size objective (minimize maximum cam radius)
            if self.targets.minimize_cam_size:
                max_cam_radius = float(np.max(r_profile))
                objective += self.targets.cam_size_weight * max_cam_radius

            # Cam profile completeness objective (encourage full 360° profile)
            cam_theta_deg = theta_deg
            cam_angle_span = float(np.max(cam_theta_deg) - np.min(cam_theta_deg))
            target_angle_span = 360.0  # Target full 360° coverage

            # Strong penalty for incomplete angular coverage
            if cam_angle_span < 300.0:  # Less than 300° is considered incomplete
                angle_penalty = (target_angle_span - cam_angle_span) * 10.0  # Strong penalty
                objective += angle_penalty
                log.warning(f"Incomplete cam profile: {cam_angle_span:.1f}° coverage (penalty: {angle_penalty:.1f})")

            # Also check radius range for profile variation
            cam_radius_range = float(np.max(r_profile) - np.min(r_profile))
            target_cam_range = 10.0  # Target 10mm range for cam profile
            cam_range_error = abs(target_cam_range - cam_radius_range)
            objective += 0.3 * cam_range_error  # Moderate weight for cam completeness

            # Curvature variation objective
            if self.targets.minimize_curvature_variation:
                # Estimate curvature variation via cam curvature from Litvin synthesis (derived from cam)
                rho = litvin_result.rho_c
                # kappa = 1 / rho where finite
                kappa = np.divide(1.0, rho, out=np.zeros_like(rho), where=np.isfinite(rho) & (rho != 0))
                curvature_variation = float(np.std(kappa[np.isfinite(kappa)]))
                objective += self.targets.curvature_weight * curvature_variation

            # Efficiency objective (simplified)
            if self.targets.maximize_efficiency:
                # Simple efficiency metric based on smoothness
                velocity = primary_data.get("velocity", np.array([]))
                if len(velocity) > 0:
                    velocity_smoothness = np.std(velocity)
                    objective += self.targets.efficiency_weight * velocity_smoothness

            return objective

        except Exception as e:
            log.warning(f"Objective function error: {e}")
            return 1e6  # Large penalty for invalid parameters

    def _define_constraints(self, theta: np.ndarray, x_theta: np.ndarray,
                          primary_data: Dict[str, np.ndarray]) -> List[Dict]:
        """Define optimization constraints."""
        constraints = []

        # Physical constraint: positive base radius
        def positive_radii_constraint(params):
            base_radius = params[0]
            return base_radius - 1.0

        constraints.append({
            "type": "ineq",
            "fun": positive_radii_constraint,
        })

        return constraints

    def _generate_final_design(self, optimized_params: Dict[str, float],
                             theta: np.ndarray, x_theta: np.ndarray,
                             primary_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate final ring design with optimized parameters."""
        # Build cam polar profile and synthesize Litvin ring profile
        base_radius = float(optimized_params["base_radius"])
        theta_deg = theta
        theta_rad = np.deg2rad(theta_deg)
        r_profile = base_radius + x_theta

        litvin = LitvinSynthesis()
        litvin_result = litvin.synthesize_from_cam_profile(
            theta=theta_rad,
            r_profile=r_profile,
            target_ratio=1.0,
        )

        # Construct cam curves dictionary consistent with previous structure
        cam_curves = {
            "theta": theta_deg,
            "pitch_radius": r_profile,
            "profile_radius": r_profile,
            "contact_radius": r_profile,
        }

        # Derive tooth-level geometry and metrics
        gear_geom = LitvinGearGeometry.from_synthesis(
            theta=theta_rad,
            r_profile=r_profile,
            psi=litvin_result.psi,
            R_psi=litvin_result.R_psi,
            target_average_radius=float(np.mean(litvin_result.R_psi)),
        )

        result = {
            "psi": litvin_result.psi,          # radians
            "R_psi": litvin_result.R_psi,
            "kappa_c": np.divide(1.0, litvin_result.rho_c, out=np.zeros_like(litvin_result.rho_c), where=np.isfinite(litvin_result.rho_c) & (litvin_result.rho_c != 0)),
            "cam_curves": cam_curves,
            "gear_geometry": {
                "base_circle_cam": gear_geom.base_circle_cam,
                "base_circle_ring": gear_geom.base_circle_ring,
                "pressure_angle_deg": gear_geom.pressure_angle_deg,
                "contact_ratio": gear_geom.contact_ratio,
                "path_of_contact_arc_length": gear_geom.path_of_contact_arc_length,
                "z_cam": gear_geom.z_cam,
                "z_ring": gear_geom.z_ring,
                "interference": gear_geom.interference_flag,
                "flanks": gear_geom.flanks,
                "undercut_flags": gear_geom.undercut_flags,
            },
        }

        # Compute assembly state with radial center stepping C(θ)=C0+x(θ)
        try:
            assembly_inputs = AssemblyInputs(
                base_circle_cam=float(gear_geom.base_circle_cam),
                base_circle_ring=float(gear_geom.base_circle_ring),
                z_cam=int(gear_geom.z_cam),
                contact_type="internal",  # ring gear with internal mesh
                psi=np.asarray(litvin_result.psi),
                R_psi=np.asarray(litvin_result.R_psi),
                theta_cam_rad=theta_rad,
                center_base_radius=float(base_radius),
                motion_theta_deg=theta_deg,
                motion_offset_mm=x_theta,
            )
            assembly_state = compute_assembly_state(assembly_inputs)
            result["assembly_state"] = {
                "center_distance": assembly_state.center_distance,
                "planet_center_angle": assembly_state.planet_center_angle,
                "planet_center_radius": assembly_state.planet_center_radius,
                "planet_spin_angle": assembly_state.planet_spin_angle,
                "contact_theta": assembly_state.contact_theta,
                "contact_radius": assembly_state.contact_radius,
                "z_cam": assembly_state.z_cam,
            }
        except Exception as e:
            log.warning(f"Assembly state computation failed: {e}")

        # Calculate average ring radius for validation
        average_ring_radius = float(np.mean(result["R_psi"]))
        ring_radius_variation = float(np.std(result["R_psi"]))

        # Add optimization metadata
        result["optimized_parameters"] = optimized_params
        result["optimization_objectives"] = {
            "minimize_ring_size": self.targets.minimize_ring_size,
            "minimize_cam_size": self.targets.minimize_cam_size,
            "maximize_efficiency": self.targets.maximize_efficiency,
            "minimize_curvature_variation": self.targets.minimize_curvature_variation,
            "target_average_ring_radius": self.targets.target_average_ring_radius,
        }
        result["ring_analysis"] = {
            "average_radius": float(average_ring_radius),
            "radius_variation": float(ring_radius_variation),
            "target_radius": self.constraints.target_average_ring_radius,
            "radius_error": float(abs(average_ring_radius - self.constraints.target_average_ring_radius)),
        }

        return result
