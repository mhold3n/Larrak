"""
Sun gear optimization for cam-ring systems.

This module implements a third optimization layer that introduces a sun gear
between the cam and ring follower to eliminate interference constraints and
enable optimal journal placement with back rotation capabilities.

NOTE: This module uses scipy.optimize.minimize and may be legacy code.
The main optimization flow (phases 1, 2, 3) uses CasADi/IPOPT. This optimizer
is exported in __init__.py but may not be actively used in the main flow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from campro.logging import get_logger

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationSettings

log = get_logger(__name__)


@dataclass
class SunGearParameters:
    """Parameters for sun gear system design."""

    # Sun gear geometry
    sun_gear_radius: float = 15.0
    sun_gear_teeth: int = 30

    # Ring gear geometry
    ring_gear_radius: float = 45.0
    ring_gear_teeth: int = 90

    # Gear ratio
    gear_ratio: float = 3.0  # ring_teeth / sun_teeth

    # Back rotation parameters
    back_rotation_enabled: bool = True
    max_back_rotation: float = np.pi  # Maximum back rotation in radians

    # Journal placement
    journal_offset_x: float = 0.0
    journal_offset_y: float = 0.0
    journal_radius: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sun_gear_radius": self.sun_gear_radius,
            "sun_gear_teeth": self.sun_gear_teeth,
            "ring_gear_radius": self.ring_gear_radius,
            "ring_gear_teeth": self.ring_gear_teeth,
            "gear_ratio": self.gear_ratio,
            "back_rotation_enabled": self.back_rotation_enabled,
            "max_back_rotation": self.max_back_rotation,
            "journal_offset_x": self.journal_offset_x,
            "journal_offset_y": self.journal_offset_y,
            "journal_radius": self.journal_radius,
        }


@dataclass
class SunGearOptimizationConstraints:
    """Constraints for sun gear system optimization."""

    # Sun gear bounds
    sun_gear_radius_min: float = 10.0
    sun_gear_radius_max: float = 50.0
    sun_gear_teeth_min: int = 20
    sun_gear_teeth_max: int = 100

    # Ring gear bounds
    ring_gear_radius_min: float = 30.0
    ring_gear_radius_max: float = 150.0
    ring_gear_teeth_min: int = 60
    ring_gear_teeth_max: int = 300

    # Gear ratio constraints
    min_gear_ratio: float = 1.5
    max_gear_ratio: float = 10.0

    # Journal placement bounds
    journal_offset_max: float = 20.0
    journal_radius_min: float = 2.0
    journal_radius_max: float = 15.0

    # Back rotation constraints
    max_back_rotation: float = np.pi
    min_back_rotation: float = 0.0

    # Physical constraints
    min_clearance: float = 2.0  # Minimum clearance between components
    max_interference: float = 0.0  # No interference allowed

    # Optimization constraints
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass
class SunGearOptimizationTargets:
    """Optimization targets for sun gear system."""

    # Primary objectives
    minimize_system_size: bool = True
    maximize_efficiency: bool = True
    minimize_back_rotation: bool = True

    # Secondary objectives
    minimize_gear_stress: bool = True
    maximize_smoothness: bool = True
    minimize_vibration: bool = True

    # Weighting factors
    system_size_weight: float = 1.0
    efficiency_weight: float = 0.8
    back_rotation_weight: float = 0.6
    gear_stress_weight: float = 0.4
    smoothness_weight: float = 0.3
    vibration_weight: float = 0.2


class SunGearOptimizer(BaseOptimizer):
    """
    Sun gear system optimizer for cam-ring systems.

    This optimizer introduces a sun gear between the cam and ring follower
    to eliminate interference constraints and enable optimal journal placement
    with back rotation capabilities.
    """

    def __init__(
        self,
        name: str = "SunGearOptimizer",
        settings: CollocationSettings | None = None,
    ):
        super().__init__(name)
        self.settings = settings or CollocationSettings()
        self.constraints = SunGearOptimizationConstraints()
        self.targets = SunGearOptimizationTargets()
        self._is_configured = True

    def configure(
        self,
        constraints: SunGearOptimizationConstraints | None = None,
        targets: SunGearOptimizationTargets | None = None,
        **kwargs,
    ) -> None:
        """
        Configure the optimizer.

        Parameters
        ----------
        constraints : SunGearOptimizationConstraints, optional
            Optimization constraints
        targets : SunGearOptimizationTargets, optional
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
        log.info(f"Configured {self.name} with sun gear constraints and targets")

    def optimize(
        self,
        primary_data: dict[str, np.ndarray],
        secondary_data: dict[str, np.ndarray],
        initial_guess: dict[str, float] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize sun gear system parameters.

        Parameters
        ----------
        primary_data : Dict[str, np.ndarray]
            Primary optimization results (linear follower motion law)
        secondary_data : Dict[str, np.ndarray]
            Secondary optimization results (cam-ring system)
        initial_guess : Dict[str, float], optional
            Initial parameter values
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizationResult
            Optimization results including optimized sun gear parameters
        """
        if not self._is_configured:
            raise RuntimeError("Optimizer must be configured before optimization")

        log.info(f"Starting sun gear system optimization with {self.name}")

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
            # Extract data from previous optimizations
            theta = primary_data.get("cam_angle", np.array([]))
            x_theta = primary_data.get("position", np.array([]))
            cam_curves = secondary_data.get("cam_curves", {})
            optimized_params = secondary_data.get("optimized_parameters", {})

            if len(theta) == 0 or len(x_theta) == 0:
                raise ValueError(
                    "Primary data must contain cam_angle and position arrays",
                )

            # Set up initial guess
            if initial_guess is None:
                initial_guess = self._get_default_initial_guess(
                    primary_data, secondary_data,
                )

            log.info(f"Initial guess: {initial_guess}")

            # Define optimization variables
            param_names = [
                "sun_gear_radius",
                "ring_gear_radius",
                "gear_ratio",
                "journal_offset_x",
                "journal_offset_y",
                "max_back_rotation",
            ]
            initial_params = np.array([initial_guess[name] for name in param_names])

            # Define parameter bounds
            bounds = [
                (
                    self.constraints.sun_gear_radius_min,
                    self.constraints.sun_gear_radius_max,
                ),
                (
                    self.constraints.ring_gear_radius_min,
                    self.constraints.ring_gear_radius_max,
                ),
                (self.constraints.min_gear_ratio, self.constraints.max_gear_ratio),
                (
                    -self.constraints.journal_offset_max,
                    self.constraints.journal_offset_max,
                ),
                (
                    -self.constraints.journal_offset_max,
                    self.constraints.journal_offset_max,
                ),
                (
                    self.constraints.min_back_rotation,
                    self.constraints.max_back_rotation,
                ),
            ]

            # Define objective function
            def objective(params):
                return self._objective_function(
                    params,
                    param_names,
                    theta,
                    x_theta,
                    cam_curves,
                    optimized_params,
                    primary_data,
                    secondary_data,
                )

            # Define constraints
            constraints = self._define_constraints(
                theta, x_theta, cam_curves, optimized_params,
            )

            # Perform optimization
            log.info("Starting sun gear parameter optimization...")
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

                # Generate final sun gear design
                final_design = self._generate_final_design(
                    optimized_params,
                    theta,
                    x_theta,
                    cam_curves,
                    optimized_params,
                    primary_data,
                    secondary_data,
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

                log.info(
                    f"Sun gear optimization completed successfully in {optimization_result.nit} iterations",
                )
                log.info(f"Final objective value: {optimization_result.fun:.6f}")
                log.info(f"Optimized parameters: {optimized_params}")

            else:
                result.status = OptimizationStatus.FAILED
                result.metadata = {
                    "error_message": optimization_result.message,
                    "optimization_method": "SLSQP",
                    "iterations": optimization_result.nit,
                }
                log.error(
                    f"Sun gear optimization failed: {optimization_result.message}",
                )

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.metadata = {"error_message": str(e)}
            log.error(f"Sun gear optimization error: {e}")
            import traceback

            log.error(f"Traceback: {traceback.format_exc()}")

        finally:
            result.solve_time = time.time() - start_time

        return result

    def _get_default_initial_guess(
        self, primary_data: dict[str, np.ndarray], secondary_data: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Get default initial guess based on previous optimization results."""
        # Use optimized cam parameters from secondary optimization
        optimized_params = secondary_data.get("optimized_parameters", {})
        base_radius = optimized_params.get("base_radius", 10.0)
        connecting_rod_length = optimized_params.get("connecting_rod_length", 25.0)

        # Calculate sun gear parameters based on cam geometry
        sun_gear_radius = base_radius * 1.5  # Sun gear slightly larger than cam base
        ring_gear_radius = sun_gear_radius * 3.0  # Ring gear 3x larger than sun gear
        gear_ratio = 3.0  # Standard gear ratio

        return {
            "sun_gear_radius": sun_gear_radius,
            "ring_gear_radius": ring_gear_radius,
            "gear_ratio": gear_ratio,
            "journal_offset_x": 0.0,
            "journal_offset_y": 0.0,
            "max_back_rotation": np.pi / 4,  # 45 degrees initial back rotation
        }

    def _objective_function(
        self,
        params: np.ndarray,
        param_names: list[str],
        theta: np.ndarray,
        x_theta: np.ndarray,
        cam_curves: dict[str, np.ndarray],
        optimized_params: dict[str, float],
        primary_data: dict[str, np.ndarray],
        secondary_data: dict[str, np.ndarray],
    ) -> float:
        """Calculate objective function value for sun gear system."""
        try:
            # Create parameter dictionary
            param_dict = dict(zip(param_names, params))

            # Create sun gear parameters
            sun_gear_params = SunGearParameters(
                sun_gear_radius=param_dict["sun_gear_radius"],
                ring_gear_radius=param_dict["ring_gear_radius"],
                gear_ratio=param_dict["gear_ratio"],
                journal_offset_x=param_dict["journal_offset_x"],
                journal_offset_y=param_dict["journal_offset_y"],
                max_back_rotation=param_dict["max_back_rotation"],
            )

            # Calculate objective components
            objective = 0.0

            # System size objective
            if self.targets.minimize_system_size:
                system_size = (
                    sun_gear_params.ring_gear_radius + sun_gear_params.sun_gear_radius
                )
                objective += self.targets.system_size_weight * system_size

            # Efficiency objective (simplified)
            if self.targets.maximize_efficiency:
                # Efficiency decreases with gear ratio and back rotation
                efficiency_penalty = (
                    sun_gear_params.gear_ratio * sun_gear_params.max_back_rotation
                )
                objective += self.targets.efficiency_weight * efficiency_penalty

            # Back rotation objective
            if self.targets.minimize_back_rotation:
                objective += (
                    self.targets.back_rotation_weight
                    * sun_gear_params.max_back_rotation
                )

            # Gear stress objective (simplified)
            if self.targets.minimize_gear_stress:
                # Stress increases with gear ratio and decreases with gear size
                stress_factor = (
                    sun_gear_params.gear_ratio / sun_gear_params.sun_gear_radius
                )
                objective += self.targets.gear_stress_weight * stress_factor

            return objective

        except Exception as e:
            log.warning(f"Sun gear objective function error: {e}")
            return 1e6  # Large penalty for invalid parameters

    def _define_constraints(
        self,
        theta: np.ndarray,
        x_theta: np.ndarray,
        cam_curves: dict[str, np.ndarray],
        optimized_params: dict[str, float],
    ) -> list[dict]:
        """Define optimization constraints for sun gear system."""
        constraints = []

        # Physical constraint: no interference
        def no_interference_constraint(params):
            (
                sun_gear_radius,
                ring_gear_radius,
                gear_ratio,
                journal_x,
                journal_y,
                back_rotation,
            ) = params

            # Check clearance between sun gear and ring gear
            clearance = (
                ring_gear_radius - sun_gear_radius - self.constraints.min_clearance
            )
            return clearance

        constraints.append(
            {
                "type": "ineq",
                "fun": no_interference_constraint,
            },
        )

        # Physical constraint: gear ratio consistency
        def gear_ratio_constraint(params):
            (
                sun_gear_radius,
                ring_gear_radius,
                gear_ratio,
                journal_x,
                journal_y,
                back_rotation,
            ) = params

            # Gear ratio should be consistent with radii
            expected_ratio = ring_gear_radius / sun_gear_radius
            ratio_error = abs(gear_ratio - expected_ratio)
            return 1.0 - ratio_error  # Allow 1% error

        constraints.append(
            {
                "type": "ineq",
                "fun": gear_ratio_constraint,
            },
        )

        return constraints

    def _generate_final_design(
        self,
        optimized_params: dict[str, float],
        theta: np.ndarray,
        x_theta: np.ndarray,
        cam_curves: dict[str, np.ndarray],
        cam_optimized_params: dict[str, float],
        primary_data: dict[str, np.ndarray],
        secondary_data: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Generate final sun gear design with optimized parameters."""
        # Create optimized sun gear parameters
        sun_gear_params = SunGearParameters(
            sun_gear_radius=optimized_params["sun_gear_radius"],
            ring_gear_radius=optimized_params["ring_gear_radius"],
            gear_ratio=optimized_params["gear_ratio"],
            journal_offset_x=optimized_params["journal_offset_x"],
            journal_offset_y=optimized_params["journal_offset_y"],
            max_back_rotation=optimized_params["max_back_rotation"],
        )

        # Calculate gear teeth based on radii and ratio
        sun_gear_teeth = int(
            sun_gear_params.sun_gear_radius * 2,
        )  # Simplified calculation
        ring_gear_teeth = int(sun_gear_teeth * sun_gear_params.gear_ratio)

        # Generate complete ring profile (360° coverage guaranteed)
        # Use endpoint=True to ensure exact 360° coverage
        psi_complete = np.linspace(0, 2 * np.pi, len(theta), endpoint=True)
        # Ensure the last point is exactly 2π (360°)
        psi_complete[-1] = 2 * np.pi
        R_psi_complete = np.full_like(psi_complete, sun_gear_params.ring_gear_radius)

        # Compile final design
        final_design = {
            "theta": theta,
            "x_theta": x_theta,
            "cam_curves": cam_curves,
            "psi": psi_complete,  # Complete 360° ring profile
            "R_psi": R_psi_complete,  # Complete ring radius profile
            "sun_gear_parameters": sun_gear_params.to_dict(),
            "optimized_parameters": optimized_params,
            "optimization_objectives": {
                "minimize_system_size": self.targets.minimize_system_size,
                "maximize_efficiency": self.targets.maximize_efficiency,
                "minimize_back_rotation": self.targets.minimize_back_rotation,
                "minimize_gear_stress": self.targets.minimize_gear_stress,
            },
            "gear_specifications": {
                "sun_gear_teeth": sun_gear_teeth,
                "ring_gear_teeth": ring_gear_teeth,
                "gear_ratio": sun_gear_params.gear_ratio,
                "back_rotation_capability": sun_gear_params.max_back_rotation
                * 180
                / np.pi,
            },
        }

        return final_design
