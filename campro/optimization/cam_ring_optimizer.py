"""
Cam-ring system optimization using collocation methods.

This module implements a secondary optimization that optimizes the cam-ring system
parameters (cam radius, connecting rod length, ring design) using the same
collocation approach as the primary motion law optimization.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from campro.freepiston.core.chem import (
    CombustionParameters,
    create_combustion_parameters,
)
from campro.litvin.config import GeometrySearchConfig
from campro.litvin.motion import RadialSlotMotion
from campro.litvin.optimization import OptimizationOrder, optimize_geometry
from campro.logging import get_logger
from campro.physics.geometry.litvin import LitvinGearGeometry, LitvinSynthesis

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationSettings
from .se2 import angle_map

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
    max_curvature: float = 10.0  # Maximum curvature

    # NEW: Gear geometry constraints
    ring_teeth_candidates: list[int] = field(
        default_factory=lambda: [40, 50, 60, 70, 80],
    )
    planet_teeth_candidates: list[int] = field(
        default_factory=lambda: [20, 25, 30, 35, 40],
    )
    pressure_angle_min: float = 15.0
    pressure_angle_max: float = 30.0
    addendum_factor_min: float = 0.8
    addendum_factor_max: float = 1.2
    samples_per_rev: int = 360

    # Optimization constraints
    max_iterations: int = 100
    tolerance: float = 1e-6

    # Combustion model parameters for section-based optimization
    combustion_params: CombustionParameters | None = None

    # Multi-threading
    n_threads: int = field(default_factory=lambda: os.cpu_count() or 4)


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

    def __init__(
        self,
        name: str = "CamRingOptimizer",
        settings: CollocationSettings | None = None,
        enable_order2_micro: bool | None = None,
    ):
        super().__init__(name)
        self.settings = settings or CollocationSettings()
        self.constraints = CamRingOptimizationConstraints()
        self.targets = CamRingOptimizationTargets()
        # Feature flag to enable CasADi ORDER2_MICRO path
        if enable_order2_micro is None:
            env_flag = os.getenv("CAMPRO_ENABLE_ORDER2_MICRO", "0").strip()
            self.enable_order2_micro = env_flag in {"1", "true", "TRUE", "yes", "on"}
        else:
            self.enable_order2_micro = bool(enable_order2_micro)
        self._is_configured = True

    def configure(
        self,
        constraints: CamRingOptimizationConstraints | None = None,
        targets: CamRingOptimizationTargets | None = None,
        **kwargs,
    ) -> None:
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
            # Allow toggling ORDER2 via configure
            if key == "enable_order2_micro":
                self.enable_order2_micro = bool(value)

        self._is_configured = True
        log.info(f"Configured {self.name} with constraints and targets")

    def optimize(
        self,
        primary_data: dict[str, NDArray[np.float64]],
        initial_guess: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Optimize cam-ring system parameters using multi-order Litvin optimization.

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

        log.info(
            f"Starting multi-order Litvin cam-ring system optimization with {self.name}",
        )

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
                raise ValueError(
                    "Primary data must contain cam_angle and position arrays",
                )

            ca_markers = primary_data.get("ca_markers")

            # Initialize combustion parameters for legacy fallback only
            if ca_markers is None and self.constraints.combustion_params is None:
                from campro.utils.progress_logger import ProgressLogger

                comb_logger = ProgressLogger("COMBUSTION", flush_immediately=True)
                comb_logger.info("Initializing default combustion parameters (gasoline)")
                self.constraints.combustion_params = create_combustion_parameters("gasoline")
                comb_logger.info(
                    f"Combustion params: start={self.constraints.combustion_params.theta_start:.2f}°, "
                    f"duration={self.constraints.combustion_params.theta_duration:.2f}°"
                )

            # Identify combustion sections for piecewise optimization
            from campro.litvin.section_analysis import identify_combustion_sections
            from campro.utils.progress_logger import ProgressLogger
            
            section_logger = ProgressLogger("COMBUSTION", flush_immediately=True)
            section_logger.step(1, 2, "Identifying combustion sections from motion law")
            
            # Prepare primary data dict for section analysis
            primary_data_for_sections = {
                "theta": theta,  # Already in degrees
                "position": x_theta,
            }
            if ca_markers:
                primary_data_for_sections["ca_markers"] = ca_markers
            
            try:
                section_boundaries = identify_combustion_sections(
                    primary_data_for_sections,
                    self.constraints.combustion_params,
                )
                section_logger.step_complete("Section identification", 0.0)
                section_analysis_available = True
                section_logger.info(
                    f"Identified {len(section_boundaries.sections)} sections for piecewise optimization"
                )
            except Exception as e:
                import traceback
                section_logger.warning(
                    f"Section analysis failed: {e}. Falling back to grid search."
                )
                section_logger.warning(f"Traceback: {traceback.format_exc()}")
                section_boundaries = None
                section_analysis_available = False

            # Set up initial guess
            if initial_guess is None:
                initial_guess = self._get_default_initial_guess(primary_data)

            log.info(f"Initial guess: {initial_guess}")

        # Create RadialSlotMotion from primary motion law data (on universal grid)
            motion = self._create_radial_slot_motion(primary_data)

            # Build GeometrySearchConfig with gear parameter candidates
            geometry_config = GeometrySearchConfig(
                ring_teeth_candidates=self.constraints.ring_teeth_candidates,
                planet_teeth_candidates=self.constraints.planet_teeth_candidates,
                pressure_angle_deg_bounds=(
                    self.constraints.pressure_angle_min,
                    self.constraints.pressure_angle_max,
                ),
                addendum_factor_bounds=(
                    self.constraints.addendum_factor_min,
                    self.constraints.addendum_factor_max,
                ),
                base_center_radius=initial_guess["base_radius"],
                samples_per_rev=self.constraints.samples_per_rev,
                motion=motion,
                section_boundaries=section_boundaries if section_analysis_available else None,
                n_threads=self.constraints.n_threads,
            )

            # Perform multi-order optimization (0→1→2→3)
            from campro.utils.progress_logger import ProgressLogger
            order_logger = ProgressLogger("CAM-RING", flush_immediately=True)

            order_logger.step(1, 3, "ORDER0_EVALUATE: Initial geometry evaluation")
            order0_start = time.time()
            order0_result = optimize_geometry(
                geometry_config, OptimizationOrder.ORDER0_EVALUATE,
            )
            order0_elapsed = time.time() - order0_start
            order_logger.info(f"ORDER0 completed: feasible={order0_result.feasible}, time={order0_elapsed:.3f}s")
            order_logger.step_complete("ORDER0_EVALUATE", order0_elapsed)

            order_logger.step(2, 3, "ORDER1_GEOMETRY: Basic geometry optimization")
            order_logger.info("Starting multi-parameter grid search with local refinement...")
            order1_start = time.time()
            order1_result = optimize_geometry(
                geometry_config, OptimizationOrder.ORDER1_GEOMETRY,
            )
            order1_elapsed = time.time() - order1_start
            obj_str = f"{order1_result.objective_value:.6f}" if order1_result.objective_value is not None else "N/A"
            order_logger.info(f"ORDER1 completed: feasible={order1_result.feasible}, "
                            f"objective={obj_str}, "
                            f"time={order1_elapsed:.3f}s")
            order_logger.step_complete("ORDER1_GEOMETRY", order1_elapsed)

            if self.enable_order2_micro:
                order_logger.step(3, 3, "ORDER2_MICRO: Advanced optimization (CasADi + Ipopt)")
                order_logger.info("Starting CasADi-based NLP optimization with Ipopt...")
                order2_start = time.time()
                try:
                    order2_result = optimize_geometry(
                        geometry_config, OptimizationOrder.ORDER2_MICRO,
                    )
                    order2_elapsed = time.time() - order2_start
                    obj_str = f"{order2_result.objective_value:.6f}" if order2_result.objective_value is not None else "N/A"
                    order_logger.info(f"ORDER2 completed: feasible={order2_result.feasible}, "
                                    f"objective={obj_str}, "
                                    f"time={order2_elapsed:.3f}s")
                    order_logger.step_complete("ORDER2_MICRO", order2_elapsed)
                except Exception as e:
                    order2_elapsed = time.time() - order2_start
                    order_logger.warning(f"ORDER2_MICRO failed: {e}; falling back to ORDER1/0")
                    order_logger.step_complete("ORDER2_MICRO (failed)", order2_elapsed)
                    order2_result = type(
                        "MockResult",
                        (),
                        {
                            "feasible": False,
                            "best_config": None,
                            "objective_value": None,
                            "ipopt_analysis": None,
                        },
                    )()
            else:
                order_logger.step(3, 3, "ORDER2_MICRO: Disabled (set enable_order2_micro=True to enable)")
                order_logger.info(
                    "ORDER2_MICRO disabled (set enable_order2_micro=True or CAMPRO_ENABLE_ORDER2_MICRO=1 to enable)",
                )
                order2_result = type(
                    "MockResult",
                    (),
                    {
                        "feasible": False,
                        "best_config": None,
                        "objective_value": None,
                        "ipopt_analysis": None,
                    },
                )()

            # Use the best result from the optimization orders
            best_result = (
                order2_result
                if order2_result.feasible
                else (order1_result if order1_result.feasible else order0_result)
            )

            if best_result.feasible and best_result.best_config is not None:
                # Generate final ring design with optimized gear geometry
                final_design = self._generate_final_design_from_gear_config(
                    best_result.best_config,
                    theta,
                    x_theta,
                    primary_data,
                )

                # Update result
                result.status = OptimizationStatus.CONVERGED
                result.solution = final_design
                result.objective_value = best_result.objective_value or float("inf")
                result.iterations = 1  # Multi-order optimization counts as 1 iteration
                result.metadata = {
                    "optimization_method": "MultiOrderLitvin",
                    "optimized_gear_config": {
                        "ring_teeth": best_result.best_config.ring_teeth,
                        "planet_teeth": best_result.best_config.planet_teeth,
                        "pressure_angle_deg": best_result.best_config.pressure_angle_deg,
                        "addendum_factor": best_result.best_config.addendum_factor,
                        "base_center_radius": best_result.best_config.base_center_radius,
                    },
                    "order_results": {
                        "order0_feasible": order0_result.feasible,
                        "order1_feasible": order1_result.feasible,
                        "order2_feasible": order2_result.feasible,
                    },
                    "initial_guess": initial_guess,
                    # Pass through analysis from litvin optimization (ORDER2_MICRO uses Ipopt)
                    "ipopt_analysis": order2_result.ipopt_analysis
                    if hasattr(order2_result, "ipopt_analysis")
                    else None,
                }

                log.info("Multi-order optimization completed successfully")
                log.info(f"Final objective value: {best_result.objective_value:.6f}")
                log.info(
                    f"Optimized gear config: {result.metadata['optimized_gear_config']}",
                )

            else:
                # Provide fallback result to allow cascaded optimization to continue
                # This is a temporary workaround while CasADi issues are resolved
                log.warning("All optimization orders failed, providing fallback result")

                # Create a simple fallback design with proper structure
                fallback_design = {
                    "optimized_parameters": {
                        "base_radius": initial_guess.get(
                            "base_radius", 20.0,
                        ),  # Use initial guess or default
                    },
                    "gear_geometry": {
                        "ring_teeth": 50,  # Default values
                        "planet_teeth": 25,
                        "pressure_angle_deg": 20.0,
                        "addendum_factor": 1.0,
                    },
                }

                result.status = OptimizationStatus.CONVERGED
                result.solution = fallback_design
                result.objective_value = float("inf")  # Indicate suboptimal
                result.iterations = 0
                result.metadata = {
                    "optimization_method": "MultiOrderLitvin_Fallback",
                    "optimized_gear_config": {
                        "ring_teeth": fallback_design["gear_geometry"]["ring_teeth"],
                        "planet_teeth": fallback_design["gear_geometry"][
                            "planet_teeth"
                        ],
                        "pressure_angle_deg": fallback_design["gear_geometry"][
                            "pressure_angle_deg"
                        ],
                        "addendum_factor": fallback_design["gear_geometry"][
                            "addendum_factor"
                        ],
                        "base_center_radius": fallback_design["optimized_parameters"][
                            "base_radius"
                        ],
                    },
                    "order_results": {
                        "order0_feasible": order0_result.feasible,
                        "order1_feasible": order1_result.feasible,
                        "order2_feasible": order2_result.feasible,
                    },
                    "fallback": True,
                    "error_message": "All optimization orders failed, using fallback values",
                }
                log.warning(
                    "Using fallback secondary optimization result to continue cascaded optimization",
                )

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.metadata = {"error_message": str(e)}
            log.error(f"Optimization error: {e}")
            import traceback

            log.error(f"Traceback: {traceback.format_exc()}")

        finally:
            result.solve_time = time.time() - start_time

        return result

    def _get_default_initial_guess(
        self, primary_data: dict[str, NDArray[np.float64]],
    ) -> dict[str, float]:
        """Get default initial guess based on primary data."""
        # Use stroke-based initial guesses
        stroke = np.max(primary_data.get("position", [20.0])) - np.min(
            primary_data.get("position", [0.0]),
        )

        return {
            "base_radius": float(stroke),
        }

    def _create_radial_slot_motion(
        self, primary_data: dict[str, NDArray[np.float64]],
    ) -> RadialSlotMotion:
        """Convert primary motion law to RadialSlotMotion for Litvin synthesis."""
        from scipy.interpolate import interp1d

        theta = primary_data["cam_angle"]  # degrees
        x_theta = primary_data["position"]  # mm

        # Create interpolators for center offset and planet angle
        theta_rad = np.deg2rad(theta)
        center_offset_interp = interp1d(
            theta_rad, x_theta, kind="cubic", fill_value="extrapolate",
        )

        # Planet angle mapping centralized (default scale 2.0, no offset)
        planet_angle_fn = lambda th: angle_map(th, scale=2.0, offset=0.0)

        return RadialSlotMotion(
            center_offset_fn=lambda th: float(center_offset_interp(th)),
            planet_angle_fn=planet_angle_fn,
        )

    def _generate_final_design_from_gear_config(
        self,
        gear_config,
        theta: NDArray[np.float64],
        x_theta: NDArray[np.float64],
        primary_data: dict[str, NDArray[np.float64]],
    ) -> dict[str, Any]:
        """Generate final design using optimized gear geometry."""
        from campro.litvin.planetary_synthesis import synthesize_planet_from_motion

        # Synthesize final profile using optimized gear configuration
        planet_profile = synthesize_planet_from_motion(gear_config)

        # Build cam polar profile from primary motion: r(θ) = r_b + x(θ)
        base_radius = gear_config.base_center_radius
        theta_rad = np.deg2rad(theta)
        r_profile = base_radius + x_theta

        # Create Litvin synthesis result for compatibility
        litvin = LitvinSynthesis()
        litvin_result = litvin.synthesize_from_cam_profile(
            theta=theta_rad,
            r_profile=r_profile,
            target_ratio=1.0,
        )

        # Compute gear geometry using LitvinGearGeometry.from_synthesis
        gear_geometry = LitvinGearGeometry.from_synthesis(
            theta=theta_rad,
            r_profile=r_profile,
            psi=litvin_result.psi,
            R_psi=litvin_result.R_psi,
            target_average_radius=self.constraints.target_average_ring_radius,
        )

        # Return complete gear design with tooth profiles
        return {
            "base_radius": base_radius,
            "ring_teeth": gear_config.ring_teeth,
            "planet_teeth": gear_config.planet_teeth,
            "pressure_angle_deg": gear_config.pressure_angle_deg,
            "addendum_factor": gear_config.addendum_factor,
            "cam_profile": {
                "theta": theta,
                "profile_radius": r_profile,
            },
            "ring_profile": {
                "psi": litvin_result.psi,
                "R_psi": litvin_result.R_psi,
            },
            "planet_profile": {
                "points": planet_profile.points,
            },
            "gear_geometry": {
                "base_circle_cam": gear_geometry.base_circle_cam,
                "base_circle_ring": gear_geometry.base_circle_ring,
                "z_cam": gear_geometry.z_cam,
                "z_ring": gear_geometry.z_ring,
                "pressure_angle_rad": gear_geometry.pressure_angle_rad,
                "flanks": {
                    "tooth": {
                        "theta": [p[0] for p in planet_profile.points],
                        "r": [p[1] for p in planet_profile.points],
                    },
                },
            },
            "litvin_result": litvin_result,
        }

    def _define_constraints(
        self,
        theta: NDArray[np.float64],
        x_theta: NDArray[np.float64],
        primary_data: dict[str, NDArray[np.float64]],
    ) -> list[dict[str, Any]]:
        """Define optimization constraints."""
        constraints = []

        # Physical constraint: positive base radius
        def positive_radii_constraint(params: np.ndarray) -> float:
            base_radius = float(params[0])
            return base_radius - 1.0

        constraints.append(
            {
                "type": "ineq",
                "fun": positive_radii_constraint,
            },
        )

        return constraints
