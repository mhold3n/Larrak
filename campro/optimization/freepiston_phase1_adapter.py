"""
Free-Piston IPOPT Phase 1 Adapter.

This adapter wraps the free-piston IPOPT optimization flow (solve_cycle) to
replace the legacy motion-law optimizer in Phase 1. It converts GUI inputs and
unified framework constraints to the free-piston parameter format and converts
results back to OptimizationResult format.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

import numpy as np

from campro.constraints.cam import CamMotionConstraints
from campro.freepiston.opt.config_factory import ConfigFactory
from campro.freepiston.opt.driver import solve_cycle
from campro.logging import get_logger
from campro.optimization.base import BaseOptimizer, OptimizationResult, OptimizationStatus

log = get_logger(__name__)


class FreePistonPhase1Adapter(BaseOptimizer):
    """
    Adapter that wraps free-piston IPOPT flow for Phase 1 optimization.

    This adapter replaces the legacy motion-law optimizer by routing optimization
    requests through the free-piston IPOPT flow with integrated combustion model.
    """

    def __init__(self, name: str = "FreePistonPhase1Adapter"):
        """Initialize the free-piston phase 1 adapter."""
        super().__init__(name)
        self._is_configured = True
        self._config: dict[str, Any] = {}

    def configure(self, **kwargs) -> None:
        """
        Configure the optimizer with problem-specific parameters.
        
        Args:
            **kwargs: Configuration parameters (stored for future use)
                - min_segments: Minimum number of collocation segments K
                - refinement_factor: K ≈ n_points / refinement_factor
                - disable_combustion: Disable combustion model for physics debugging
        """
        # Store configuration if needed
        self._config.update(kwargs)
        # Extract discretization parameters
        self._min_segments = kwargs.get("min_segments", None)
        self._refinement_factor = kwargs.get("refinement_factor", None)
        self._disable_combustion = kwargs.get("disable_combustion", False)
        self._is_configured = True

    def optimize(
        self,
        objective: Callable,
        constraints: Any,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Solve an optimization problem.
        
        This method provides compatibility with BaseOptimizer interface.
        For cam motion law problems, use solve_cam_motion_law() directly.
        
        Args:
            objective: Objective function (not used directly - handled by free-piston flow)
            constraints: Constraint system (should be CamMotionConstraints)
            initial_guess: Initial guess (optional)
            **kwargs: Additional parameters (cycle_time, motion_type, etc.)
            
        Returns:
            OptimizationResult object
        """
        # If constraints are CamMotionConstraints, delegate to solve_cam_motion_law
        if isinstance(constraints, CamMotionConstraints):
            cycle_time = kwargs.get("cycle_time", 1.0)
            motion_type = kwargs.get("motion_type", "minimum_jerk")
            n_points = kwargs.get("n_points", None)
            afr = kwargs.get("afr", None)
            ignition_timing = kwargs.get("ignition_timing", None)
            fuel_mass = kwargs.get("fuel_mass", None)
            ca50_target_deg = kwargs.get("ca50_target_deg", None)
            ca50_weight = kwargs.get("ca50_weight", None)
            duration_target_deg = kwargs.get("duration_target_deg", None)
            duration_weight = kwargs.get("duration_weight", None)
            
            return self.solve_cam_motion_law(
                cam_constraints=constraints,
                motion_type=motion_type,
                cycle_time=cycle_time,
                n_points=n_points,
                afr=afr,
                ignition_timing=ignition_timing,
                fuel_mass=fuel_mass,
                ca50_target_deg=ca50_target_deg,
                ca50_weight=ca50_weight,
                duration_target_deg=duration_target_deg,
                duration_weight=duration_weight,
            )
        else:
            raise NotImplementedError(
                f"FreePistonPhase1Adapter.optimize() only supports CamMotionConstraints, "
                f"got {type(constraints)}"
            )

    def solve_custom_objective(
        self,
        objective_function: Callable,
        constraints: CamMotionConstraints,
        distance: float,
        time_horizon: float | None = None,
        n_points: int | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Solve cam motion law with custom objective function.
        
        Note: The free-piston IPOPT flow uses its own objective function, so the
        custom objective_function parameter is accepted for compatibility but
        not directly used. The optimization uses the free-piston IPOPT objective
        which includes thermal efficiency and motion constraints.
        
        Args:
            objective_function: Custom objective function (accepted for compatibility)
            constraints: Cam motion constraints
            distance: Total stroke distance (used as stroke in constraints)
            time_horizon: Cycle time in seconds
            n_points: Number of collocation points
            **kwargs: Additional parameters (afr, ignition_timing, initial_guess, etc.)
            
        Returns:
            OptimizationResult object
        """
        # Extract additional parameters from kwargs
        afr = kwargs.get("afr", None)
        ignition_timing = kwargs.get("ignition_timing", None)
        fuel_mass = kwargs.get("fuel_mass", None)
        ca50_target_deg = kwargs.get("ca50_target_deg", None)
        ca50_weight = kwargs.get("ca50_weight", None)
        duration_target_deg = kwargs.get("duration_target_deg", None)
        duration_weight = kwargs.get("duration_weight", None)
        
        # Use time_horizon if provided, otherwise use cycle_time from constraints
        cycle_time = time_horizon if time_horizon is not None else 1.0
        
        # Delegate to solve_cam_motion_law (free-piston flow has its own objective)
        log.info(
            f"solve_custom_objective: delegating to solve_cam_motion_law "
            f"(custom objective not used by free-piston IPOPT flow)"
        )
        
        return self.solve_cam_motion_law(
            cam_constraints=constraints,
            motion_type="pcurve_te",  # Default motion type
            cycle_time=cycle_time,
            n_points=n_points,
            afr=afr,
            ignition_timing=ignition_timing,
            fuel_mass=fuel_mass,
            ca50_target_deg=ca50_target_deg,
            ca50_weight=ca50_weight,
            duration_target_deg=duration_target_deg,
            duration_weight=duration_weight,
        )

    def solve_cam_motion_law(
        self,
        cam_constraints: CamMotionConstraints,
        motion_type: str = "minimum_jerk",
        cycle_time: float = 1.0,
        *,
        n_points: int | None = None,
        afr: float | None = None,
        ignition_timing: float | None = None,
        fuel_mass: float | None = None,
        ca50_target_deg: float | None = None,
        ca50_weight: float | None = None,
        duration_target_deg: float | None = None,
        duration_weight: float | None = None,
    ) -> OptimizationResult:
        """
        Solve cam motion law using free-piston IPOPT flow.

        Args:
            cam_constraints: Cam-specific constraints
            motion_type: Type of motion law (for future use)
            cycle_time: Total cycle time in seconds
            n_points: Number of collocation points (used to determine K)
            afr: Air-fuel ratio (optional, from GUI)
            ignition_timing: Ignition timing in seconds (optional, from GUI)

        Returns:
            OptimizationResult with motion profile
        """
        log.info(
            f"Free-piston IPOPT Phase 1: stroke={cam_constraints.stroke}mm, "
            f"cycle_time={cycle_time}s, motion_type={motion_type}",
        )
        print(
            f"[FREE-PISTON] Free-piston IPOPT Phase 1: stroke={cam_constraints.stroke}mm, "
            f"cycle_time={cycle_time}s, motion_type={motion_type}",
            file=sys.stderr,
            flush=True,
        )

        try:
            # Convert constraints and inputs to free-piston parameter dict
            print(
                f"[FREE-PISTON] Building problem dictionary: "
                f"n_points={n_points}, afr={afr}, ignition_timing={ignition_timing}",
                file=sys.stderr,
                flush=True,
            )
            build_start = time.time()
            # Extract discretization parameters from config if available
            min_segments = getattr(self, '_min_segments', None)
            refinement_factor = getattr(self, '_refinement_factor', None)
            disable_combustion = getattr(self, '_disable_combustion', False)
            
            P = self._build_problem_dict(
                cam_constraints=cam_constraints,
                cycle_time=cycle_time,
                n_points=n_points,
                afr=afr,
                ignition_timing=ignition_timing,
                fuel_mass=fuel_mass,
                ca50_target_deg=ca50_target_deg,
                ca50_weight=ca50_weight,
                duration_target_deg=duration_target_deg,
                duration_weight=duration_weight,
                min_segments=min_segments,
                refinement_factor=refinement_factor,
                disable_combustion=disable_combustion,
            )
            build_elapsed = time.time() - build_start
            # Extract problem dimensions for logging
            num = P.get("num", {})
            K = int(num.get("K", 10))
            C = int(num.get("C", 1))
            # Estimate variable count: K * C * 6 (states) + initial states + controls
            estimated_vars = K * C * 6 + 6 + K * C * 2  # Rough estimate
            print(
                f"[FREE-PISTON] Problem dictionary built in {build_elapsed:.3f}s: "
                f"K={K}, C={C}, estimated_vars≈{estimated_vars}",
                file=sys.stderr,
                flush=True,
            )
            log.info(
                f"Problem dictionary built in {build_elapsed:.3f}s: "
                f"K={K}, C={C}, estimated_vars≈{estimated_vars}"
            )

            # Solve using free-piston IPOPT flow
            print(
                f"[FREE-PISTON] Calling free-piston IPOPT solve_cycle()...",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[FREE-PISTON] Problem dimensions: K={K}, C={C}, "
                f"estimated_vars≈{estimated_vars}, estimated_constraints≈{K * C * 4}",
                file=sys.stderr,
                flush=True,
            )
            solve_start = time.time()
            solution = solve_cycle(P)
            solve_elapsed = time.time() - solve_start
            print(
                f"[FREE-PISTON] Free-piston IPOPT solve_cycle() completed in {solve_elapsed:.3f}s",
                file=sys.stderr,
                flush=True,
            )
            log.info(f"Free-piston IPOPT solve_cycle() completed in {solve_elapsed:.3f}s")

            # Convert solution to OptimizationResult
            result = self._convert_solution_to_result(
                solution=solution,
                cam_constraints=cam_constraints,
                cycle_time=cycle_time,
                n_points=n_points or 360,
            )

            log.info(
                f"Free-piston IPOPT optimization completed: "
                f"status={result.status}, iterations={result.iterations}",
            )

            return result

        except Exception as e:
            log.error(f"Free-piston IPOPT optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                error_message=str(e),
                solution={},
            )

    def _build_problem_dict(
        self,
        cam_constraints: CamMotionConstraints,
        cycle_time: float,
        n_points: int | None = None,
        afr: float | None = None,
        ignition_timing: float | None = None,
        fuel_mass: float | None = None,
        ca50_target_deg: float | None = None,
        ca50_weight: float | None = None,
        duration_target_deg: float | None = None,
        duration_weight: float | None = None,
        min_segments: int | None = None,
        refinement_factor: int | None = None,
        disable_combustion: bool = False,
    ) -> dict:
        """
        Build free-piston IPOPT parameter dictionary from GUI inputs.

        Args:
            cam_constraints: Cam motion constraints
            cycle_time: Cycle time in seconds
            n_points: Number of collocation points
            afr: Air-fuel ratio (optional)
            ignition_timing: Ignition timing in seconds (optional)

        Returns:
            Parameter dictionary P for solve_cycle()
        """
        # Use ConfigFactory to create base config
        config = ConfigFactory.create_default_config()

        # Convert stroke from mm to m
        stroke_m = cam_constraints.stroke / 1000.0

        # Update geometry
        config.geometry["stroke"] = stroke_m
        config.geometry["bore"] = stroke_m * 0.8  # Estimate bore from stroke
        config.geometry["compression_ratio"] = 10.0  # Default compression ratio

        # Calculate clearance volume from stroke and compression ratio
        bore = config.geometry["bore"]
        area_m2 = np.pi * (bore / 2.0) ** 2
        stroke_volume_m3 = area_m2 * stroke_m
        clearance_volume_m3 = stroke_volume_m3 / (config.geometry["compression_ratio"] - 1.0)
        config.geometry["clearance_volume"] = clearance_volume_m3

        # Set collocation grid size
        # K = number of finite elements, roughly n_points / C
        # Use n_points if provided, otherwise default
        # Parameterize with min_segments and refinement_factor for experimentation
        min_segments_val = min_segments if min_segments is not None else 10
        refinement_factor_val = refinement_factor if refinement_factor is not None else 4
        
        if n_points is not None:
            # For collocation, we typically use fewer elements than points
            # Use parameterized mapping: K = max(min_segments, n_points / refinement_factor)
            # This allows incremental refinement: start with coarse mesh, then refine
            K = max(min_segments_val, int(n_points / refinement_factor_val))
        else:
            K = max(min_segments_val, 20)  # Default with safety floor

        config.num = {"K": K, "C": 1, "min_segments": min_segments_val, "refinement_factor": refinement_factor_val}  # Radau collocation with C=1

        # Enable/disable integrated combustion model (for physics debugging)
        # When disable_combustion=True, use simplified mechanical model to verify solver health
        if disable_combustion:
            config.combustion["use_integrated_model"] = False
            log.info("Combustion model disabled for physics debugging (mechanical core only)")
        else:
            config.combustion["use_integrated_model"] = True
        config.combustion["cycle_time_s"] = cycle_time
        config.combustion["fuel_type"] = "diesel"  # Default fuel type

        # Set AFR and ignition timing if provided (from GUI)
        if afr is not None:
            config.combustion["afr"] = float(afr)
        if ignition_timing is not None:
            config.combustion["ignition_initial_s"] = float(ignition_timing)
            # Set bounds around ignition timing
            ignition_bounds = (
                max(0.001, float(ignition_timing) - 0.002),
                min(cycle_time * 0.5, float(ignition_timing) + 0.002),
            )
            config.combustion["ignition_bounds_s"] = ignition_bounds

        # Estimate fuel mass from geometry and AFR
        # This is a rough estimate - will be refined in next phase
        fuel_mass_kg = 5e-4  # Default
        if fuel_mass is not None:
            fuel_mass_kg = max(1e-9, float(fuel_mass))
        config.combustion["fuel_mass_kg"] = fuel_mass_kg

        if ca50_target_deg is not None:
            config.combustion["ca50_target_deg"] = float(ca50_target_deg)
        if ca50_weight is not None:
            config.combustion["w_ca50"] = float(ca50_weight)
        if duration_target_deg is not None:
            config.combustion["ca_duration_target_deg"] = float(duration_target_deg)
        if duration_weight is not None:
            config.combustion["w_ca_duration"] = float(duration_weight)

        # Set initial conditions
        config.combustion["initial_temperature_K"] = 900.0
        config.combustion["initial_pressure_Pa"] = 1e5

        # Convert config to dict format for solve_cycle
        P = {
            "geometry": config.geometry,
            "thermodynamics": config.thermodynamics,
            "bounds": config.bounds,
            "constraints": config.constraints,
            "num": config.num,
            "solver": config.solver,
            "objective": config.objective,
            "combustion": config.combustion,
            "flow": {"use_1d_gas": False},  # Use 0D gas model
            "walls": {},
        }

        return P

    def _convert_solution_to_result(
        self,
        solution,
        cam_constraints: CamMotionConstraints,
        cycle_time: float,
        n_points: int,
    ) -> OptimizationResult:
        """
        Convert free-piston Solution to OptimizationResult.

        Args:
            solution: Solution object from solve_cycle()
            cam_constraints: Original cam constraints
            cycle_time: Cycle time
            n_points: Number of points for output

        Returns:
            OptimizationResult with motion profile
        """
        # Extract optimization metadata
        opt_meta = solution.meta.get("optimization", {})
        success = opt_meta.get("success", False)
        iterations = opt_meta.get("iterations", 0)
        cpu_time = opt_meta.get("cpu_time", 0.0)
        f_opt = opt_meta.get("f_opt", float("inf"))
        message = opt_meta.get("message", "")

        # Map solver status
        if success:
            status = OptimizationStatus.CONVERGED
        else:
            status = OptimizationStatus.FAILED

        # Extract solution variables
        x_opt = opt_meta.get("x_opt")
        if x_opt is None:
            log.warning("No solution variables found in free-piston result")
            return OptimizationResult(
                status=status,
                objective_value=f_opt,
                iterations=iterations,
                solve_time=cpu_time,
                error_message=message if not success else None,
                solution={},
            )

        # Extract motion profile from collocation solution
        # The free-piston NLP has states: xL, xR, vL, vR at each collocation point
        # We need to extract these and map to cam angle and position arrays
        grid_meta = solution.meta.get("grid", {})
        meta_nlp = solution.meta.get("meta", {})

        # Get collocation grid info
        K = meta_nlp.get("K", 20)
        C = meta_nlp.get("C", 1)

        # Extract position and velocity from solution
        # Variable structure: [xL0, xR0, vL0, vR0, ... (for each collocation point)]
        # For now, create a simplified motion profile
        # TODO: Extract actual collocation states properly in next phase
        try:
            # Create a basic motion profile based on constraints
            # This is a placeholder - full extraction will be done in next phase
            cam_angle = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
            
            # Simple sinusoidal motion profile as placeholder
            # Position goes from 0 to stroke over upstroke portion
            upstroke_frac = cam_constraints.upstroke_duration_percent / 100.0
            position = np.zeros(n_points)
            
            for i, theta in enumerate(cam_angle):
                theta_deg = np.degrees(theta)
                if theta_deg < 180 * upstroke_frac:
                    # Upstroke: position increases
                    progress = theta_deg / (180 * upstroke_frac)
                    position[i] = cam_constraints.stroke * (1 - np.cos(np.pi * progress)) / 2.0
                else:
                    # Downstroke: position decreases
                    progress = (theta_deg - 180 * upstroke_frac) / (180 * (1 - upstroke_frac))
                    position[i] = cam_constraints.stroke * (1 + np.cos(np.pi * progress)) / 2.0

            # Compute velocity and acceleration from position
            dt = cycle_time / n_points
            velocity = np.gradient(position, dt)
            acceleration = np.gradient(velocity, dt)

            # Store in solution dict
            solution_dict = {
                "cam_angle": cam_angle,
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
            }

            # Extract CA markers from combustion model if available
            combustion_meta = meta_nlp.get("combustion_model", {})
            ca_markers: dict[str, float] = {}
            if combustion_meta:
                ca_marker_expr = combustion_meta.get("ca_markers", {})
                if ca_marker_expr:
                    try:
                        import casadi as ca  # type: ignore

                        nlp_dict = solution.data.get("nlp") if hasattr(solution, "data") else None
                        x_opt_vec = np.array(opt_meta.get("x_opt", []), dtype=float)
                        if nlp_dict is not None and x_opt_vec.size > 0:
                            marker_names = sorted(ca_marker_expr.keys())
                            expr_list = [ca_marker_expr[name] for name in marker_names]
                            eval_fun = ca.Function(
                                "eval_ca_markers",
                                [nlp_dict["x"]],
                                [ca.vertcat(*expr_list)],
                            )
                            values = np.array(eval_fun(ca.DM(x_opt_vec))).flatten()
                            for idx, name in enumerate(marker_names):
                                ca_markers[name] = float(values[idx])
                        else:
                            for name, value in ca_marker_expr.items():
                                if hasattr(value, "is_constant") and value.is_constant():
                                    ca_markers[name] = float(value)
                                elif isinstance(value, (int, float)):
                                    ca_markers[name] = float(value)
                    except Exception as ca_exc:  # pragma: no cover - best effort
                        log.debug(f"CA marker evaluation failed: {ca_exc}")

            # Build metadata
            metadata = {
                "free_piston_optimization": True,
                "combustion_model": "integrated",
                "ca_markers": ca_markers,
                "grid_info": {
                    "K": K,
                    "C": C,
                    "n_points": n_points,
                },
                "iterations": iterations,
                "cpu_time": cpu_time,
            }

            return OptimizationResult(
                status=status,
                objective_value=f_opt,
                solution=solution_dict,
                iterations=iterations,
                solve_time=cpu_time,
                error_message=message if not success else None,
                metadata=metadata,
            )

        except Exception as e:
            log.error(f"Failed to extract motion profile from solution: {e}", exc_info=True)
            return OptimizationResult(
                status=status,
                objective_value=f_opt,
                iterations=iterations,
                solve_time=cpu_time,
                error_message=f"Solution extraction failed: {e}" if success else message,
                solution={},
            )
