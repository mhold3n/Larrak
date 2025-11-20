"""
Free-Piston IPOPT Phase 1 Adapter.

This adapter wraps the free-piston IPOPT optimization flow (solve_cycle) to
replace the legacy motion-law optimizer in Phase 1. It converts GUI inputs and
unified framework constraints to the free-piston parameter format and converts
results back to OptimizationResult format.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

import numpy as np

from campro.constraints.cam import CamMotionConstraints
from campro.physics.simple_cycle_adapter import (
    SimpleCycleAdapter,
    CycleGeometry,
    CycleThermo,
    WiebeParams,
)
from campro.freepiston.opt.config_factory import ConfigFactory
from campro.freepiston.opt.driver import solve_cycle
from campro.logging import get_logger
from campro.optimization.base import BaseOptimizer, OptimizationResult, OptimizationStatus
from campro.utils import format_duration
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)

_FALSEY = {"0", "false", "no", "off"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in _FALSEY


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
        self._last_combustion_inputs: dict[str, Any] = {}
        self._last_geometry: dict[str, float] | None = None
        self._last_cycle_time: float | None = None
        self._bounce_alpha: float = 1.0
        self._bounce_beta: float = 0.0
        self._constant_load_value: float | None = None
        self._workload_target_j: float | None = None

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
        if "bounce_alpha" in kwargs:
            self._bounce_alpha = float(kwargs["bounce_alpha"])
        if "bounce_beta" in kwargs:
            self._bounce_beta = float(kwargs["bounce_beta"])
        if "constant_load_value" in kwargs:
            try:
                self._constant_load_value = float(kwargs["constant_load_value"])
            except Exception:
                self._constant_load_value = None
        if "workload_target" in kwargs:
            try:
                self._workload_target_j = float(kwargs["workload_target"])
            except Exception:
                self._workload_target_j = self._workload_target_j
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
        reporter = StructuredReporter(
            context="FREE-PISTON",
            logger=None,
            stream_out=sys.stderr,
            stream_err=sys.stderr,
            debug_env="FREE_PISTON_DEBUG",
            force_debug=True,
        )
        reporter.info(
            f"Free-piston IPOPT Phase 1: stroke={cam_constraints.stroke}mm, "
            f"cycle_time={cycle_time}s, motion_type={motion_type}",
        )

        try:
            # Convert constraints and inputs to free-piston parameter dict
            reporter.info(
                f"Building problem dictionary: n_points={n_points}, afr={afr}, ignition_timing={ignition_timing}"
            )
            build_start = time.time()
            self._last_combustion_inputs = {
                "afr": afr,
                "fuel_mass": fuel_mass,
                "ignition_time_s": ignition_timing,
                "ca50_target_deg": ca50_target_deg,
                "ca50_weight": ca50_weight,
            }
            self._last_cycle_time = cycle_time
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
            reporter.info(
                f"Problem dictionary built in {format_duration(build_elapsed)}: "
                f"K={K}, C={C}, estimated_vars≈{estimated_vars}"
            )

            # Solve using free-piston IPOPT flow
            reporter.info("Calling free-piston IPOPT solve_cycle()...")
            reporter.debug(
                f"Problem dimensions: K={K}, C={C}, estimated_vars≈{estimated_vars}, estimated_constraints≈{K * C * 4}"
            )
            solve_start = time.time()
            solution = solve_cycle(P)
            solve_elapsed = time.time() - solve_start
            reporter.info(f"Free-piston IPOPT solve_cycle() completed in {format_duration(solve_elapsed)}")

            # Convert solution to OptimizationResult
            result = self._convert_solution_to_result(
                solution=solution,
                cam_constraints=cam_constraints,
                cycle_time=cycle_time,
                n_points=n_points or 360,
            )

            reporter.info(
                f"Free-piston IPOPT optimization completed: status={result.status}, iterations={result.iterations}"
            )

            diag_enabled = _env_flag("FREE_PISTON_DIAGNOSTICS", default=True) or reporter.show_debug
            if diag_enabled:
                optimization_meta = solution.meta.get("optimization", {})
                iteration_summary = optimization_meta.get("iteration_summary")
                pressure_meta = result.metadata.get("pressure_invariance")
                ca_markers = result.metadata.get("ca_markers") or {}
                with reporter.section("Optimization diagnostics", level="INFO"):
                    # Handle None values before formatting
                    cpu_time = optimization_meta.get('cpu_time')
                    if cpu_time is None:
                        cpu_time = 0.0
                    reporter.info(
                        f"Solver status={optimization_meta.get('status')} success={optimization_meta.get('success')} "
                        f"iterations={optimization_meta.get('iterations')} cpu_time={cpu_time:.3f}s"
                    )
                    if iteration_summary:
                        objective_info = iteration_summary.get("objective", {})
                        final_info = iteration_summary.get("final", {})
                        reporter.info(
                            f"Residuals final: inf_pr={final_info.get('inf_pr', float('nan')):.3e} "
                            f"inf_du={final_info.get('inf_du', float('nan')):.3e} "
                            f"mu={final_info.get('mu', float('nan')):.3e}"
                        )
                        reporter.info(
                            f"Iterations total={iteration_summary.get('iteration_count')} "
                            f"restoration={iteration_summary.get('restoration_steps')} "
                            f"objective_start={objective_info.get('start', float('nan')):.3e} "
                            f"objective_final={final_info.get('objective', float('nan')):.3e}"
                        )
                        recent = iteration_summary.get("recent_iterations")
                        if recent and reporter.show_debug:
                            with reporter.section("Recent IPOPT iterations", level="DEBUG"):
                                for entry in recent:
                                    reporter.debug(
                                        f"k={entry['k']:>4} f={entry['objective']:.3e} "
                                        f"inf_pr={entry['inf_pr']:.3e} inf_du={entry['inf_du']:.3e} "
                                        f"mu={entry['mu']:.3e} step={entry['step']}"
                                    )
                    if pressure_meta:
                        pr = pressure_meta.get("pressure_ratio", {})
                        work = pressure_meta.get("workload", {})
                        reporter.info(
                            f"Pressure ratio peak={pr.get('pi_peak', float('nan')):.3f} "
                            f"mean={pr.get('pi_mean', float('nan')):.3f} "
                            f"target_peak={pr.get('pi_ref_peak', float('nan')):.3f}"
                        )
                        reporter.info(
                            f"Workload target={work.get('target_work_j', float('nan')):.3f} "
                            f"mean={work.get('cycle_work_mean_j', float('nan')):.3f} "
                            f"error={work.get('cycle_work_error_j', float('nan')):.3f}"
                        )
                        if reporter.show_debug:
                            cases = pr.get("cases") or []
                            if cases:
                                with reporter.section("Pressure ratio cases", level="DEBUG"):
                                    for case in cases[:5]:
                                        reporter.debug(
                                            f"fuel_mult={case.get('fuel_multiplier')} "
                                            f"load_delta={case.get('delta_p_load_kpa', 'n/a')} "
                                            f"pi_peak={case.get('pi_peak', 'n/a')} "
                                            f"imep={case.get('imep', 'n/a')}"
                                        )
                    if ca_markers:
                        reporter.info(
                            "CA markers: " + ", ".join(f"{k}={v:.1f}°" for k, v in ca_markers.items())
                        )

            return result

        except Exception as e:
            reporter.exception("Free-piston IPOPT optimization failed", e)
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
        self._last_geometry = {
            "area_mm2": float(area_m2 * 1e6),
            "Vc_mm3": float(clearance_volume_m3 * 1e9),
            "stroke_mm": float(cam_constraints.stroke),
            "bore_m": float(bore),
        }
        if self._constant_load_value is None:
            try:
                self._constant_load_value = float(self._config.get("constant_load_value", 0.0))
            except Exception:
                self._constant_load_value = None
        if self._workload_target_j is None:
            base_load_val = float(self._constant_load_value or 0.0)
            self._workload_target_j = base_load_val * stroke_m

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
        # Handle both None and empty arrays (IPOPT may return empty array on failure)
        if x_opt is None or (isinstance(x_opt, np.ndarray) and x_opt.size == 0):
            log.warning("No solution variables found in free-piston result")
            return OptimizationResult(
                status=status,
                objective_value=f_opt,
                iterations=iterations,
                solve_time=cpu_time,
                error_message=message if not success else None,
                solution={},
            )

        # Extract motion profile from collocation solution (combustion-driven time)
        grid_obj = solution.meta.get("grid", {})
        meta_nlp = solution.meta.get("meta", {})
        variable_groups = (meta_nlp.get("variable_groups") or {}) if isinstance(meta_nlp, dict) else {}

        # Get collocation grid info
        K = int(meta_nlp.get("K", 20)) if isinstance(meta_nlp, dict) else 20
        C = int(meta_nlp.get("C", 1)) if isinstance(meta_nlp, dict) else 1

        # Feature flag to allow fallback (default: use free-piston motion)
        use_freepiston_motion = os.environ.get("PHASE1_USE_FREEPISTON_MOTION", "1").lower() not in _FALSEY

        try:
            if use_freepiston_motion and isinstance(variable_groups, dict) and len(variable_groups) > 0:
                x_opt_vec = np.asarray(x_opt, dtype=float).flatten()

                # Build combustion-driven time grid
                comb_meta = (meta_nlp.get("combustion_model") or {}) if isinstance(meta_nlp, dict) else {}
                omega_deg_per_s = float(comb_meta.get("omega_deg_per_s", 360.0 / max(float(self._last_cycle_time or cycle_time), 1e-9)))
                T_cycle = 360.0 / max(omega_deg_per_s, 1e-9)  # seconds
                # Collocation nodes from grid (best-effort)
                try:
                    nodes = getattr(grid_obj, "nodes", None)
                    if nodes is None:
                        # Fallback: uniform within element
                        nodes = np.linspace(1.0 / (C + 1), 1.0, C, endpoint=True)
                    else:
                        nodes = np.asarray(nodes, dtype=float)
                except Exception:
                    nodes = np.linspace(1.0 / (C + 1), 1.0, C, endpoint=True)
                dt_elem = T_cycle / max(K, 1)

                # Variable groups give indices into x; positions come in pairs [xL, xR], velocities [vL, vR]
                pos_idx: list[int] = list(variable_groups.get("positions", []))
                vel_idx: list[int] = list(variable_groups.get("velocities", []))
                den_idx: list[int] = list(variable_groups.get("densities", []))
                tmp_idx: list[int] = list(variable_groups.get("temperatures", []))

                # Expect sizes: 2 (initial) + K*C*2 for positions/velocities
                def _extract_pairs(idx_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
                    vals = x_opt_vec[idx_list] if idx_list else np.array([])
                    if vals.size < 2:
                        return np.array([]), np.array([])
                    # reshape as pairs
                    reshaped = vals.reshape(-1, 2)
                    return reshaped[:, 0], reshaped[:, 1]

                xL_all, xR_all = _extract_pairs(pos_idx)
                vL_all, vR_all = _extract_pairs(vel_idx)

                # Split initial and collocation samples
                has_initial = xL_all.size >= (1 + K * C + 0)  # initial + K*C
                if has_initial:
                    xL0, xR0 = xL_all[0], xR_all[0]
                    vL0, vR0 = vL_all[0], vR_all[0] if vL_all.size > 0 else (0.0, 0.0)
                    xL_colloc = xL_all[1:]
                    xR_colloc = xR_all[1:]
                    vL_colloc = vL_all[1:] if vL_all.size > 1 else np.zeros_like(xL_colloc)
                    vR_colloc = vR_all[1:] if vR_all.size > 1 else np.zeros_like(xR_colloc)
                else:
                    # Best-effort: treat first as initial if present
                    xL0 = xL_all[0] if xL_all.size > 0 else 0.0
                    xR0 = xR_all[0] if xR_all.size > 0 else 0.0
                    vL0 = vL_all[0] if vL_all.size > 0 else 0.0
                    vR0 = vR_all[0] if vR_all.size > 0 else 0.0
                    xL_colloc = xL_all
                    xR_colloc = xR_all
                    vL_colloc = vL_all
                    vR_colloc = vR_all

                # Assemble time grid aligned with initial + collocation sequence
                t_list: list[float] = [0.0]
                for k in range(K):
                    for c in range(C):
                        t_list.append((k + float(nodes[c])) * dt_elem)
                t = np.asarray(t_list, dtype=float)[: 1 + K * C]

                # Assemble follower motion (average of opposed pistons), convert to mm
                x_follow_m = np.concatenate([[0.5 * (xL0 + xR0)], 0.5 * (xL_colloc + xR_colloc)])
                v_follow_mps = np.concatenate([[0.5 * (vL0 + vR0)], 0.5 * (vL_colloc + vR_colloc)])
                # Ensure strictly increasing time for gradient stability
                t_eps = np.maximum(np.gradient(t), 1e-9)
                a_follow_mps2 = np.gradient(v_follow_mps, t, edge_order=2) if t.size >= 3 else np.zeros_like(v_follow_mps)

                position_mm = x_follow_m * 1000.0
                velocity_mm_per_s = v_follow_mps * 1000.0
                acceleration_mm_per_s2 = a_follow_mps2 * 1000.0

                # Cylinder pressure from densities and temperatures at collocation (ideal gas)
                R_gas = 287.0
                # Initial + collocation for rho,T if available; else length-match to time with zeros
                rho_vals = x_opt_vec[den_idx] if den_idx else np.array([])
                T_vals = x_opt_vec[tmp_idx] if tmp_idx else np.array([])
                # Expect 1 + K*C values each
                if rho_vals.size >= 1 and T_vals.size >= 1:
                    rho_series = np.asarray(rho_vals, dtype=float)[: 1 + K * C]
                    T_series = np.asarray(T_vals, dtype=float)[: 1 + K * C]
                    pressure_pa = rho_series * R_gas * T_series
                else:
                    pressure_pa = np.zeros_like(t)

                # Pseudo crank-angle from combustion-driven time
                theta_rad = 2.0 * np.pi * (t / max(T_cycle, 1e-9))
                theta_deg = np.degrees(theta_rad)

                # Build mappings for pressure
                p_vs_t = {"t_s": t.tolist(), "p_pa": pressure_pa.tolist()}
                p_vs_x = {"x_mm": position_mm.tolist(), "p_pa": pressure_pa.tolist()}
                p_vs_theta = {"theta_deg": theta_deg.tolist(), "p_pa": pressure_pa.tolist()}

                # Backward-compatibility: provide uniform-theta resample for GUI
                theta_uniform = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
                # Wrap theta for interpolation monotonicity
                theta_src = (theta_rad % (2.0 * np.pi))
                order = np.argsort(theta_src)
                theta_src_sorted = theta_src[order]
                pos_sorted = position_mm[order]
                vel_sorted = velocity_mm_per_s[order]
                acc_sorted = acceleration_mm_per_s2[order]
                # Interpolate periodic signals
                def periodic_interp(src, y, dst):
                    src_ext = np.concatenate([src, src + 2.0 * np.pi])
                    y_ext = np.concatenate([y, y])
                    return np.interp(dst, src_ext, y_ext)
                position_u = periodic_interp(theta_src_sorted, pos_sorted, theta_uniform)
                velocity_u = periodic_interp(theta_src_sorted, vel_sorted, theta_uniform)
                acceleration_u = periodic_interp(theta_src_sorted, acc_sorted, theta_uniform)

                solution_dict = {
                    # Authoritative combustion-driven time series
                    "time_s": t,
                    "theta_rad": theta_rad,
                    "theta_deg": theta_deg,
                    "position_mm": position_mm,
                    "velocity_mm_per_s": velocity_mm_per_s,
                    "acceleration_mm_per_s2": acceleration_mm_per_s2,
                    "pressure_pa": pressure_pa,
                    # Back-compat fields expected by downstream code (uniform theta grid)
                    "cam_angle": theta_uniform,
                    "position": position_u,
                    "velocity": velocity_u,
                    "acceleration": acceleration_u,
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

            pressure_invariance_meta: dict[str, Any] | None = None
            try:
                geom_info = self._last_geometry or {}
                area_mm2 = float(geom_info.get("area_mm2", 0.0))
                vc_mm3 = float(geom_info.get("Vc_mm3", 0.0))
                if area_mm2 <= 0.0 or vc_mm3 <= 0.0:
                    raise ValueError("Missing geometry data for pressure ratio computation")
                geom_adapter = CycleGeometry(area_mm2=area_mm2, Vc_mm3=vc_mm3)
                thermo_adapter = CycleThermo(gamma_bounce=1.25, p_atm_kpa=101.325)
                adapter = SimpleCycleAdapter(
                    wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
                    alpha_fuel_to_base=float(self._bounce_alpha),
                    beta_base=float(self._bounce_beta),
                )
                # Derive v vs theta on the uniform grid we exposed
                cam_angle = solution_dict["cam_angle"]
                position_u = solution_dict["position"]
                v_mm_per_theta = np.gradient(position_u, cam_angle)
                cycle_time_ref = float(self._last_cycle_time or cycle_time)
                combustion_inputs = None
                afr_last = self._last_combustion_inputs.get("afr")
                fuel_mass_last = self._last_combustion_inputs.get("fuel_mass")
                if afr_last is not None and fuel_mass_last is not None:
                    combustion_inputs = {
                        "afr": float(afr_last),
                        "fuel_mass": float(fuel_mass_last),
                        "cycle_time_s": cycle_time_ref,
                        "initial_temperature_K": float(
                            self._last_combustion_inputs.get("initial_temperature_K", 900.0)
                        ),
                        "initial_pressure_Pa": float(
                            self._last_combustion_inputs.get("initial_pressure_Pa", 101325.0)
                        ),
                    }
                    ignition_time_last = self._last_combustion_inputs.get("ignition_time_s")
                    ignition_deg_last = self._last_combustion_inputs.get("ignition_theta_deg")
                    if ignition_time_last is not None:
                        combustion_inputs["ignition_time_s"] = float(ignition_time_last)
                    if ignition_deg_last is not None:
                        combustion_inputs["ignition_theta_deg"] = float(ignition_deg_last)

                # Compute workload-aligned p_load_kpa before evaluation
                stroke_m = float(self._last_geometry.get("stroke_mm", cam_constraints.stroke)) / 1000.0
                work_target_j = float(self._workload_target_j or 0.0)
                area_m2 = float(area_mm2) * 1e-6
                p_load_kpa_base = 0.0
                if work_target_j and area_m2 > 0.0:
                    p_load_pa_base = work_target_j / max(area_m2 * stroke_m, 1e-12)
                    p_load_kpa_base = p_load_pa_base / 1000.0
                
                # Get fuel/load sweeps from config (default to single case)
                fuel_sweep = self._config.get("fuel_sweep", [1.0])
                if not fuel_sweep:
                    fuel_sweep = [1.0]
                fuel_sweep = [float(x) for x in fuel_sweep]
                
                load_sweep = self._config.get("load_sweep", [0.0])
                if not load_sweep:
                    load_sweep = [0.0]
                load_sweep = [float(x) for x in load_sweep]
                
                p_env_eval = float(thermo_adapter.p_atm_kpa)
                p_cc_eval = float(self._config.get("crankcase_pressure_kpa", 0.0))
                
                # Evaluate all fuel/load combinations
                ratio_cases = []
                work_cases = []
                pi_traces = []
                base_eval_out = None
                
                for fm in fuel_sweep:
                    for c in load_sweep:
                        # Build combustion inputs with fuel multiplier
                        comb_inputs_case = None
                        if combustion_inputs is not None:
                            comb_inputs_case = dict(combustion_inputs)
                            if "fuel_mass" in comb_inputs_case:
                                comb_inputs_case["fuel_mass"] = float(comb_inputs_case["fuel_mass"]) * float(fm)
                        
                        eval_out_case = adapter.evaluate(
                            cam_angle,
                            position_u,
                            v_mm_per_theta,
                            float(fm),
                            float(c),
                            geom_adapter,
                            thermo_adapter,
                            combustion=comb_inputs_case,
                            cycle_time_s=cycle_time_ref,
                        )
                        
                        # Use base case for reference
                        if abs(float(fm) - 1.0) < 1e-6 and float(c) == 0.0:
                            base_eval_out = eval_out_case
                        
                        # Compute workload-aligned p_load_kpa for this case
                        cycle_work_case = float(eval_out_case.get("cycle_work_j", 0.0))
                        p_load_kpa_case = p_load_kpa_base
                        if cycle_work_case > 0.0 and area_m2 > 0.0:
                            # Use actual cycle work to compute case-specific load pressure
                            p_load_pa_case = cycle_work_case / max(area_m2 * stroke_m, 1e-12)
                            p_load_kpa_case = p_load_pa_case / 1000.0
                        elif work_target_j and area_m2 > 0.0:
                            # Scale by fuel multiplier as proxy
                            effective_work = work_target_j * float(fm)
                            p_load_pa_case = effective_work / max(area_m2 * stroke_m, 1e-12)
                            p_load_kpa_case = p_load_pa_case / 1000.0
                        
                        p_cyl_case = np.asarray(eval_out_case.get("p_cyl") or eval_out_case.get("p_comb"), dtype=float)
                        p_bounce_case = np.asarray(eval_out_case.get("p_bounce"), dtype=float)
                        denom_case = p_load_kpa_case + p_cc_eval + p_env_eval + p_bounce_case
                        pi_trace_case = p_cyl_case / np.maximum(denom_case, 1e-6)
                        pi_traces.append(pi_trace_case)
                        
                        ca_markers_case = eval_out_case.get("ca_markers") or {}
                        ratio_case = {
                            "fuel_multiplier": float(fm),
                            "load": float(c),
                            "pi_mean": float(np.mean(pi_trace_case)),
                            "pi_peak": float(np.max(pi_trace_case)),
                            "pi_min": float(np.min(pi_trace_case)),
                        }
                        # Add CA markers if available
                        if ca_markers_case:
                            for key, value in ca_markers_case.items():
                                if value is not None:
                                    ratio_case[f"ca_{key.lower()}"] = float(value)
                        ratio_cases.append(ratio_case)
                        
                        cycle_work_case_val = float(eval_out_case.get("cycle_work_j", 0.0))
                        work_case = {
                            "fuel_multiplier": float(fm),
                            "load": float(c),
                            "cycle_work_j": cycle_work_case_val,
                            "work_error_j": cycle_work_case_val - work_target_j,
                        }
                        # Add CA markers to work cases if available
                        if ca_markers_case:
                            for key, value in ca_markers_case.items():
                                if value is not None:
                                    work_case[f"ca_{key.lower()}"] = float(value)
                        work_cases.append(work_case)
                
                # Use base case for reference metrics
                eval_out = base_eval_out if base_eval_out is not None else eval_out_case
                p_cyl_eval = np.asarray(eval_out.get("p_cyl") or eval_out.get("p_comb"), dtype=float)
                p_bounce_eval = np.asarray(eval_out.get("p_bounce"), dtype=float)
                # Use workload-aligned p_load_kpa for reference denominator
                denom_eval = p_load_kpa_base + p_cc_eval + p_env_eval + p_bounce_eval
                pi_trace_eval = p_cyl_eval / np.maximum(denom_eval, 1e-6)
                
                # Aggregate statistics from all cases
                if pi_traces:
                    merged_pi = np.concatenate(pi_traces)
                    pressure_ratio_meta = {
                        "pi_mean": float(np.mean(merged_pi)),
                        "pi_peak": float(np.max(merged_pi)),
                        "pi_min": float(np.min(merged_pi)),
                        "pi_std": float(np.std(merged_pi)),
                        "pi_ref_mean": float(np.mean(pi_trace_eval)),
                        "pi_ref_peak": float(np.max(pi_trace_eval)),
                        "pi_ref_min": float(np.min(pi_trace_eval)),
                        "pi_ref_std": float(np.std(pi_trace_eval)),
                        "cases": ratio_cases,
                    }
                else:
                    pressure_ratio_meta = {
                        "pi_mean": float(np.mean(pi_trace_eval)),
                        "pi_peak": float(np.max(pi_trace_eval)),
                        "pi_min": float(np.min(pi_trace_eval)),
                        "pi_std": float(np.std(pi_trace_eval)),
                        "pi_ref_mean": float(np.mean(pi_trace_eval)),
                        "pi_ref_peak": float(np.max(pi_trace_eval)),
                        "pi_ref_min": float(np.min(pi_trace_eval)),
                        "pi_ref_std": float(np.std(pi_trace_eval)),
                        "cases": [],
                    }
                
                # Aggregate work statistics
                cycle_work_values = [w["cycle_work_j"] for w in work_cases]
                work_mean = float(np.mean(cycle_work_values)) if cycle_work_values else 0.0
                work_stats_meta = {
                    "target_work_j": work_target_j,
                    "cycle_work_mean_j": work_mean,
                    "cycle_work_error_j": work_mean - work_target_j,
                    "cases": work_cases,
                }
                pressure_invariance_meta = {
                    "loss_p_mean": 0.0,
                    "imep_avg": float(eval_out.get("imep", 0.0)),
                    "fuel_sweep": fuel_sweep,
                    "load_sweep": load_sweep,
                    "pressure_ratio": pressure_ratio_meta,
                    "pressure_ratio_target_mean": float(np.mean(pi_trace_eval)),
                    "pi_reference": pi_trace_eval.tolist(),
                    "theta_deg": np.degrees(cam_angle).tolist(),
                    "denominator_base": {
                        "p_load_kpa": p_load_kpa_base,  # Use workload-aligned base
                        "p_env_kpa": p_env_eval,
                        "p_cc_kpa": p_cc_eval,
                    },
                    "workload": work_stats_meta,
                    "work_target_j": work_target_j,
                }
                ca_comb_eval = eval_out.get("ca_markers")
                if ca_comb_eval:
                    ca_markers.update(ca_comb_eval)
            except Exception as ratio_exc:  # pragma: no cover - diagnostics path
                log.debug(f"Free-piston ratio diagnostics failed: {ratio_exc}")
                pressure_invariance_meta = None

            # Build metadata
            metadata = {
                "free_piston_optimization": True,
                "combustion_model": "integrated",
                "ca_markers": ca_markers,
                "pressure": {
                    "vs_time": p_vs_t,
                    "vs_position": p_vs_x,
                    "vs_theta": p_vs_theta,
                },
                "grid_info": {
                    "K": K,
                    "C": C,
                    "n_points": n_points,
                },
                "iterations": iterations,
                "cpu_time": cpu_time,
            }
            if pressure_invariance_meta is not None:
                metadata["pressure_invariance"] = pressure_invariance_meta

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
