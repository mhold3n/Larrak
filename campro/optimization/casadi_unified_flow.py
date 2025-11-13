"""
CasADi unified optimization flow manager.

This module orchestrates Phase 1 optimization using CasADi Opti stack
with warm-starting capabilities and integration with the existing
unified optimization framework.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from campro.logging import get_logger
from campro.optimization.base import OptimizationResult, OptimizationStatus
from campro.optimization.casadi_motion_optimizer import (
    CasADiMotionOptimizer,
    CasADiMotionProblem,
)
from campro.optimization.initial_guess import InitialGuessBuilder
from campro.physics.thermal_efficiency_simple import (
    SimplifiedThermalModel,
    ThermalEfficiencyConfig,
)

log = get_logger(__name__)


@dataclass
class CasADiOptimizationSettings:
    """Settings for CasADi unified optimization flow."""

    # Warm-starting settings
    enable_warmstart: bool = True
    max_history: int = 50
    tolerance: float = 0.1
    storage_path: str | None = None

    # Collocation settings
    poly_order: int = 3
    collocation_method: str = "legendre"

    # Adaptive resolution settings
    coarse_resolution_segments: tuple[int, ...] = (40, 80, 160)
    resolution_ladder: tuple[int, ...] | None = None
    min_segments: int = 20
    max_angle_resolution_segments: int = 4096  # Maximum segments for target angle resolution
    target_angle_resolution_deg: float = 0.1  # Target final angular step
    refinement_improvement_threshold: float = 1e-4
    retry_failed_level: bool = True

    # Thermal efficiency settings
    efficiency_target: float = 0.55
    heat_transfer_coeff: float = 0.1
    friction_coeff: float = 0.01

    # Solver settings
    solver_options: dict[str, Any] = None

    def __post_init__(self):
        if self.solver_options is None:
            self.solver_options = {
                "ipopt.linear_solver": "ma57",
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                "ipopt.warm_start_init_point": "yes",
            }

        # Handle case where coarse_resolution_segments is an integer or not an iterable
        if isinstance(self.coarse_resolution_segments, int):
            self.coarse_resolution_segments = (max(self.min_segments, int(self.coarse_resolution_segments)),)
        elif not isinstance(self.coarse_resolution_segments, Iterable) or not self.coarse_resolution_segments:
            self.coarse_resolution_segments = (self.min_segments,)
        else:
            self.coarse_resolution_segments = tuple(
                max(self.min_segments, int(seg)) for seg in self.coarse_resolution_segments
            )

        if self.resolution_ladder:
            # Handle case where resolution_ladder is an integer instead of an iterable
            if isinstance(self.resolution_ladder, int):
                self.resolution_ladder = (max(self.min_segments, int(self.resolution_ladder)),)
            elif isinstance(self.resolution_ladder, Iterable):
                self.resolution_ladder = tuple(
                    max(self.min_segments, int(seg)) for seg in self.resolution_ladder
                )
            else:
                self.resolution_ladder = None


class CasADiUnifiedFlow:
    """
    Unified optimization flow manager for CasADi-based Phase 1 optimization.

    This class orchestrates the complete optimization flow including:
    - Problem setup and validation
    - Warm-starting from previous solutions
    - CasADi Opti stack optimization
    - Thermal efficiency evaluation
    - Solution storage for future warm-starts
    """

    def __init__(self, settings: CasADiOptimizationSettings | None = None):
        """
        Initialize CasADi unified flow.

        Parameters
        ----------
        settings : Optional[CasADiOptimizationSettings]
            Optimization settings
        """
        self.settings = settings or CasADiOptimizationSettings()

        # Initialize components using the coarsest available level
        initial_segments = self._initial_segments()
        self.motion_optimizer = CasADiMotionOptimizer(
            n_segments=initial_segments,
            poly_order=self.settings.poly_order,
            collocation_method=self.settings.collocation_method,
            solver_options=self.settings.solver_options,
        )
        self.initial_guess_builder = InitialGuessBuilder(initial_segments)
        self._current_segments = initial_segments

        self.thermal_model = SimplifiedThermalModel(
            config=ThermalEfficiencyConfig(
                efficiency_target=self.settings.efficiency_target,
                heat_transfer_coeff=self.settings.heat_transfer_coeff,
                friction_coeff=self.settings.friction_coeff,
            ),
        )

        log.info(
            f"Initialized CasADiUnifiedFlow: "
            f"seed_polish={self.settings.enable_warmstart}, "
            f"initial_segments={initial_segments}, "
            f"efficiency_target={self.settings.efficiency_target}",
        )

    def optimize_phase1(
        self, constraints: dict[str, Any], targets: dict[str, Any], **kwargs,
    ) -> OptimizationResult:
        """
        Optimize Phase 1 motion law with thermal efficiency.

        Parameters
        ----------
        constraints : Dict[str, Any]
            Optimization constraints
        targets : Dict[str, Any]
            Optimization targets
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        start_time = time.time()

        try:
            problem = self._create_problem_from_constraints(constraints, targets)
            resolution_schedule = self._build_resolution_schedule(problem, constraints)
            
            # Explicit logging for ladder resolution and warm-start visibility
            schedule_str = " -> ".join(str(seg) for seg in resolution_schedule)
            warmstart_status = "enabled" if self.settings.enable_warmstart else "disabled"
            
            # Use warning level to ensure visibility (INFO may be filtered)
            log.warning("=" * 70)
            log.warning("CasADi ladder-resolution: %s", schedule_str)
            log.warning("  (automatic refinement to target angle resolution)")
            log.warning("CasADi-deterministic-warmstart: %s", warmstart_status)
            log.warning("  (interpolation between levels, seed polishing)")
            log.warning("=" * 70)

            ladder_history: list[dict[str, Any]] = []
            previous_success: OptimizationResult | None = None
            last_attempt: OptimizationResult | None = None

            # Extract universal_theta_rad if provided for final level
            universal_theta_rad = constraints.get("universal_theta_rad")
            target_theta_rad = None
            if universal_theta_rad is not None and len(resolution_schedule) > 0:
                # Check if final level matches universal grid size
                final_segments = resolution_schedule[-1]
                expected_points = final_segments + 1  # CasADi uses n_segments + 1 points
                if len(universal_theta_rad) == expected_points:
                    target_theta_rad = universal_theta_rad
                    log.info(
                        f"Final ladder level matches universal grid: {expected_points} points, "
                        "will use universal grid theta directly"
                    )
            
            for level_index, segments in enumerate(resolution_schedule):
                log.warning(
                    "[LADDER] Level %d/%d: %d segments",
                    level_index + 1,
                    len(resolution_schedule),
                    segments,
                )
                # Only use target_theta_rad for the final level
                use_target_theta = (
                    target_theta_rad is not None
                    and level_index == len(resolution_schedule) - 1
                )
                result = self._run_resolution_level(
                    problem,
                    segments,
                    previous_success,
                    allow_interpolation=True,
                    target_theta_rad=target_theta_rad if use_target_theta else None,
                )
                level_record = self._build_level_record(
                    level_index,
                    len(resolution_schedule),
                    segments,
                    result,
                    retried=False,
                )
                ladder_history.append(level_record)
                last_attempt = result

                if result.successful:
                    previous_success = result
                    continue

                if not self.settings.retry_failed_level:
                    log.warning(
                        "Resolution level %s failed and retries are disabled. "
                        "Returning last successful coarse result.",
                        segments,
                    )
                    break

                log.warning(
                    "Resolution level %s failed; retrying with polished deterministic seed",
                    segments,
                )
                # Only use target_theta_rad for the final level retry too
                use_target_theta_retry = (
                    target_theta_rad is not None
                    and level_index == len(resolution_schedule) - 1
                )
                retry_result = self._run_resolution_level(
                    problem,
                    segments,
                    previous_success,
                    allow_interpolation=False,
                    target_theta_rad=target_theta_rad if use_target_theta_retry else None,
                )
                ladder_history.append(
                    self._build_level_record(
                        level_index,
                        len(resolution_schedule),
                        segments,
                        retry_result,
                        retried=True,
                    ),
                )
                last_attempt = retry_result

                if retry_result.successful:
                    previous_success = retry_result
                    continue

                log.error(
                    "Resolution level %s failed twice; escalating to next resolution level",
                    segments,
                )
                continue

            final_result = previous_success or last_attempt
            if final_result is None:
                raise RuntimeError("Adaptive resolution failed before producing any result")

            if final_result.successful:
                self._attach_thermal_metrics(final_result)

            total_wall = time.time() - start_time
            final_result.metadata.setdefault("resolution_levels", ladder_history)
            final_result.metadata["resolution_levels"] = ladder_history
            final_result.metadata["adaptive_resolution_schedule"] = resolution_schedule
            final_result.metadata["adaptive_resolution_total_time"] = total_wall
            final_result.metadata["resolution_strategy"] = "adaptive_angle"
            final_result.metadata["finest_success_segments"] = (
                final_result.metadata.get("n_segments") or self._current_segments
            )

            return final_result

        except Exception as e:
            log.error(f"Phase 1 optimization failed: {e}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solve_time=time.time() - start_time,
                solution={},
                error_message=str(e),
                metadata={
                    "error": str(e),
                    "resolution_strategy": "adaptive_angle",
                },
            )

    def _build_level_record(
        self,
        level_index: int,
        total_levels: int,
        segments: int,
        result: OptimizationResult,
        *,
        retried: bool,
    ) -> dict[str, Any]:
        return {
            "level": level_index,
            "of": total_levels,
            "segments": segments,
            "status": result.status.value if isinstance(result.status, OptimizationStatus) else str(result.status),
            "objective": result.objective_value,
            "solve_time": result.solve_time,
            "retry": retried,
        }

    def _run_resolution_level(
        self,
        problem: CasADiMotionProblem,
        n_segments: int,
        previous_result: OptimizationResult | None,
        *,
        allow_interpolation: bool,
        target_theta_rad: np.ndarray | None = None,
    ) -> OptimizationResult:
        self._ensure_components(n_segments)

        seed: dict[str, np.ndarray] | None = None
        if (
            allow_interpolation
            and previous_result is not None
            and previous_result.successful
        ):
            prev_segments = previous_result.metadata.get("n_segments", 0)
            log.warning(
                "[WARM-START] Interpolating from %d to %d segments",
                prev_segments,
                n_segments,
            )
            seed = self._interpolate_solution_to_seed(
                previous_result,
                n_segments,
                problem.duration_angle_deg,
            )
            if seed is not None:
                log.warning("[WARM-START] Interpolation successful")
            else:
                log.warning("[WARM-START] Interpolation returned None, using fresh seed")

        if seed is None:
            log.warning("[WARM-START] Building fresh deterministic seed for %d segments", n_segments)
            seed = self.initial_guess_builder.build_seed(problem)

        if self.settings.enable_warmstart:
            log.warning("[WARM-START] Polishing seed with deterministic smoothing")
            seed = self.initial_guess_builder.polish_seed(problem, seed)
        else:
            log.warning("[WARM-START] Disabled, using unpolished seed")

        result = self.motion_optimizer.solve(problem, seed, target_theta_rad=target_theta_rad)
        result.metadata.setdefault("n_segments", n_segments)
        result.metadata["n_segments"] = n_segments
        return result

    def _build_resolution_schedule(
        self,
        problem: CasADiMotionProblem,
        constraints: dict[str, Any],
    ) -> list[int]:
        manual_schedule = constraints.get("resolution_ladder") or self.settings.resolution_ladder
        if manual_schedule:
            # Handle case where manual_schedule is an integer instead of an iterable
            if isinstance(manual_schedule, int):
                schedule_source = (max(self.settings.min_segments, int(manual_schedule)),)
            elif isinstance(manual_schedule, Iterable):
                schedule_source = tuple(max(self.settings.min_segments, int(seg)) for seg in manual_schedule)
            else:
                # Fallback to default
                schedule_source = self.settings.coarse_resolution_segments
        else:
            schedule_source = self.settings.coarse_resolution_segments

        # Check if universal_n_points is provided - required for grid alignment
        # CasADi uses n_segments + 1 points, so final_segments = universal_n_points - 1
        universal_n_points = constraints.get("universal_n_points")
        if universal_n_points is None:
            raise ValueError(
                "universal_n_points is required for CasADi optimization. "
                "The CasADi ladder resolution schedule uses legacy universal grid logic. "
                "Update UnifiedOptimizationFramework to pass universal_n_points in constraints before optimization."
            )
        universal_n_points = int(universal_n_points)
        if universal_n_points <= 0:
            raise ValueError(
                f"universal_n_points must be positive, got {universal_n_points}. "
                "The CasADi ladder resolution schedule uses legacy universal grid logic. "
                "Update UnifiedOptimizationFramework to pass valid universal_n_points in constraints before optimization."
            )
        # Use universal grid as final mesh
        final_segments = max(self.settings.min_segments, universal_n_points - 1)
        log.info(
            f"Using universal grid as final mesh: universal_n_points={universal_n_points}, "
            f"final_segments={final_segments}"
        )

        schedule: list[int] = []
        for seg in schedule_source:
            seg = max(self.settings.min_segments, int(seg))
            if schedule and seg <= schedule[-1]:
                continue
            if seg >= final_segments:
                break
            schedule.append(seg)

        if not schedule or schedule[-1] != final_segments:
            schedule.append(final_segments)

        return schedule

    def _compute_angle_resolution_segments(self, total_angle_deg: float, step_deg: float) -> int:
        step = max(1e-9, float(step_deg))
        raw_segments = max(1, math.ceil(total_angle_deg / step))
        bounded = max(self.settings.min_segments, raw_segments)
        max_segments = getattr(self.settings, "max_angle_resolution_segments", None)
        if max_segments is not None:
            return min(int(max_segments), bounded)
        return bounded

    def _interpolate_solution_to_seed(
        self,
        result: OptimizationResult,
        target_segments: int,
        total_angle_deg: float,
    ) -> dict[str, np.ndarray] | None:
        if not result.solution:
            return None

        position = result.solution.get("position")
        velocity = result.solution.get("velocity")
        acceleration = result.solution.get("acceleration")

        if position is None or velocity is None or acceleration is None:
            return None

        prev_segments = len(position) - 1
        if prev_segments <= 0:
            return None

        if target_segments == prev_segments:
            jerk = result.solution.get("jerk")
            if jerk is None or len(jerk) != target_segments:
                jerk = self._derive_jerk_from_acceleration(acceleration, total_angle_deg)
            return {
                "x": np.array(position, dtype=float),
                "v": np.array(velocity, dtype=float),
                "a": np.array(acceleration, dtype=float),
                "j": np.array(jerk, dtype=float),
            }

        prev_grid = np.linspace(0.0, total_angle_deg, prev_segments + 1)
        new_grid = np.linspace(0.0, total_angle_deg, target_segments + 1)

        x_new = np.interp(new_grid, prev_grid, position)
        v_new = np.interp(new_grid, prev_grid, velocity)
        a_new = np.interp(new_grid, prev_grid, acceleration)

        jerk = result.solution.get("jerk")
        if jerk is not None and len(jerk) == prev_segments:
            prev_dt = total_angle_deg / prev_segments
            prev_mid = np.linspace(prev_dt / 2, total_angle_deg - prev_dt / 2, prev_segments)
            target_dt = total_angle_deg / target_segments
            target_mid = np.linspace(target_dt / 2, total_angle_deg - target_dt / 2, target_segments)
            j_new = np.interp(target_mid, prev_mid, jerk)
        else:
            j_new = self._derive_jerk_from_acceleration(a_new, total_angle_deg)

        return {
            "x": x_new,
            "v": v_new,
            "a": a_new,
            "j": j_new,
        }

    def _derive_jerk_from_acceleration(self, acceleration: np.ndarray, total_angle_deg: float) -> np.ndarray:
        if len(acceleration) < 2:
            return np.zeros(max(1, len(acceleration)))
        n_segments = max(1, len(acceleration) - 1)
        dt = total_angle_deg / max(n_segments, 1)
        jerk_nodes = np.gradient(acceleration, dt)
        if len(jerk_nodes) < 2:
            return np.zeros(n_segments)
        return 0.5 * (jerk_nodes[:-1] + jerk_nodes[1:])

    def _attach_thermal_metrics(self, result: OptimizationResult) -> None:
        try:
            efficiency_metrics = self.thermal_model.evaluate_efficiency(
                result.variables["position"],
                result.variables["velocity"],
                result.variables["acceleration"],
            )
            result.metadata.update(efficiency_metrics)
            log.info(
                "Phase 1 optimization completed: efficiency=%.3f target=%.3f",
                efficiency_metrics["total_efficiency"],
                self.settings.efficiency_target,
            )
        except Exception as err:
            log.warning("Thermal efficiency evaluation failed: %s", err)

    def _ensure_components(self, n_segments: int) -> None:
        n_segments = max(self.settings.min_segments, int(n_segments))
        if (
            getattr(self, "_current_segments", None) == n_segments
            and self.motion_optimizer is not None
            and self.initial_guess_builder is not None
        ):
            return

        if self.motion_optimizer is None:
            self.motion_optimizer = CasADiMotionOptimizer(
                n_segments=n_segments,
                poly_order=self.settings.poly_order,
                collocation_method=self.settings.collocation_method,
                solver_options=self.settings.solver_options,
            )
        else:
            self.motion_optimizer.configure(
                n_segments=n_segments,
                solver_options=self.settings.solver_options,
            )

        if self.initial_guess_builder is None:
            self.initial_guess_builder = InitialGuessBuilder(n_segments)
        else:
            self.initial_guess_builder.update_segments(n_segments)

        self._current_segments = n_segments

    def _initial_segments(self) -> int:
        ladder = self.settings.resolution_ladder or self.settings.coarse_resolution_segments
        if not ladder:
            return self.settings.min_segments
        # Handle case where ladder is an integer instead of an iterable
        if isinstance(ladder, int):
            return max(self.settings.min_segments, int(ladder))
        # Handle iterable (tuple, list, etc.)
        try:
            if isinstance(ladder, Iterable):
                first_item = next(iter(ladder))
                return max(self.settings.min_segments, int(first_item))
        except (StopIteration, TypeError, ValueError):
            pass
        # Fallback
        return self.settings.min_segments

    def _normalize_ladder(self, ladder: Iterable[int] | None) -> tuple[int, ...] | None:
        if not ladder:
            return None
        normalized: list[int] = []
        for seg in ladder:
            if seg is None:
                continue
            try:
                seg_value = int(seg)
            except (TypeError, ValueError):
                continue
            if seg_value <= 0:
                continue
            normalized.append(max(self.settings.min_segments, seg_value))
        return tuple(normalized) or None

    def _create_problem_from_constraints(
        self, constraints: dict[str, Any], targets: dict[str, Any],
    ) -> CasADiMotionProblem:
        """
        Create CasADi problem from constraints and targets.
        
        Raises
        ------
        ValueError
            If duration_angle_deg is missing or invalid.
        """
        # Require duration_angle_deg - no fallback
        duration_angle_deg = constraints.get("duration_angle_deg")
        if duration_angle_deg is None:
            raise ValueError(
                "duration_angle_deg is required for Phase 1 per-degree optimization. "
                "It must be provided in constraints dict. "
                "No fallback is allowed to prevent unit mixing."
            )
        duration_angle_deg = float(duration_angle_deg)
        if duration_angle_deg <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {duration_angle_deg}. "
                "Phase 1 optimization requires angle-based units, not time-based."
            )
        
        # Extract problem parameters
        # Default to None for max_velocity/max_acceleration/max_jerk (unbounded)
        stroke = constraints.get("stroke", 0.1)
        
        # Compression ratio limits based on clearance geometry
        # CR = V_max / V_min = (stroke + clearance) / clearance
        # Default clearance: 2mm (0.002m) for typical engines
        # Maximum CR is determined by geometry: (stroke + clearance) / clearance
        # Minimum CR is always 1.0 (at BDC) and is determined by geometry, not constrained
        default_clearance_m = 0.002
        max_cr_from_geometry = (stroke + default_clearance_m) / default_clearance_m
        min_cr_default = 1.0  # Minimum is always 1.0 at BDC (geometry-determined)
        max_cr_default = min(100.0, max_cr_from_geometry * 1.2)  # Up to 120% of max, cap at 100
        
        problem_params = {
            "stroke": stroke,
            "cycle_time": constraints.get("cycle_time", 0.0385),  # Derived from engine_speed_rpm and duration_angle_deg
            "duration_angle_deg": duration_angle_deg,
            "upstroke_percent": constraints.get("upstroke_percent", 50.0),
            "max_velocity": constraints.get("max_velocity"),  # None if not provided
            "max_acceleration": constraints.get("max_acceleration"),  # None if not provided
            "max_jerk": constraints.get("max_jerk"),  # None if not provided
            "compression_ratio_limits": constraints.get(
                "compression_ratio_limits", (min_cr_default, max_cr_default),
            ),
        }

        # Extract objective weights
        weights = targets.get("weights", {})
        if not weights:
            weights = {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            }

        # Create problem
        problem = CasADiMotionProblem(
            stroke=problem_params["stroke"],
            cycle_time=problem_params["cycle_time"],
            duration_angle_deg=problem_params["duration_angle_deg"],
            upstroke_percent=problem_params["upstroke_percent"],
            max_velocity=problem_params["max_velocity"],
            max_acceleration=problem_params["max_acceleration"],
            max_jerk=problem_params["max_jerk"],
            compression_ratio_limits=problem_params["compression_ratio_limits"],
            minimize_jerk=targets.get("minimize_jerk", True),
            maximize_thermal_efficiency=targets.get(
                "maximize_thermal_efficiency", True,
            ),
            weights=weights,
        )

        return problem

    def get_warmstart_stats(self) -> dict[str, Any]:
        """Get warm-start statistics."""
        return {
            "strategy": "deterministic_seed",
            "current_segments": self._current_segments,
            "planned_ladder": list(
                self.settings.resolution_ladder or self.settings.coarse_resolution_segments
            ),
            "target_angle_resolution_deg": self.settings.target_angle_resolution_deg,
        }

    def clear_warmstart_history(self) -> None:
        """Clear warm-start history."""
        log.info("Deterministic seed has no history to clear")

    def update_settings(self, **kwargs) -> None:
        """Update optimization settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                if key in {"coarse_resolution_segments", "resolution_ladder"}:
                    normalized = self._normalize_ladder(value)
                    setattr(self.settings, key, normalized)
                else:
                    setattr(self.settings, key, value)

                if key in {"poly_order", "collocation_method"}:
                    self.motion_optimizer.configure(**{key: value})
                if key == "solver_options":
                    self.motion_optimizer.configure(
                        solver_options=self.settings.solver_options,
                    )

                if key in {"min_segments", "coarse_resolution_segments", "resolution_ladder"}:
                    self._ensure_components(self._initial_segments())

                log.debug(f"Updated setting {key} = {value}")
            else:
                log.warning(f"Unknown setting: {key}")

    def setup_multiple_shooting_fallback(self) -> None:
        """
        Setup multiple shooting fallback for stiff cases.

        This is a stub for future implementation of multiple shooting
        as a fallback when direct collocation fails.
        """
        message = (
            "Multiple shooting fallback is not implemented. "
            "Direct collocation failures must be handled upstream."
        )
        log.error(message)
        raise NotImplementedError(message)

    def optimize_with_fallback(
        self, constraints: dict[str, Any], targets: dict[str, Any], **kwargs,
    ) -> OptimizationResult:
        """
        Optimize with fallback to multiple shooting if direct collocation fails.

        Parameters
        ----------
        constraints : Dict[str, Any]
            Optimization constraints
        targets : Dict[str, Any]
            Optimization targets
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        # Try direct collocation first
        result = self.optimize_phase1(constraints, targets, **kwargs)

        # If failed, try fallback (when implemented)
        if not result.successful:
            message = (
                "Direct collocation failed and multiple shooting fallback "
                "is unavailable. Aborting optimization."
            )
            log.error(message)
            raise RuntimeError(message)

        return result

    def benchmark_optimization(self, problem_specs: list) -> dict[str, Any]:
        """
        Benchmark optimization performance across multiple problem specifications.

        Parameters
        ----------
        problem_specs : list
            List of problem specifications to benchmark

        Returns
        -------
        Dict[str, Any]
            Benchmark results
        """
        results = []

        for i, spec in enumerate(problem_specs):
            log.info(f"Benchmarking problem {i + 1}/{len(problem_specs)}")

            start_time = time.time()
            result = self.optimize_phase1(spec["constraints"], spec["targets"])
            solve_time = time.time() - start_time

            results.append(
                {
                    "problem_id": i,
                    "successful": result.successful,
                    "solve_time": solve_time,
                    "objective_value": result.objective_value,
                    "n_iterations": result.metadata.get("n_iterations", 0),
                    "efficiency": result.metadata.get("total_efficiency", 0.0),
                },
            )

        # Compute statistics
        successful_results = [r for r in results if r["successful"]]

        benchmark_stats = {
            "total_problems": len(problem_specs),
            "successful_problems": len(successful_results),
            "success_rate": len(successful_results) / len(problem_specs),
            "avg_solve_time": np.mean([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "min_solve_time": np.min([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "max_solve_time": np.max([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "avg_efficiency": np.mean([r["efficiency"] for r in successful_results])
            if successful_results
            else 0,
            "results": results,
        }

        log.info(
            f"Benchmark completed: {benchmark_stats['success_rate']:.1%} success rate, "
            f"avg solve time: {benchmark_stats['avg_solve_time']:.3f}s",
        )

        return benchmark_stats
