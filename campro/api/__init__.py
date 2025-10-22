"""Public API for Larrak.

Exposes data classes and façade functions for common workflows.
"""

from typing import Any, Dict, Tuple

from campro.constraints.cam import CamMotionConstraints
from campro.diagnostics.run_metadata import RUN_ID, log_run_metadata
from campro.optimization.motion import MotionOptimizer

from .problem_spec import ProblemSpec
from .solve_report import SolveReport
from .adapters import motion_result_to_solve_report


def solve_motion(spec: ProblemSpec) -> SolveReport:
    """Unified entry point for motion law optimization.

    Adapts ProblemSpec into existing MotionOptimizer and returns a SolveReport.
    """
    # Map ProblemSpec to cam constraints
    phases = spec.phases or {}
    bounds = spec.bounds or {}

    up_pct = phases.get("upstroke_percent") or phases.get("upstroke", 60.0)
    zero_pct = phases.get("zero_accel_percent") or phases.get("zero_accel", 0.0)

    def _pick(*names: str, default: float | None = None):
        for n in names:
            if n in bounds and bounds[n] is not None:
                return float(bounds[n])
        return default

    max_v = _pick("max_velocity", "v_max")
    max_a = _pick("max_acceleration", "a_max")
    max_j = _pick("max_jerk", "j_max")

    constraints = CamMotionConstraints(
        stroke=float(spec.stroke),
        upstroke_duration_percent=float(up_pct),
        zero_accel_duration_percent=float(zero_pct) if zero_pct is not None else None,
        max_velocity=max_v,
        max_acceleration=max_a,
        max_jerk=max_j,
    )

    # Normalize objective naming
    obj_map = {
        "min_jerk": "minimum_jerk",
        "minimum_jerk": "minimum_jerk",
        "min_time": "minimum_time",
        "minimum_time": "minimum_time",
        "min_energy": "minimum_energy",
        "minimum_energy": "minimum_energy",
    }
    motion_type = obj_map.get(spec.objective, "minimum_jerk")

    optimizer = MotionOptimizer()
    result = optimizer.solve_cam_motion_law(
        cam_constraints=constraints, motion_type=motion_type, cycle_time=float(spec.cycle_time)
    )

    # Persist lightweight run metadata
    meta: Dict[str, Any] = {
        "run_id": RUN_ID,
        "objective": motion_type,
        "stroke": spec.stroke,
        "cycle_time": spec.cycle_time,
        "phases": spec.phases,
        "bounds": spec.bounds,
        "status": getattr(result.status, "value", str(result.status)),
        "iterations": getattr(result, "iterations", 0),
    }
    try:
        log_run_metadata(meta)
    except Exception:
        pass

    return motion_result_to_solve_report(result)


def design_gear(spec: ProblemSpec, motion: SolveReport) -> Tuple[dict, SolveReport]:
    """Stub for future Litvin synthesis integration."""
    raise NotImplementedError("design_gear is scheduled for a future sprint")


def evaluate_tribology(gear: Dict[str, Any], duty: Dict[str, Any]) -> Dict[str, Any]:
    """Stub for future EHL/Λ analysis."""
    raise NotImplementedError("evaluate_tribology is scheduled for a future sprint")


__all__ = [
    "ProblemSpec",
    "SolveReport",
    "solve_motion",
    "design_gear",
    "evaluate_tribology",
]
