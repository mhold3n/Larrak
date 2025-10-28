"""Public API for Larrak.

Exposes data classes and façade functions for common workflows.
"""
from __future__ import annotations

from dataclasses import asdict as _asdict
from typing import Any

from campro.constraints.cam import CamMotionConstraints
from campro.diagnostics.feasibility import check_feasibility_nlp
from campro.diagnostics.run_metadata import RUN_ID, log_run_metadata, set_global_seeds
from campro.logging import get_logger
from campro.optimization.motion import MotionOptimizer

from .adapters import motion_result_to_solve_report
from .problem_spec import ProblemSpec
from .solve_report import SolveReport

log = get_logger(__name__)


def solve_motion(spec: ProblemSpec) -> SolveReport:
    """Unified entry point for motion law optimization.

    Adapts ProblemSpec into existing MotionOptimizer and returns a SolveReport.
    """
    # Determinism: seed RNGs once per façade entry
    try:
        set_global_seeds(1337)
    except Exception:
        pass

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

    # Phase-0 feasibility check (fast infeasibility detection)
    feas_constraints: dict[str, Any] = {
        "stroke": float(spec.stroke),
        "cycle_time": float(spec.cycle_time),
        "upstroke_percent": float(up_pct),
        "zero_accel_percent": float(zero_pct) if zero_pct is not None else None,
    }
    feas_bounds: dict[str, Any] = {}
    for k in ("max_velocity", "max_acceleration", "max_jerk"):
        if k in bounds and bounds[k] is not None:
            try:
                feas_bounds[k] = float(bounds[k])
            except Exception:
                pass

    try:
        feas = check_feasibility_nlp(feas_constraints, feas_bounds)
    except Exception as exc:  # pragma: no cover - feasibility failures shouldn't crash
        log.warning(f"Feasibility pre-check failed: {exc}")
        feas = None

    # Prepare simple scaling stats (O(1) normalization hints)
    def _infer_scales() -> dict[str, float]:
        x_s = max(1.0, abs(float(spec.stroke)))
        v_s = max(1.0, abs(float(bounds.get("max_velocity", 1.0) or 1.0)))
        a_s = max(1.0, abs(float(bounds.get("max_acceleration", 1.0) or 1.0)))
        j_s = max(1.0, abs(float(bounds.get("max_jerk", 1.0) or 1.0)))
        t_s = max(1e-2, float(spec.cycle_time))
        return {"x": x_s, "v": v_s, "a": a_s, "j": j_s, "t": t_s}

    scaling_stats = {"scaling_primary": _infer_scales()}

    # Early exit on infeasible spec
    if feas is not None and not getattr(feas, "feasible", True):
        from campro.optimization.base import OptimizationResult, OptimizationStatus

        # Create a minimal failed result to flow through the adapter
        fake = OptimizationResult(
            status=OptimizationStatus.INFEASIBLE,
            iterations=0,
            convergence_info={"feasibility_primary": _asdict(feas)},
        )
        report = motion_result_to_solve_report(fake)
        # Enrich report with feasibility + scaling telemetry
        try:
            report.residuals["feas.max_violation"] = float(
                getattr(feas, "max_violation", 0.0),
            )
            for k, v in (getattr(feas, "violations", {}) or {}).items():
                if isinstance(v, (int, float)):
                    report.residuals[f"feas.violation.{k}"] = float(v)
        except Exception:
            pass
        report.scaling_stats.update(scaling_stats)

        # Persist lightweight run metadata
        meta: dict[str, Any] = {
            "run_id": RUN_ID,
            "objective": motion_type,
            "stroke": spec.stroke,
            "cycle_time": spec.cycle_time,
            "phases": spec.phases,
            "bounds": spec.bounds,
            "feasibility": _asdict(feas),
            "status": report.status,
        }
        try:
            log_run_metadata(meta)
        except Exception:
            pass
        return report

    optimizer = MotionOptimizer()
    result = optimizer.solve_cam_motion_law(
        cam_constraints=constraints,
        motion_type=motion_type,
        cycle_time=float(spec.cycle_time),
    )

    # Persist lightweight run metadata
    meta: dict[str, Any] = {
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

    # Convert to public report and augment with feasibility + scaling telemetry
    report = motion_result_to_solve_report(result)
    if feas is not None:
        try:
            report.residuals["feas.max_violation"] = float(
                getattr(feas, "max_violation", 0.0),
            )
            for k, v in (getattr(feas, "violations", {}) or {}).items():
                if isinstance(v, (int, float)):
                    report.residuals[f"feas.violation.{k}"] = float(v)
        except Exception:
            pass
    report.scaling_stats.update(scaling_stats)

    return report


def design_gear(spec: ProblemSpec, motion: SolveReport) -> tuple[dict, SolveReport]:
    """Stub for future Litvin synthesis integration."""
    raise NotImplementedError("design_gear is scheduled for a future sprint")


def evaluate_tribology(gear: dict[str, Any], duty: dict[str, Any]) -> dict[str, Any]:
    """Stub for future EHL/Λ analysis."""
    raise NotImplementedError("evaluate_tribology is scheduled for a future sprint")


__all__ = [
    "ProblemSpec",
    "SolveReport",
    "design_gear",
    "evaluate_tribology",
    "solve_motion",
]
