from __future__ import annotations

from typing import Any

from campro.api.solve_report import SolveReport
from campro.diagnostics import RUN_ID
from campro.diagnostics.ipopt_logger import get_ipopt_log_stats
from campro.diagnostics.run_metadata import log_run_metadata


def motion_result_to_solve_report(result: OptimizationResult) -> SolveReport:  # type: ignore[name-defined]
    """Convert a Motion/OptimizationResult to a SolveReport."""
    status = getattr(result, "status", None)
    status_str = (
        getattr(status, "value", str(status)) if status is not None else "unknown"
    )

    # Extract basic residuals if present
    residuals: dict[str, float] = {}
    conv = getattr(result, "convergence_info", {}) or {}
    for k in (
        "primal_inf",
        "dual_inf",
        "complementarity",
        "constraint_violation",
        "kkt_error",
    ):
        v = conv.get(k)
        if isinstance(v, (int, float)):
            residuals[k] = float(v)

    # Derive iterations if available
    n_iter = 0
    try:
        n_iter = int(getattr(result, "iterations", 0) or 0)
    except Exception:
        pass

    # Parse Ipopt log if present and merge residuals
    log_stats = get_ipopt_log_stats()
    if log_stats:
        for k_src, k_dst in (
            ("primal_inf", "primal_inf"),
            ("dual_inf", "dual_inf"),
            ("compl_inf", "compl_inf"),
        ):
            v = log_stats.get(k_src)
            if isinstance(v, (int, float)):
                residuals[k_dst] = float(v)

    # Artifacts are populated by diagnostics elsewhere; include runs log path
    artifacts: dict[str, Any] = {
        "ipopt_log": f"runs/{RUN_ID}-ipopt.log",
        "run_meta": f"runs/{RUN_ID}.json",
    }

    # Map common backend statuses to public report status
    public_status = (
        "Solve_Success"
        if status_str == "converged"
        else ("Infeasible" if status_str == "infeasible" else "Failed")
    )

    # Build extended payloads if available
    motion_payload: dict[str, Any] = {}
    pressure_payload: dict[str, Any] = {}
    thermo_payload: dict[str, Any] = {}
    try:
        sol = getattr(result, "solution", {}) or {}
        meta = getattr(result, "metadata", {}) or {}
        # Motion: prefer combustion-time series; include back-compat fields if present
        for key in (
            "time_s",
            "theta_rad",
            "theta_deg",
            "position_mm",
            "velocity_mm_per_s",
            "acceleration_mm_per_s2",
            "cam_angle",
            "position",
            "velocity",
            "acceleration",
        ):
            if key in sol:
                motion_payload[key] = sol[key]
        # Pressure mappings if provided in metadata
        pressure_meta = meta.get("pressure")
        if isinstance(pressure_meta, dict):
            pressure_payload = pressure_meta
        # Thermo scalars if present
        pi_meta = (meta.get("pressure_invariance") or {}) if isinstance(meta, dict) else {}
        if isinstance(pi_meta, dict):
            # Common scalars to expose
            for k in ("imep_avg", "loss_p_mean", "pressure_ratio_target_mean"):
                if k in pi_meta:
                    thermo_payload[k] = pi_meta[k]
    except Exception:
        pass

    report = SolveReport(
        run_id=RUN_ID,
        status=public_status,
        kkt={
            k: residuals.get(k)
            for k in ("primal_inf", "dual_inf", "compl_inf")
            if k in residuals
        },
        n_iter=n_iter,
        scaling_stats={},
        residuals=residuals,
        artifacts=artifacts,
        motion=motion_payload,
        pressure=pressure_payload,
        thermo=thermo_payload,
    )

    # Persist metadata
    try:
        log_run_metadata(
            {
                "run_id": RUN_ID,
                "status": report.status,
                "n_iter": report.n_iter,
                "kkt": report.kkt,
                "residuals": report.residuals,
                "artifacts": report.artifacts,
            },
        )
    except Exception:
        pass

    return report


def unified_data_to_solve_report(data: UnifiedOptimizationData) -> SolveReport:  # type: ignore[name-defined]
    """Convert UnifiedOptimizationData to a SolveReport summary.

    Aggregates convergence info to a single high-level status.
    """
    # Aggregate status: prefer tertiary, else secondary, else primary
    ci = getattr(data, "convergence_info", {}) or {}
    status_primary = (ci.get("primary", {}) or {}).get("status")
    status_secondary = (ci.get("secondary", {}) or {}).get("status")
    status_tertiary = (ci.get("tertiary", {}) or {}).get("status")

    final_status = status_tertiary or status_secondary or status_primary or "unknown"
    status_str = str(final_status).lower()

    # Collect residual-like values if available via analyses
    residuals: dict[str, float] = {}
    for phase_attr in (
        "primary_ipopt_analysis",
        "secondary_ipopt_analysis",
        "tertiary_ipopt_analysis",
    ):
        report = getattr(data, phase_attr, None)
        if report and getattr(report, "stats", None):
            stats = report.stats  # type: ignore[assignment]
            for key in ("primal_inf", "dual_inf"):
                v = stats.get(key)
                if isinstance(v, (int, float)):
                    residuals[f"{phase_attr}.{key}"] = float(v)

    # Include feasibility pre-check results if present
    feas = (getattr(data, "convergence_info", {}) or {}).get(
        "feasibility_primary",
    ) or {}
    if isinstance(feas, dict):
        mv = feas.get("max_violation")
        if isinstance(mv, (int, float)):
            residuals["feas_primary.max_violation"] = float(mv)

    # Parse Ipopt log and add residuals/kkt if available
    log_stats = get_ipopt_log_stats()
    if log_stats:
        for k_src, k_dst in (
            ("primal_inf", "primal_inf"),
            ("dual_inf", "dual_inf"),
            ("compl_inf", "compl_inf"),
        ):
            v = log_stats.get(k_src)
            if isinstance(v, (int, float)):
                residuals[k_dst] = float(v)

    # Iterations: prefer tertiary
    n_iter = 0
    for phase in ("tertiary", "secondary", "primary"):
        it = (ci.get(phase, {}) or {}).get("iterations")
        if isinstance(it, int):
            n_iter = it
            break

    artifacts: dict[str, Any] = {
        "ipopt_log": f"runs/{RUN_ID}-ipopt.log",
        "run_meta": f"runs/{RUN_ID}.json",
    }

    # Scaling stats if recorded by framework
    scaling_stats = {}
    ci = getattr(data, "convergence_info", {}) or {}
    for key in ("scaling_primary", "scaling_secondary", "scaling_tertiary"):
        if key in ci and isinstance(ci[key], dict):
            scaling_stats[key] = ci[key]

    report = SolveReport(
        run_id=RUN_ID,
        status="Solve_Success"
        if status_str == "converged"
        else ("Infeasible" if status_str == "infeasible" else "Failed"),
        kkt={
            k: residuals.get(k)
            for k in ("primal_inf", "dual_inf", "compl_inf")
            if k in residuals
        },
        n_iter=n_iter,
        scaling_stats=scaling_stats,
        residuals=residuals,
        artifacts=artifacts,
    )

    # Persist metadata (feasibility and scaling included)
    try:
        feas = (getattr(data, "convergence_info", {}) or {}).get("feasibility_primary")
        meta = {
            "run_id": RUN_ID,
            "status": report.status,
            "n_iter": report.n_iter,
            "kkt": report.kkt,
            "residuals": report.residuals,
            "scaling_stats": report.scaling_stats,
            "feasibility_primary": feas,
            "artifacts": report.artifacts,
        }
        log_run_metadata(meta)
    except Exception:
        pass

    return report
