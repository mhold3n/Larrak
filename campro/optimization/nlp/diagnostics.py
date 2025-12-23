"""NLP diagnostic utilities for constraint Jacobian and variable/objective evaluation.

This module provides functions for diagnosing issues in NLP formulations,
such as NaN entries in Jacobians and evaluating NLPs at initial guesses.

Functions here are extracted from driver.py to reduce its complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from campro.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def diagnose_nan_in_jacobian(
    nlp: dict[str, Any],
    x0: np.ndarray[Any, Any],
    row_idx: int,
    col_idx: int,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Diagnose NaN in Jacobian at specific row/column indices.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess
        row_idx: Constraint row index (0-based)
        col_idx: Variable column index (0-based)
        meta: Problem metadata (optional, for constraint/variable mapping)

    Returns:
        Dictionary with diagnostic information
    """
    import casadi as ca

    diagnostics: dict[str, Any] = {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "constraint_type": "unknown",
        "variable_group": "unknown",
        "constraint_value": None,
        "variable_value": None,
        "jacobian_entry": None,
        "constraint_expr_info": None,
        "constraint_index_in_group": None,
        "variable_index_in_group": None,
    }

    if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
        return diagnostics

    x_sym = nlp["x"]
    g_expr = nlp["g"]

    # Get variable value
    if col_idx < len(x0):
        diagnostics["variable_value"] = float(x0[col_idx])

    # Try to get variable group from metadata
    if meta and "variable_groups" in meta:
        var_groups = meta["variable_groups"]
        for group_name, indices in var_groups.items():
            if col_idx in indices:
                diagnostics["variable_group"] = group_name
                # Find position within group
                try:
                    pos_in_group = indices.index(col_idx)
                    diagnostics["variable_index_in_group"] = pos_in_group
                except ValueError:
                    pass
                break

    # Try to get constraint type from metadata
    if meta and "constraint_groups" in meta:
        con_groups = meta["constraint_groups"]
        for con_type, indices in con_groups.items():
            if row_idx in indices:
                diagnostics["constraint_type"] = con_type
                # Find position within group
                try:
                    pos_in_group = indices.index(row_idx)
                    diagnostics["constraint_index_in_group"] = pos_in_group
                except ValueError:
                    pass
                break

    # Evaluate constraint value
    try:
        g_func = ca.Function("g_func", [x_sym], [g_expr])
        g0 = g_func(x0)
        g_arr = np.array(g0).flatten()
        if row_idx < len(g_arr):
            diagnostics["constraint_value"] = float(g_arr[row_idx])
    except Exception as e:
        diagnostics["constraint_value"] = f"Error: {e}"

    # Evaluate Jacobian entry
    try:
        jac_g_expr = ca.jacobian(g_expr, x_sym)
        jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
        jac_g0 = jac_g_func(x0)
        jac_g0_arr = np.array(jac_g0)
        if row_idx < jac_g0_arr.shape[0] and col_idx < jac_g0_arr.shape[1]:
            diagnostics["jacobian_entry"] = float(jac_g0_arr[row_idx, col_idx])
    except Exception as e:
        diagnostics["jacobian_entry"] = f"Error: {e}"

    # Try to get individual constraint expression
    try:
        if hasattr(g_expr, "elements") or hasattr(g_expr, "get_elements"):
            # CasADi SX/MX may have element access
            if row_idx < g_expr.numel():
                # Try to extract individual constraint
                g_elem = g_expr[row_idx] if hasattr(g_expr, "__getitem__") else None
                if g_elem is not None:
                    diagnostics["constraint_expr_info"] = str(g_elem)
    except Exception:
        pass

    return diagnostics


def evaluate_nlp_at_x0(
    nlp: dict[str, Any],
    x0: np.ndarray[Any, Any],
    meta: dict[str, Any] | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    float,
    np.ndarray[Any, Any] | None,
    np.ndarray[Any, Any] | None,
]:
    """
    Evaluate NLP at x0: constraints, objective, constraint Jacobian, and objective gradient.

    Single evaluation point for all scaling computations.

    Args:
        nlp: CasADi NLP dict with 'x', 'g', and 'f' keys
        x0: Initial guess for variables
        meta: Problem metadata (optional, for NaN diagnosis)

    Returns:
        Tuple of (g0_arr, f0_val, jac_g0_arr, grad_f0_arr):
        - g0_arr: Constraint values at x0 (array)
        - f0_val: Objective value at x0 (scalar)
        - jac_g0_arr: Constraint Jacobian at x0 (array, or None if unavailable)
        - grad_f0_arr: Objective gradient at x0 (array, or None if unavailable)
    """
    import casadi as ca

    g0_arr: np.ndarray[Any, Any] = np.array([])
    f0_val = 0.0
    jac_g0_arr: np.ndarray[Any, Any] | None = None
    grad_f0_arr: np.ndarray[Any, Any] | None = None

    if not isinstance(nlp, dict):
        return g0_arr, f0_val, jac_g0_arr, grad_f0_arr

    x_sym = nlp.get("x")
    if x_sym is None:
        return g0_arr, f0_val, jac_g0_arr, grad_f0_arr

    # Evaluate constraints
    if "g" in nlp and nlp["g"] is not None:
        try:
            g_expr = nlp["g"]
            if g_expr.numel() > 0:
                g_func = ca.Function("g_func", [x_sym], [g_expr])
                g0 = g_func(x0)
                g0_arr = np.array(g0).flatten()

                # Compute constraint Jacobian
                jac_g_expr = ca.jacobian(g_expr, x_sym)
                jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
                jac_g0 = jac_g_func(x0)
                jac_g0_arr = np.array(jac_g0)

                # DIAGNOSTIC: Check for NaN and diagnose
                if np.any(np.isnan(jac_g0_arr)):
                    nan_locations = np.where(np.isnan(jac_g0_arr))
                    if len(nan_locations[0]) > 0:
                        # Diagnose first NaN location
                        first_row = int(nan_locations[0][0])
                        first_col = int(nan_locations[1][0])
                        log.warning(
                            f"NaN detected in Jacobian at row {first_row}, col {first_col}. "
                            f"Total NaN entries: {len(nan_locations[0])}"
                        )
                        # Use metadata passed as parameter, or try to get from nlp
                        meta_for_diag = meta
                        if meta_for_diag is None:
                            if hasattr(nlp, "meta"):
                                meta_for_diag = nlp.meta  # type: ignore[union-attr]
                            elif isinstance(nlp, dict) and "meta" in nlp:
                                meta_for_diag = nlp["meta"]

                        # Diagnose the NaN location
                        try:
                            diag = diagnose_nan_in_jacobian(
                                nlp, x0, first_row, first_col, meta_for_diag
                            )
                            log.warning(
                                f"  Constraint type: {diag.get('constraint_type', 'unknown')}, "
                                f"Variable group: {diag.get('variable_group', 'unknown')}"
                            )
                            if diag.get("constraint_index_in_group") is not None:
                                log.warning(
                                    f"  Constraint index in group: {diag.get('constraint_index_in_group')}"
                                )
                            if diag.get("variable_index_in_group") is not None:
                                log.warning(
                                    f"  Variable index in group: {diag.get('variable_index_in_group')}"
                                )
                            if diag.get("constraint_value") is not None:
                                log.warning(f"  Constraint value: {diag.get('constraint_value')}")
                            if diag.get("variable_value") is not None:
                                log.warning(f"  Variable value: {diag.get('variable_value')}")
                        except Exception as diag_exc:
                            log.debug(f"  Could not diagnose NaN location: {diag_exc}")
        except Exception as e:
            log.debug(f"Failed to evaluate constraints/Jacobian at x0: {e}")

    # Evaluate objective
    if "f" in nlp and nlp["f"] is not None:
        try:
            f_expr = nlp["f"]
            f_func = ca.Function("f_func", [x_sym], [f_expr])
            f0 = f_func(x0)
            f0_val = float(f0) if hasattr(f0, "__float__") else float(np.array(f0).item())

            # Compute objective gradient
            grad_f_expr = ca.gradient(f_expr, x_sym)
            grad_f_func = ca.Function("grad_f_func", [x_sym], [grad_f_expr])
            grad_f0 = grad_f_func(x0)
            grad_f0_arr = np.array(grad_f0).flatten()
        except Exception as e:
            log.debug(f"Failed to evaluate objective/gradient at x0: {e}")

    return g0_arr, f0_val, jac_g0_arr, grad_f0_arr


def summarize_ipopt_iterations(stats: dict[str, Any], reporter: Any) -> dict[str, Any] | None:
    """
    Summarize IPOPT iteration statistics for reporting.

    Args:
        stats: IPOPT solver statistics dict containing 'iterations' key
        reporter: StructuredReporter for output

    Returns:
        Summary dict with iteration metrics, or None if no data available
    """

    def _to_numpy_array(data: Any) -> np.ndarray[Any, Any]:
        """Convert data to numpy array, handling None and various types."""
        if data is None:
            return np.array([], dtype=float)
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=float)
        if isinstance(data, np.ndarray):
            return data.astype(float)
        try:
            return np.array([data], dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)

    iterations = stats.get("iterations")
    if not iterations or not isinstance(iterations, dict):
        if hasattr(reporter, "show_debug") and reporter.show_debug:
            reporter.debug("No IPOPT iteration diagnostics available.")
        return None

    k = _to_numpy_array(iterations.get("k"))
    obj_source = iterations.get("obj")
    if obj_source is None:
        obj_source = iterations.get("f")
    obj = _to_numpy_array(obj_source)
    inf_pr = _to_numpy_array(iterations.get("inf_pr"))
    inf_du = _to_numpy_array(iterations.get("inf_du"))
    mu = _to_numpy_array(iterations.get("mu"))
    step_types = iterations.get("type") or iterations.get("step_type") or []
    if hasattr(step_types, "tolist"):
        step_types = step_types.tolist()
    step_types = [str(s) for s in step_types]

    total_iters = len(k) if k.size else 0
    if total_iters == 0:
        return None

    restoration_steps = sum(1 for step in step_types if "r" in step.lower())
    final_idx = total_iters - 1

    def _safe_get(arr: np.ndarray[Any, Any], idx: int, default: float = float("nan")) -> float:
        if arr.size == 0:
            return default
        try:
            return float(arr[idx])
        except Exception:
            return default

    summary: dict[str, Any] = {
        "iteration_count": total_iters,
        "restoration_steps": restoration_steps,
        "max_inf_pr": float(np.max(np.abs(inf_pr))) if inf_pr.size else float("nan"),
        "max_inf_du": float(np.max(np.abs(inf_du))) if inf_du.size else float("nan"),
        "objective": {
            "start": _safe_get(obj, 0),
            "min": float(np.min(obj)) if obj.size else float("nan"),
            "max": float(np.max(obj)) if obj.size else float("nan"),
        },
        "final": {
            "objective": _safe_get(obj, final_idx),
            "inf_pr": _safe_get(inf_pr, final_idx),
            "inf_du": _safe_get(inf_du, final_idx),
            "mu": _safe_get(mu, final_idx),
            "step": step_types[final_idx] if step_types else None,
        },
    }

    reporter.info(
        f"IPOPT iterations={summary['iteration_count']} restoration={summary['restoration_steps']} "
        f"objective(start={summary['objective']['start']:.3e}, final={summary['final']['objective']:.3e})",
    )
    reporter.info(
        f"Final residuals: inf_pr={summary['final']['inf_pr']:.3e} inf_du={summary['final']['inf_du']:.3e} "
        f"mu={summary['final']['mu']:.3e}",
    )

    if hasattr(reporter, "show_debug") and reporter.show_debug and total_iters > 0:
        recent_start = max(0, total_iters - 5)
        recent_entries = []
        with reporter.section("Recent IPOPT iterations", level="DEBUG"):
            for idx in range(recent_start, total_iters):
                entry = {
                    "k": int(_safe_get(k, idx, default=float(idx))),
                    "objective": _safe_get(obj, idx),
                    "inf_pr": _safe_get(inf_pr, idx),
                    "inf_du": _safe_get(inf_du, idx),
                    "mu": _safe_get(mu, idx),
                    "step": step_types[idx] if idx < len(step_types) else "",
                }
                recent_entries.append(entry)
                reporter.debug(
                    f"k={entry['k']:>4} f={entry['objective']:.3e} inf_pr={entry['inf_pr']:.3e} "
                    f"inf_du={entry['inf_du']:.3e} mu={entry['mu']:.3e} step={entry['step']}",
                )
        summary["recent_iterations"] = recent_entries

    return summary


def compute_scaled_jacobian(
    nlp: dict[str, Any],
    x0: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
    """
    Compute scaled Jacobian to identify problematic rows/columns.

    Args:
        nlp: CasADi NLP dict with 'x' and 'g' keys
        x0: Initial guess
        scale: Variable scaling factors
        scale_g: Constraint scaling factors

    Returns:
        Tuple of (jac_g0_arr, jac_g0_scaled) or (None, None) if computation fails
    """
    try:
        import casadi as ca

        if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
            return None, None

        g_expr = nlp["g"]
        x_sym = nlp["x"]

        if g_expr is None or g_expr.numel() == 0:
            return None, None

        # Compute Jacobian
        jac_g_expr = ca.jacobian(g_expr, x_sym)
        jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
        jac_g0 = jac_g_func(x0)
        jac_g0_arr = np.array(jac_g0)

        # Check for NaN in inputs before scaling
        if np.any(np.isnan(jac_g0_arr)):
            log.warning("NaN detected in unscaled Jacobian in compute_scaled_jacobian")
            jac_g0_arr = np.nan_to_num(jac_g0_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        if scale_g is not None and len(scale_g) > 0:
            if np.any(np.isnan(scale_g)):
                log.warning("NaN detected in scale_g in compute_scaled_jacobian")
                scale_g = np.nan_to_num(scale_g, nan=1.0, posinf=1e8, neginf=1e-8)

        if scale is not None and len(scale) > 0:
            if np.any(np.isnan(scale)):
                log.warning("NaN detected in scale in compute_scaled_jacobian")
                scale = np.nan_to_num(scale, nan=1.0, posinf=1e8, neginf=1e-8)

        # Compute scaled Jacobian: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
        jac_g0_scaled = np.zeros_like(jac_g0_arr)
        for i in range(min(jac_g0_arr.shape[0], len(scale_g))):
            for j in range(min(jac_g0_arr.shape[1], len(scale))):
                if scale[j] > 1e-10 and scale_g[i] > 1e-10:
                    jac_g0_scaled[i, j] = scale_g[i] * jac_g0_arr[i, j] / scale[j]
                else:
                    jac_g0_scaled[i, j] = 0.0

        # Check for NaN in output
        if np.any(np.isnan(jac_g0_scaled)):
            log.warning("NaN detected in scaled Jacobian output in compute_scaled_jacobian")
            jac_g0_scaled = np.nan_to_num(jac_g0_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        return jac_g0_arr, jac_g0_scaled
    except Exception:
        return None, None
