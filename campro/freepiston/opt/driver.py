from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

from campro.freepiston.core.states import MechState
from campro.freepiston.io.save import save_json
from campro.freepiston.opt.colloc import make_grid
from campro.freepiston.opt.ipopt_solver import (
    IPOPTOptions,
    IPOPTSolver,
    get_default_ipopt_options,
    get_robust_ipopt_options,
)
from campro.freepiston.opt.nlp import build_collocation_nlp
from campro.freepiston.opt.solution import Solution
from campro.freepiston.zerod.cv import cv_residual
from campro.diagnostics.scaling import (
    build_scaled_nlp,
    scale_value,
    scale_bounds,
    unscale_value,
)
from campro.logging import get_logger
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)

_FALSEY = {"0", "false", "no", "off"}


def _to_numpy_array(data: Any) -> np.ndarray:
    if data is None:
        return np.array([])
    try:
        arr = np.asarray(data, dtype=float)
        return arr.flatten()
    except Exception:
        try:
            return np.array([float(x) for x in data], dtype=float)
        except Exception:
            return np.array([])


def _summarize_ipopt_iterations(stats: dict[str, Any], reporter: StructuredReporter) -> dict[str, Any] | None:
    iterations = stats.get("iterations")
    if not iterations or not isinstance(iterations, dict):
        if reporter.show_debug:
            reporter.debug("No IPOPT iteration diagnostics available.")
        return None

    k = _to_numpy_array(iterations.get("k"))
    obj = _to_numpy_array(iterations.get("obj") or iterations.get("f"))
    inf_pr = _to_numpy_array(iterations.get("inf_pr"))
    inf_du = _to_numpy_array(iterations.get("inf_du"))
    mu = _to_numpy_array(iterations.get("mu"))
    step_types = iterations.get("type") or iterations.get("step_type") or []
    if hasattr(step_types, "tolist"):
        step_types = step_types.tolist()
    step_types = [str(s) for s in step_types]

    total_iters = int(len(k)) if k.size else 0
    if total_iters == 0:
        return None

    restoration_steps = sum(1 for step in step_types if "r" in step.lower())
    final_idx = total_iters - 1

    def _safe_get(arr: np.ndarray, idx: int, default: float = float("nan")) -> float:
        if arr.size == 0:
            return default
        try:
            return float(arr[idx])
        except Exception:
            return default

    summary = {
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
        f"objective(start={summary['objective']['start']:.3e}, final={summary['final']['objective']:.3e})"
    )
    reporter.info(
        f"Final residuals: inf_pr={summary['final']['inf_pr']:.3e} inf_du={summary['final']['inf_du']:.3e} "
        f"mu={summary['final']['mu']:.3e}"
    )

    if reporter.show_debug and total_iters > 0:
        recent_start = max(0, total_iters - 5)
        recent_entries = []
        with reporter.section("Recent IPOPT iterations", level="DEBUG"):
            for idx in range(recent_start, total_iters):
                entry = {
                    "k": int(_safe_get(k, idx, default=idx)),
                    "objective": _safe_get(obj, idx),
                    "inf_pr": _safe_get(inf_pr, idx),
                    "inf_du": _safe_get(inf_du, idx),
                    "mu": _safe_get(mu, idx),
                    "step": step_types[idx] if idx < len(step_types) else "",
                }
                recent_entries.append(entry)
                reporter.debug(
                    f"k={entry['k']:>4} f={entry['objective']:.3e} inf_pr={entry['inf_pr']:.3e} "
                    f"inf_du={entry['inf_du']:.3e} mu={entry['mu']:.3e} step={entry['step']}"
                )
        summary["recent_iterations"] = recent_entries

    return summary


def solve_cycle(P: dict[str, Any]) -> dict[str, Any]:
    """
    Solve OP engine cycle optimization using IPOPT.

    This function builds the collocation NLP and solves it using IPOPT
    with appropriate options for OP engine optimization.

    Args:
        P: Problem parameters dictionary

    Returns:
        Solution object with optimization results
    """
    num = P.get("num", {})
    K = int(num.get("K", 10))
    C = int(num.get("C", 3))
    grid = make_grid(K, C, kind="radau")

    iteration_summary: dict[str, Any] = {}

    # Check if combustion model is enabled
    combustion_cfg = P.get("combustion", {})
    use_combustion = bool(combustion_cfg.get("use_integrated_model", False))
    
    reporter = StructuredReporter(
        context="FREE-PISTON",
        logger=None,
        stream_out=sys.stderr,
        stream_err=sys.stderr,
        debug_env="FREE_PISTON_DEBUG",
        force_debug=True,
    )

    # Log problem parameters before building NLP
    reporter.info(
        f"Building collocation NLP: K={K}, C={C}, combustion_model={'enabled' if use_combustion else 'disabled'}"
    )
    # Estimate complexity
    estimated_vars = K * C * 6 + 6 + K * C * 2  # Rough estimate
    estimated_constraints = K * C * 4  # Rough estimate
    reporter.info(
        f"Estimated problem size: vars≈{estimated_vars}, constraints≈{estimated_constraints}"
    )
    
    nlp_build_start = time.time()

    # Minimal residual evaluation at a nominal state (placeholder)
    mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
    gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5}
    res = cv_residual(mech, gas, {"geom": P.get("geom", {}), "flows": {}})

    # Build NLP
    try:
        nlp, meta = build_collocation_nlp(P)
        nlp_build_elapsed = time.time() - nlp_build_start
        
        # Extract actual problem size from meta
        n_vars = meta.get("n_vars", 0) if meta else 0
        n_constraints = meta.get("n_constraints", 0) if meta else 0
        reporter.info(
            f"NLP built in {nlp_build_elapsed:.3f}s: n_vars={n_vars}, n_constraints={n_constraints}"
        )
        if meta:
            reporter.info(
                f"Problem characteristics: K={meta.get('K', K)}, C={meta.get('C', C)}, "
                f"combustion_model={'integrated' if use_combustion else 'none'}"
            )

        # Get solver options
        solver_opts = P.get("solver", {}).get("ipopt", {})
        warm_start = P.get("warm_start", {})
        
        # Check for diagnostic mode via environment variable
        if os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1":
            solver_opts.setdefault("ipopt.derivative_test", "first-order")
            solver_opts.setdefault("ipopt.print_level", 12)
            reporter.info("Diagnostic mode enabled: derivative test and verbose output activated")

        # Create IPOPT solver
        reporter.info("Creating IPOPT solver with options...")
        solver_create_start = time.time()
        ipopt_options = _create_ipopt_options(solver_opts, P)
        solver = IPOPTSolver(ipopt_options)
        solver_create_elapsed = time.time() - solver_create_start
        reporter.info(
            f"IPOPT solver created in {solver_create_elapsed:.3f}s: "
            f"max_iter={ipopt_options.max_iter}, tol={ipopt_options.tol:.2e}, "
            f"print_level={ipopt_options.print_level}"
        )
        if os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1":
            reporter.info("Derivative test enabled. To disable: unset FREE_PISTON_DIAGNOSTIC_MODE")

        # Set up initial guess and bounds
        reporter.info("Setting up optimization bounds and initial guess...")
        x0, lbx, ubx, lbg, ubg, p = _setup_optimization_bounds(nlp, P, warm_start)
        
        if x0 is None or lbx is None or ubx is None:
            raise ValueError("Failed to set up optimization bounds and initial guess")
        
        # Compute variable scaling factors (using group-based scaling with initial guess)
        reporter.info("Computing variable scaling factors (group-based)...")
        # Get variable groups from metadata if available
        variable_groups = meta.get("variable_groups", {}) if meta else {}
        scale = _compute_variable_scaling(lbx, ubx, x0=x0, variable_groups=variable_groups)
        
        # Log scaling statistics
        scale_min = scale.min()
        scale_max = scale.max()
        scale_mean = scale.mean()
        reporter.info(
            f"Variable scaling factors: min={scale_min:.3e}, max={scale_max:.3e}, mean={scale_mean:.3e}"
        )
        
        # Group-wise diagnostics
        if variable_groups:
            with reporter.section("Variable scaling by group"):
                for group_name, indices in variable_groups.items():
                    if indices and len(indices) > 0:
                        group_scales = scale[np.array(indices)]
                        group_min = group_scales.min()
                        group_max = group_scales.max()
                        group_mean = group_scales.mean()
                        group_range_ratio = group_max / group_min if group_min > 1e-10 else np.inf
                        reporter.info(
                            f"Group '{group_name}' ({len(indices)} vars): "
                            f"range=[{group_min:.3e}, {group_max:.3e}], mean={group_mean:.3e}, "
                            f"ratio={group_range_ratio:.2e}"
                        )
                        if group_range_ratio > 1e6:
                            reporter.warning(
                                f"Group '{group_name}' has extreme scale ratio ({group_range_ratio:.2e}). "
                                f"Consider adjusting unit references or bounds."
                            )
        
        # Compute constraint scaling factors (using evaluation-based scaling)
        reporter.info("Computing constraint scaling factors (evaluation-based with Jacobian)...")
        try:
            scale_g = _compute_constraint_scaling_from_evaluation(nlp, x0, lbg, ubg, scale=scale)
            scaling_method = "evaluation-based with Jacobian"
        except Exception as e:
            reporter.warning(f"Constraint evaluation-based scaling failed: {e}, falling back to bounds-based")
            scale_g = _compute_constraint_scaling(lbg, ubg)
            scaling_method = "bounds-based"
        
        if len(scale_g) > 0:
            scale_g_min = scale_g.min()
            scale_g_max = scale_g.max()
            scale_g_mean = scale_g.mean()
            reporter.info(
                f"Constraint scaling factors ({scaling_method}): "
                f"min={scale_g_min:.3e}, max={scale_g_max:.3e}, mean={scale_g_mean:.3e}"
            )
        
        # Compute objective scaling factor
        scale_f = 1.0  # Initialize default (no scaling)
        reporter.info("Computing objective scaling factor...")
        try:
            scale_f = _compute_objective_scaling(nlp, x0)
            if scale_f != 1.0:
                reporter.info(f"Objective scaling factor: {scale_f:.6e}")
        except Exception as e:
            reporter.warning(f"Objective scaling failed: {e}, using no objective scaling")
            scale_f = 1.0
        
        # Apply scaling to NLP
        if isinstance(nlp, dict):
            nlp_scaled = build_scaled_nlp(
                nlp, 
                scale, 
                constraint_scale=scale_g if len(scale_g) > 0 else None,
                objective_scale=scale_f if scale_f != 1.0 else None,
            )
        else:
            # If nlp is a CasADi Function, convert to dict format first
            # This is a fallback - ideally nlp should be a dict
            reporter.warning("NLP is not a dict, attempting to use unscaled NLP")
            nlp_scaled = nlp
            scale_f = 1.0  # No objective scaling if NLP not a dict
        
        # Scale initial guess and bounds
        x0_scaled = scale_value(x0, scale)
        lbx_scaled, ubx_scaled = scale_bounds((lbx, ubx), scale)
        
        # IMPORTANT: Re-clamp scaled initial guess to scaled bounds
        # The new scaling may be very different from what the initial guess assumed,
        # so we need to ensure x0_scaled satisfies the scaled bounds
        x0_scaled = np.clip(x0_scaled, lbx_scaled, ubx_scaled)
        
        # Scale constraint bounds
        if lbg is not None and ubg is not None and len(scale_g) > 0:
            lbg_scaled = scale_value(lbg, scale_g)
            ubg_scaled = scale_value(ubg, scale_g)
        else:
            lbg_scaled = lbg
            ubg_scaled = ubg
        
        if x0 is not None:
            reporter.info(
                f"Initial guess (unscaled): n_vars={len(x0)}, "
                f"x0_range=[{x0.min():.3e}, {x0.max():.3e}], mean={x0.mean():.3e}"
            )
            reporter.info(
                f"Initial guess (scaled): n_vars={len(x0_scaled)}, "
                f"x0_range=[{x0_scaled.min():.3e}, {x0_scaled.max():.3e}], mean={x0_scaled.mean():.3e}"
            )
        if lbx_scaled is not None and ubx_scaled is not None:
            bounded_vars = ((lbx_scaled > -np.inf) & (ubx_scaled < np.inf)).sum()
            reporter.info(
                f"Bounded variables: {bounded_vars}/{len(lbx_scaled) if lbx_scaled is not None else 0}"
            )

        # Verify scaling quality (diagnostics)
        _verify_scaling_quality(nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter)
        
        # Solve optimization problem with scaled NLP
        reporter.info("Starting IPOPT optimization (with comprehensive scaling)...")
        reporter.debug(
            f"Problem dimensions: n_vars={len(x0_scaled) if x0_scaled is not None else 0}, "
            f"n_constraints={len(lbg_scaled) if lbg_scaled is not None else 0}"
        )
        solve_start = time.time()
        result = solver.solve(nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled, p)
        solve_elapsed = time.time() - solve_start
        reporter.info(f"IPOPT solve completed in {solve_elapsed:.3f}s")
        stats = solver.stats()
        iteration_summary = _summarize_ipopt_iterations(stats, reporter) or {}

        # Unscale solution
        if result.success and result.x_opt is not None and len(result.x_opt) > 0:
            x_opt_unscaled = unscale_value(result.x_opt, scale)
            # Unscale objective if it was scaled
            f_opt_unscaled = result.f_opt / scale_f if scale_f != 1.0 else result.f_opt
            reporter.info(
                f"Solution unscaled: x_opt_range=[{x_opt_unscaled.min():.3e}, {x_opt_unscaled.max():.3e}], "
                f"f_opt={f_opt_unscaled:.6e}"
            )
        else:
            x_opt_unscaled = result.x_opt if result.x_opt is not None else None
            f_opt_unscaled = result.f_opt / scale_f if scale_f != 1.0 else result.f_opt

        if result.success:
            reporter.info(f"Optimization successful: {result.message}")
            reporter.info(f"Iterations: {result.iterations}, CPU time: {result.cpu_time:.2f}s")
            reporter.info(f"Objective value: {result.f_opt:.6e}")
            reporter.info(f"KKT error: {result.kkt_error:.2e}")
            reporter.info(f"Feasibility error: {result.feasibility_error:.2e}")
        else:
            reporter.warning(f"Optimization failed: {result.message}")
            reporter.warning(f"Status: {result.status}, Iterations: {result.iterations}")

        # Store results (with unscaled solution)
        optimization_result = {
            "success": result.success,
            "x_opt": x_opt_unscaled,
            "f_opt": f_opt_unscaled if result.success else result.f_opt,
            "iterations": result.iterations,
            "cpu_time": result.cpu_time,
            "message": result.message,
            "status": result.status,
            "kkt_error": result.kkt_error,
            "feasibility_error": result.feasibility_error,
            "iteration_summary": iteration_summary,
        }

        # Optional checkpoint save per iteration group (best-effort minimal)
        run_dir = P.get("run_dir")
        if run_dir:
            try:
                save_json(
                    {"meta": meta, "opt": optimization_result},
                    run_dir,
                    filename="checkpoint.json",
                )
            except Exception as exc:  # pragma: no cover
                reporter.warning(f"Checkpoint save failed: {exc}")

    except Exception as e:
        reporter.error(f"Failed to build or solve NLP: {e!s}")
        nlp, meta = None, None
        optimization_result = {
            "success": False,
            "error": str(e),
            "x_opt": None,
            "f_opt": float("inf"),
            "iterations": 0,
            "cpu_time": 0.0,
            "message": f"NLP build/solve failed: {e!s}",
            "status": -1,
            "iteration_summary": iteration_summary,
        }

    return Solution(
        meta={"grid": grid, "meta": meta, "optimization": optimization_result},
        data={"residual_sample": res, "nlp": nlp},
    )


def _create_ipopt_options(
    solver_opts: dict[str, Any], P: dict[str, Any],
) -> IPOPTOptions:
    """Create IPOPT options from problem parameters."""
    # Get problem type to select appropriate options
    problem_type = P.get("problem_type", "default")

    if problem_type == "robust":
        options = get_robust_ipopt_options()
    else:
        options = get_default_ipopt_options()

    # Override with user-specified options
    for key, value in solver_opts.items():
        # Handle ipopt. prefix in config keys
        if key.startswith("ipopt."):
            attr_name = key[6:]  # Remove 'ipopt.' prefix
        else:
            attr_name = key

        if hasattr(options, attr_name):
            setattr(options, attr_name, value)
        else:
            log.warning(f"Unknown IPOPT option: {key} (attribute: {attr_name})")

    # Adjust options based on problem size
    num = P.get("num", {})
    K = int(num.get("K", 10))
    C = int(num.get("C", 3))

    # Estimate problem size
    n_vars = (
        K * C * 6
    )  # Rough estimate: K collocation points, C stages, 6 variables per point
    n_constraints = K * C * 4  # Rough estimate: 4 constraints per collocation point

    if n_vars > 1000 or n_constraints > 1000:
        # Large problem - use more robust settings
        options.hessian_approximation = "limited-memory"
        options.max_iter = 10000
        log.info(
            f"Large problem detected ({n_vars} vars, {n_constraints} constraints), using robust settings",
        )
    
    # Tune barrier parameter initialization based on problem scale
    # Increased mu_init values for better initial convergence
    if n_vars > 500:
        options.mu_init = 1e-2  # Increased from 1e-3 for large problems
    elif n_vars > 100:
        options.mu_init = 5e-2  # Increased from 5e-3 for medium problems
    else:
        options.mu_init = 1e-1  # Increased from 1e-2 for small problems
    
    log.debug(
        f"Barrier parameter tuning: mu_init={options.mu_init:.2e}, mu_max={options.mu_max:.2e}, "
        f"mu_strategy={options.mu_strategy}"
    )

    return options


def _setup_optimization_bounds(
    nlp: Any,
    P: dict[str, Any],
    warm_start: dict[str, Any],
) -> tuple:
    """Set up optimization bounds and initial guess."""
    if nlp is None:
        return None, None, None, None, None, None

    try:
        # Get problem dimensions
        # Handle CasADi dict format: {"x": ..., "f": ..., "g": ...}
        if isinstance(nlp, dict):
            n_vars = nlp["x"].size1()
            n_constraints = nlp["g"].size1()
        else:
            # Handle CasADi Function format
            n_vars = nlp.size1_in(0)
            n_constraints = nlp.size1_out(0)

        # Set up bounds
        lbx = -np.inf * np.ones(n_vars)
        ubx = np.inf * np.ones(n_vars)
        lbg = -np.inf * np.ones(n_constraints)
        ubg = np.inf * np.ones(n_constraints)

        # Set up initial guess
        if warm_start and "x0" in warm_start:
            x0 = np.array(warm_start["x0"])
            if len(x0) != n_vars:
                log.warning(f"Warm start x0 length {len(x0)} != problem size {n_vars}")
                x0 = _generate_physics_based_initial_guess(n_vars, P)
        else:
            x0 = _generate_physics_based_initial_guess(n_vars, P)

        # Set up parameters
        p = np.array([])  # No parameters for now

        # Apply problem-specific bounds
        _apply_problem_bounds(lbx, ubx, lbg, ubg, P)
        
        # Clamp initial guess to bounds
        x0 = _clamp_initial_guess(x0, lbx, ubx)

        return x0, lbx, ubx, lbg, ubg, p

    except Exception as e:
        log.error(f"Failed to set up optimization bounds: {e!s}")
        return None, None, None, None, None, None


def _compute_interior_point(
    lb: float,
    ub: float,
    margin: float = 0.05,
) -> float:
    """
    Compute interior point within bounds with given margin.
    
    Returns a point that is margin% inside the interval from the lower bound.
    For unbounded cases, returns a reasonable default.
    
    Args:
        lb: Lower bound (can be -inf)
        ub: Upper bound (can be inf)
        margin: Margin factor (0.05 = 5% inside from lower bound)
        
    Returns:
        Interior point value
    """
    if np.isfinite(lb) and np.isfinite(ub):
        if ub > lb:
            return lb + margin * (ub - lb)
        else:
            return lb  # Degenerate case
    elif np.isfinite(lb):
        return lb + 1.0  # Default offset from lower bound
    elif np.isfinite(ub):
        return ub - 1.0  # Default offset from upper bound
    else:
        return 0.0  # Unbounded case


def _clamp_initial_guess(
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
) -> np.ndarray:
    """
    Clamp initial guess to bounds and validate.
    
    Ensures all values in x0 are within [lbx, ubx] bounds.
    Logs warnings when values are clamped.
    
    Args:
        x0: Initial guess array
        lbx: Lower bounds
        ubx: Upper bounds
        
    Returns:
        Clamped initial guess array
    """
    x0_clamped = x0.copy()
    n_clamped = 0
    
    for i in range(len(x0)):
        original = x0[i]
        clamped = False
        
        # Check for NaN or Inf
        if not np.isfinite(original):
            log.warning(
                f"Initial guess[{i}] is not finite ({original}), setting to midpoint of bounds"
            )
            if np.isfinite(lbx[i]) and np.isfinite(ubx[i]):
                x0_clamped[i] = 0.5 * (lbx[i] + ubx[i])
            elif np.isfinite(lbx[i]):
                x0_clamped[i] = lbx[i] + 1.0
            elif np.isfinite(ubx[i]):
                x0_clamped[i] = ubx[i] - 1.0
            else:
                x0_clamped[i] = 0.0
            clamped = True
        # Clamp to lower bound
        elif np.isfinite(lbx[i]) and original < lbx[i]:
            x0_clamped[i] = lbx[i]
            clamped = True
        # Clamp to upper bound
        elif np.isfinite(ubx[i]) and original > ubx[i]:
            x0_clamped[i] = ubx[i]
            clamped = True
        
        if clamped:
            n_clamped += 1
            if n_clamped <= 10:  # Log first 10 clamped values
                log.debug(
                    f"Clamped initial guess[{i}]: {original:.6e} -> {x0_clamped[i]:.6e} "
                    f"(bounds: [{lbx[i]:.6e}, {ubx[i]:.6e}])"
                )
    
    if n_clamped > 0:
        log.info(
            f"Clamped {n_clamped}/{len(x0)} initial guess values to bounds "
            f"({100.0 * n_clamped / len(x0):.1f}%)"
        )
        if n_clamped > 10:
            log.debug(
                f"Many initial guess values ({n_clamped}) were clamped. "
                f"Consider improving initial guess generation."
            )
    
    return x0_clamped


def _generate_physics_based_initial_guess(n_vars: int, P: dict[str, Any]) -> np.ndarray:
    """
    Generate physics-based initial guess using ONLY:
    - Basic cylinder geometry (stroke, bore, clearance_volume, compression_ratio)
    - Combustion model inputs (cycle_time, initial T/P, AFR, fuel_mass, ignition_timing)
    - Thermodynamic properties (gamma, R, cp)
    
    Seeds variables at interior points (5% inside bounds) for better feasibility.
    NO motion profile assumptions - only free-piston physics and geometry.
    """
    x0 = np.zeros(n_vars)

    # Get inputs from problem parameters
    geom = P.get("geometry", {})
    thermo = P.get("thermodynamics", {})
    bounds = P.get("bounds", {})
    combustion_cfg = P.get("combustion", {})
    
    # Extract variable group bounds for interior point seeding
    # Positions
    xL_min = bounds.get("xL_min", -0.1)
    xL_max = bounds.get("xL_max", 0.1)
    xR_min = bounds.get("xR_min", 0.0)
    xR_max = bounds.get("xR_max", 0.2)
    # Velocities
    vL_min = bounds.get("vL_min", -10.0)
    vL_max = bounds.get("vL_max", 10.0)
    vR_min = bounds.get("vR_min", -10.0)
    vR_max = bounds.get("vR_max", 10.0)
    # Densities
    rho_min = bounds.get("rho_min", 0.1)
    rho_max = bounds.get("rho_max", 10.0)
    # Temperatures
    T_min = bounds.get("T_min", 200.0)
    T_max = bounds.get("T_max", 2000.0)
    # Valve areas (lower bound is 0)
    Ain_max = bounds.get("Ain_max", 0.01)
    Aex_max = bounds.get("Aex_max", 0.01)
    # Ignition timing
    ignition_bounds = combustion_cfg.get(
        "ignition_bounds_s",
        (0.0, max(combustion_cfg.get("cycle_time_s", 1.0), 1e-6)),
    )
    t_ign_min = ignition_bounds[0] if isinstance(ignition_bounds, (tuple, list)) else 0.0
    t_ign_max = ignition_bounds[1] if isinstance(ignition_bounds, (tuple, list)) else max(combustion_cfg.get("cycle_time_s", 1.0), 1e-6)
    
    # Compute interior point values for each group (5% inside bounds)
    xL_interior = _compute_interior_point(xL_min, xL_max, margin=0.05)
    xR_interior = _compute_interior_point(xR_min, xR_max, margin=0.05)
    vL_interior = _compute_interior_point(vL_min, vL_max, margin=0.05)
    vR_interior = _compute_interior_point(vR_min, vR_max, margin=0.05)
    rho_interior = _compute_interior_point(rho_min, rho_max, margin=0.05)
    T_interior = _compute_interior_point(T_min, T_max, margin=0.05)
    Ain_interior = _compute_interior_point(0.0, Ain_max, margin=0.05)
    Aex_interior = _compute_interior_point(0.0, Aex_max, margin=0.05)
    t_ign_interior = _compute_interior_point(t_ign_min, t_ign_max, margin=0.05)

    # Extract geometry inputs
    stroke = geom.get("stroke", 0.1)  # m
    bore = geom.get("bore", 0.1)  # m
    compression_ratio = geom.get("compression_ratio", 10.0)
    clearance_volume = geom.get("clearance_volume", 1e-4)  # m^3
    gap_min = bounds.get("gap_min", 0.0008)  # Minimum gap constraint

    # Extract thermodynamic properties
    gamma = thermo.get("gamma", 1.4)
    R = thermo.get("R", 287.0)  # J/(kg K)
    cp = thermo.get("cp", 1005.0)  # J/(kg K)
    cv = cp / gamma

    # Extract combustion model inputs
    cycle_time = combustion_cfg.get("cycle_time_s", 1.0)  # Cycle time from user
    p_initial = combustion_cfg.get("initial_pressure_Pa", 1e5)  # Initial pressure from user
    T_initial = combustion_cfg.get("initial_temperature_K", 300.0)  # Initial temperature from user
    ignition_timing = combustion_cfg.get("ignition_initial_s", None)  # Ignition timing from user
    
    # Calculate initial gas density from ideal gas law
    rho_initial = p_initial / (R * T_initial)  # kg/m^3

    # Calculate cylinder geometry
    A_piston = np.pi * (bore / 2.0) ** 2  # Piston area
    V_max = clearance_volume + A_piston * stroke  # Maximum volume (at TDC)
    V_min = clearance_volume  # Minimum volume (at BDC)
    
    # Calculate volumes at compression ratio
    V_compressed = V_max / compression_ratio  # Volume at end of compression

    # Estimate thermodynamic states through cycle using isentropic relations
    # Compression phase: isentropic compression from initial state
    p_compressed = p_initial * ((V_max / V_compressed) ** gamma)
    T_compressed = T_initial * ((V_max / V_compressed) ** (gamma - 1))
    rho_compressed = p_compressed / (R * T_compressed)

    # Expansion phase: isentropic expansion (simplified - no combustion heat addition yet)
    # At maximum expansion (back to V_max), pressure and temperature drop
    p_expanded = p_compressed * ((V_compressed / V_max) ** gamma)
    T_expanded = T_compressed * ((V_compressed / V_max) ** (gamma - 1))
    rho_expanded = p_expanded / (R * T_expanded)

    # Variable ordering per collocation point: x_L, v_L, x_R, v_R, rho, T
    n_per_point = 6
    n_points = n_vars // n_per_point

    # Calculate time grid (assuming uniform spacing)
    dt = cycle_time / max(1, n_points - 1) if n_points > 1 else cycle_time

    # Estimate average piston velocity based on cycle time and stroke
    # For a free-piston, pistons move in opposite directions during compression/expansion
    # Simple kinematic estimate: v_avg ≈ stroke / (cycle_time / 2) for one direction
    v_avg = stroke / (cycle_time / 2) if cycle_time > 0 else 0.0

    for i in range(n_points):
        idx = i * n_per_point
        t = i * dt  # Time at this collocation point
        phase = (t / cycle_time) if cycle_time > 0 else 0.0  # Normalized phase [0, 1]

        # Estimate piston positions based on simple kinematic cycle
        # Free-piston: pistons compress (move toward center) then expand (move apart)
        # Use simple sinusoidal approximation based on cycle time
        if phase < 0.5:
            # Compression phase: pistons moving toward center
            compression_phase = phase * 2.0  # [0, 1] over compression
            x_L_center = stroke * 0.5  # Center position
            x_R_center = x_L_center + gap_min  # Right piston center (with gap)
            # Move pistons toward center during compression
            x_L = x_L_center + stroke * 0.4 * (1.0 - compression_phase)
            x_R = x_R_center - stroke * 0.4 * (1.0 - compression_phase)
        else:
            # Expansion phase: pistons moving apart
            expansion_phase = (phase - 0.5) * 2.0  # [0, 1] over expansion
            x_L_center = stroke * 0.5
            x_R_center = x_L_center + gap_min
            # Move pistons apart during expansion
            x_L = x_L_center - stroke * 0.4 * expansion_phase
            x_R = x_R_center + stroke * 0.4 * expansion_phase

        # Ensure minimum gap constraint
        if x_R - x_L < gap_min:
            x_R = x_L + gap_min

        # Estimate velocities: simple kinematic approximation
        # Velocities change sign during compression/expansion
        if phase < 0.5:
            # Compression: v_L positive (moving right), v_R negative (moving left)
            v_L = v_avg * (1.0 - compression_phase * 2.0)
            v_R = -v_avg * (1.0 - compression_phase * 2.0)
        else:
            # Expansion: v_L negative (moving left), v_R positive (moving right)
            v_L = -v_avg * (expansion_phase * 2.0 - 1.0)
            v_R = v_avg * (expansion_phase * 2.0 - 1.0)

        # Estimate gas state based on phase and thermodynamic relations
        if phase < 0.25:
            # Early compression: close to initial conditions
            rho = rho_initial + (rho_compressed - rho_initial) * (phase / 0.25)
            T = T_initial + (T_compressed - T_initial) * (phase / 0.25)
        elif phase < 0.5:
            # Late compression: approaching compressed state
            compression_progress = (phase - 0.25) / 0.25
            rho = rho_compressed * (1.0 - compression_progress * 0.1)  # Slight variation
            T = T_compressed * (1.0 - compression_progress * 0.1)
        elif phase < 0.75:
            # Early expansion: high pressure/temperature
            expansion_progress = (phase - 0.5) / 0.25
            rho = rho_compressed * (1.0 - expansion_progress * 0.5)
            T = T_compressed * (1.0 - expansion_progress * 0.3)
        else:
            # Late expansion: approaching initial conditions
            expansion_progress = (phase - 0.75) / 0.25
            rho = rho_expanded + (rho_initial - rho_expanded) * expansion_progress
            T = T_expanded + (T_initial - T_expanded) * expansion_progress

        # Apply to variables with NaN guards and interior point fallbacks
        if idx < n_vars:
            x0[idx] = x_L if np.isfinite(x_L) else xL_interior
        if idx + 1 < n_vars:
            x0[idx + 1] = v_L if np.isfinite(v_L) else vL_interior
        if idx + 2 < n_vars:
            x0[idx + 2] = x_R if np.isfinite(x_R) else xR_interior
        if idx + 3 < n_vars:
            x0[idx + 3] = v_R if np.isfinite(v_R) else vR_interior
        if idx + 4 < n_vars:
            x0[idx + 4] = rho if np.isfinite(rho) else rho_interior
        if idx + 5 < n_vars:
            x0[idx + 5] = T if np.isfinite(T) else T_interior

    # Final safety check: ensure all gaps satisfy minimum constraint and replace NaN/Inf
    for i in range(n_points):
        idx = i * n_per_point
        if idx + 2 < n_vars:  # Both x_L and x_R are available
            # Replace NaN/Inf with interior points
            if not np.isfinite(x0[idx]):
                x0[idx] = xL_interior
            if not np.isfinite(x0[idx + 2]):
                x0[idx + 2] = xR_interior
            gap = x0[idx + 2] - x0[idx]  # x_R - x_L
            if gap < gap_min:
                x0[idx + 2] = x0[idx] + gap_min
                log.debug(
                    f"Adjusted piston gap at point {i}: gap={gap:.6f} -> {gap_min:.6f}",
                )
    
    # Final NaN guard: replace any remaining NaN/Inf with interior points
    for i in range(n_vars):
        if not np.isfinite(x0[i]):
            # Determine which group this variable belongs to based on position
            var_idx_in_group = i % n_per_point
            if var_idx_in_group == 0:  # x_L
                x0[i] = xL_interior
            elif var_idx_in_group == 1:  # v_L
                x0[i] = vL_interior
            elif var_idx_in_group == 2:  # x_R
                x0[i] = xR_interior
            elif var_idx_in_group == 3:  # v_R
                x0[i] = vR_interior
            elif var_idx_in_group == 4:  # rho
                x0[i] = rho_interior
            elif var_idx_in_group == 5:  # T
                x0[i] = T_interior
            else:
                x0[i] = 0.0  # Fallback

    log.info(
        f"Generated physics-based initial guess for {n_vars} variables using "
        f"geometry (stroke={stroke*1000:.1f}mm, bore={bore*1000:.1f}mm, CR={compression_ratio:.1f}) "
        f"and combustion inputs (cycle_time={cycle_time:.3f}s, T_init={T_initial:.0f}K, p_init={p_initial:.0f}Pa)"
    )

    return x0


def _compute_variable_scaling(
    lbx: np.ndarray,
    ubx: np.ndarray,
    x0: np.ndarray | None = None,
    variable_groups: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """
    Compute variable scaling factors with standardized per-group references
    and clamped ratios to keep scales within 10² range.
    
    Uses group-based per-unit scales combined with value-based scaling:
    1. Apply per-unit reference scales based on variable groups (positions, velocities, etc.)
    2. Normalize each group to cap ratio ≤ 10²
    3. Apply value-based scaling with clamped adjustments
    4. Final global normalization to ensure all scales within 10² range
    
    This ensures scaling accounts for both physical units and typical operating values
    while preventing extreme ratios that lead to poor Jacobian conditioning.
    
    Args:
        lbx: Lower bounds on variables
        ubx: Upper bounds on variables
        x0: Initial guess values (optional, falls back to bounds-only if None)
        variable_groups: Dict mapping group names to variable indices (optional)
        
    Returns:
        Array of scale factors (one per variable)
    """
    n_vars = len(lbx)
    scale = np.ones(n_vars)
    
    # Standardized per-group reference units with physical anchors
    unit_references = {
        "positions": 0.05,      # meters → scale ≈ 1/0.05 = 20 (typical position ~50mm)
        "velocities": 10.0,     # m/s → scale ≈ 1/10 = 0.1 (typical velocity ~10 m/s)
        "densities": 1.0,       # kg/m³ (already reasonable)
        "temperatures": 1000.0, # K (normalize to 0.001-2.0 range)
        "pressures": 1e6,       # MPa (normalize to 0.01-10 range)
        "valve_areas": 1e-4,    # m² → scale ≈ 1/1e-4 = 1e4 (typical area ~0.1 mm²)
        "ignition": 1.0,        # seconds (already in base units)
        "penalties": 1.0,       # dimensionless
    }
    
    # Positive-only groups (can use sqrt transform)
    positive_only_groups = {"valve_areas", "densities"}
    
    # Map variable indices to their groups
    var_to_group = {}
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if group_name in unit_references:
                for idx in indices:
                    if 0 <= idx < n_vars:
                        var_to_group[idx] = group_name
    
    # First pass: apply unit-based scaling
    for i in range(n_vars):
        group = var_to_group.get(i)
        if group and group in unit_references:
            ref = unit_references[group]
            if group in positive_only_groups:
                # Use sqrt transform for positive-only variables
                scale[i] = 1.0 / np.sqrt(ref)
            else:
                scale[i] = 1.0 / ref
    
    # Normalize each group to cap ratio ≤ 10²
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if indices and len(indices) > 0:
                group_indices = np.array([i for i in indices if 0 <= i < n_vars])
                if len(group_indices) > 0:
                    group_scales = scale[group_indices]
                    group_min = group_scales.min()
                    group_max = group_scales.max()
                    
                    # Cap ratio to 10²
                    if group_min > 1e-10 and group_max / group_min > 1e2:
                        # Normalize to median, then clamp to ±1 log unit
                        group_median = np.median(group_scales)
                        for idx in group_indices:
                            # Clamp to [0.1 * median, 10 * median]
                            scale[idx] = np.clip(scale[idx], 0.1 * group_median, 10.0 * group_median)
    
    # Second pass: apply value-based scaling with clamped adjustments
    for i in range(n_vars):
        lb = lbx[i]
        ub = ubx[i]
        
        # Skip if unbounded
        if lb == -np.inf and ub == np.inf:
            continue  # Keep unit-based scale
        
        # Compute magnitude from bounds
        if lb == -np.inf:
            magnitude = abs(ub)
        elif ub == np.inf:
            magnitude = abs(lb)
        else:
            magnitude = max(abs(lb), abs(ub))
        
        # Incorporate initial guess value if available
        if x0 is not None and i < len(x0) and np.isfinite(x0[i]):
            magnitude = max(magnitude, abs(x0[i]))
        
        # Apply value-based adjustment with clamping
        if magnitude > 1e-6:
            adjustment = scale[i] / magnitude
            # Clamp adjustment to [0.1, 10] to prevent extreme ratios
            adjustment = np.clip(adjustment, 0.1, 10.0)
            scale[i] = adjustment
    
    # Final normalization: ensure all scales are within 10² range globally
    scale_min = scale[scale > 1e-10].min() if (scale > 1e-10).any() else 1e-10
    scale_max = scale.max()
    if scale_min > 1e-10 and scale_max / scale_min > 1e2:
        scale_median = np.median(scale[scale > 1e-10])
        for i in range(n_vars):
            if scale[i] > 1e-10:
                scale[i] = np.clip(scale[i], 0.1 * scale_median, 10.0 * scale_median)
    
    return scale


def _compute_constraint_scaling(
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
) -> np.ndarray:
    """
    Compute constraint scaling factors to normalize constraints to O(1) magnitude.
    
    For each constraint, uses max(|lb|, |ub|) to compute a scale factor s_g
    such that the scaled constraint g_scaled = s_g * g has magnitude ~O(1).
    This ensures the constraint Jacobian is properly scaled relative to variables.
    
    Args:
        lbg: Lower bounds on constraints (can be None)
        ubg: Upper bounds on constraints (can be None)
        
    Returns:
        Array of constraint scale factors (one per constraint)
    """
    if lbg is None and ubg is None:
        return np.array([])
    
    if lbg is None:
        n_cons = len(ubg) if ubg is not None else 0
        lbg = -np.inf * np.ones(n_cons)
    if ubg is None:
        n_cons = len(lbg) if lbg is not None else 0
        ubg = np.inf * np.ones(n_cons)
    
    n_cons = len(lbg)
    scale_g = np.ones(n_cons)
    
    for i in range(n_cons):
        lb = lbg[i]
        ub = ubg[i]
        
        # Skip if unbounded
        if lb == -np.inf and ub == np.inf:
            scale_g[i] = 1.0
            continue
        
        # Compute magnitude from bounds
        if lb == -np.inf:
            magnitude = abs(ub)
        elif ub == np.inf:
            magnitude = abs(lb)
        else:
            magnitude = max(abs(lb), abs(ub))
        
        # For equality constraints (lb == ub), use the absolute value
        if lb == ub and np.isfinite(lb):
            magnitude = abs(lb)
        
        # Compute scale factor: s_g = 1 / magnitude, but avoid division by very small numbers
        if magnitude > 1e-6:
            scale_g[i] = 1.0 / magnitude
        else:
            scale_g[i] = 1.0
    
    return scale_g


def _verify_scaling_quality(
    nlp: Any,
    x0: np.ndarray,
    scale: np.ndarray,
    scale_g: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    reporter: StructuredReporter | None = None,
) -> None:
    """
    Verify scaling quality by checking constraint Jacobian at initial guess.
    
    Computes the constraint Jacobian and checks element magnitudes to ensure
    proper scaling. Logs warnings if scaling appears insufficient.
    
    Args:
        nlp: CasADi NLP dict
        x0: Initial guess (unscaled)
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
    """
    try:
        import casadi as ca
        
        if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
            return
        
        g_expr = nlp["g"]
        x_sym = nlp["x"]
        
        if g_expr is None or g_expr.numel() == 0:
            return
        
        # Create Jacobian function
        try:
            jac_g_expr = ca.jacobian(g_expr, x_sym)
            jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
            jac_g0 = jac_g_func(x0)
            jac_g0_arr = np.array(jac_g0)
        except Exception as e:
            if reporter:
                reporter.debug(f"Could not evaluate constraint Jacobian for scaling verification: {e}")
            else:
                log.debug(f"Could not evaluate constraint Jacobian for scaling verification: {e}")
            return
        
        # Apply scaling to Jacobian: J_scaled = scale_g * J * (1/scale)
        # Check if scaled Jacobian elements are O(1)
        if len(scale_g) > 0 and jac_g0_arr.size > 0:
            # Reshape scale_g for broadcasting
            scale_g_col = scale_g.reshape(-1, 1) if len(scale_g.shape) == 1 else scale_g
            scale_row = scale.reshape(1, -1) if len(scale.shape) == 1 else scale
            
            # Scaled Jacobian: scale_g * J * (1/scale) for each element
            # For element-wise: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
            jac_g0_scaled = jac_g0_arr.copy()
            for i in range(min(jac_g0_arr.shape[0], len(scale_g))):
                for j in range(min(jac_g0_arr.shape[1], len(scale))):
                    if scale[j] > 1e-10:  # Avoid division by zero
                        jac_g0_scaled[i, j] = scale_g[i] * jac_g0_arr[i, j] / scale[j]
            
            # Check magnitude statistics
            jac_mag = np.abs(jac_g0_scaled)
            jac_max = jac_mag.max()
            jac_mean = jac_mag.mean()
            jac_min = jac_mag[jac_mag > 0].min() if (jac_mag > 0).any() else 0.0
            
            # Log diagnostics
            msg = f"Scaled Jacobian statistics: min={jac_min:.3e}, mean={jac_mean:.3e}, max={jac_max:.3e}"
            if reporter:
                reporter.debug(msg)
            else:
                log.debug(msg)
            
            # Warn if scaling appears insufficient
            if jac_max > 1e3:
                warn_msg = (
                    f"Scaled Jacobian has large elements (max={jac_max:.3e}). "
                    f"Scaling may be insufficient. Consider adjusting scale factors."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)
            if jac_min > 0 and jac_max / jac_min > 1e6:
                warn_msg = (
                    f"Scaled Jacobian has large condition number (max/min={jac_max/jac_min:.3e}). "
                    f"Consider more uniform scaling."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)
            
            # Check for very small elements that might cause numerical issues
            very_small = (jac_mag > 0) & (jac_mag < 1e-10)
            if very_small.sum() > 0:
                debug_msg = (
                    f"Scaled Jacobian has {very_small.sum()} very small elements (<1e-10). "
                    f"This may indicate over-scaling."
                )
                if reporter:
                    reporter.debug(debug_msg)
                else:
                    log.debug(debug_msg)
        
    except Exception as e:
        if reporter:
            reporter.debug(f"Scaling verification failed: {e}")
        else:
            log.debug(f"Scaling verification failed: {e}")


def _compute_objective_scaling(
    nlp: Any,
    x0: np.ndarray,
) -> float:
    """
    Compute objective scaling factor from actual objective evaluation at initial guess.
    
    Evaluates objective at x0 and computes scale factor:
    scale_f = 1.0 / max(|f0|, 1.0)
    
    This ensures objective is O(1) magnitude to match variable/constraint scaling.
    
    Args:
        nlp: CasADi NLP dict with 'x' and 'f' keys
        x0: Initial guess for variables
        
    Returns:
        Objective scale factor (scalar)
    """
    try:
        import casadi as ca
        
        if not isinstance(nlp, dict) or "f" not in nlp or "x" not in nlp:
            log.warning("NLP does not support objective evaluation, skipping objective scaling")
            return 1.0
        
        f_expr = nlp["f"]
        x_sym = nlp["x"]
        
        # Create function to evaluate objective
        try:
            f_func = ca.Function("f_func", [x_sym], [f_expr])
            f0 = f_func(x0)
            f0_val = float(f0) if hasattr(f0, '__float__') else float(np.array(f0).item())
        except Exception as e:
            log.warning(f"Failed to evaluate objective at initial guess: {e}, skipping objective scaling")
            return 1.0
        
        # Compute scale factor: s_f = 1.0 / max(|f0|, 1.0)
        magnitude = max(abs(f0_val), 1.0)
        scale_f = 1.0 / magnitude
        
        log.debug(f"Objective scaling: f0={f0_val:.6e}, scale_f={scale_f:.6e}")
        return scale_f
        
    except Exception as e:
        log.warning(f"Error in objective scaling: {e}, skipping objective scaling")
        return 1.0


def _compute_constraint_scaling_from_evaluation(
    nlp: Any,
    x0: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute constraint scaling factors from actual constraint evaluation and Jacobian at initial guess.
    
    Uses robust scaling that accounts for both constraint values and Jacobian sensitivity:
    1. Computes constraint Jacobian at initial guess
    2. For each constraint, uses max(|g0[i]|, typical_jacobian_row_magnitude[i])
    3. For equality constraints (lb == ub): uses reference scale based on typical residual magnitude
    4. Caps all scale factors to prevent extreme ratios (1e-3 to 1e3 range)
    5. Uses percentiles for more robust scaling
    
    Args:
        nlp: CasADi NLP dict with 'x' and 'g' keys
        x0: Initial guess for variables
        lbg: Lower bounds on constraints (can be None)
        ubg: Upper bounds on constraints (can be None)
        scale: Variable scaling factors (used to compute scaled Jacobian magnitude)
        
    Returns:
        Array of constraint scale factors (one per constraint)
    """
    try:
        import casadi as ca
        
        # Fall back to bounds-based scaling if evaluation fails
        bounds_scale = _compute_constraint_scaling(lbg, ubg)
        
        if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
            log.warning("NLP does not support constraint evaluation, using bounds-based scaling")
            return bounds_scale
        
        g_expr = nlp["g"]
        x_sym = nlp["x"]
        
        # Handle empty constraints
        if g_expr is None or g_expr.numel() == 0:
            return bounds_scale
        
        # Create function to evaluate constraints
        try:
            g_func = ca.Function("g_func", [x_sym], [g_expr])
            g0 = g_func(x0)
            g0_arr = np.array(g0).flatten()
        except Exception as e:
            log.warning(f"Failed to evaluate constraints at initial guess: {e}, using bounds-based scaling")
            return bounds_scale
        
        # Compute constraint Jacobian to account for sensitivity
        jac_row_norms = None
        try:
            jac_g_expr = ca.jacobian(g_expr, x_sym)
            jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
            jac_g0 = jac_g_func(x0)
            jac_g0_arr = np.array(jac_g0)
            
            # Compute row norms accounting for variable scaling
            # The scaled Jacobian is: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
            # For row norm calculation, we need: ||row_i|| = sqrt(sum_j (J[i,j] / scale[j])^2)
            # This represents the sensitivity of constraint i to scaled variables
            if scale is not None and len(scale) > 0:
                # Compute scaled Jacobian row norms: sqrt(sum_j (J[i,j] / scale[j])^2)
                jac_row_norms = np.zeros(jac_g0_arr.shape[0])
                for i in range(jac_g0_arr.shape[0]):
                    row_norm_sq = 0.0
                    for j in range(min(jac_g0_arr.shape[1], len(scale))):
                        if scale[j] > 1e-10:
                            row_norm_sq += (jac_g0_arr[i, j] / scale[j]) ** 2
                    jac_row_norms[i] = np.sqrt(row_norm_sq)
            else:
                # Without variable scales, use unscaled row norms
                jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
            
            # Use median of non-zero row norms as reference
            non_zero_norms = jac_row_norms[jac_row_norms > 1e-10]
            if len(non_zero_norms) > 0:
                ref_jac_norm = np.median(non_zero_norms)
            else:
                ref_jac_norm = 1.0
        except Exception as e:
            log.debug(f"Could not compute constraint Jacobian: {e}, using constraint values only")
            jac_row_norms = None
            ref_jac_norm = 1.0
        
        # Combine bounds and actual constraint values
        if lbg is None and ubg is None:
            n_cons = len(g0_arr)
            lbg = -np.inf * np.ones(n_cons)
            ubg = np.inf * np.ones(n_cons)
        
        if lbg is None:
            n_cons = len(g0_arr)
            lbg = -np.inf * np.ones(n_cons)
        if ubg is None:
            n_cons = len(g0_arr)
            ubg = np.inf * np.ones(n_cons)
        
        n_cons = len(g0_arr)
        scale_g = np.ones(n_cons)
        
        # Identify equality constraints (typically collocation residuals)
        equality_mask = np.zeros(n_cons, dtype=bool)
        for i in range(n_cons):
            lb = lbg[i] if i < len(lbg) else -np.inf
            ub = ubg[i] if i < len(ubg) else np.inf
            if lb == ub and np.isfinite(lb):
                equality_mask[i] = True
        
        # Compute reference scale for equality constraints based on typical residual magnitude
        # Use median of absolute values of equality constraint residuals (excluding zeros)
        equality_g0 = g0_arr[equality_mask]
        if len(equality_g0) > 0:
            abs_equality = np.abs(equality_g0)
            # Use median of non-zero values, or 1e-3 as default
            non_zero = abs_equality[abs_equality > 1e-10]
            if len(non_zero) > 0:
                ref_magnitude = np.median(non_zero)
            else:
                ref_magnitude = 1e-3  # Default reference for zero residuals
        else:
            ref_magnitude = 1e-3
        
        # Compute scale factors with robust handling
        for i in range(n_cons):
            lb = lbg[i] if i < len(lbg) else -np.inf
            ub = ubg[i] if i < len(ubg) else np.inf
            g_val = g0_arr[i] if i < len(g0_arr) else 0.0
            
            # For equality constraints, use reference magnitude
            if equality_mask[i]:
                magnitude = ref_magnitude
                # Detect large-magnitude combustion residuals (>1e4) and apply aggressive scaling
                if np.isfinite(g_val) and abs(g_val) > 1e4:
                    # For large combustion residuals, use more aggressive normalization
                    # Scale down by normalizing to typical residual magnitude
                    magnitude = max(ref_magnitude, abs(g_val) / 1e6)  # Normalize large residuals
            else:
                # For inequality constraints, compute from bounds and values
                if lb == -np.inf and ub == np.inf:
                    magnitude = abs(g_val) if np.isfinite(g_val) else 1.0
                elif lb == -np.inf:
                    magnitude = abs(ub)
                elif ub == np.inf:
                    magnitude = abs(lb)
                else:
                    magnitude = max(abs(lb), abs(ub))
                
                # Incorporate actual constraint value (but don't let tiny values dominate)
                if np.isfinite(g_val):
                    # Use max of bounds and value, but cap value influence
                    magnitude = max(magnitude, min(abs(g_val), max(abs(lb), abs(ub)) * 10))
                    # Detect large-magnitude combustion/energy constraints (>1e4) and apply aggressive scaling
                    if abs(g_val) > 1e4 or magnitude > 1e4:
                        # For large energy/combustion constraints, normalize more aggressively
                        # This helps bring Jacobian entries from O(1e6-1e8) to O(1)
                        magnitude = max(magnitude / 1e6, 1e-3)  # Normalize large constraints
            
            # Incorporate Jacobian row norm to account for constraint sensitivity
            # If constraint i has a large Jacobian row norm, it needs a smaller scale factor
            # to keep the scaled Jacobian elements O(1)
            if jac_row_norms is not None and i < len(jac_row_norms):
                jac_norm_i = jac_row_norms[i]
                if jac_norm_i > 1e-10:
                    # Identify penalty constraints (rows with very large Jacobian norms)
                    # These are likely clearance penalties with 1e6 stiffness
                    penalty_threshold = ref_jac_norm * 1e3  # 1000x larger than typical
                    if jac_norm_i > penalty_threshold:
                        # For penalty constraints, use a more conservative scale
                        # Scale down the penalty contribution to keep Jacobian entries O(1)
                        magnitude = magnitude * 1e3  # Scale down penalty contributions
                    else:
                        # Use geometric mean of constraint magnitude and Jacobian norm
                        # This balances constraint value scaling with sensitivity scaling
                        magnitude = np.sqrt(magnitude * jac_norm_i) if magnitude > 1e-10 else jac_norm_i
                        # But don't let Jacobian dominate if constraint value is much larger
                        magnitude = max(magnitude, jac_norm_i * 0.1)
            
            # Compute scale factor with capping to prevent extreme ratios
            if magnitude > 1e-6:
                scale_g[i] = 1.0 / magnitude
            else:
                scale_g[i] = 1.0
            
            # Cap scale factors to tighter range to limit condition number
            # Range 1e-3 to 1e3 gives max condition number of 1e6
            scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
        
        # Log scaling statistics before normalization
        scale_g_pre = scale_g.copy()
        log.debug(
            f"Constraint scaling (pre-normalization): range=[{scale_g_pre.min():.3e}, {scale_g_pre.max():.3e}], "
            f"mean={scale_g_pre.mean():.3e}, equality_ref={ref_magnitude:.3e}, "
            f"n_equality={equality_mask.sum()}/{n_cons}"
        )
        
        # Normalize constraint scales relative to variable scales
        # Ensure constraint scales stay within reasonable range relative to variable scales
        if scale is not None and len(scale) > 0:
            # Get typical variable scale magnitude
            scale_median = np.median(scale[scale > 1e-10])
            
            # Normalize constraint scales relative to variable scales
            # Constraint scales should be roughly comparable to variable scales
            scale_g_median = np.median(scale_g[scale_g > 1e-10])
            if scale_g_median > 1e-10 and scale_median > 1e-10:
                # Adjust constraint scales to be comparable to variable scales
                # Limit adjustment to prevent extreme changes that break feasibility
                scale_ratio = scale_median / scale_g_median
                scale_ratio = np.clip(scale_ratio, 0.1, 10.0)
                scale_g = scale_g * scale_ratio
        
        # Normalize scale factors to reduce extreme ratios
        # Use percentile-based normalization to prevent outliers
        scale_g_log = np.log10(np.maximum(scale_g, 1e-10))
        p25 = np.percentile(scale_g_log, 25)
        p75 = np.percentile(scale_g_log, 75)
        iqr = p75 - p25
        
        # Clip outliers (more than 2 IQRs from median for tighter range)
        median_log = np.median(scale_g_log)
        # Use tighter bounds: 2 IQRs instead of 3, and cap to ±2 log units from median
        lower_bound = max(median_log - 2 * iqr if iqr > 0 else median_log - 2, median_log - 2.0)
        upper_bound = min(median_log + 2 * iqr if iqr > 0 else median_log + 2, median_log + 2.0)
        
        # Ensure bounds stay within reasonable range (log10 scale: -3 to 3, i.e., 1e-3 to 1e3)
        lower_bound = max(lower_bound, -3.0)  # 1e-3
        upper_bound = min(upper_bound, 3.0)   # 1e3
        
        n_clipped = 0
        for i in range(n_cons):
            if scale_g_log[i] < lower_bound:
                scale_g[i] = 10.0 ** lower_bound
                n_clipped += 1
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = 10.0 ** upper_bound
                n_clipped += 1
        
        # Apply clamping strategy similar to variables: ensure constraint scales stay within [0.1×median, 10×median]
        scale_g_median = np.median(scale_g[scale_g > 1e-10])
        if scale_g_median > 1e-10:
            for i in range(n_cons):
                if scale_g[i] > 1e-10:
                    scale_g[i] = np.clip(scale_g[i], 0.1 * scale_g_median, 10.0 * scale_g_median)
        
        log.info(
            f"Constraint scaling (post-normalization): range=[{scale_g.min():.3e}, {scale_g.max():.3e}], "
            f"mean={scale_g.mean():.3e}, clipped={n_clipped}/{n_cons} outliers"
        )
        
        return scale_g
        
    except Exception as e:
        log.warning(f"Error in constraint evaluation-based scaling: {e}, using bounds-based scaling")
        return _compute_constraint_scaling(lbg, ubg)


def _apply_problem_bounds(
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray,
    ubg: np.ndarray,
    P: dict[str, Any],
) -> None:
    """Apply problem-specific bounds."""
    # Get problem parameters
    geom = P.get("geom", {})
    constraints = P.get("constraints", {})

    # Piston position bounds
    x_L_min = constraints.get("x_L_min", 0.01)
    x_L_max = constraints.get("x_L_max", 0.1)
    x_R_min = constraints.get("x_R_min", 0.1)
    x_R_max = constraints.get("x_R_max", 0.2)

    # Piston velocity bounds
    v_max = constraints.get("v_max", 10.0)

    # Gas pressure bounds
    p_min = constraints.get("p_min", 1e3)
    p_max = constraints.get("p_max", 1e7)

    # Gas temperature bounds
    T_min = constraints.get("T_min", 200.0)
    T_max = constraints.get("T_max", 3000.0)

    # Apply bounds to variables (assuming specific ordering)
    # This is a simplified version - in practice, you'd need to know the exact variable ordering
    n_vars = len(lbx)
    n_per_point = 6  # x_L, v_L, x_R, v_R, rho, T

    for i in range(0, n_vars, n_per_point):
        if i < n_vars:
            lbx[i] = x_L_min  # x_L
            ubx[i] = x_L_max
        if i + 1 < n_vars:
            lbx[i + 1] = -v_max  # v_L
            ubx[i + 1] = v_max
        if i + 2 < n_vars:
            lbx[i + 2] = x_R_min  # x_R
            ubx[i + 2] = x_R_max
        if i + 3 < n_vars:
            lbx[i + 3] = -v_max  # v_R
            ubx[i + 3] = v_max
        if i + 4 < n_vars:
            lbx[i + 4] = 0.1  # rho (density)
            ubx[i + 4] = 100.0
        if i + 5 < n_vars:
            lbx[i + 5] = T_min  # T (temperature)
            ubx[i + 5] = T_max


def solve_cycle_robust(P: dict[str, Any]) -> dict[str, Any]:
    """
    Solve OP engine cycle with robust IPOPT settings.

    This function uses more conservative IPOPT settings for difficult problems.

    Args:
        P: Problem parameters dictionary

    Returns:
        Solution object with optimization results
    """
    # Set robust problem type
    P_robust = P.copy()
    P_robust["problem_type"] = "robust"

    return solve_cycle(P_robust)


def solve_cycle_with_warm_start(
    P: dict[str, Any],
    x0: np.ndarray,
) -> dict[str, Any]:
    """
    Solve OP engine cycle with warm start.

    Args:
        P: Problem parameters dictionary
        x0: Initial guess for optimization variables

    Returns:
        Solution object with optimization results
    """
    # Add warm start information
    P_warm = P.copy()
    P_warm["warm_start"] = {"x0": x0.tolist()}

    return solve_cycle(P_warm)


def solve_cycle_with_refinement(
    P: dict[str, Any],
    refinement_strategy: str = "adaptive",
) -> dict[str, Any]:
    """
    Solve cycle with 0D to 1D refinement switching.

    Args:
        P: Problem parameters
        refinement_strategy: Refinement strategy ("adaptive", "fixed", "error_based")

    Returns:
        Solution dictionary
    """
    log.info(f"Starting cycle solve with {refinement_strategy} refinement strategy")

    # Initial 0D solve
    P_0d = P.copy()
    P_0d["model_type"] = "0d"
    P_0d["num"] = P_0d.get("num", {})
    P_0d["num"]["K"] = P_0d["num"].get("K_0d", 10)

    log.info("Solving with 0D model...")
    result_0d = solve_cycle(P_0d)

    if not result_0d["success"]:
        log.warning("0D solve failed, trying 1D directly")
        return solve_cycle(P)

    # Check if refinement is needed
    if refinement_strategy == "fixed":
        # Always refine to 1D
        refine = True
    elif refinement_strategy == "error_based":
        # Refine based on error estimates
        refine = _should_refine_error_based(result_0d, P)
    else:  # adaptive
        # Refine based on problem characteristics
        refine = _should_refine_adaptive(result_0d, P)

    if not refine:
        log.info("0D solution is sufficient, no refinement needed")
        return result_0d

    # Refine to 1D
    log.info("Refining to 1D model...")
    P_1d = P.copy()
    P_1d["model_type"] = "1d"
    P_1d["num"] = P_1d.get("num", {})
    P_1d["num"]["K"] = P_1d["num"].get("K_1d", 30)

    # Use 0D solution as warm start
    warm_start = _create_warm_start_from_0d(result_0d, P_1d)
    P_1d["warm_start"] = warm_start

    result_1d = solve_cycle(P_1d)

    if result_1d["success"]:
        log.info("1D refinement successful")
        return result_1d
    log.warning("1D refinement failed, returning 0D solution")
    return result_0d


def _should_refine_error_based(result_0d: dict[str, Any], P: dict[str, Any]) -> bool:
    """Determine if refinement is needed based on error estimates."""
    # Check convergence criteria
    if result_0d.get("kkt_error", float("inf")) > 1e-4:
        return True

    # Check objective function value
    f_opt = result_0d.get("f_opt", float("inf"))
    if f_opt > 1e6:  # High objective value might indicate poor solution
        return True

    # Check problem size
    K = P.get("num", {}).get("K", 10)
    if K < 20:  # Small problem might benefit from refinement
        return True

    return False


def _should_refine_adaptive(result_0d: dict[str, Any], P: dict[str, Any]) -> bool:
    """Determine if refinement is needed based on problem characteristics."""
    # Check problem complexity
    if P.get("complex_geometry", False):
        return True

    # Check if high accuracy is required
    if P.get("high_accuracy", False):
        return True

    # Check if 1D effects are important
    if P.get("1d_effects_important", False):
        return True

    # Check solution quality
    if result_0d.get("kkt_error", float("inf")) > 1e-5:
        return True

    return False


def _create_warm_start_from_0d(
    result_0d: dict[str, Any],
    P_1d: dict[str, Any],
) -> dict[str, Any]:
    """Create warm start for 1D solve from 0D solution."""
    if not result_0d["success"] or result_0d["x_opt"] is None:
        return {}

    x_0d = result_0d["x_opt"]
    n_0d = len(x_0d)
    K_1d = P_1d.get("num", {}).get("K", 30)
    C = P_1d.get("num", {}).get("C", 3)
    n_1d = K_1d * C * 6  # Assuming 6 variables per collocation point

    # Interpolate 0D solution to 1D grid
    if n_1d > n_0d:
        # Upsample using linear interpolation
        x_1d = _interpolate_solution(x_0d, n_1d)
    else:
        # Downsample using averaging
        x_1d = _downsample_solution(x_0d, n_1d)

    return {
        "x0": x_1d,
        "lambda0": result_0d.get("lambda_opt", []),
        "mu0": result_0d.get("mu_opt", []),
    }


def _interpolate_solution(x_0d: List[float], n_1d: int) -> List[float]:
    """Interpolate solution from 0D to 1D grid."""
    x_0d_array = np.array(x_0d)
    n_0d = len(x_0d_array)

    # Create interpolation points
    x_0d_indices = np.linspace(0, n_0d - 1, n_0d)
    x_1d_indices = np.linspace(0, n_0d - 1, n_1d)

    # Linear interpolation
    x_1d = np.interp(x_1d_indices, x_0d_indices, x_0d_array)

    return x_1d.tolist()


def _downsample_solution(x_0d: List[float], n_1d: int) -> List[float]:
    """Downsample solution from 0D to 1D grid."""
    x_0d_array = np.array(x_0d)
    n_0d = len(x_0d_array)

    # Average over groups
    group_size = n_0d // n_1d
    x_1d = []

    for i in range(n_1d):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, n_0d)
        x_1d.append(np.mean(x_0d_array[start_idx:end_idx]))

    return x_1d


def solve_cycle_adaptive(
    P: dict[str, Any],
    max_refinements: int = 3,
) -> dict[str, Any]:
    """
    Solve cycle with adaptive refinement strategy.

    Args:
        P: Problem parameters
        max_refinements: Maximum number of refinements

    Returns:
        Solution dictionary
    """
    log.info(f"Starting adaptive cycle solve with max {max_refinements} refinements")

    # Start with 0D model
    current_model = "0d"
    current_result = None

    for refinement in range(max_refinements + 1):
        log.info(f"Refinement {refinement}: Solving with {current_model} model")

        # Set up problem for current model
        P_current = P.copy()
        P_current["model_type"] = current_model
        P_current["num"] = P_current.get("num", {})

        if current_model == "0d":
            P_current["num"]["K"] = P_current["num"].get("K_0d", 10)
        else:
            P_current["num"]["K"] = P_current["num"].get("K_1d", 30)

        # Use previous result as warm start
        if current_result is not None and current_result["success"]:
            warm_start = _create_warm_start_from_0d(current_result, P_current)
            P_current["warm_start"] = warm_start

        # Solve current model
        current_result = solve_cycle(P_current)

        if not current_result["success"]:
            log.warning(f"{current_model} solve failed at refinement {refinement}")
            if refinement == 0:
                return current_result
            # Return previous successful result
            break

        # Check if refinement is needed
        if refinement < max_refinements:
            if _should_refine_adaptive(current_result, P_current):
                current_model = "1d"
                log.info(f"Refining to 1D model for refinement {refinement + 1}")
            else:
                log.info("No further refinement needed")
                break
        else:
            log.info("Maximum refinements reached")
            break

    return current_result


def get_driver_function(driver_type: str = "standard"):
    """
    Get driver function by type.

    Args:
        driver_type: Type of driver function

    Returns:
        Driver function
    """
    functions = {
        "standard": solve_cycle,
        "robust": solve_cycle_robust,
        "refinement": solve_cycle_with_refinement,
        "warm_start": solve_cycle_with_warm_start,
        "adaptive": solve_cycle_adaptive,
    }

    if driver_type not in functions:
        raise ValueError(f"Unknown driver type: {driver_type}")

    return functions[driver_type]
