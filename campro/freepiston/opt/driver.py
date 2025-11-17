from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
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


def _get_available_hsl_solvers() -> set[str]:
    """Return the set of available HSL solvers (best-effort detection)."""
    try:
        from campro.environment.hsl_detector import detect_available_solvers, clear_cache
    except Exception:
        return set()

    try:
        # Clear cache to ensure we get fresh detection results
        # This is important because the library path (from ipopt.opt) might have changed
        clear_cache()
        # Use runtime detection to verify symbols actually exist in the HSL library
        # This prevents selecting solvers (like MA57) that aren't in the library
        solvers = detect_available_solvers(test_runtime=True) or []
    except Exception as exc:
        log.debug("HSL solver detection failed: %s", exc)
        return set()

    return {solver.lower() for solver in solvers}


def _configure_ma27_memory(
    options: IPOPTOptions,
    n_vars: int,
    n_constraints: int,
) -> None:
    """
    Configure MA27 memory using supported IPOPT options.

    IPOPT exposes memory controls via *_init_factor and meminc_factor knobs
    rather than absolute liw/la values. We estimate a scale factor based on
    problem size and apply it consistently.
    """
    baseline_vars = 1906
    baseline_constraints = 3154

    var_scale = n_vars / baseline_vars if baseline_vars else 1.0
    cons_scale = n_constraints / baseline_constraints if baseline_constraints else 1.0
    combined_scale = max(1.0, max(var_scale, cons_scale ** 0.5))
    safety = 1.5  # ensure extra headroom
    init_factor = max(2.0, combined_scale * safety)

    options.linear_solver_options["ma27_liw_init_factor"] = init_factor
    options.linear_solver_options["ma27_la_init_factor"] = init_factor
    options.linear_solver_options["ma27_meminc_factor"] = max(init_factor, 2.0)

    log.info(
        "Configured MA27 memory factors (n_vars=%d, n_constraints=%d): "
        "liw_init_factor=%.2f, la_init_factor=%.2f, meminc_factor=%.2f",
        n_vars,
        n_constraints,
        init_factor,
        init_factor,
        max(init_factor, 2.0),
    )


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
        solver_wrapper = IPOPTSolver(ipopt_options)
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
        
        # Get variable groups from metadata if available
        variable_groups = meta.get("variable_groups", {}) if meta else {}
        
        # Iteratively refine scaling to achieve target condition number
        reporter.info("Computing and refining scaling factors (iterative)...")
        scale, scale_g, scaling_quality = _refine_scaling_iteratively(
            nlp, x0, lbx, ubx, lbg, ubg, variable_groups, meta=meta, reporter=reporter
        )
        
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
        
        if len(scale_g) > 0:
            scale_g_min = scale_g.min()
            scale_g_max = scale_g.max()
            scale_g_mean = scale_g.mean()
            reporter.info(
                f"Constraint scaling factors: "
                f"min={scale_g_min:.3e}, max={scale_g_max:.3e}, mean={scale_g_mean:.3e}"
            )
        
        # Log final scaling quality
        condition_number = scaling_quality.get("condition_number", np.inf)
        quality_score = scaling_quality.get("quality_score", 0.0)
        reporter.info(
            f"Final scaling quality: condition_number={condition_number:.3e}, "
            f"quality_score={quality_score:.3f}"
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
        
        # Solve optimization problem with scaled NLP
        reporter.info("Starting IPOPT optimization (with comprehensive scaling)...")
        reporter.debug(
            f"Problem dimensions: n_vars={len(x0_scaled) if x0_scaled is not None else 0}, "
            f"n_constraints={len(lbg_scaled) if lbg_scaled is not None else 0}"
        )
        solve_start = time.time()
        
        # Try to solve with selected solver, fall back to MA27 if MA57 symbols not found
        selected_solver = ipopt_options.linear_solver.lower()
        result = solver_wrapper.solve(nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled, p)
        
        # Check if solve failed immediately (0 iterations) with MA57 - likely symbol loading failure
        # If MA57 was selected and solve fails immediately without any iterations, fall back to MA27
        if (selected_solver == "ma57" and not result.success and result.iterations == 0):
            reporter.warning(
                f"MA57 solver failed immediately (status={result.status}, message={result.message}). "
                "Likely symbols not found in HSL library. Falling back to MA27..."
            )
            # Recreate solver with MA27 - modify options and recreate solver
            ipopt_options.linear_solver = "ma27"
            n_vars_actual = len(x0_scaled) if x0_scaled is not None else 0
            n_constraints_actual = len(lbg_scaled) if lbg_scaled is not None else 0
            _configure_ma27_memory(ipopt_options, n_vars_actual, n_constraints_actual)
            solver_wrapper = IPOPTSolver(ipopt_options)
            result = solver_wrapper.solve(nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled, p)
        
        solve_elapsed = time.time() - solve_start
        reporter.info(f"IPOPT solve completed in {solve_elapsed:.3f}s")
        # Access stats through the result object (IPOPTSolver.solve() already extracted stats)
        # Create a stats dict compatible with _summarize_ipopt_iterations
        stats = {
            "iterations": {
                "k": np.array([result.iterations]) if result.iterations > 0 else np.array([]),
                "obj": np.array([result.f_opt]) if result.success else np.array([]),
            } if result.iterations > 0 else {}
        }
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
    
    # Use adaptive barrier strategy for better convergence with improved scaling
    # Adaptive strategy works better with well-scaled problems
    if not hasattr(options, "mu_strategy") or options.mu_strategy == "monotone":
        # Only override if not already set by user or if still using monotone
        options.mu_strategy = "adaptive"
    
    # Tune barrier parameter initialization based on problem scale
    # Increased mu_init values for better initial convergence
    if n_vars > 500:
        options.mu_init = max(options.mu_init, 1e-1)  # Increased from 1e-2 for large problems
    elif n_vars > 100:
        options.mu_init = max(options.mu_init, 5e-2)  # Increased from 5e-2 for medium problems
    else:
        options.mu_init = max(options.mu_init, 1e-1)  # Use higher default for small problems too
    
    # Increase mu_max to allow more barrier parameter growth
    options.mu_max = max(options.mu_max, 1e4)
    
    log.debug(
        f"Barrier parameter tuning: mu_init={options.mu_init:.2e}, mu_max={options.mu_max:.2e}, "
        f"mu_strategy={options.mu_strategy}"
    )

    available_solvers = _get_available_hsl_solvers()
    available_display = ", ".join(sorted(available_solvers)) if available_solvers else "unknown"
    options.linear_solver_options = dict(options.linear_solver_options or {})
    
    # Prefer MA57 when available (better performance and memory efficiency)
    # Fall back to MA27 with memory configuration if MA57 unavailable
    solver_choice = "ma27"
    if "ma57" in available_solvers:
        solver_choice = "ma57"
        log.info(
            "Using MA57 (n_vars=%d, n_constraints=%d, available=%s)",
            n_vars,
            n_constraints,
            available_display,
        )
    else:
        log.info(
            "Using MA27 (MA57 unavailable, available=%s, n_vars=%d, n_constraints=%d)",
            available_display,
            n_vars,
            n_constraints,
        )
        _configure_ma27_memory(options, n_vars, n_constraints)

    options.linear_solver = solver_choice

    log.info(
        "Selected IPOPT linear solver '%s' (n_vars=%d, n_constraints=%d, available=%s)",
        solver_choice,
        n_vars,
        n_constraints,
        available_display,
    )

    env_print_level = os.getenv("CAMPRO_IPOPT_PRINT_LEVEL")
    if env_print_level:
        try:
            options.print_level = max(0, min(12, int(env_print_level)))
            log.info(
                "Overriding IPOPT print_level via CAMPRO_IPOPT_PRINT_LEVEL=%s",
                env_print_level,
            )
        except ValueError:
            log.warning(
                "Invalid CAMPRO_IPOPT_PRINT_LEVEL=%s (expected integer). Using default.",
                env_print_level,
            )
    
    # Ensure minimum verbosity for convergence monitoring (unless explicitly set lower via env var)
    # print_level 8+ shows detailed iteration info including convergence metrics
    # Only enforce if not explicitly set via environment variable (allows test overrides)
    if env_print_level is None and options.print_level < 8:
        options.print_level = 8
        log.debug(f"Increased print_level to {options.print_level} for verbose convergence monitoring")
    
    # Ensure print_frequency_iter is 1 to show every iteration
    if options.print_frequency_iter > 1:
        options.print_frequency_iter = 1
        log.debug(f"Set print_frequency_iter to 1 to show every iteration")

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
    and clamped ratios to keep scales within 10¹ range.
    
    Uses group-based per-unit scales combined with value-based scaling:
    1. Apply per-unit reference scales based on variable groups (positions, velocities, etc.)
    2. Normalize each group to cap ratio ≤ 10¹
    3. Apply value-based scaling with clamped adjustments
    4. Final global normalization to ensure all scales within 10¹ range
    
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
    
    # Normalize each group to cap ratio ≤ 10¹
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if indices and len(indices) > 0:
                group_indices = np.array([i for i in indices if 0 <= i < n_vars])
                if len(group_indices) > 0:
                    group_scales = scale[group_indices]
                    group_min = group_scales.min()
                    group_max = group_scales.max()
                    
                    # Cap ratio to 10¹ (tighter than before)
                    if group_min > 1e-10 and group_max / group_min > 1e1:
                        # Normalize to median, then clamp to tighter range: [0.316 * median, 3.16 * median]
                        # This gives ratio of 10¹ (sqrt(10) ≈ 3.16)
                        group_median = np.median(group_scales)
                        sqrt_10 = np.sqrt(10.0)
                        for idx in group_indices:
                            # Clamp to [median/sqrt(10), median*sqrt(10)] ≈ [0.316*median, 3.16*median]
                            scale[idx] = np.clip(scale[idx], group_median / sqrt_10, group_median * sqrt_10)
    
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
            # Clamp adjustment to tighter range [0.316, 3.16] to prevent extreme ratios (10¹ ratio)
            sqrt_10 = np.sqrt(10.0)
            adjustment = np.clip(adjustment, 1.0 / sqrt_10, sqrt_10)
            scale[i] = adjustment
    
    # Final normalization: ensure all scales are within 10¹ range globally
    scale_min = scale[scale > 1e-10].min() if (scale > 1e-10).any() else 1e-10
    scale_max = scale.max()
    if scale_min > 1e-10 and scale_max / scale_min > 1e1:
        scale_median = np.median(scale[scale > 1e-10])
        sqrt_10 = np.sqrt(10.0)
        for i in range(n_vars):
            if scale[i] > 1e-10:
                # Clamp to [median/sqrt(10), median*sqrt(10)] for 10¹ ratio
                scale[i] = np.clip(scale[i], scale_median / sqrt_10, scale_median * sqrt_10)
    
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
    meta: dict[str, Any] | None = None,
) -> dict[str, float]:
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
        reporter: Optional reporter for logging
        
    Returns:
        Dictionary with quality metrics:
        - condition_number: max/min ratio of scaled Jacobian (target < 1e6)
        - jac_max: Maximum absolute value in scaled Jacobian
        - jac_mean: Mean absolute value in scaled Jacobian
        - jac_min: Minimum non-zero absolute value in scaled Jacobian
        - quality_score: Normalized quality score (0-1, higher is better)
    """
    # Default return values if verification fails
    default_metrics = {
        "condition_number": np.inf,
        "jac_max": np.inf,
        "jac_mean": 0.0,
        "jac_min": 0.0,
        "quality_score": 0.0,
    }
    
    try:
        import casadi as ca
        
        if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
            return default_metrics
        
        g_expr = nlp["g"]
        x_sym = nlp["x"]
        
        if g_expr is None or g_expr.numel() == 0:
            return default_metrics
        
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
            return default_metrics
        
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
            
            # Compute condition number (max/min ratio)
            condition_number = jac_max / jac_min if jac_min > 0 else np.inf
            
            # Compute quality score: normalized measure of scaling quality
            # Score is based on:
            # 1. Condition number (target < 1e6, penalty if > 1e6)
            # 2. Maximum Jacobian element (target < 1e2, penalty if > 1e2)
            # Score ranges from 0 (poor) to 1 (excellent)
            condition_score = 1.0 / (1.0 + np.log10(max(condition_number / 1e6, 1.0)))
            max_score = 1.0 / (1.0 + np.log10(max(jac_max / 1e2, 1.0)))
            quality_score = 0.5 * condition_score + 0.5 * max_score
            
            # Compute percentile statistics for detailed diagnostics
            jac_mag_nonzero = jac_mag[jac_mag > 0]
            if len(jac_mag_nonzero) > 0:
                p25 = np.percentile(jac_mag_nonzero, 25)
                p50 = np.percentile(jac_mag_nonzero, 50)
                p75 = np.percentile(jac_mag_nonzero, 75)
                p95 = np.percentile(jac_mag_nonzero, 95)
                p99 = np.percentile(jac_mag_nonzero, 99)
            else:
                p25 = p50 = p75 = p95 = p99 = 0.0
            
            # Log diagnostics with percentile statistics
            msg = (
                f"Scaled Jacobian statistics: min={jac_min:.3e}, p25={p25:.3e}, "
                f"p50={p50:.3e}, p75={p75:.3e}, p95={p95:.3e}, p99={p99:.3e}, "
                f"max={jac_max:.3e}, mean={jac_mean:.3e}, condition_number={condition_number:.3e}, "
                f"quality_score={quality_score:.3f}"
            )
            if reporter:
                reporter.info(msg)
            else:
                log.info(msg)
            
            # Report statistics per constraint type if available
            if meta and "constraint_groups" in meta:
                constraint_groups = meta["constraint_groups"]
                if reporter:
                    with reporter.section("Scaled Jacobian statistics by constraint type"):
                        for con_type, indices in constraint_groups.items():
                            if len(indices) == 0:
                                continue
                            # Get Jacobian entries for this constraint type
                            type_jac_mag = []
                            for idx in indices:
                                if idx < jac_mag.shape[0]:
                                    type_jac_mag.extend(jac_mag[idx, :].flatten())
                            
                            if len(type_jac_mag) > 0:
                                type_jac_mag = np.array(type_jac_mag)
                                type_jac_mag_nonzero = type_jac_mag[type_jac_mag > 0]
                                if len(type_jac_mag_nonzero) > 0:
                                    type_max = type_jac_mag_nonzero.max()
                                    type_mean = type_jac_mag_nonzero.mean()
                                    type_p95 = np.percentile(type_jac_mag_nonzero, 95)
                                    type_p99 = np.percentile(type_jac_mag_nonzero, 99)
                                    reporter.info(
                                        f"  {con_type} ({len(indices)} constraints): "
                                        f"max={type_max:.3e}, mean={type_mean:.3e}, "
                                        f"p95={type_p95:.3e}, p99={type_p99:.3e}"
                                    )
                                    # Warn if this type has extreme entries
                                    if type_max > 1e2:
                                        reporter.debug(
                                            f"    Warning: {con_type} has large max entry ({type_max:.3e} > 1e2)"
                                        )
            
            # Lower warning thresholds (more sensitive)
            if jac_max > 1e2:  # Lowered from 1e3
                warn_msg = (
                    f"Scaled Jacobian has large elements (max={jac_max:.3e} > 1e2). "
                    f"Scaling may be insufficient. Consider adjusting scale factors."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)
            if condition_number > 1e6:  # Target condition number
                warn_msg = (
                    f"Scaled Jacobian has large condition number (max/min={condition_number:.3e} > 1e6). "
                    f"Consider more uniform scaling."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)
            elif condition_number > 1e4:  # Lower threshold for warning
                warn_msg = (
                    f"Scaled Jacobian condition number is elevated (max/min={condition_number:.3e} > 1e4). "
                    f"May benefit from tighter scaling."
                )
                if reporter:
                    reporter.debug(warn_msg)
                else:
                    log.debug(warn_msg)
            
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
            
            return {
                "condition_number": condition_number,
                "jac_max": jac_max,
                "jac_mean": jac_mean,
                "jac_min": jac_min,
                "quality_score": quality_score,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
                "p99": p99,
            }
        else:
            # No constraint scaling available
            return default_metrics
        
    except Exception as e:
        if reporter:
            reporter.debug(f"Scaling verification failed: {e}")
        else:
            log.debug(f"Scaling verification failed: {e}")
    
    return default_metrics


def _identify_constraint_types(
    meta: dict[str, Any] | None,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    jac_g0_arr: np.ndarray | None = None,
    g0_arr: np.ndarray | None = None,
) -> dict[int, str]:
    """
    Identify constraint types from metadata or heuristics.
    
    Args:
        meta: NLP metadata dict (may contain constraint_groups)
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        jac_g0_arr: Constraint Jacobian at initial guess (optional, for heuristics)
        g0_arr: Constraint values at initial guess (optional, for heuristics)
        
    Returns:
        Dict mapping constraint index to constraint type string
    """
    constraint_types: dict[int, str] = {}
    
    # Try to use metadata constraint groups first (explicit identification)
    if meta and "constraint_groups" in meta:
        constraint_groups = meta["constraint_groups"]
        for con_type, indices in constraint_groups.items():
            for idx in indices:
                constraint_types[idx] = con_type
        return constraint_types
    
    # Fall back to heuristic identification if metadata unavailable
    if lbg is None or ubg is None:
        return constraint_types
    
    n_cons = len(lbg)
    
    # Heuristic: Identify constraint types based on bounds patterns
    for i in range(n_cons):
        lb = lbg[i]
        ub = ubg[i]
        
        # Equality constraints (lb == ub)
        if lb == ub and np.isfinite(lb):
            # Check if it's a periodicity constraint (typically 0.0)
            if abs(lb) < 1e-6:
                constraint_types[i] = "periodicity"
            else:
                constraint_types[i] = "continuity"  # Default for equality
        # Inequality constraints
        elif lb == -np.inf and ub == np.inf:
            constraint_types[i] = "collocation_residuals"  # Unbounded = residuals
        elif lb > 0 and np.isfinite(lb) and ub == np.inf:
            # Lower bound only, positive
            if lb > 1e3:  # Large lower bound suggests pressure (Pa)
                constraint_types[i] = "path_pressure"
            elif lb > 1e2:  # Medium lower bound suggests combustion (J)
                constraint_types[i] = "combustion"
            else:
                constraint_types[i] = "path_clearance"  # Small positive = clearance
        elif abs(lb) < 1e-6 and ub > 1e5:
            # Near-zero lower bound, large upper bound suggests pressure
            constraint_types[i] = "path_pressure"
        elif abs(lb) < 1e-6 and ub > 1e3:
            # Near-zero lower bound, medium upper bound suggests combustion
            constraint_types[i] = "combustion"
        elif abs(lb) < 50 and abs(ub) < 50:
            # Small bounds suggest velocity/acceleration
            constraint_types[i] = "path_velocity"
        else:
            # Default to path constraint
            constraint_types[i] = "path_constraint"
    
    # Refine using Jacobian row norms if available
    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
        if len(jac_row_norms) == n_cons:
            # Penalty constraints have very large row norms
            penalty_threshold = np.percentile(jac_row_norms[jac_row_norms > 0], 95) * 10
            for i in range(n_cons):
                if jac_row_norms[i] > penalty_threshold:
                    constraint_types[i] = "path_clearance"  # Likely penalty constraint
    
    return constraint_types


def _compute_scaled_jacobian(
    nlp: Any,
    x0: np.ndarray,
    scale: np.ndarray,
    scale_g: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Compute scaled Jacobian to identify problematic rows/columns.
    
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
        
        # Compute scaled Jacobian: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
        jac_g0_scaled = jac_g0_arr.copy()
        for i in range(min(jac_g0_arr.shape[0], len(scale_g))):
            for j in range(min(jac_g0_arr.shape[1], len(scale))):
                if scale[j] > 1e-10:
                    jac_g0_scaled[i, j] = scale_g[i] * jac_g0_arr[i, j] / scale[j]
        
        return jac_g0_arr, jac_g0_scaled
    except Exception:
        return None, None


def _compute_constraint_scaling_by_type(
    constraint_types: dict[int, str],
    nlp: Any,
    x0: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    scale: np.ndarray,
    jac_g0_arr: np.ndarray | None = None,
    g0_arr: np.ndarray | None = None,
    current_scale_g: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute constraint scaling factors using constraint-type-aware strategies.
    
    Applies specialized scaling per constraint type:
    - Penalty/Clearance: Aggressive scaling (target max entry < 1e1)
    - Pressure: Account for 1e6 unit conversion, normalize to O(1)
    - Combustion: Account for 1e3 unit conversion, normalize to O(1)
    - Collocation residuals: Standard scaling
    - Path constraints: Standard scaling based on bounds and Jacobian sensitivity
    - Periodicity: Equality constraints, scale based on typical violation magnitude
    
    Args:
        constraint_types: Dict mapping constraint index to type string
        nlp: CasADi NLP dict
        x0: Initial guess
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        scale: Variable scaling factors
        jac_g0_arr: Unscaled Jacobian (optional)
        g0_arr: Constraint values at initial guess (optional)
        
    Returns:
        Array of constraint scaling factors
    """
    if lbg is None or ubg is None:
        return np.array([])
    
    n_cons = len(lbg)
    # Start with current constraint scales (if available from previous iteration)
    # This allows constraint-type-aware to refine existing scaling rather than starting from scratch
    # If no current scales provided, start with ones
    if current_scale_g is not None and len(current_scale_g) == n_cons:
        scale_g = current_scale_g.copy()
    else:
        scale_g = np.ones(n_cons)
    
    # Group constraints by type for batch processing
    type_to_indices: dict[str, list[int]] = {}
    for idx, con_type in constraint_types.items():
        if con_type not in type_to_indices:
            type_to_indices[con_type] = []
        type_to_indices[con_type].append(idx)
    
    # Compute Jacobian row norms and max entries if available
    # IMPORTANT: Use scaled row norms AND max entries (accounting for variable scaling) to properly
    # assess constraint sensitivity. Max entries are more important than norms for extreme cases.
    jac_row_norms = None
    jac_row_max_entries = None  # Max absolute entry per row (after variable scaling, before constraint scaling)
    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        if scale is not None and len(scale) > 0:
            # Compute scaled row norms: sqrt(sum_j (J[i,j] / scale[j])^2)
            # Also compute max absolute entry per row: max_j |J[i,j] / scale[j]|
            # This represents the sensitivity of constraint i to scaled variables
            jac_row_norms = np.zeros(jac_g0_arr.shape[0])
            jac_row_max_entries = np.zeros(jac_g0_arr.shape[0])
            for i in range(jac_g0_arr.shape[0]):
                row_norm_sq = 0.0
                row_max = 0.0
                for j in range(min(jac_g0_arr.shape[1], len(scale))):
                    if scale[j] > 1e-10:
                        scaled_entry = abs(jac_g0_arr[i, j] / scale[j])
                        row_norm_sq += scaled_entry ** 2
                        row_max = max(row_max, scaled_entry)
                jac_row_norms[i] = np.sqrt(row_norm_sq)
                jac_row_max_entries[i] = row_max
        else:
            # Without variable scales, use unscaled row norms and max entries
            jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
            jac_row_max_entries = np.abs(jac_g0_arr).max(axis=1)
    
    # Process each constraint type with specialized scaling
    import logging
    log = logging.getLogger(__name__)
    
    # Store jac_sensitivity values for verification step
    stored_jac_sensitivity = {}
    
    # Check if we need to force recalculation due to poor current scaling
    if current_scale_g is not None and len(current_scale_g) == n_cons and jac_g0_arr is not None and scale is not None:
        # Quick check: compute a few scaled entries to see if they're too large
        max_scaled_entry_estimate = 0.0
        for i in range(min(100, n_cons)):  # Sample first 100 constraints
            if i < len(scale_g) and i < jac_g0_arr.shape[0]:
                row_max = 0.0
                for j in range(min(jac_g0_arr.shape[1], len(scale))):
                    if scale[j] > 1e-10:
                        scaled_entry = abs(scale_g[i] * jac_g0_arr[i, j] / scale[j])
                        row_max = max(row_max, scaled_entry)
                max_scaled_entry_estimate = max(max_scaled_entry_estimate, row_max)
        # If estimated max entry is very large (>1e2), force recalculation by starting from ones
        if max_scaled_entry_estimate > 1e2:
            log.debug(f"Current scaling produces large entries (est. max={max_scaled_entry_estimate:.3e} > 1e2), forcing recalculation")
            scale_g = np.ones(n_cons)
    
    for con_type, indices in type_to_indices.items():
        log.debug(f"Processing constraint type '{con_type}' with {len(indices)} constraints")
        for i in indices:
            if i >= n_cons:
                log.debug(f"  Skipping constraint {i}: index >= n_cons ({n_cons})")
                continue
            
            lb = lbg[i]
            ub = ubg[i]
            
            # Get constraint value magnitude
            g0_mag = 0.0
            if g0_arr is not None and i < len(g0_arr):
                g0_mag = abs(g0_arr[i])
            
            # Get Jacobian row norm and max entry if available
            jac_norm = 0.0
            jac_max_entry = 0.0
            if jac_row_norms is not None and i < len(jac_row_norms):
                jac_norm = jac_row_norms[i]
            if jac_row_max_entries is not None and i < len(jac_row_max_entries):
                jac_max_entry = jac_row_max_entries[i]
            # Use max entry for aggressive scaling decisions (more accurate than norm)
            # Fall back to norm if max entry not available
            jac_sensitivity = max(jac_max_entry, jac_norm) if jac_max_entry > 0 else jac_norm
            # Store for verification step
            stored_jac_sensitivity[i] = jac_sensitivity
            
            # Compute magnitude from bounds
            if lb == -np.inf and ub == np.inf:
                magnitude = max(g0_mag, jac_norm) if jac_norm > 0 else g0_mag
            elif lb == -np.inf:
                magnitude = max(abs(ub), g0_mag, jac_norm)
            elif ub == np.inf:
                magnitude = max(abs(lb), g0_mag, jac_norm)
            else:
                magnitude = max(abs(lb), abs(ub), g0_mag, jac_norm)
            
            # Apply type-specific scaling strategies
            if con_type == "path_clearance":
                # Penalty constraints: most aggressive scaling
                # Target max scaled Jacobian entry < 1e1
                if jac_norm > 1e6:
                    # Very large Jacobian norm indicates penalty stiffness
                    # Scale aggressively to bring max entry to O(1)
                    scale_g[i] = 1e1 / max(jac_norm, 1e-10)
                    scale_g[i] = np.clip(scale_g[i], 1e-8, 1e2)
                elif magnitude > 1e-6:
                    scale_g[i] = 1.0 / magnitude
                    scale_g[i] = np.clip(scale_g[i], 1e-6, 1e2)
                else:
                    scale_g[i] = 1.0
            
            elif con_type == "path_pressure":
                # Pressure constraints: account for 1e6 unit conversion (MPa → Pa)
                # Normalize to O(1) by accounting for unit conversion
                if magnitude > 1e6:
                    # Large magnitude suggests Pa units, normalize to MPa scale
                    scale_g[i] = 1e-6 / max(magnitude / 1e6, 1e-10)
                elif magnitude > 1e-6:
                    scale_g[i] = 1.0 / magnitude
                else:
                    scale_g[i] = 1.0
                scale_g[i] = np.clip(scale_g[i], 1e-4, 1e2)
            
            elif con_type == "combustion":
                # Combustion constraints: account for 1e3 unit conversion (kJ → J)
                # Normalize to O(1) by accounting for unit conversion
                if magnitude > 1e3:
                    # Large magnitude suggests J units, normalize to kJ scale
                    scale_g[i] = 1e-3 / max(magnitude / 1e3, 1e-10)
                elif magnitude > 1e-6:
                    scale_g[i] = 1.0 / magnitude
                else:
                    scale_g[i] = 1.0
                scale_g[i] = np.clip(scale_g[i], 1e-4, 1e2)
            
            elif con_type == "collocation_residuals":
                # Collocation residuals: should be ~0, scale based on Jacobian sensitivity
                # These can have extreme Jacobian entries due to force calculations with penalty terms
                # Use max entry (not norm) for more accurate scaling decisions
                old_scale = scale_g[i]
                if jac_sensitivity > 1e6:
                    # Very extreme Jacobian entries - need very aggressive scaling
                    # Target O(1) max entry for extreme cases
                    target_max_entry = 1e0
                    scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                    if i < 5:  # Log first few for debugging
                        log.debug(f"  collocation_residuals[{i}]: jac_sensitivity={jac_sensitivity:.3e}, old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}, expected_scaled_max={scale_g[i] * jac_sensitivity:.3e}")
                    # Allow very small scales for extreme cases, but prevent overscaling (< 1e-10)
                    # Compute minimum scale needed: if jac_sensitivity is huge, we need tiny scale
                    min_scale_needed = target_max_entry / jac_sensitivity
                    scale_g[i] = np.clip(scale_g[i], max(min_scale_needed * 0.1, 1e-10), 1e2)
                elif jac_sensitivity > 1e2:
                    # Large scaled Jacobian max entry - need aggressive scaling
                    # Target O(1) max entry
                    target_max_entry = 1e0
                    scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                    if i < 5:  # Log first few for debugging
                        log.debug(f"  collocation_residuals[{i}]: jac_sensitivity={jac_sensitivity:.3e}, old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}, expected_scaled_max={scale_g[i] * jac_sensitivity:.3e}")
                    scale_g[i] = np.clip(scale_g[i], 1e-8, 1e2)  # Much wider range for extreme cases
                elif jac_sensitivity > 1e-10:
                    # Moderate Jacobian sensitivity - scale to normalize max entry to O(1)
                    target_entry = 1e0
                    scale_g[i] = target_entry / max(jac_sensitivity, 1e-10)
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
                else:
                    # Small Jacobian sensitivity - keep existing scaling or use default
                    if scale_g[i] <= 1e-10:
                        scale_g[i] = 1.0
            
            elif con_type == "periodicity":
                # Periodicity constraints: equality constraints, scale based on violation
                if g0_mag > 1e-10:
                    scale_g[i] = 1.0 / max(g0_mag, 1e-10)
                elif jac_norm > 1e-10:
                    scale_g[i] = 1.0 / max(jac_norm, 1e-10)
                else:
                    scale_g[i] = 1.0
                scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
            
            elif con_type == "continuity":
                # Continuity constraints: equality constraints
                # These can also have extreme Jacobian entries
                if jac_sensitivity > 1e6:
                    # Very extreme Jacobian entries - need very aggressive scaling
                    target_max_entry = 1e0
                    scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                    # Allow very small scales for extreme cases, but prevent overscaling (< 1e-10)
                    min_scale_needed = target_max_entry / jac_sensitivity
                    scale_g[i] = np.clip(scale_g[i], max(min_scale_needed * 0.1, 1e-10), 1e2)
                elif jac_sensitivity > 1e2:
                    # Large scaled Jacobian max entry - need aggressive scaling
                    # Target O(1) max entry
                    target_max_entry = 1e0
                    scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                    scale_g[i] = np.clip(scale_g[i], 1e-8, 1e2)  # Much wider range
                elif g0_mag > 1e-10:
                    # Scale based on constraint violation magnitude
                    # For equality constraints: scale_g[i] * g0_mag should be O(1)
                    scale_g[i] = 1.0 / max(g0_mag, 1e-10)
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
                elif jac_sensitivity > 1e-10:
                    # Moderate Jacobian sensitivity - scale to normalize max entry to O(1)
                    target_entry = 1e0
                    scale_g[i] = target_entry / max(jac_sensitivity, 1e-10)
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
                else:
                    # Small Jacobian sensitivity - keep existing scaling or use default
                    if scale_g[i] <= 1e-10:
                        scale_g[i] = 1.0
            
            else:
                # Default path constraints: standard scaling
                if magnitude > 1e-6:
                    scale_g[i] = 1.0 / magnitude
                else:
                    scale_g[i] = 1.0
                scale_g[i] = np.clip(scale_g[i], 1e-3, 1e2)
    
    # Verification step: Check actual scaled max entries for collocation_residuals and continuity
    # Use stored jac_sensitivity values to verify scaling is correct
    for i in range(n_cons):
        con_type = constraint_types.get(i, "path_constraint")
        if con_type in {"collocation_residuals", "continuity"}:
            jac_sens = stored_jac_sensitivity.get(i, 0.0)
            if jac_sens > 1e-10 and scale_g[i] > 1e-10:
                # Compute expected scaled max entry using stored jac_sensitivity
                expected_scaled_max = scale_g[i] * jac_sens
                # Target max entry: O(1) for all cases now
                target_max = 1e0
                # If expected scaled max is still above target, apply additional reduction
                if expected_scaled_max > target_max * 1.1:  # 10% tolerance
                    safety_factor = target_max / expected_scaled_max
                    scale_g[i] = scale_g[i] * safety_factor
                    if i < 5:  # Log first few for debugging
                        log.debug(
                            f"  Verification[{i}]: {con_type}, jac_sens={jac_sens:.3e}, "
                            f"expected_scaled_max={expected_scaled_max:.3e}, target={target_max:.3e}, "
                            f"applied_safety_factor={safety_factor:.3e}, final_scale={scale_g[i]:.3e}"
                        )
    
    # Overscaling detection and correction: Prevent very small scaled Jacobian entries
    # Very small entries (<1e-8) can cause numerical precision issues
    # Balance: target max entry O(1-10), min entry >= 1e-8
    min_scaled_entry_threshold = 1e-8  # Minimum acceptable scaled Jacobian entry
    max_scaled_entry_target = 1e1      # Target max entry (O(10))
    
    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        # Compute current scaled Jacobian to detect overscaling
        _, jac_g0_scaled_check = _compute_scaled_jacobian(nlp, x0, scale, scale_g)
        
        if jac_g0_scaled_check is not None:
            jac_mag_check = np.abs(jac_g0_scaled_check)
            n_corrected = 0
            
            for i in range(min(jac_g0_scaled_check.shape[0], len(scale_g))):
                # Get row's scaled entries
                row_entries = jac_mag_check[i, :]
                row_nonzero = row_entries[row_entries > 0]
                
                if len(row_nonzero) > 0:
                    row_min = row_nonzero.min()
                    row_max = row_nonzero.max()
                    
                    # Check if row has overscaling (min entry too small)
                    if row_min < min_scaled_entry_threshold:
                        # Compute adjustment to bring min entry to threshold
                        # But don't increase max entry too much
                        adjustment_factor = min_scaled_entry_threshold / max(row_min, 1e-12)
                        
                        # Check if adjustment would create too large max entry
                        new_max = row_max * adjustment_factor
                        if new_max <= max_scaled_entry_target * 10:  # Allow up to 10x target
                            # Apply adjustment
                            old_scale = scale_g[i]
                            scale_g[i] = scale_g[i] * adjustment_factor
                            n_corrected += 1
                            
                            if i < 5:  # Log first few for debugging
                                con_type = constraint_types.get(i, "path_constraint")
                                log.debug(
                                    f"  Overscaling correction[{i}]: {con_type}, "
                                    f"row_min={row_min:.3e}, row_max={row_max:.3e}, "
                                    f"adjustment={adjustment_factor:.3e}, "
                                    f"old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}, "
                                    f"new_max={new_max:.3e}"
                                )
                        else:
                            # Compromise: adjust less aggressively to balance min and max
                            # Target: bring min to threshold while keeping max reasonable
                            compromise_factor = (min_scaled_entry_threshold / row_min) ** 0.5
                            if compromise_factor > 1.0:
                                old_scale = scale_g[i]
                                scale_g[i] = scale_g[i] * compromise_factor
                                n_corrected += 1
                                
                                if i < 5:  # Log first few for debugging
                                    con_type = constraint_types.get(i, "path_constraint")
                                    log.debug(
                                        f"  Overscaling compromise[{i}]: {con_type}, "
                                        f"row_min={row_min:.3e}, row_max={row_max:.3e}, "
                                        f"compromise_factor={compromise_factor:.3e}, "
                                        f"old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}"
                                    )
            
            if n_corrected > 0:
                log.debug(f"Overscaling correction: adjusted {n_corrected} constraints to prevent very small entries")
    
    # Normalize constraint scales to maintain reasonable range
    # BUT: Preserve aggressive scaling for extreme constraint types (collocation_residuals, continuity)
    # These constraint types need very small scale factors to normalize large Jacobian entries
    scale_g_log = np.log10(np.maximum(scale_g, 1e-10))
    median_log = np.median(scale_g_log)
    sqrt_10_log = np.log10(np.sqrt(10.0))
    
    # Identify extreme constraint types that need aggressive scaling preserved
    extreme_types = {"collocation_residuals", "continuity", "path_clearance"}
    
    # Clip outliers with constraint-type-aware bounds
    # For extreme constraint types, allow wider range (1e-8 to 1e2)
    # For other types, use tighter range (1e-3 to 1e2)
    for i in range(n_cons):
        con_type = constraint_types.get(i, "path_constraint")
        is_extreme = con_type in extreme_types
        
        if is_extreme:
            # Preserve aggressive scaling for extreme constraint types
            # Allow very small scales (down to 1e-10) for extreme cases, but prevent overscaling
            lower_bound_extreme = -10.0  # 1e-10 (allow very small scales for extreme cases)
            upper_bound_extreme = 2.0    # 1e2
            if scale_g_log[i] < lower_bound_extreme:
                scale_g[i] = 10.0 ** lower_bound_extreme
            elif scale_g_log[i] > upper_bound_extreme:
                scale_g[i] = 10.0 ** upper_bound_extreme
            # Otherwise, preserve the aggressive scaling (don't clip)
        else:
            # For normal constraint types, use percentile-based clipping
            lower_bound = max(median_log - 2.0 * sqrt_10_log, -3.0)  # Allow down to 1e-3
            upper_bound = min(median_log + 2.0 * sqrt_10_log, 2.0)   # Allow up to 1e2
            if scale_g_log[i] < lower_bound:
                scale_g[i] = max(10.0 ** lower_bound, scale_g[i] * 0.1)
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = min(10.0 ** upper_bound, scale_g[i] * 10.0)
    
    return scale_g


def _try_scaling_strategy(
    strategy_name: str,
    nlp: Any,
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    scale: np.ndarray,
    scale_g: np.ndarray,
    jac_g0_arr: np.ndarray,
    jac_g0_scaled: np.ndarray,
    variable_groups: dict[str, list[int]] | None,
    constraint_types: dict[int, str] | None = None,
    meta: dict[str, Any] | None = None,
    g0_arr: np.ndarray | None = None,
    target_max_entry: float = 1e2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Try a specific scaling strategy and return refined scales.
    
    Args:
        strategy_name: Name of strategy to try
        nlp: CasADi NLP dict
        x0: Initial guess
        lbx, ubx: Variable bounds
        lbg, ubg: Constraint bounds
        scale: Current variable scales
        scale_g: Current constraint scales
        jac_g0_arr: Unscaled Jacobian
        jac_g0_scaled: Current scaled Jacobian
        variable_groups: Variable group mapping
        target_max_entry: Target maximum scaled Jacobian entry
        
    Returns:
        Tuple of (new_scale, new_scale_g)
    """
    new_scale = scale.copy()
    new_scale_g = scale_g.copy()
    
    jac_mag = np.abs(jac_g0_scaled)
    
    if strategy_name == "tighten_ratios":
        # Strategy 1: Tighten variable and constraint scaling ratios
        scale_median = np.median(new_scale[new_scale > 1e-10])
        tight_factor = np.sqrt(np.sqrt(10.0))
        for i in range(len(new_scale)):
            if new_scale[i] > 1e-10:
                new_scale[i] = np.clip(new_scale[i], scale_median / tight_factor, scale_median * tight_factor)
        
        if len(new_scale_g) > 0:
            scale_g_median = np.median(new_scale_g[new_scale_g > 1e-10])
            for i in range(len(new_scale_g)):
                if new_scale_g[i] > 1e-10:
                    new_scale_g[i] = np.clip(new_scale_g[i], scale_g_median / tight_factor, scale_g_median * tight_factor)
    
    elif strategy_name == "row_max_scaling":
        # Strategy 2: Scale constraint rows based on max entry per row
        # Target: max scaled entry per row should be <= target_max_entry
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_max = jac_mag[i, :].max() if jac_g0_scaled.shape[1] > 0 else 0.0
            if row_max > target_max_entry:
                # Reduce scale_g[i] to bring max entry down to target
                reduction_factor = target_max_entry / row_max
                new_scale_g[i] = new_scale_g[i] * reduction_factor
                # Allow more aggressive scaling for very large entries
                if row_max > 1e10:
                    # Extra aggressive for extreme entries
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-6, 1e3)
                else:
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-3, 1e3)
    
    elif strategy_name == "column_max_scaling":
        # Strategy 3: Scale variable columns based on max entry per column
        # Target: max scaled entry per column should be <= target_max_entry
        # Use more conservative scaling to avoid over-scaling variables
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_max = jac_mag[:, j].max() if jac_g0_scaled.shape[0] > 0 else 0.0
            if col_max > target_max_entry:
                # Increase scale[j] to bring max entry down to target
                # But be conservative - don't increase too much at once
                increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x increase
                new_scale[j] = new_scale[j] * increase_factor
                # More conservative clamping to prevent extreme variable scaling
                if col_max > 1e10:
                    # Extra aggressive for extreme entries, but still bounded
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                else:
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)
        
        # Re-normalize variable scales to maintain 10¹ ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10)
    
    elif strategy_name == "extreme_entry_targeting":
        # Strategy 4: Aggressively target extreme entries (>1e2)
        # Find rows and columns with extreme entries and scale them down aggressively
        # Use adaptive threshold based on current max entry
        global_max = jac_mag.max() if jac_g0_scaled.size > 0 else 0.0
        if global_max > 1e10:
            # For very extreme entries, use more aggressive threshold
            extreme_threshold = 1e1  # Target O(10) for extreme cases
        else:
            extreme_threshold = 1e2  # Target O(100) for moderate cases
        
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_max = jac_mag[i, :].max() if jac_g0_scaled.shape[1] > 0 else 0.0
            if row_max > extreme_threshold:
                # Aggressively reduce scale_g[i] to bring max entry to threshold
                reduction_factor = extreme_threshold / row_max
                new_scale_g[i] = new_scale_g[i] * reduction_factor
                # Allow very aggressive scaling for extreme entries
                if row_max > 1e10:
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                else:
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-6, 1e3)
        
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_max = jac_mag[:, j].max() if jac_g0_scaled.shape[0] > 0 else 0.0
            if col_max > extreme_threshold:
                # Aggressively increase scale[j] to bring max entry to threshold
                # But cap the increase to prevent over-scaling variables
                increase_factor = min(col_max / extreme_threshold, 100.0)  # Cap at 100x
                new_scale[j] = new_scale[j] * increase_factor
                # Allow aggressive scaling but with tighter bounds
                if col_max > 1e10:
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                else:
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)
        
        # Re-normalize variable scales to maintain 10¹ ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10)
    
    elif strategy_name == "percentile_based":
        # Strategy 5: Scale based on percentiles - target p95 entries
        # Bring p95 entries down to target_max_entry
        if jac_g0_scaled.size > 0:
            jac_mag_nonzero = jac_mag[jac_mag > 0]
            if len(jac_mag_nonzero) > 0:
                p95_value = np.percentile(jac_mag_nonzero, 95)
                if p95_value > target_max_entry:
                    global_reduction = target_max_entry / p95_value
                    # Apply reduction to constraint scales
                    for i in range(len(new_scale_g)):
                        new_scale_g[i] = new_scale_g[i] * global_reduction
                        # Allow more aggressive scaling for high percentiles
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)
    
    elif strategy_name == "combined_row_column":
        # Strategy 6: Combined aggressive row and column scaling
        # Apply both row and column scaling together for maximum effect
        # First apply row scaling
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_max = jac_mag[i, :].max() if jac_g0_scaled.shape[1] > 0 else 0.0
            if row_max > target_max_entry:
                reduction_factor = target_max_entry / row_max
                new_scale_g[i] = new_scale_g[i] * reduction_factor
                if row_max > 1e10:
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                else:
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)
        
        # Then apply column scaling (more conservative to avoid over-scaling)
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_max = jac_mag[:, j].max() if jac_g0_scaled.shape[0] > 0 else 0.0
            if col_max > target_max_entry:
                # Cap increase factor to prevent extreme variable scaling
                increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x
                new_scale[j] = new_scale[j] * increase_factor
                if col_max > 1e10:
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                else:
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)
        
        # Re-normalize variable scales to maintain 10¹ ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10)
    
    elif strategy_name == "constraint_type_aware":
        # Strategy: Constraint-type-aware scaling
        # Use specialized scaling strategies per constraint type
        import logging
        log = logging.getLogger(__name__)
        
        if constraint_types is None or len(constraint_types) == 0:
            # Fall back to standard scaling if types unavailable
            log.debug("Constraint-type-aware: No constraint types available, falling back to standard scaling")
            return new_scale, new_scale_g
        
        log.debug(f"Constraint-type-aware: Processing {len(constraint_types)} constraint types")
        
        # Compute constraint values if not provided
        if g0_arr is None:
            try:
                import casadi as ca
                if isinstance(nlp, dict) and "g" in nlp and "x" in nlp:
                    g_expr = nlp["g"]
                    x_sym = nlp["x"]
                    g_func = ca.Function("g_func", [x_sym], [g_expr])
                    g0 = g_func(x0)
                    g0_arr = np.array(g0)
                    log.debug(f"Constraint-type-aware: Computed constraint values, shape={g0_arr.shape}")
            except Exception as e:
                log.debug(f"Constraint-type-aware: Failed to compute constraint values: {e}")
                g0_arr = None
        
        # Apply constraint-type-aware scaling
        # Pass current scale_g so it can refine existing scaling rather than starting from scratch
        log.debug(f"Constraint-type-aware: Computing scaling with current scale_g range=[{scale_g.min():.3e}, {scale_g.max():.3e}]")
        try:
            new_scale_g = _compute_constraint_scaling_by_type(
                constraint_types, nlp, x0, lbg, ubg, new_scale,
                jac_g0_arr=jac_g0_arr, g0_arr=g0_arr, current_scale_g=scale_g
            )
            log.debug(f"Constraint-type-aware: Computed new scale_g, range=[{new_scale_g.min():.3e}, {new_scale_g.max():.3e}]")
        except Exception as e:
            import traceback
            log.error(f"Constraint-type-aware: Failed in _compute_constraint_scaling_by_type: {e}")
            log.error(f"Exception traceback:\n{traceback.format_exc()}")
            raise
        
        # Debug: Check if constraint-type-aware actually modified scales
        if not np.allclose(new_scale_g, scale_g, rtol=1e-6):
            # Log which constraint types were modified
            modified_types = {}
            n_modified = 0
            for idx, con_type in constraint_types.items():
                if idx < len(scale_g) and idx < len(new_scale_g):
                    if abs(new_scale_g[idx] - scale_g[idx]) > 1e-6 * max(abs(scale_g[idx]), 1.0):
                        n_modified += 1
                        if con_type not in modified_types:
                            modified_types[con_type] = {"count": 0, "max_change": 0.0}
                        modified_types[con_type]["count"] += 1
                        change_ratio = abs(new_scale_g[idx] / (scale_g[idx] + 1e-10))
                        modified_types[con_type]["max_change"] = max(modified_types[con_type]["max_change"], change_ratio)
            
            log.debug(f"Constraint-type-aware: Modified {n_modified} constraints")
            for con_type, stats in modified_types.items():
                log.debug(f"  {con_type}: {stats['count']} constraints, max_change_ratio={stats['max_change']:.3e}")
        else:
            log.debug("Constraint-type-aware: No constraints were modified (scales unchanged)")
    
    elif strategy_name == "percentile_based":
        # Strategy 5: Scale based on percentiles - target p95 entries
        # Bring p95 entries down to target_max_entry
        if jac_g0_scaled.size > 0:
            jac_mag_nonzero = jac_mag[jac_mag > 0]
            if len(jac_mag_nonzero) > 0:
                p95_value = np.percentile(jac_mag_nonzero, 95)
                if p95_value > target_max_entry:
                    global_reduction = target_max_entry / p95_value
                    # Apply reduction to constraint scales
                    for i in range(len(new_scale_g)):
                        new_scale_g[i] = new_scale_g[i] * global_reduction
                        # Allow more aggressive scaling for high percentiles
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)
    
    return new_scale, new_scale_g


def _get_scaling_cache_path() -> Path:
    """Get the path to the scaling cache file."""
    cache_dir = Path.home() / ".campro" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "scaling_cache.json"


def _generate_scaling_cache_key(
    n_vars: int,
    n_constraints: int,
    variable_groups: dict[str, list[int]] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """
    Generate a cache key from problem characteristics.
    
    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        variable_groups: Variable group mapping
        meta: Problem metadata
        
    Returns:
        Cache key string (hash)
    """
    # Create a dictionary of problem characteristics
    key_data = {
        "n_vars": n_vars,
        "n_constraints": n_constraints,
    }
    
    # Add variable group info if available
    if variable_groups:
        key_data["variable_groups"] = {
            group: len(indices) for group, indices in variable_groups.items()
        }
    
    # Add constraint type info if available
    if meta and "constraint_groups" in meta:
        constraint_groups = meta["constraint_groups"]
        key_data["constraint_groups"] = {
            group: len(indices) for group, indices in constraint_groups.items()
        }
    
    # Create hash from key data
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _load_scaling_cache(
    cache_key: str,
    n_vars: int,
    n_constraints: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float] | None]:
    """
    Load scaling factors from cache if available.
    
    Args:
        cache_key: Cache key for this problem
        n_vars: Expected number of variables
        n_constraints: Expected number of constraints
        
    Returns:
        Tuple of (scale, scale_g, quality) or (None, None, None) if not found
    """
    cache_path = _get_scaling_cache_path()
    
    if not cache_path.exists():
        return None, None, None
    
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        
        if cache_key not in cache:
            return None, None, None
        
        entry = cache[cache_key]
        
        # Verify dimensions match
        cached_scale = np.array(entry["scale"])
        cached_scale_g = np.array(entry["scale_g"])
        
        if len(cached_scale) != n_vars or len(cached_scale_g) != n_constraints:
            return None, None, None
        
        quality = entry.get("quality", {})
        
        return cached_scale, cached_scale_g, quality
        
    except Exception as e:
        log.debug(f"Failed to load scaling cache: {e}")
        return None, None, None


def _save_scaling_cache(
    cache_key: str,
    scale: np.ndarray,
    scale_g: np.ndarray,
    quality: dict[str, float],
) -> None:
    """
    Save scaling factors to cache.
    
    Args:
        cache_key: Cache key for this problem
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        quality: Quality metrics
    """
    cache_path = _get_scaling_cache_path()
    
    try:
        # Load existing cache
        cache = {}
        if cache_path.exists():
            with open(cache_path, "r") as f:
                cache = json.load(f)
        
        # Update cache entry
        cache[cache_key] = {
            "scale": scale.tolist(),
            "scale_g": scale_g.tolist(),
            "quality": quality,
            "timestamp": time.time(),
        }
        
        # Limit cache size (keep only most recent 10 entries)
        if len(cache) > 10:
            # Sort by timestamp and keep most recent
            entries = list(cache.items())
            entries.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            cache = dict(entries[:10])
        
        # Save cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
            
    except Exception as e:
        log.debug(f"Failed to save scaling cache: {e}")


def _refine_scaling_iteratively(
    nlp: Any,
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    variable_groups: dict[str, list[int]] | None,
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
    max_iterations: int = 5,
    target_condition_number: float = 1e6,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Iteratively refine scaling factors to achieve target condition number.
    
    Tries multiple scaling strategies per iteration and selects the best one.
    
    Args:
        nlp: CasADi NLP dict
        x0: Initial guess (unscaled)
        lbx: Lower variable bounds
        ubx: Upper variable bounds
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        variable_groups: Variable group mapping
        reporter: Optional reporter for logging
        max_iterations: Maximum refinement iterations
        target_condition_number: Target condition number (default 1e6)
        
    Returns:
        Tuple of (refined_scale, refined_scale_g, quality_metrics)
    """
    # Generate cache key for this problem
    n_vars = len(x0) if x0 is not None else len(lbx)
    n_constraints = len(lbg) if lbg is not None else 0
    cache_key = _generate_scaling_cache_key(n_vars, n_constraints, variable_groups, meta)
    
    # Track whether we're starting from cached scaling
    skip_initial_scaling = False
    
    # Try to load cached scaling
    cached_scale, cached_scale_g, cached_quality = _load_scaling_cache(cache_key, n_vars, n_constraints)
    
    if cached_scale is not None and cached_scale_g is not None:
        # Verify cached scaling quality
        cached_quality_check = _verify_scaling_quality(
            nlp, x0, cached_scale, cached_scale_g, lbg, ubg, reporter=None, meta=meta
        )
        cached_condition = cached_quality_check.get("condition_number", np.inf)
        cached_score = cached_quality_check.get("quality_score", 0.0)
        
        # Minimum quality_score threshold
        min_quality_score = 0.7
        
        # Use cached scaling as starting point if it meets both targets, otherwise iterate to improve
        if cached_condition <= target_condition_number and cached_score >= min_quality_score:
            if reporter:
                reporter.info(
                    f"Using cached scaling (meets targets): condition_number={cached_condition:.3e}, "
                    f"quality_score={cached_score:.3f} >= {min_quality_score:.3f}"
                )
            return cached_scale, cached_scale_g, cached_quality_check
        else:
            # Use cached scaling as starting point for iteration
            if reporter:
                condition_msg = f"condition_number={cached_condition:.3e} > {target_condition_number:.3e}" if cached_condition > target_condition_number else ""
                quality_msg = f"quality_score={cached_score:.3f} < {min_quality_score:.3f}" if cached_score < min_quality_score else ""
                reason = " or ".join(filter(None, [condition_msg, quality_msg]))
                reporter.info(
                    f"Using cached scaling as starting point: condition_number={cached_condition:.3e}, "
                    f"quality_score={cached_score:.3f} ({reason}), will iterate to improve"
                )
            # Start with cached scaling instead of computing initial scaling
            scale = cached_scale.copy()
            scale_g = cached_scale_g.copy()
            quality = cached_quality_check
            condition_number = cached_condition
            initial_condition_number = condition_number
            # Skip initial scaling computation and go straight to iteration
            skip_initial_scaling = True
    
    if not skip_initial_scaling:
        # Compute initial scaling
        scale = _compute_variable_scaling(lbx, ubx, x0=x0, variable_groups=variable_groups)
        try:
            scale_g = _compute_constraint_scaling_from_evaluation(nlp, x0, lbg, ubg, scale=scale)
        except Exception:
            scale_g = _compute_constraint_scaling(lbg, ubg)
        
        # Check initial quality
        quality = _verify_scaling_quality(nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta)
        condition_number = quality.get("condition_number", np.inf)
        initial_condition_number = condition_number  # Save for comparison at end
        
        if reporter:
            reporter.info(f"Initial scaling quality: condition_number={condition_number:.3e}, quality_score={quality.get('quality_score', 0.0):.3f}")
        
        # If condition number is acceptable, save and return initial scaling
        if condition_number <= target_condition_number:
            _save_scaling_cache(cache_key, scale, scale_g, quality)
            return scale, scale_g, quality
    
    # Get unscaled and scaled Jacobian for strategy evaluation
    jac_g0_arr, jac_g0_scaled = _compute_scaled_jacobian(nlp, x0, scale, scale_g)
    if jac_g0_arr is None or jac_g0_scaled is None:
        # Can't compute Jacobian, fall back to simple tightening
        if reporter:
            reporter.warning("Cannot compute Jacobian for refinement, using simple ratio tightening")
        return scale, scale_g, quality
    
    # Identify constraint types once at the start
    constraint_types = _identify_constraint_types(meta, lbg, ubg, jac_g0_arr=jac_g0_arr)
    
    # Get constraint values for type-aware scaling
    g0_arr = None
    try:
        import casadi as ca
        if isinstance(nlp, dict) and "g" in nlp and "x" in nlp:
            g_expr = nlp["g"]
            x_sym = nlp["x"]
            g_func = ca.Function("g_func", [x_sym], [g_expr])
            g0 = g_func(x0)
            g0_arr = np.array(g0)
    except Exception:
        pass
    
    # Define strategies to try (in order of preference)
    # Prioritize constraint-type-aware if constraint groups are available
    has_constraint_groups = meta and "constraint_groups" in meta and len(constraint_types) > 0
    
    if condition_number > 1e20:
        # For extremely ill-conditioned problems, prioritize aggressive strategies
        if has_constraint_groups:
            strategies = [
                "constraint_type_aware",  # Most targeted - use constraint type information
                "combined_row_column",      # Combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",          # Scale rows with large max entries
                "column_max_scaling",       # Scale columns with large max entries
                "percentile_based",         # Scale based on percentiles
            ]
        else:
            strategies = [
                "combined_row_column",      # Most aggressive - combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",          # Scale rows with large max entries
                "column_max_scaling",       # Scale columns with large max entries
                "percentile_based",         # Scale based on percentiles
            ]
    else:
        # For moderately ill-conditioned problems, include conservative strategies
        if has_constraint_groups:
            strategies = [
                "constraint_type_aware",  # Most targeted - use constraint type information
                "combined_row_column",      # Combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",          # Scale rows with large max entries
                "column_max_scaling",       # Scale columns with large max entries
                "percentile_based",         # Scale based on percentiles
                "tighten_ratios",           # Conservative - tighten ratios
            ]
        else:
            strategies = [
                "combined_row_column",      # Most aggressive - combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",          # Scale rows with large max entries
                "column_max_scaling",       # Scale columns with large max entries
                "percentile_based",         # Scale based on percentiles
                "tighten_ratios",           # Conservative - tighten ratios
            ]
    
    # Track previous iteration's condition number for improvement detection
    prev_condition_number = condition_number
    
    # Iteratively refine scaling
    for iteration in range(max_iterations):
        if reporter:
            reporter.info(f"Scaling refinement iteration {iteration + 1}/{max_iterations}")
        
        best_scale = scale.copy()
        best_scale_g = scale_g.copy()
        best_condition_number = condition_number  # Start with current condition number
        best_quality_score = quality.get("quality_score", 0.0)
        best_strategy = None
        
        # Try each strategy and pick the best one
        for strategy in strategies:
            if reporter:
                reporter.debug(f"Trying strategy: {strategy}")
            try:
                # Try this strategy
                if reporter:
                    reporter.debug(f"  Input scales: scale range=[{scale.min():.3e}, {scale.max():.3e}], scale_g range=[{scale_g.min():.3e}, {scale_g.max():.3e}]")
                test_scale, test_scale_g = _try_scaling_strategy(
                    strategy, nlp, x0, lbx, ubx, lbg, ubg,
                    scale, scale_g, jac_g0_arr, jac_g0_scaled,
                    variable_groups,
                    constraint_types=constraint_types if has_constraint_groups else None,
                    meta=meta,
                    g0_arr=g0_arr,
                    target_max_entry=1e2
                )
                if reporter:
                    reporter.debug(f"  Output scales: scale range=[{test_scale.min():.3e}, {test_scale.max():.3e}], scale_g range=[{test_scale_g.min():.3e}, {test_scale_g.max():.3e}]")
                    scale_changed = not np.allclose(test_scale, scale, rtol=1e-6)
                    scale_g_changed = not np.allclose(test_scale_g, scale_g, rtol=1e-6)
                    reporter.debug(f"  Strategy modified scales: scale={scale_changed}, scale_g={scale_g_changed}")
                
                # Recompute constraint scaling with new variable scales if needed
                # (some strategies modify variable scales, which affects constraint scaling)
                # BUT: preserve aggressive constraint scaling from strategies that modified scale_g
                scale_g_was_modified = not np.allclose(test_scale_g, scale_g, rtol=1e-6)
                scale_was_modified = not np.allclose(test_scale, scale, rtol=1e-6)
                
                # Special handling for constraint-type-aware strategy: don't recompute, it already
                # computed optimal scaling based on constraint types
                if strategy == "constraint_type_aware" and scale_g_was_modified:
                    # Constraint-type-aware already computed optimal scaling, keep it
                    # Even if variable scales changed, the constraint-type-aware scaling
                    # is based on constraint types and Jacobian norms, not variable scales
                    pass  # Keep test_scale_g as computed by constraint-type-aware
                elif scale_was_modified and not scale_g_was_modified:
                    # Variable scales changed but constraint scales weren't modified by strategy
                    # Need to recompute constraint scaling with new variable scales
                    try:
                        test_scale_g = _compute_constraint_scaling_from_evaluation(
                            nlp, x0, lbg, ubg, scale=test_scale
                        )
                    except Exception:
                        test_scale_g = _compute_constraint_scaling(lbg, ubg)
                elif scale_was_modified and scale_g_was_modified:
                    # Both were modified - need to recompute constraint scaling but preserve
                    # the aggressive adjustments from the strategy
                    old_scale_g = scale_g.copy()
                    try:
                        new_base_scale_g = _compute_constraint_scaling_from_evaluation(
                            nlp, x0, lbg, ubg, scale=test_scale
                        )
                        # Preserve relative adjustments: apply the ratio of old strategy-modified
                        # scale_g to the new base scale_g
                        if len(old_scale_g) > 0 and len(new_base_scale_g) > 0 and len(test_scale_g) > 0:
                            # Compute what the strategy adjustment was
                            strategy_adjustment = test_scale_g / (old_scale_g + 1e-10)
                            # Apply that adjustment to the new base scaling
                            test_scale_g = new_base_scale_g * strategy_adjustment
                            # Re-clamp to reasonable ranges
                            test_scale_g = np.clip(test_scale_g, 1e-6, 1e3)
                        else:
                            test_scale_g = new_base_scale_g
                    except Exception:
                        # Keep the strategy-modified scale_g
                        pass
                
                # IMPORTANT: Recompute scaled Jacobian with NEW scales to properly evaluate quality
                # The old jac_g0_scaled was computed with old scales, so it's not accurate for evaluation
                test_jac_g0_arr, test_jac_g0_scaled = _compute_scaled_jacobian(
                    nlp, x0, test_scale, test_scale_g
                )
                if test_jac_g0_scaled is not None:
                    # Use the new scaled Jacobian for quality evaluation
                    test_jac_mag = np.abs(test_jac_g0_scaled)
                    test_jac_max = test_jac_mag.max()
                    test_jac_min = test_jac_mag[test_jac_mag > 0].min() if (test_jac_mag > 0).any() else 1e-10
                    test_condition_number = test_jac_max / test_jac_min if test_jac_min > 0 else np.inf
                    # Compute quality score
                    condition_score = 1.0 / (1.0 + np.log10(max(test_condition_number / 1e6, 1.0)))
                    max_score = 1.0 / (1.0 + np.log10(max(test_jac_max / 1e2, 1.0)))
                    test_quality_score = 0.5 * condition_score + 0.5 * max_score
                else:
                    # Fallback to full quality evaluation
                    test_quality = _verify_scaling_quality(
                        nlp, x0, test_scale, test_scale_g, lbg, ubg, reporter=None, meta=meta
                    )
                    test_condition_number = test_quality.get("condition_number", np.inf)
                    test_quality_score = test_quality.get("quality_score", 0.0)
                
                # Log evaluation for all strategies
                if reporter:
                    # Format values safely (handle inf/nan)
                    jac_max_str = f"{test_jac_max:.3e}" if (test_jac_g0_scaled is not None and np.isfinite(test_jac_max)) else ("N/A" if test_jac_g0_scaled is None else str(test_jac_max))
                    jac_min_str = f"{test_jac_min:.3e}" if (test_jac_g0_scaled is not None and np.isfinite(test_jac_min)) else ("N/A" if test_jac_g0_scaled is None else str(test_jac_min))
                    condition_str = f"{test_condition_number:.3e}" if np.isfinite(test_condition_number) else str(test_condition_number)
                    reporter.debug(
                        f"Strategy '{strategy}' evaluation: condition_number={condition_str}, "
                        f"quality_score={test_quality_score:.3f}, "
                        f"jac_max={jac_max_str}, "
                        f"jac_min={jac_min_str}"
                    )
                    # Log per-constraint-type statistics for constraint-type-aware
                    if strategy == "constraint_type_aware" and meta and "constraint_groups" in meta and test_jac_g0_scaled is not None:
                        constraint_groups = meta["constraint_groups"]
                        test_jac_mag = np.abs(test_jac_g0_scaled)
                        for con_type, indices in constraint_groups.items():
                            if len(indices) == 0:
                                continue
                            type_max_entries = []
                            for idx in indices:
                                if idx < test_jac_mag.shape[0]:
                                    type_max_entries.append(test_jac_mag[idx, :].max())
                            if len(type_max_entries) > 0:
                                type_max = max(type_max_entries)
                                type_mean = np.mean(type_max_entries)
                                type_max_str = f"{type_max:.3e}" if np.isfinite(type_max) else str(type_max)
                                type_mean_str = f"{type_mean:.3e}" if np.isfinite(type_mean) else str(type_mean)
                                reporter.debug(
                                    f"  {con_type} ({len(indices)} constraints): "
                                    f"max_entry={type_max_str}, mean_max_entry={type_mean_str}"
                                )
                
                # Check if this is better (lower condition number or higher quality score)
                # Handle inf/nan comparisons safely
                condition_better = (np.isfinite(test_condition_number) and np.isfinite(best_condition_number) and 
                                   test_condition_number < best_condition_number) or \
                                  (np.isfinite(test_condition_number) and not np.isfinite(best_condition_number))
                quality_better = (np.isfinite(test_condition_number) and np.isfinite(best_condition_number) and
                                 test_condition_number == best_condition_number and 
                                 test_quality_score > best_quality_score)
                is_better = condition_better or quality_better
                
                if reporter:
                    reporter.debug(
                        f"Strategy '{strategy}' comparison: "
                        f"condition_better={condition_better}, "
                        f"quality_better={quality_better}, "
                        f"is_best={is_better}"
                    )
                if is_better:
                    best_scale = test_scale
                    best_scale_g = test_scale_g
                    best_condition_number = test_condition_number
                    best_quality_score = test_quality_score
                    best_strategy = strategy
                    if reporter:
                        reporter.debug(f"Strategy '{strategy}' is now the best strategy")
                    
            except Exception as e:
                import traceback
                if reporter:
                    reporter.debug(f"Strategy '{strategy}' failed with exception: {e}")
                    reporter.debug(f"Exception traceback:\n{traceback.format_exc()}")
                else:
                    import logging
                    log = logging.getLogger(__name__)
                    log.error(f"Strategy '{strategy}' failed: {e}")
                    log.error(f"Exception traceback:\n{traceback.format_exc()}")
                continue
        
        # Update scales with best strategy
        scale = best_scale
        scale_g = best_scale_g
        
        # Recompute Jacobian with new scales
        jac_g0_arr, jac_g0_scaled = _compute_scaled_jacobian(nlp, x0, scale, scale_g)
        if jac_g0_arr is None or jac_g0_scaled is None:
            break
        
        # Check quality again
        quality = _verify_scaling_quality(nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta)
        condition_number = quality.get("condition_number", np.inf)
        
        if reporter:
            strategy_msg = f" (best: {best_strategy})" if best_strategy else ""
            improvement = prev_condition_number / condition_number if condition_number > 0 else 1.0
            reporter.info(
                f"Iteration {iteration + 1} quality: condition_number={condition_number:.3e}, "
                f"quality_score={quality.get('quality_score', 0.0):.3f}, "
                f"improvement={improvement:.2f}x{strategy_msg}"
            )
        
        # If condition number is acceptable, save and return refined scaling
        if condition_number <= target_condition_number:
            if reporter:
                reporter.info(f"Scaling refinement converged after {iteration + 1} iterations")
            _save_scaling_cache(cache_key, scale, scale_g, quality)
            return scale, scale_g, quality
        
        # Check if we made significant improvement
        # For very high condition numbers, require at least 10x improvement
        # For moderate condition numbers, require at least 1.01x improvement
        improvement_ratio = prev_condition_number / condition_number if condition_number > 0 else 1.0
        required_improvement = 10.0 if prev_condition_number > 1e20 else 1.01
        
        if improvement_ratio < required_improvement:
            # No significant improvement
            if best_strategy is None:
                if reporter:
                    reporter.debug("No strategy improved scaling, stopping refinement")
                break
            # If we improved but not enough, continue to next iteration
            # (might need multiple iterations to reach target)
        else:
            # We improved significantly, update previous condition number
            prev_condition_number = condition_number
    
    # Max iterations reached
    if reporter:
        reporter.warning(
            f"Scaling refinement reached max iterations ({max_iterations}). "
            f"Final condition_number={condition_number:.3e} (target < {target_condition_number:.3e})"
        )
    
    # Save best scaling found (even if not perfect)
    # Only save if it's better than what we started with
    if condition_number < initial_condition_number:
        _save_scaling_cache(cache_key, scale, scale_g, quality)
    
    return scale, scale_g, quality


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
                    # Normalize scaled Jacobian row to O(1): scale_g[i] should be 1.0 / jac_norm_i
                    # But we also need to account for constraint value magnitude
                    # Use max of constraint value magnitude and Jacobian row norm
                    magnitude = max(magnitude, jac_norm_i)
                    
                    # Identify penalty constraints (rows with very large Jacobian norms)
                    # These are likely clearance penalties with 1e6 stiffness
                    penalty_threshold = ref_jac_norm * 1e3  # 1000x larger than typical
                    if jac_norm_i > penalty_threshold:
                        # For penalty constraints, aggressively normalize Jacobian entries
                        # Target: scaled Jacobian elements should be O(1)
                        magnitude = max(magnitude, jac_norm_i / 1e3)  # More aggressive normalization
            
            # Compute scale factor: scale_g[i] = 1.0 / max(|g0[i]|, ||J_scaled_row[i]||)
            # This ensures scaled Jacobian elements are O(1)
            if magnitude > 1e-6:
                scale_g[i] = 1.0 / magnitude
            else:
                scale_g[i] = 1.0
            
            # Cap scale factors to tighter range to limit condition number
            # Range 1e-2 to 1e2 gives max condition number of 1e4 (tighter than before)
            # This helps achieve target condition number < 1e6
            scale_g[i] = np.clip(scale_g[i], 1e-2, 1e2)
        
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
                # Limit adjustment to tighter range: [0.316, 3.16] for 10¹ ratio
                scale_ratio = scale_median / scale_g_median
                sqrt_10 = np.sqrt(10.0)
                scale_ratio = np.clip(scale_ratio, 1.0 / sqrt_10, sqrt_10)
                scale_g = scale_g * scale_ratio
        
        # Normalize scale factors to reduce extreme ratios
        # Use percentile-based normalization to prevent outliers
        scale_g_log = np.log10(np.maximum(scale_g, 1e-10))
        p25 = np.percentile(scale_g_log, 25)
        p75 = np.percentile(scale_g_log, 75)
        iqr = p75 - p25
        
        # Clip outliers with tighter bounds: cap to ±1 log unit from median (10¹ ratio)
        median_log = np.median(scale_g_log)
        sqrt_10_log = np.log10(np.sqrt(10.0))  # ≈ 0.5 log units
        lower_bound = max(median_log - sqrt_10_log, -2.0)  # At least 1e-2
        upper_bound = min(median_log + sqrt_10_log, 2.0)   # At most 1e2
        
        n_clipped = 0
        for i in range(n_cons):
            if scale_g_log[i] < lower_bound:
                scale_g[i] = 10.0 ** lower_bound
                n_clipped += 1
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = 10.0 ** upper_bound
                n_clipped += 1
        
        # Apply clamping strategy similar to variables: ensure constraint scales stay within [median/sqrt(10), median*sqrt(10)]
        # This gives 10¹ ratio (tighter than before)
        scale_g_median = np.median(scale_g[scale_g > 1e-10])
        if scale_g_median > 1e-10:
            sqrt_10 = np.sqrt(10.0)
            for i in range(n_cons):
                if scale_g[i] > 1e-10:
                    scale_g[i] = np.clip(scale_g[i], scale_g_median / sqrt_10, scale_g_median * sqrt_10)
        
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
