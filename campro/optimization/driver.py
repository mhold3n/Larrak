from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from campro.diagnostics.scaling import (
    build_scaled_nlp,
    scale_bounds,
    scale_value,
    unscale_value,
)

# from campro_unaligned.freepiston.zerod.cv import cv_residual
from campro.logging import get_logger
from campro.optimization.core.solution import Solution
from campro.optimization.core.states import MechState
from campro.optimization.initialization.manager import InitializationManager
from campro.optimization.initialization.setup import (
    SCALING_GROUP_CONFIG,
    setup_optimization_bounds,
)
from campro.optimization.io.save import save_json
from campro.optimization.nlp import build_collocation_nlp
from campro.optimization.solvers.ipopt_solver import (
    IPOPTOptions,
    IPOPTSolver,
    get_default_ipopt_options,
    get_robust_ipopt_options,
)
from campro.utils import format_duration
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)

_FALSEY = {"0", "false", "no", "off"}

# Constraint type category targets for scaling
# These targets represent the desired magnitude of scaled constraints per category
CONSTRAINT_TYPE_TARGETS = {
    "kinematic": 1.0,  # Position, velocity, acceleration constraints
    "thermodynamic": 1.0,  # Reduced from 10.0 to lower weight/priority
    "boundary": 0.1,  # Boundary conditions (tighter)
    "continuity": 1.0,  # Continuity constraints
}


def _get_available_hsl_solvers() -> set[str]:
    """Return the set of available HSL solvers (best-effort detection)."""
    try:
        from campro.environment.hsl_detector import (
            clear_cache,
            detect_available_solvers,
        )
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
    combined_scale = max(1.0, max(var_scale, cons_scale**0.5))
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


def _to_numpy_array(data: Any) -> np.ndarray[Any, Any]:
    if data is None:
        return np.array([], dtype=float)  # type: ignore[return-value]
    try:
        arr = np.asarray(data, dtype=float)
        return arr.flatten()  # type: ignore[return-value]
    except Exception:
        try:
            return np.array([float(x) for x in data], dtype=float)  # type: ignore[return-value]
        except Exception:
            return np.array([], dtype=float)  # type: ignore[return-value]


def _summarize_ipopt_iterations(
    stats: dict[str, Any], reporter: StructuredReporter
) -> dict[str, Any] | None:
    iterations = stats.get("iterations")
    if not iterations or not isinstance(iterations, dict):
        if reporter.show_debug:
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
    # Type ignore for dict access - summary is properly typed but mypy doesn't understand nested dict access

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
                    f"inf_du={entry['inf_du']:.3e} mu={entry['mu']:.3e} step={entry['step']}",
                )
        summary["recent_iterations"] = recent_entries

    return summary


class CombinedDenseOutput:
    """Helper to combine geometry and thermo dense outputs for NLP."""

    def __init__(self, geom_res: dict[str, Any], sol_thermo: Any, cycle_time: float) -> None:
        self.geom_res = geom_res
        self.sol_thermo = sol_thermo
        self.cycle_time = cycle_time
        # Setup Geometry Interp

        if "spline" in geom_res:
            self.x_func = lambda t: geom_res["spline"](t)
            self.v_func = lambda t: geom_res["spline"](t, nu=1)
        else:
            self.x_func = interp1d(
                geom_res["t"], geom_res["x"], kind="cubic", fill_value="extrapolate"
            )
            self.v_func = interp1d(
                geom_res["t"], geom_res["v"], kind="cubic", fill_value="extrapolate"
            )

    def sol(self, t: Any) -> np.ndarray:
        # t can be scalar or array
        t_arr = np.asarray(t)
        # Clip/Mod time? nlp.py handles bounds, but periodicity might be needed if t > cycle_time
        # nlp.py usually queries within [0, cycle_time] or close to it.
        t_mod = t_arr % self.cycle_time

        # Kinematics
        x_val = self.x_func(t_mod)
        v_val = self.v_func(t_mod)

        # Thermo [rho, T] or [rho_vec, u_vec, E_vec]
        therm = self.sol_thermo.sol(t_mod)

        # Combine [x, v, therm...]
        # Handle shapes
        if t_arr.ndim == 0:
            return np.hstack([x_val, v_val, therm])
        else:
            return np.vstack([x_val, v_val, therm])


def solve_cycle(params: dict[str, Any]) -> dict[str, Any]:
    """
    Solve OP engine cycle optimization using IPOPT.

    This function builds the collocation NLP and solves it using IPOPT
    with appropriate options for OP engine optimization.

    Args:
        params: Problem parameters dictionary

    Returns:
        Solution object with optimization results
    """
    num = params.get("num", {})
    num_intervals = int(num.get("K", 10))
    poly_degree = int(num.get("C", 3))
    # grid = make_grid(num_intervals, poly_degree, kind="radau")
    # Placeholder grid since make_grid is missing
    grid = {"K": num_intervals, "C": poly_degree, "kind": "radau"}

    iteration_summary: dict[str, Any] = {}

    # Check if combustion model is enabled
    combustion_cfg = params.get("combustion", {})
    use_combustion = bool(combustion_cfg.get("use_integrated_model", False))

    # Configure logging
    log_file = "optimization.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)

    # Get logger and add handler
    logger = logging.getLogger("campro.optimization")
    logger.setLevel(logging.DEBUG)

    # Force specific loggers to DEBUG (overriding get_logger default INFO)
    logging.getLogger("campro.optimization.solvers.ipopt_solver").setLevel(logging.DEBUG)
    logging.getLogger("campro.optimization.driver").setLevel(logging.DEBUG)
    # Also set the module-level log variable in driver to DEBUG
    log.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates if called multiple times
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler) and h.baseFilename.endswith(log_file):
            logger.removeHandler(h)
    logger.addHandler(file_handler)

    reporter = StructuredReporter(
        context="FREE-PISTON",
        logger=logger,
        stream_out=sys.stderr,
        stream_err=sys.stderr,
        debug_env="FREE_PISTON_DEBUG",
        force_debug=False,
        show_debug=True,  # Generate debug messages for the logger
        console_min_level="INFO",  # Only show INFO+ on console
    )

    # Log problem parameters before building NLP
    reporter.info(
        f"Building collocation NLP: K={num_intervals}, C={poly_degree}, combustion_model={'enabled' if use_combustion else 'disabled'}",
    )
    # Estimate complexity
    estimated_vars = (
        num_intervals * poly_degree * 6 + 6 + num_intervals * poly_degree * 2
    )  # Rough estimate
    estimated_constraints = num_intervals * poly_degree * 4  # Rough estimate
    reporter.info(
        f"Estimated problem size: vars≈{estimated_vars}, constraints≈{estimated_constraints}",
    )

    nlp_build_start = time.time()

    # Run Ensemble Initialization Suite
    initial_trajectory = None
    pr_cfg = params.get("planet_ring", {})
    use_load_model = pr_cfg.get("use_load_model", False)

    # Phase 3 Detection
    is_phase3_mechanical = "load_profile" in params

    # Skip ensemble initialization when using load model - it works better with
    # the hypocycloid-aware trajectory generator in generate_physics_trajectory()
    # ALSO Skip if problem_type is kinematic (no thermo)
    problem_type = params.get("problem_type", "default")

    if (
        "planet_ring" in params
        and not use_load_model
        and problem_type != "kinematic"
        and not is_phase3_mechanical
    ):
        reporter.info(
            "Planet-Ring configuration detected. Running Ensemble Initialization Suite..."
        )
        try:
            init_manager = InitializationManager(params)
            init_result = init_manager.solve()

            if init_result["success"]:
                best_cand = init_result["best_candidate"]
                reporter.info(
                    f"Ensemble Initialization converged! Selected: {best_cand['geometry']['method']} + {best_cand['thermo']['method']}"
                )

                # Convert to grid format expected by NLP (K+1 points)
                # Use the dense output solver from the best candidate
                sol_thermo = best_cand["thermo"]["sol"]
                geom_res = best_cand["geometry"]
                cycle_time = geom_res["t"][-1]

                # Create Combined Wrapper
                sol = CombinedDenseOutput(geom_res, sol_thermo, cycle_time)
                t_eval = np.linspace(0, cycle_time, num_intervals + 1)

                # Evaluate solution on grid (Shape: [n_vars, K+1])
                y_eval = sol.sol(t_eval)

                initial_trajectory = {
                    "t": t_eval.tolist(),
                    "xL": y_eval[0].tolist(),
                    "vL": y_eval[1].tolist(),
                    "xR": (-y_eval[0]).tolist(),
                    "vR": (-y_eval[1]).tolist(),
                    "_sol": sol,
                }

                # Unpack Thermo
                n_rows = y_eval.shape[0]
                if n_rows == 4:
                    # 0D Case: x, v, rho, T
                    initial_trajectory["rho"] = y_eval[2].tolist()
                    initial_trajectory["T"] = y_eval[3].tolist()
                elif n_rows > 4:
                    # 1D Case: x, v, rho_vec, u_vec, E_vec
                    n_cells = (n_rows - 2) // 3
                    for i in range(n_cells):
                        initial_trajectory[f"rho_{i}"] = y_eval[2 + i].tolist()
                        initial_trajectory[f"u_{i}"] = y_eval[2 + n_cells + i].tolist()
                        initial_trajectory[f"E_{i}"] = y_eval[2 + 2 * n_cells + i].tolist()

                    # Provide generic keys for fallback/logging (use Cell 0)
                    initial_trajectory["rho"] = y_eval[2].tolist()
                    # T is not directly available in E vector (E = Cv*T).
                    # nlp.py should compute E from T if T provided, or take E directly.
                    # We provided E_i, so nlp.py will use E_i.
                    # We can calc T approx for debug
                    Cv = 718.0
                    initial_trajectory["T"] = (y_eval[2 + 2 * n_cells] / Cv).tolist()
            else:
                reporter.warning(
                    "Ensemble Initialization failed. Falling back to default initialization."
                )
        except Exception as e:
            reporter.warning(f"Ensemble Initialization encountered an error: {e}. Falling back.")

    # Phase 3 Mechanical Initialization
    if is_phase3_mechanical and initial_trajectory is None:
        reporter.info(
            "Phase 3 Mechanical Optimization detected. Generating linear kinematic guess."
        )
        # Psi ramps from 0 to 2*pi*(MeanRatio-1)
        mean_ratio = float(pr_cfg.get("mean_ratio", 2.0))
        target_delta_psi = 2.0 * np.pi * (mean_ratio - 1.0)

        # Grid of K+1 points
        psi_guess = np.linspace(0, target_delta_psi, num_intervals + 1).tolist()

        initial_trajectory = {"psi": psi_guess}

    # Minimal residual evaluation at a nominal state (placeholder)
    # Only meaningful if not Phase 3
    res = None
    if not is_phase3_mechanical:
        mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
        gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5}
        try:
            from campro_unaligned.freepiston.zerod.cv import cv_residual

            res = cv_residual(mech, gas, {"geom": params.get("geom", {}), "flows": {}})
        except Exception:
            # Ignore if signature mismatch in Phase 3 transition
            pass

    # Load Golden Shape for Ratio Tracking
    shape_file = params.get("shape_file")
    target_ratio_profile = None

    if shape_file:
        try:
            import json
            from pathlib import Path

            shape_path = Path(shape_file)
            if not shape_path.exists():
                # Try default location
                shape_path = Path("shapes") / shape_file

            with open(shape_path) as f:
                shape_data = json.load(f)

            raw_profile = shape_data.get("ratio_profile")
            if raw_profile:
                # Interpolate to match K+1 nodes
                # Current logic assumes profile is over time/angle 0..T
                num_intervals = params.get("num", {}).get("K", 20)
                # target needed at K+1 points

                N_raw = len(raw_profile)
                N_target = num_intervals + 1

                if N_raw == N_target:
                    target_ratio_profile = raw_profile
                else:
                    x_raw = np.linspace(0, 1, N_raw)
                    x_target = np.linspace(0, 1, N_target)
                    target_ratio_profile = np.interp(x_target, x_raw, raw_profile).tolist()

                reporter.info(
                    f"Loaded golden shape from {shape_path} (interpolated to {N_target} points)"
                )

                # Update params with the processed profile
                params["target_ratio_profile"] = target_ratio_profile

        except Exception as e:
            reporter.warning(f"Failed to load shape file {shape_file}: {e}")

    # Build NLP
    try:
        nlp, meta = build_collocation_nlp(params, initial_trajectory=initial_trajectory)
        nlp_build_elapsed = time.time() - nlp_build_start

        # Extract actual problem size from meta
        n_vars = meta.get("n_vars", 0) if meta else 0
        n_constraints = meta.get("n_constraints", 0) if meta else 0
        reporter.info(
            f"NLP built in {format_duration(nlp_build_elapsed)}: n_vars={n_vars}, n_constraints={n_constraints}",
        )
        if meta:
            reporter.info(
                f"Problem characteristics: K={meta.get('K', num_intervals)}, C={meta.get('C', poly_degree)}, "
                f"combustion_model={'integrated' if use_combustion else 'none'}",
            )

        # Get solver options
        ipopt_opts_dict = params.get("solver", {}).get("ipopt", {})
        warm_start = params.get("warm_start", {})

        # Check for diagnostic mode via environment variable
        if os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1":
            ipopt_opts_dict.setdefault("ipopt.derivative_test", "first-order")
            ipopt_opts_dict.setdefault("ipopt.print_level", 12)
            reporter.info("Diagnostic mode enabled: derivative test and verbose output activated")

        # Create IPOPT solver
        reporter.info("Creating IPOPT solver with options...")
        solver_create_start = time.time()
        ipopt_options = _create_ipopt_options(ipopt_opts_dict, params)
        solver_wrapper = IPOPTSolver(ipopt_options)
        # Metadata will be stored later after NLP is built
        solver_create_elapsed = time.time() - solver_create_start
        reporter.info(
            f"IPOPT solver created in {format_duration(solver_create_elapsed)}: "
            f"max_iter={ipopt_options.max_iter}, tol={ipopt_options.tol:.2e}, "
            f"print_level={ipopt_options.print_level}",
        )
        if os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1":
            reporter.info("Derivative test enabled. To disable: unset FREE_PISTON_DIAGNOSTIC_MODE")

        # Set up initial guess and bounds
        reporter.info("Setting up optimization bounds and initial guess...")
        # Ensure dimensions are available
        if n_vars == 0 or n_constraints == 0:
            if isinstance(nlp, dict):
                if "x" in nlp:
                    n_vars = nlp["x"].shape[0] if hasattr(nlp["x"], "shape") else nlp["x"].size1()
                if "g" in nlp:
                    n_constraints = (
                        nlp["g"].shape[0] if hasattr(nlp["g"], "shape") else nlp["g"].size1()
                    )
            elif hasattr(nlp, "size1_in"):
                n_vars = nlp.size1_in(0)
                n_constraints = nlp.size1_out(0)

        # Set up initial guess and bounds
        reporter.info("Setting up optimization bounds and initial guess...")
        # Note: builder is None here as we are using the legacy driver flow which builds the NLP directly
        x0, lbx, ubx, lbg, ubg, p = setup_optimization_bounds(
            n_vars,
            n_constraints,
            params,
            builder=None,
            warm_start=warm_start,
            meta=meta,
        )

        if x0 is None or lbx is None or ubx is None:
            raise ValueError("Failed to set up optimization bounds and initial guess")

        # Get variable groups from metadata if available
        variable_groups = meta.get("variable_groups", {}) if meta else {}

        # Store metadata in solver for diagnostics (before scaling, in case scaling fails)
        if meta is not None:
            solver_wrapper._meta_for_diagnostics = meta

        # Compute objective scaling (Gerschgorin estimate for Lagrangian Hessian balance)
        reporter.info("Computing objective scaling factor...")
        # Compute unified data-driven scaling (analyzes distribution, normalizes to median center)
        scale, scale_g, _, scaling_quality = _compute_unified_data_driven_scaling(
            nlp,
            x0,
            lbx,
            ubx,
            lbg,
            ubg,
            variable_groups=variable_groups,
            meta=meta,
            reporter=reporter,
        )

        # Phase 2: Audit constraint rank (SVD) to identify redundant constraints
        # Only run if reporter is available (diagnostic mode)
        if reporter:
            _analyze_constraint_rank(nlp, x0, scale, scale_g, meta=meta, reporter=reporter)

        # Combine objective scaling (from Gerschgorin) with constraint scaling
        # Final objective scale_f = scale_f_obj * scale_f_constraint

        # REDESIGN: Global objective scaling based on gradient magnitude
        # 1. Compute scaled gradient: g_tilde = S^-1 * grad_f(x0)
        # 2. Compute max element: g_max = max(|g_tilde|)
        # 3. Compute w_0 = target / max(g_max, g_min)
        # 4. Clamp w_0 to [1e-3, 1e3]

        scale_f = 1.0
        try:
            import casadi as ca

            if isinstance(nlp, dict) and "f" in nlp and "x" in nlp:
                f_expr = nlp["f"]
                x_sym = nlp["x"]
                grad_f_expr = ca.gradient(f_expr, x_sym)
                grad_f_func = ca.Function("grad_f_func", [x_sym], [grad_f_expr])

                # Evaluate gradient at unscaled initial guess
                grad_f0 = np.array(grad_f_func(x0)).flatten()

                # Apply variable scaling to get scaled gradient: g_tilde = grad_f / scale
                # (Chain rule: d(f)/d(x_scaled) = d(f)/dx * dx/dx_scaled = grad_f * scale_factor)
                # Wait, x = x_scaled * scale_factor => dx/dx_scaled = scale_factor
                # So g_tilde = grad_f * scale
                # Let's verify: f(x) = f(S * x_tilde).
                # grad_tilde = df/dx_tilde = df/dx * dx/dx_tilde = grad_f * S
                # Where S is the diagonal matrix of scale factors (scale array).
                # So g_tilde[i] = grad_f[i] * scale[i]

                g_tilde = grad_f0 * scale
                g_max = np.max(np.abs(g_tilde))

                target_grad = 10.0
                g_min_threshold = 1e-2

                # Compute raw scaling factor
                raw_w0 = target_grad / max(g_max, g_min_threshold)

                # Clamp to safe range [1e-3, 1e3]
                scale_f = float(np.clip(raw_w0, 1e-3, 1e3))

                reporter.info(
                    f"Global objective scaling: g_max={g_max:.2e}, target={target_grad}, "
                    f"raw_w0={raw_w0:.2e}, clamped_w0={scale_f:.2e}"
                )
        except Exception as e:
            reporter.warning(f"Failed to compute global objective scaling: {e}")
            scale_f = 1.0

        reporter.info(f"Final objective scaling factor: {scale_f:.2e}")

        # Log scaling statistics
        if np.any(np.isnan(scale)) or np.any(np.isinf(scale)):
            reporter.error("NaN or Inf detected in variable scaling factors (scale)!")
            # Find which indices are NaN/Inf
            bad_indices = np.where(np.isnan(scale) | np.isinf(scale))[0]
            reporter.error(f"Bad indices in scale: {bad_indices}")
            # Identify groups
            if variable_groups:
                for group_name, indices in variable_groups.items():
                    if any(idx in bad_indices for idx in indices):
                        reporter.error(f"Group '{group_name}' has NaN/Inf scaling factors.")

        scale_min = scale.min()
        scale_max = scale.max()
        scale_mean = scale.mean()
        reporter.info(
            f"Variable scaling factors: min={scale_min:.3e}, max={scale_max:.3e}, mean={scale_mean:.3e}",
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
                            f"ratio={group_range_ratio:.2e}",
                        )
                        if group_range_ratio > 1e6:
                            reporter.warning(
                                f"Group '{group_name}' has extreme scale ratio ({group_range_ratio:.2e}). "
                                f"Consider adjusting unit references or bounds.",
                            )

        if len(scale_g) > 0:
            if np.any(np.isnan(scale_g)) or np.any(np.isinf(scale_g)):
                reporter.error("NaN or Inf detected in constraint scaling factors (scale_g)!")
                bad_indices = np.where(np.isnan(scale_g) | np.isinf(scale_g))[0]
                reporter.error(f"Bad indices in scale_g: {bad_indices}")
                if meta and "constraint_groups" in meta:
                    for group_name, indices in meta["constraint_groups"].items():
                        if any(idx in bad_indices for idx in indices):
                            reporter.error(
                                f"Constraint group '{group_name}' has NaN/Inf scaling factors."
                            )

            scale_g_min = scale_g.min()
            scale_g_max = scale_g.max()
            scale_g_mean = scale_g.mean()
            reporter.info(
                f"Constraint scaling factors: "
                f"min={scale_g_min:.3e}, max={scale_g_max:.3e}, mean={scale_g_mean:.3e}",
            )

            # Group-wise constraint diagnostics
            if meta and "constraint_groups" in meta:
                with reporter.section("Constraint scaling by group"):
                    for group_name, indices in meta["constraint_groups"].items():
                        if indices and len(indices) > 0:
                            # Filter indices to ensure they are within bounds
                            valid_indices = [i for i in indices if 0 <= i < len(scale_g)]
                            if not valid_indices:
                                continue

                            group_scales = scale_g[np.array(valid_indices)]
                            group_min = group_scales.min()
                            group_max = group_scales.max()
                            group_mean = group_scales.mean()
                            reporter.info(
                                f"Group '{group_name}' ({len(valid_indices)} constr): "
                                f"min={group_min:.3e}, max={group_max:.3e}, mean={group_mean:.3e}"
                            )

        # Log final scaling quality
        condition_number = scaling_quality.get("condition_number", np.inf)
        quality_score = scaling_quality.get("quality_score", 0.0)
        reporter.info(
            f"Final scaling quality: condition_number={condition_number:.3e}, "
            f"quality_score={quality_score:.3f}",
        )

        # Objective scaling is already computed in unified function
        if scale_f != 1.0:
            reporter.info(f"Objective scaling factor: {scale_f:.6e}")

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
        x0_scaled: np.ndarray[Any, Any] = np.asarray(scale_value(x0, scale))
        lbx_scaled, ubx_scaled = scale_bounds((lbx, ubx), scale)

        # IMPORTANT: Re-clamp scaled initial guess to scaled bounds
        # The new scaling may be very different from what the initial guess assumed,
        # so we need to ensure x0_scaled satisfies the scaled bounds
        x0_scaled = np.clip(x0_scaled, lbx_scaled, ubx_scaled)

        # Scale constraint bounds
        if lbg is not None and ubg is not None and len(scale_g) > 0:
            lbg_scaled: np.ndarray[Any, Any] = np.asarray(scale_value(lbg, scale_g))
            ubg_scaled: np.ndarray[Any, Any] = np.asarray(scale_value(ubg, scale_g))
        else:
            lbg_scaled = lbg if lbg is not None else np.array([])
            ubg_scaled = ubg if ubg is not None else np.array([])

        if x0 is not None:
            reporter.info(
                f"Initial guess (unscaled): n_vars={len(x0)}, "
                f"x0_range=[{x0.min():.3e}, {x0.max():.3e}], mean={x0.mean():.3e}",
            )
            reporter.info(
                f"Initial guess (scaled): n_vars={len(x0_scaled)}, "
                f"x0_range=[{x0_scaled.min():.3e}, {x0_scaled.max():.3e}], mean={x0_scaled.mean():.3e}",
            )
        # Diagnostic constraint evaluation: check for NaN/Inf before solving
        diagnostic_mode = os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1"
        if diagnostic_mode:
            reporter.info("=" * 80)
            reporter.info("DIAGNOSTIC MODE: Evaluating constraints and Jacobian at initial guess")
            reporter.info("=" * 80)
            try:
                import casadi as ca

                # Evaluate constraints at initial guess
                if isinstance(nlp_scaled, dict) and "g" in nlp_scaled and "x" in nlp_scaled:
                    g_expr = nlp_scaled["g"]
                    x_sym = nlp_scaled["x"]
                    g_func = ca.Function("g_func", [x_sym], [g_expr])
                    g0 = g_func(x0_scaled)
                    g0_arr = np.array(g0).flatten()

                    # Check for NaN/Inf in constraints
                    nan_mask = np.isnan(g0_arr)
                    inf_mask = np.isinf(g0_arr)

                    if np.any(nan_mask) or np.any(inf_mask):
                        reporter.warning("=" * 80)
                        reporter.warning(
                            "NaN/Inf DETECTED in constraint evaluation at initial guess!"
                        )
                        reporter.warning("=" * 80)

                        # Find problematic constraints
                        problematic_rows = np.where(nan_mask | inf_mask)[0]
                        reporter.warning(f"Found {len(problematic_rows)} problematic constraint(s)")

                        # Report first 20 problematic constraints with details
                        for idx, row_idx in enumerate(problematic_rows[:20]):
                            con_type = "unknown"
                            con_name = f"constraint_{row_idx}"

                            # Identify constraint type from metadata
                            if meta and "constraint_groups" in meta:
                                for group_name, indices in meta["constraint_groups"].items():
                                    if row_idx in indices:
                                        con_type = group_name
                                        # Try to get more specific info
                                        if group_name == "collocation_residuals":
                                            # Calculate which time step and collocation point
                                            K = meta.get("K", 0)
                                            C = meta.get("C", 0)
                                            if K > 0 and C > 0:
                                                # Collocation residuals are grouped by (k, c)
                                                # Each collocation point has ~6 constraints
                                                constraints_per_point = 6  # xL, xR, vL, vR, rho, T
                                                k_idx = row_idx // (C * constraints_per_point)
                                                c_idx = (
                                                    row_idx % (C * constraints_per_point)
                                                ) // constraints_per_point
                                                con_name = f"colloc_k{k_idx}_c{c_idx}"
                                        elif group_name == "continuity":
                                            K = meta.get("K", 0)
                                            if K > 0:
                                                k_idx = (
                                                    row_idx // 13
                                                )  # ~13 constraints per continuity
                                                con_name = f"continuity_k{k_idx}"
                                        break

                            value = g0_arr[row_idx]
                            value_str = f"{value:.6e}" if np.isfinite(value) else str(value)
                            reporter.warning(
                                f"  Row {row_idx:4d} ({con_type:20s}, {con_name:25s}): "
                                f"g[{row_idx}] = {value_str} "
                                f"(NaN={nan_mask[row_idx]}, Inf={inf_mask[row_idx]})",
                            )

                        if len(problematic_rows) > 20:
                            reporter.warning(
                                f"  ... and {len(problematic_rows) - 20} more problematic constraints"
                            )

                        # Evaluate Jacobian to find problematic variable columns
                        reporter.warning("-" * 80)
                        reporter.warning(
                            "Evaluating constraint Jacobian to identify problematic variables..."
                        )
                        try:
                            jac_g_expr = ca.jacobian(g_expr, x_sym)
                            jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
                            jac_g0 = jac_g_func(x0_scaled)
                            jac_g0_arr = np.array(jac_g0)

                            # Find columns with NaN/Inf
                            jac_nan_mask = np.isnan(jac_g0_arr)
                            jac_inf_mask = np.isinf(jac_g0_arr)

                            if np.any(jac_nan_mask) or np.any(jac_inf_mask):
                                reporter.warning("NaN/Inf DETECTED in constraint Jacobian!")

                                # Find problematic (row, col) pairs
                                problematic_pairs = np.where(jac_nan_mask | jac_inf_mask)
                                n_problematic = len(problematic_pairs[0])
                                reporter.warning(
                                    f"Found {n_problematic} problematic Jacobian entries"
                                )

                                # Group by column to identify problematic variables
                                problematic_cols: dict[int, list[tuple[int, float]]] = {}
                                for i in range(min(100, n_problematic)):  # Limit to first 100
                                    row_idx = problematic_pairs[0][i]
                                    col_idx = problematic_pairs[1][i]
                                    if col_idx not in problematic_cols:
                                        problematic_cols[col_idx] = []
                                    problematic_cols[col_idx].append(
                                        (row_idx, jac_g0_arr[row_idx, col_idx])
                                    )

                                # Report problematic variables
                                reporter.warning("-" * 80)
                                reporter.warning(
                                    "Problematic variables (columns with NaN/Inf in Jacobian):"
                                )
                                for col_idx in sorted(problematic_cols.keys())[:20]:
                                    var_type = "unknown"
                                    var_name = f"x_{col_idx}"

                                    # Identify variable type from metadata
                                    if meta and "variable_groups" in meta:
                                        for group_name, indices in meta["variable_groups"].items():
                                            if col_idx in indices:
                                                var_type = group_name
                                                # Try to get more specific info
                                                if group_name == "densities":
                                                    # Check if it's log-space or physical-space
                                                    if "use_log_density" in str(meta):
                                                        var_name = f"rho_log_{col_idx}"
                                                    else:
                                                        var_name = f"rho_{col_idx}"
                                                elif group_name == "positions":
                                                    var_name = f"x_{col_idx}"
                                                elif group_name == "velocities":
                                                    var_name = f"v_{col_idx}"
                                                elif group_name == "temperatures":
                                                    var_name = f"T_{col_idx}"
                                                elif group_name == "valve_areas":
                                                    var_name = f"A_{col_idx}"
                                                break

                                    x0_val = (
                                        x0_scaled[col_idx] if col_idx < len(x0_scaled) else np.nan
                                    )
                                    x0_str = f"{x0_val:.6e}" if np.isfinite(x0_val) else str(x0_val)
                                    n_entries = len(problematic_cols[col_idx])
                                    reporter.warning(
                                        f"  Col {col_idx:4d} ({var_type:15s}, {var_name:20s}): "
                                        f"x0[{col_idx}] = {x0_str}, "
                                        f"{n_entries} NaN/Inf entries in Jacobian",
                                    )

                                    # Show first few problematic rows for this variable
                                    for row_idx, jac_val in problematic_cols[col_idx][:3]:
                                        jac_str = (
                                            f"{jac_val:.6e}"
                                            if np.isfinite(jac_val)
                                            else str(jac_val)
                                        )
                                        reporter.warning(
                                            f"    -> J[{row_idx:4d}, {col_idx:4d}] = {jac_str}",
                                        )

                                if len(problematic_cols) > 20:
                                    reporter.warning(
                                        f"  ... and {len(problematic_cols) - 20} more problematic variables"
                                    )
                            else:
                                reporter.info("No NaN/Inf detected in constraint Jacobian")
                        except Exception as jac_exc:
                            reporter.warning(f"Failed to evaluate Jacobian: {jac_exc}")
                    else:
                        reporter.info("✓ No NaN/Inf detected in constraint evaluation")
                        try:
                            # Compute constraint violations
                            if g0_arr.shape != lbg_scaled.shape:
                                reporter.warning(
                                    f"Shape mismatch: g0_arr={g0_arr.shape}, lbg_scaled={lbg_scaled.shape}, ubg_scaled={ubg_scaled.shape}"
                                )
                                # Try to reshape if size matches
                                if g0_arr.size == lbg_scaled.size:
                                    g0_arr = g0_arr.reshape(lbg_scaled.shape)

                            viol_lb = np.maximum(0, lbg_scaled - g0_arr)
                            viol_ub = np.maximum(0, g0_arr - ubg_scaled)
                            viol = np.maximum(viol_lb, viol_ub)
                            max_viol = np.max(viol)
                            mean_viol = np.mean(viol)
                            num_viol = np.sum(viol > 1e-6)

                            reporter.info("  Primal Infeasibility (Initial Guess):")
                            reporter.info(f"    Max Violation: {float(max_viol):.6e}")
                            reporter.info(f"    Mean Violation: {float(mean_viol):.6e}")
                            reporter.info(
                                f"    Violated Constraints (>1e-6): {num_viol}/{len(g0_arr)}"
                            )
                        except Exception as e:
                            reporter.warning(f"Failed to compute violations: {e}")
                            reporter.warning(
                                f"Shapes: g0_arr={g0_arr.shape}, lbg_scaled={lbg_scaled.shape}"
                            )

                        if max_viol > 1e-6:
                            # Show top violations
                            top_indices = np.argsort(viol)[-5:][::-1]
                            reporter.info("    Top 5 Violations:")
                            for idx in top_indices:
                                reporter.info(
                                    f"      Con {idx}: viol={viol[idx]:.6e} (val={g0_arr[idx]:.6e}, bounds=[{lbg_scaled[idx]:.6e}, {ubg_scaled[idx]:.6e}])"
                                )
                else:
                    reporter.warning("Cannot evaluate constraints: NLP not in expected dict format")
            except Exception as diag_exc:
                reporter.warning(f"Diagnostic evaluation failed: {diag_exc}")
                import traceback

                reporter.debug(traceback.format_exc())

            reporter.info("=" * 80)

        if lbx_scaled is not None and ubx_scaled is not None:
            bounded_vars = ((lbx_scaled > -np.inf) & (ubx_scaled < np.inf)).sum()
            reporter.info(
                f"Bounded variables: {bounded_vars}/{len(lbx_scaled) if lbx_scaled is not None else 0}",
            )

        # Solve optimization problem with scaled NLP
        reporter.info("Starting IPOPT optimization (with comprehensive scaling)...")
        reporter.debug(
            f"Problem dimensions: n_vars={len(x0_scaled) if x0_scaled is not None else 0}, "
            f"n_constraints={len(lbg_scaled) if lbg_scaled is not None else 0}",
        )
        solve_start = time.time()

        # Try to solve with selected solver, fall back to MA27 if MA57 symbols not found
        selected_solver = ipopt_options.linear_solver.lower()
        result = solver_wrapper.solve(
            nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled, p
        )

        # Check if solve failed immediately (0 iterations) with MA57 - likely symbol loading failure
        # If MA57 was selected and solve fails immediately without any iterations, fall back to MA27
        if selected_solver == "ma57" and not result.success and result.iterations == 0:
            reporter.warning(
                f"MA57 solver failed immediately (status={result.status}, message={result.message}). "
                "Likely symbols not found in HSL library. Falling back to MA27...",
            )
            # Recreate solver with MA27 - modify options and recreate solver
            ipopt_options.linear_solver = "ma27"
            n_vars_actual = len(x0_scaled) if x0_scaled is not None else 0
            n_constraints_actual = len(lbg_scaled) if lbg_scaled is not None else 0
            _configure_ma27_memory(ipopt_options, n_vars_actual, n_constraints_actual)
            solver_wrapper = IPOPTSolver(ipopt_options)
            result = solver_wrapper.solve(
                nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled, p
            )

        solve_elapsed = time.time() - solve_start
        reporter.info(f"IPOPT solve completed in {format_duration(solve_elapsed)}")
        # Access stats through the result object (IPOPTSolver.solve() already extracted stats)
        # Create a stats dict compatible with _summarize_ipopt_iterations
        # Access stats through the result object (IPOPTSolver.solve() already extracted stats)
        # Create a stats dict compatible with _summarize_ipopt_iterations
        stats = {
            "iterations": {
                "k": np.array([result.iterations]) if result.iterations > 0 else np.array([]),
                "obj": np.array([result.f_opt]),
                "inf_pr": np.array([result.feasibility_error]),
                "inf_du": np.array([result.kkt_error]),
            }
            if result.iterations > 0
            else {},
        }
        iteration_summary = _summarize_ipopt_iterations(stats, reporter) or {}

        # Initialize optimization_result variable before try block
        optimization_result: dict[str, Any] = {}

        # Unscale solution
        # Unscale solution (always, even if failed, so we can debug)
        # Unscale solution for interpretation
        x_opt_unscaled: np.ndarray | None = None
        if result.x_opt is not None and len(result.x_opt) > 0:
            x_opt_unscaled = np.asarray(unscale_value(result.x_opt, scale))

            # Unscale objective if it was scaled
            f_opt_unscaled = result.f_opt / scale_f if scale_f != 1.0 else result.f_opt
            if x_opt_unscaled is not None:
                reporter.info(
                    f"Solution unscaled: x_opt_range=[{x_opt_unscaled.min():.3e}, {x_opt_unscaled.max():.3e}], "
                    f"f_opt={f_opt_unscaled:.6e}",
                )
        else:
            x_opt_unscaled = None
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
            # Add multipliers and infeasibilities for diagnostic analysis
            "primal_inf": result.primal_inf,
            "dual_inf": result.dual_inf,
            "lam_g": result.lambda_opt,  # Constraint multipliers
            "lam_x": getattr(result, "lam_x", None),  # Bound multipliers (if available)
            "complementarity": result.complementarity,
            "constraint_violation": result.constraint_violation,
        }

        # Extract T_cycle if available
        if x_opt_unscaled is not None and "variable_groups" in meta:
            cycle_time_indices = meta["variable_groups"].get("cycle_time", [])
            if cycle_time_indices and len(cycle_time_indices) > 0:
                t_cycle_idx = cycle_time_indices[0]
                if 0 <= t_cycle_idx < len(x_opt_unscaled):
                    t_cycle_val = float(x_opt_unscaled[t_cycle_idx])
                    optimization_result["T_cycle"] = t_cycle_val
                    reporter.info(f"Found Cycle Time: {t_cycle_val:.6f} s")

        # --- Generate Conjugate Profiles ---
        if x_opt_unscaled is not None and "get_profiles" in meta and "variables_detailed" in meta:
            try:
                reporter.info("Generating Conjugate Gear Profiles...")

                indices_map = meta["variables_detailed"]

                # Extract PSI (High Res State)
                if "psi" in indices_map:
                    psi_idx = indices_map["psi"]
                    psi_vals = x_opt_unscaled[psi_idx]
                else:
                    raise KeyError("psi state not found")

                # Extract RADIUS CONTROLS (Low Res)
                if "r_planet" in indices_map and "R_ring" in indices_map:
                    r_idx = indices_map["r_planet"]
                    R_idx = indices_map["R_ring"]
                    r_vals_coarse = x_opt_unscaled[r_idx]
                    R_vals_coarse = x_opt_unscaled[R_idx]
                else:
                    # Fallback for old ratio based? Or error?
                    # Since we updated NLP, we expect radii.
                    # But check if i_ratio exists for legacy support?
                    if "i_ratio" in indices_map:
                        # Legacy mode not supported by new get_profiles signature
                        raise KeyError(
                            "Legacy i_ratio found but get_profiles expects radii. Model mismatch."
                        )
                    raise KeyError("Radius controls (r_planet, R_ring) not found")

                # Extract TIME (High Res)
                if "t_cycle" in indices_map:
                    t_idx = indices_map["t_cycle"]
                    t_vals = x_opt_unscaled[t_idx]
                else:
                    # Fallback to linear grid
                    K_est = len(r_vals_coarse)
                    # Use T_cycle if found, else guess
                    cycle_time_est = optimization_result.get("T_cycle", 0.02)
                    t_vals = np.linspace(0, cycle_time_est, len(psi_vals))

                # Upsample Radii to match psi length
                from scipy.interpolate import interp1d

                # Reconstruct time grid for intervals
                K_est = len(r_vals_coarse)
                t_grid_coarse = np.linspace(t_vals[0], t_vals[-1], K_est + 1)

                # Use 'previous' interpolation: val[k] holds for [t_k, t_k+1)
                f_r = interp1d(
                    t_grid_coarse[:-1],
                    r_vals_coarse,
                    kind="previous",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                f_R = interp1d(
                    t_grid_coarse[:-1],
                    R_vals_coarse,
                    kind="previous",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

                r_vals_fine = f_r(t_vals)
                R_vals_fine = f_R(t_vals)

                # Computed Ratio for plotting
                i_vals_fine = R_vals_fine / (r_vals_fine + 1e-9)

                # Evaluate CasADi function
                f_prof = meta["get_profiles"]
                # New Signature: [r, R, t, psi]
                res = f_prof(r_vals_fine, R_vals_fine, t_vals, psi_vals)

                # Unpack and store coordinates
                profiles = {
                    "Px": np.array(res[0]).flatten().tolist(),
                    "Py": np.array(res[1]).flatten().tolist(),
                    "Rx": np.array(res[2]).flatten().tolist(),
                    "Ry": np.array(res[3]).flatten().tolist(),
                    # Store vectors for plotting
                    "t": t_vals.tolist(),
                    "i": i_vals_fine.tolist(),
                    "r": r_vals_fine.tolist(),
                    "R": R_vals_fine.tolist(),
                    "psi": psi_vals.tolist(),
                }

                optimization_result["profiles"] = profiles
                reporter.info(f"Generated conjugate profiles (N={len(t_vals)})")

            except Exception as e:
                reporter.warning(f"Failed to generate conjugate profiles: {e}")

        # --- Phase 3: Mechanical Optimization Results ---
        if is_phase3_mechanical and x_opt_unscaled is not None and "variable_groups" in meta:
            try:
                reporter.info("Processing Phase 3 Mechanical Results...")
                var_groups = meta["variable_groups"]

                # Helper to extract variable by name
                def get_var(name):
                    idx = var_groups.get(name)
                    if idx is None:
                        return None
                    if isinstance(idx, slice):
                        return x_opt_unscaled[idx]
                    return x_opt_unscaled[idx]

                psi_vals = get_var("psi")
                r_vals = get_var("r_planet")
                R_vals = get_var("R_ring")

                if psi_vals is not None and r_vals is not None and R_vals is not None:
                    # Grid (Angle Domain)
                    phi_grid = meta.get("time_grid")
                    if phi_grid is None:
                        K_val = len(r_vals)
                        phi_grid = np.linspace(0, 2 * np.pi, K_val + 1)

                    # Handle Collocation Points in psi_vals
                    # psi_vals contains [X0, Xc_0_1...Xc_0_C, X1, ...]
                    # We only want the grid points X0, X1, ...
                    C_val = meta.get("C", 3)
                    psi_grid_points = psi_vals[:: C_val + 1]

                    # Ensure shapes align
                    # Controls r, R are K points
                    # psi_grid_points should be K+1 points

                    K_intervals = len(r_vals)
                    phi_eval = phi_grid[:-1]  # Start of intervals

                    # Truncate psi to K points (start of intervals)
                    if len(psi_grid_points) > K_intervals:
                        psi_eval = psi_grid_points[:K_intervals]
                    else:
                        # Fallback if logic mismatch
                        psi_eval = psi_grid_points

                    # Interpolate Load Profile
                    F_gas_eval = np.zeros(K_intervals)
                    load_prof = params.get("load_profile")
                    if load_prof:
                        # Linear interp: assumes load_prof['angle'] is sorted 0..2pi
                        F_gas_eval = np.interp(phi_eval, load_prof["angle"], load_prof["F_gas"])

                    # Kinematics
                    # xL = (R - r) * cos(psi)
                    x_L_eval = (R_vals - r_vals) * np.cos(psi_eval)

                    # dx/dphi approx = - (R - r) * sin(psi) * (R/r - 1)
                    i_ratio = R_vals / (r_vals + 1e-9)
                    dx_dphi_eval = -(R_vals - r_vals) * np.sin(psi_eval) * (i_ratio - 1.0)

                    # Torque Output (Ideal)
                    T_out_ideal = F_gas_eval * dx_dphi_eval

                    # Friction
                    alpha = 20.0 * (np.pi / 180.0)
                    mu = 0.05
                    N_c = np.abs(F_gas_eval) / np.cos(alpha)
                    T_loss = mu * N_c * r_vals

                    # Net Torque
                    T_out_net = T_out_ideal - np.sign(T_out_ideal) * np.abs(
                        T_loss
                    )  # Opposes motion?
                    # Actually T_out is driven by Gas. T_loss consumes torque.
                    # Work_Net = Work_Gas - Work_Friction
                    # T_net = T_ideal - T_loss (assuming T_ideal > 0 and loss reduces it?)
                    # If T_ideal < 0 (compression), we need to put IN torque.
                    # T_in_required = T_ideal - T_loss (more negative? or simpler logic?)
                    # Let's just report T_loss magnitude.

                    # Efficiency (Instantaneous, where T_out > 0)
                    # eta = (T_out - T_loss) / T_out
                    # Handle divide by zero
                    efficiency = np.ones_like(T_out_ideal)
                    mask_power = np.abs(T_out_ideal) > 1e-6
                    # Simple efficiency: 1 - |T_loss / T_ideal|
                    efficiency[mask_power] = 1.0 - np.abs(
                        T_loss[mask_power] / T_out_ideal[mask_power]
                    )

                    # Stress Proxy: Conformity (1/r + 1/R)
                    curvature_sum = 1.0 / r_vals + 1.0 / R_vals

                    mech_results = {
                        "phi": phi_eval.tolist(),
                        "psi": psi_eval.tolist(),
                        "r": r_vals.tolist(),
                        "R": R_vals.tolist(),
                        "i_ratio": i_ratio.tolist(),
                        "x_L": x_L_eval.tolist(),
                        "F_gas": F_gas_eval.tolist(),
                        "T_out_ideal": T_out_ideal.tolist(),
                        "T_loss": T_loss.tolist(),
                        "efficiency": efficiency.tolist(),
                        "curvature_sum": curvature_sum.tolist(),
                        "mean_efficiency": float(np.mean(efficiency[efficiency > 0])),
                        "mean_torque": float(np.mean(T_out_ideal)),
                    }

                    optimization_result["mechanical"] = mech_results
                    reporter.info(
                        f"Mechanical Summary: Mean Eff={mech_results['mean_efficiency']:.2%}, "
                        f"Mean Torque={mech_results['mean_torque']:.2f} Nm"
                    )

            except Exception as e:
                reporter.warning(f"Failed to process mechanical results: {e}")

        # Optional checkpoint save per iteration group (best-effort minimal)
        # Optional checkpoint save per iteration group (best-effort minimal)
        run_dir = params.get("run_dir")
        if run_dir:
            try:
                save_json(
                    {"meta": meta, "opt": optimization_result},
                    run_dir,
                    filename="checkpoint.json",
                )
            except Exception as exc:  # pragma: no cover
                reporter.warning(f"Checkpoint save failed: {exc}")

        # Prepare Solution Object
        final_meta = {
            "meta": meta,
            "optimization": optimization_result,
        }
        final_data = {
            "x": optimization_result["x_opt"],
        }
        final_solution = Solution(meta=final_meta, data=final_data)

        # --- Auto-Plot Generation ---
        # Automatically generate plots if requested in configuration
        # This ensures visualization happens "once the collocation converges"
        if params.get("auto_plot", False):
            try:
                # Lazy import to avoid circular dependency at top level if any
                # Lazy import to avoid circular dependency at top level if any
                # from plot_planet_ring_results import plot_results

                # Get output directories from config or default
                plot_dirs = params.get("plot_dirs", ["plots"])

                reporter.info("Auto-generating plots...")
                # plot_results(final_solution, output_dirs=plot_dirs)

            except Exception as e:
                reporter.warning(f"Auto-plot failed: {e}")

        return final_solution

    except Exception as e:
        reporter.error(f"Failed to build or solve NLP: {e!s}")
        nlp, meta = None, None
        # optimization_result is already initialized before try block
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

    # Return Solution object (not dict) - type annotation is incorrect but kept for compatibility
    solution = Solution(
        meta={"grid": grid, "meta": meta, "optimization": optimization_result},
        data={
            "residual_sample": res,
            "nlp": nlp,
            # Add optimization variables and multipliers for diagnostics
            "x": optimization_result.get("x_opt"),
            "lam_g": optimization_result.get("lam_g"),
            "lam_x": optimization_result.get("lam_x"),
            "initial_trajectory": initial_trajectory,
        },
    )
    return solution  # type: ignore[return-value]


def _create_ipopt_options(ipopt_opts_dict: dict[str, Any], params: dict[str, Any]) -> IPOPTOptions:
    """Create IPOPTOptions from dictionary and params."""
    # Start with robust options as baseline
    options = get_robust_ipopt_options()

    # Get problem parameters
    num = params.get("num", {})
    num_intervals = int(num.get("K", 10))
    poly_degree = int(num.get("C", 3))

    # Override with user-specified options

    for key, value in ipopt_opts_dict.items():
        # Handle ipopt. prefix in config keys
        if key.startswith("ipopt."):
            attr_name = key[6:]  # Remove 'ipopt.' prefix
        else:
            attr_name = key

        if hasattr(options, attr_name):
            setattr(options, attr_name, value)
        else:
            log.warning(f"Unknown IPOPT option: {key} (attribute: {attr_name})")

    # Set default options for common parameters if not already set by user
    # These are applied after user overrides to ensure user-specified values take precedence
    if not hasattr(options, "max_iter") or options.max_iter is None:
        options.max_iter = int(ipopt_opts_dict.get("max_iter", 500))
    if not hasattr(options, "tol") or options.tol is None:
        options.tol = float(ipopt_opts_dict.get("tol", 1e-6))
    if not hasattr(options, "print_level") or options.print_level is None:
        options.print_level = int(ipopt_opts_dict.get("print_level", 5))

    # Remove CPU time limit unless explicitly requested
    # User requested to only stop on plateau
    if "max_cpu_time" not in ipopt_opts_dict:
        options.max_cpu_time = 1e10

    # Add scaling options if not already set by user
    if not hasattr(options, "nlp_scaling_method") or options.nlp_scaling_method is None:
        options.nlp_scaling_method = str(ipopt_opts_dict.get("nlp_scaling_method", "none"))
    if not hasattr(options, "nlp_scaling_max_gradient") or options.nlp_scaling_max_gradient is None:
        options.nlp_scaling_max_gradient = float(
            ipopt_opts_dict.get("nlp_scaling_max_gradient", 100.0)
        )
    if not hasattr(options, "obj_scaling_factor") or options.obj_scaling_factor is None:
        options.obj_scaling_factor = float(ipopt_opts_dict.get("obj_scaling_factor", 1.0))

    # Set linear solver if not already set by user
    if not hasattr(options, "linear_solver") or options.linear_solver is None:
        options.linear_solver = str(ipopt_opts_dict.get("linear_solver", "ma57"))

    # Handle solver wrapper options (not IPOPT options, but IPOPTSolver options)
    solver_cfg = params.get("solver", {})
    if "plateau_check_enabled" in solver_cfg:
        options.plateau_check_enabled = bool(solver_cfg["plateau_check_enabled"])
    if "plateau_eps" in solver_cfg:
        options.plateau_eps = float(solver_cfg["plateau_eps"])
    if "plateau_window_size" in solver_cfg:
        options.plateau_window_size = int(solver_cfg["plateau_window_size"])

    # Adjust options based on problem size
    num = params.get("num", {})
    num_intervals = int(num.get("K", 10))
    poly_degree = int(num.get("C", 3))

    # Estimate problem size
    n_vars = (
        num_intervals * poly_degree * 6
    )  # Rough estimate: K collocation points, C stages, 6 variables per point
    n_constraints = (
        num_intervals * poly_degree * 4
    )  # Rough estimate: 4 constraints per collocation point

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
        f"mu_strategy={options.mu_strategy}",
    )

    # KKT Regularization for Stiff Problems (Phase 1.2)
    # Per Biegler: essential for κ ~ 10^14 problems
    # ma57_automatic_scaling compounds with external Betts scaling
    if ipopt_opts_dict.get("linear_solver", "ma86") == "ma57":
        options.ma57_automatic_scaling = "yes"
        options.ma57_pre_alloc = 3.0  # More memory for pivoting in ill-conditioned systems
        options.ma57_pivot_order = 5  # METIS ordering, better for stiffness
        log.debug("Enabled MA57 automatic scaling and METIS pivot ordering for stiff KKT")

    # MA86 settings for stiff KKT matrices (default solver)
    # mc77 equilibration scaling helps with ill-conditioned systems
    if ipopt_opts_dict.get("linear_solver", "ma86") == "ma86":
        # Use linear_solver_options dict for solver-specific settings
        if options.linear_solver_options is None:
            options.linear_solver_options = {}
        options.linear_solver_options["ma86_scaling"] = "mc77"  # Equilibration scaling
        options.linear_solver_options["ma86_order"] = "metis"  # METIS ordering
        log.debug("Enabled MA86 mc77 scaling and METIS ordering for stiff KKT")

    # Corrector type: primal-dual more robust for ill-conditioned KKT
    if "corrector_type" not in ipopt_opts_dict:
        if options.linear_solver_options is None:
            options.linear_solver_options = {}
        options.linear_solver_options["corrector_type"] = "primal-dual"
        log.debug("Using primal-dual corrector for stiff problem robustness")

    # Relaxed dual tolerance to accept noise floor from high condition number
    # With κ ~ 10^14, dual accuracy limited to ~1e-2 even with perfect linear solve
    if "dual_inf_tol" not in ipopt_opts_dict:
        options.dual_inf_tol = 1e-4  # Relaxed from default 1e-8
        options.acceptable_dual_inf_tol = 1e-2  # Fallback tolerance
        log.debug(
            f"Relaxed dual_inf_tol to {options.dual_inf_tol:.1e} "
            f"(acceptable: {options.acceptable_dual_inf_tol:.1e}) for high-κ problem"
        )

    available_solvers = _get_available_hsl_solvers()
    available_display = ", ".join(sorted(available_solvers)) if available_solvers else "unknown"
    options.linear_solver_options = dict(options.linear_solver_options or {})

    # Use the configured linear solver (user selection or default, e.g. "ma86")
    solver_choice = options.linear_solver

    # Fallback only if requested solver is definitely missing and we have a known alternative
    if (
        solver_choice == "ma57"
        and available_solvers
        and "ma57" not in available_solvers
        and "ma27" in available_solvers
    ):
        log.warning("MA57 requested but not found. Falling back to MA27.")
        solver_choice = "ma27"
        options.linear_solver = "ma27"

    if solver_choice == "ma57":
        log.info(
            "Using MA57 (n_vars=%d, n_constraints=%d, available=%s)",
            n_vars,
            n_constraints,
            available_display,
        )
    elif solver_choice == "ma27":
        log.info(
            "Using MA27 (available=%s, n_vars=%d, n_constraints=%d)",
            available_display,
            n_vars,
            n_constraints,
        )
        _configure_ma27_memory(options, n_vars, n_constraints)
    else:
        log.info(
            "Using linear solver '%s' (available=%s, n_vars=%d, n_constraints=%d)",
            solver_choice,
            available_display,
            n_vars,
            n_constraints,
        )

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
    # Disabled by default due to IDE crash risk (massive output)
    # if env_print_level is None and options.print_level < 8:
    #     options.print_level = 8
    #     log.debug(
    #         f"Increased print_level to {options.print_level} for verbose convergence monitoring"
    #     )

    # Ensure print_frequency_iter is 1 to show every iteration
    if options.print_frequency_iter > 1:
        options.print_frequency_iter = 1
        log.debug("Set print_frequency_iter to 1 to show every iteration")

    return options


# _setup_optimization_bounds extracted to campro.optimization.initialization.setup


def _compute_variable_scaling(
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
    x0: np.ndarray[Any, Any] | None = None,
    variable_groups: dict[str, list[int]] | None = None,
    jac_g0_arr: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """
    Compute variable scaling factors with unit-based initialization and Jacobian refinement.

    Scaling phases:
    1. Phase 1: Unit-based scaling - apply per-group reference scales (physical units)
    2. Phase 1b: Group normalization - clamp scales within group using configurable max_ratio
    3. Phase 2: Jacobian-based refinement - adjust scales based on constraint sensitivity
       (only if Jacobian available; equalizes variable influence on constraints)
    4. Phase 3: Bounds/x0 fallback - value-based scaling (only if Jacobian unavailable)
    5. Final normalization: ensure global scale ratio within maximum allowed

    Note: Log-space transformation for wide-range positive variables (pressures, densities,
    valve_areas) is handled in the NLP formulation (nlp.py), not in this scaling function.

    Args:
        lbx: Lower bounds on variables
        ubx: Upper bounds on variables
        x0: Initial guess values (optional, falls back to bounds-only if None)
        variable_groups: Dict mapping group names to variable indices (optional)
        jac_g0_arr: Constraint Jacobian at initial guess (optional, enables Phase 2 refinement)

    Returns:
        Array of scale factors (one per variable)
    """
    n_vars = len(lbx)
    scale = np.ones(n_vars)

    # Standardized per-group reference units with physical anchors
    unit_references = {
        "positions": 0.05,  # meters -> scale ~ 1/0.05 = 20 (typical position ~50mm)
        "velocities": 10.0,  # m/s -> scale ~ 1/10 = 0.1 (typical velocity ~10 m/s)
        "densities": 1.0,  # kg/m^3 (already reasonable)
        "temperatures": 1000.0,  # K (normalize to 0.001-2.0 range)
        "pressures": 1e6,  # MPa (normalize to 0.01-10 range)
        "valve_areas": 1e-4,  # m^2 -> scale ~ 1/1e-4 = 1e4 (typical area ~0.1 mm^2)
        "ignition": 1.0,  # seconds (already in base units)
        "scavenging_fractions": 1.0,  # dimensionless (yF)
        "scavenging_masses": 0.01,  # kg (Mdel, Mlost)
        "scavenging_area_integrals": 1e-4,  # m^2*s (AinInt, AexInt)
        "scavenging_time_moments": 5e-5,  # m^2*s^2 (AinTmom, AexTmom)
        "cycle_time": 0.05,  # s (T_cycle)
    }

    # Map variable indices to their groups
    var_to_group = {}
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if group_name in unit_references:
                for idx in indices:
                    if 0 <= idx < n_vars:
                        var_to_group[idx] = group_name

    # Phase 1: Unit-based scaling (simplified - use 1/ref for all, no sqrt transform)
    for i in range(n_vars):
        group = var_to_group.get(i)
        if group and group in unit_references:
            ref = unit_references[group]
            # CRITICAL FIX: For log-space variables, the values are already O(10) (e.g., log(1e-4) ~ -9.2).
            # Using physical unit reference (e.g., 1e-4) would result in huge scaling factors (1e4),
            # making the scaled variable -92000, which is bad for conditioning.
            # If the group uses log scale, use a reference of 1.0 (or slightly larger) to keep scaled values O(1-10).
            group_cfg = SCALING_GROUP_CONFIG.get(group, {})
            if group_cfg.get("use_log_scale", False):
                ref = 1.0  # Log-space variables don't need unit conversion scaling
            scale[i] = 1.0 / ref

    # Phase 1b: Group normalization with configurable max_ratio
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if indices and len(indices) > 0:
                group_indices = np.array([i for i in indices if 0 <= i < n_vars])
                if len(group_indices) > 0:
                    group_scales = scale[group_indices]
                    group_min = group_scales.min()
                    group_max = group_scales.max()

                    # Get max_ratio from config (default 10.0 if not specified)
                    group_cfg = SCALING_GROUP_CONFIG.get(group_name, {})
                    max_ratio = group_cfg.get("max_ratio", 10.0)

                    # Cap ratio to configured max_ratio
                    if group_min > 1e-10 and group_max / group_min > max_ratio:
                        # Normalize to median, then clamp to symmetric range
                        group_median = np.median(group_scales)
                        sqrt_ratio = np.sqrt(max_ratio)
                        lo = group_median / sqrt_ratio
                        hi = group_median * sqrt_ratio
                        for idx in group_indices:
                            scale[idx] = np.clip(scale[idx], lo, hi)

    # Phase 2: Bounds/x0 fallback (value-based scaling)
    # Apply value-based scaling based on bounds and initial guess
    # STRICT BETTS-STYLE: Use physical ranges, no gradient info
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
            target_scale = 1.0 / magnitude
            # Calculate ratio between target scale and current unit-based scale
            ratio = target_scale / scale[i]
            # Clamp ratio to reasonable range [0.1, 10.0] to respect unit-based priors
            # but allow some adaptation to actual bounds
            ratio = np.clip(ratio, 0.1, 10.0)
            scale[i] = scale[i] * ratio

    # Final clamping: Ensure all scale factors are within [1e-3, 1e3]
    # This prevents extreme scaling factors that can cause numerical issues
    # as recommended by Betts/Biegler
    scale = np.clip(scale, 1e-3, 1e3)

    return scale  # type: ignore[return-value]


def _compute_constraint_scaling(
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
) -> np.ndarray[Any, Any]:
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
        return np.array([])  # type: ignore[return-value]

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

    return scale_g  # type: ignore[return-value]


def _verify_scaling_quality(
    nlp: Any,
    x0: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    reporter: StructuredReporter | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        - condition_number: max/min ratio of scaled Jacobian (target < 1e3)
        - jac_max: Maximum absolute value in scaled Jacobian
        - jac_mean: Mean absolute value in scaled Jacobian
        - jac_min: Minimum non-zero absolute value in scaled Jacobian
        - quality_score: Normalized quality score (0-1, higher is better)
        - over_scaled_variable_groups: List of variable group names
        - over_scaled_constraint_types: List of constraint type names
    """
    # Default return values if verification fails
    default_metrics = {
        "condition_number": np.inf,
        "jac_max": np.inf,
        "jac_mean": 0.0,
        "jac_min": 0.0,
        "quality_score": 0.0,
        "over_scaled_variable_groups": [],
        "over_scaled_constraint_types": [],
        "n_overscaled": 0,
        "n_underscaled": 0,
        "overscaling_ratio": 0.0,
        "underscaling_ratio": 0.0,
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
                reporter.debug(
                    f"Could not evaluate constraint Jacobian for scaling verification: {e}"
                )
            else:
                log.debug(f"Could not evaluate constraint Jacobian for scaling verification: {e}")
            return default_metrics

        # Apply scaling to Jacobian: J_scaled = scale_g * J * (1/scale)
        # Check if scaled Jacobian elements are O(1)
        if len(scale_g) > 0 and jac_g0_arr.size > 0:
            # DIAGNOSTIC: Check for NaN in inputs
            if np.any(np.isnan(jac_g0_arr)):
                if reporter:
                    reporter.warning(
                        f"NaN detected in unscaled Jacobian: {np.sum(np.isnan(jac_g0_arr))} entries"
                    )
                jac_g0_arr = np.nan_to_num(jac_g0_arr, nan=0.0, posinf=1e10, neginf=-1e10)

            if np.any(np.isnan(scale_g)):
                if reporter:
                    reporter.warning(
                        f"NaN detected in scale_g: {np.sum(np.isnan(scale_g))} entries"
                    )
                scale_g = np.nan_to_num(scale_g, nan=1.0, posinf=1e8, neginf=1e-8)

            if np.any(np.isnan(scale)):
                if reporter:
                    reporter.warning(f"NaN detected in scale: {np.sum(np.isnan(scale))} entries")
                scale = np.nan_to_num(scale, nan=1.0, posinf=1e8, neginf=1e-8)

            # Reshape scale_g for broadcasting
            scale_g_col = scale_g.reshape(-1, 1) if len(scale_g.shape) == 1 else scale_g
            scale_row = scale.reshape(1, -1) if len(scale.shape) == 1 else scale

            # Scaled Jacobian: scale_g * J * (1/scale) for each element
            # For element-wise: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
            jac_g0_scaled = np.zeros_like(jac_g0_arr)
            for i in range(min(jac_g0_arr.shape[0], len(scale_g))):
                for j in range(min(jac_g0_arr.shape[1], len(scale))):
                    if scale[j] > 1e-10 and scale_g[i] > 1e-10:  # Avoid division by zero
                        jac_g0_scaled[i, j] = scale_g[i] * jac_g0_arr[i, j] / scale[j]
                    else:
                        jac_g0_scaled[i, j] = 0.0

            # DIAGNOSTIC: Check for NaN in scaled Jacobian
            if np.any(np.isnan(jac_g0_scaled)):
                if reporter:
                    reporter.warning(
                        f"NaN detected in scaled Jacobian: {np.sum(np.isnan(jac_g0_scaled))} entries"
                    )
                jac_g0_scaled = np.nan_to_num(jac_g0_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

            # Check magnitude statistics
            jac_mag = np.abs(jac_g0_scaled)

            # DIAGNOSTIC: Check for NaN/Inf in magnitude
            if np.any(np.isnan(jac_mag)) or np.any(np.isinf(jac_mag)):
                if reporter:
                    reporter.warning(
                        f"NaN/Inf in jac_mag: NaN={np.sum(np.isnan(jac_mag))}, Inf={np.sum(np.isinf(jac_mag))}"
                    )
                jac_mag = np.nan_to_num(jac_mag, nan=0.0, posinf=1e10, neginf=-1e10)

            jac_max = jac_mag.max()
            jac_mean = jac_mag.mean()
            jac_min = jac_mag[jac_mag > 0].min() if (jac_mag > 0).any() else 0.0

            # DIAGNOSTIC: Validate computed values
            if np.isnan(jac_max) or np.isinf(jac_max):
                if reporter:
                    reporter.warning(f"Invalid jac_max: {jac_max}, replacing with 1.0")
                jac_max = 1.0 if np.isnan(jac_max) or np.isinf(jac_max) else jac_max

            if np.isnan(jac_mean) or np.isinf(jac_mean):
                if reporter:
                    reporter.warning(f"Invalid jac_mean: {jac_mean}, replacing with 0.0")
                jac_mean = 0.0 if np.isnan(jac_mean) or np.isinf(jac_mean) else jac_mean

            # Compute nonzero entries for overscaling checks (needed before condition number calculation)
            jac_mag_nonzero = jac_mag[jac_mag > 0]

            # Compute condition number (max/min ratio)
            if jac_min > 0 and np.isfinite(jac_max) and np.isfinite(jac_min):
                condition_number = jac_max / jac_min
            else:
                condition_number = np.inf
                if reporter:
                    reporter.warning(
                        f"Cannot compute condition number: jac_max={jac_max}, jac_min={jac_min}"
                    )

            # Check for overscaling: entries <1e-10 indicate severe overscaling
            critical_overscaling_threshold = 1e-10
            min_entry_threshold = 1e-6  # Minimum acceptable entry
            n_overscaled = (
                np.sum(jac_mag_nonzero < critical_overscaling_threshold)
                if len(jac_mag_nonzero) > 0
                else 0
            )
            n_underscaled = (
                np.sum(jac_mag_nonzero < min_entry_threshold) if len(jac_mag_nonzero) > 0 else 0
            )
            overscaling_ratio = (
                n_overscaled / len(jac_mag_nonzero) if len(jac_mag_nonzero) > 0 else 0.0
            )
            underscaling_ratio = (
                n_underscaled / len(jac_mag_nonzero) if len(jac_mag_nonzero) > 0 else 0.0
            )

            # Compute quality score: normalized measure of scaling quality
            # Score is based on:
            # 1. Condition number (target < 1e3, penalty if > 1e3)
            # 2. Maximum Jacobian element (target < 1e2, penalty if > 1e2)
            # 3. Overscaling penalty (entries <1e-10 are severely overscaled)
            # 4. Underscaling penalty (entries <1e-6 are moderately overscaled)
            # Score ranges from 0 (poor) to 1 (excellent)
            condition_score = 1.0 / (1.0 + np.log10(max(condition_number / 1e3, 1.0)))
            max_score = 1.0 / (1.0 + np.log10(max(jac_max / 1e2, 1.0)))
            # Overscaling penalty: severe penalty if any entries <1e-10
            overscaling_score = (
                1.0 / (1.0 + 10.0 * overscaling_ratio) if overscaling_ratio > 0 else 1.0
            )
            # Underscaling penalty: moderate penalty if entries <1e-6
            underscaling_score = (
                1.0 / (1.0 + 5.0 * underscaling_ratio) if underscaling_ratio > 0 else 1.0
            )
            # Weighted combination: condition and max are most important, but overscaling is critical
            quality_score = (
                0.35 * condition_score
                + 0.35 * max_score
                + 0.20 * overscaling_score
                + 0.10 * underscaling_score
            )

            # Compute percentile statistics for detailed diagnostics
            # (jac_mag_nonzero already computed above)
            if len(jac_mag_nonzero) > 0:
                p25 = np.percentile(jac_mag_nonzero, 25)
                p50 = np.percentile(jac_mag_nonzero, 50)
                p75 = np.percentile(jac_mag_nonzero, 75)
                p95 = np.percentile(jac_mag_nonzero, 95)
                p99 = np.percentile(jac_mag_nonzero, 99)
            else:
                p25 = p50 = p75 = p95 = p99 = 0.0

            # Log diagnostics with percentile statistics and overscaling info
            msg = (
                f"Scaled Jacobian statistics: min={jac_min:.3e}, p25={p25:.3e}, "
                f"p50={p50:.3e}, p75={p75:.3e}, p95={p95:.3e}, p99={p99:.3e}, "
                f"max={jac_max:.3e}, mean={jac_mean:.3e}, condition_number={condition_number:.3e}, "
                f"quality_score={quality_score:.3f}, "
                f"overscaled_entries={n_overscaled} ({overscaling_ratio * 100:.1f}%), "
                f"underscaled_entries={n_underscaled} ({underscaling_ratio * 100:.1f}%)"
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
                            type_jac_mag: list[float] = []
                            for idx in indices:
                                if idx < jac_mag.shape[0]:
                                    type_jac_mag.extend(jac_mag[idx, :].flatten().tolist())

                            if len(type_jac_mag) > 0:
                                type_jac_mag_arr = np.array(type_jac_mag)
                                type_jac_mag_nonzero = type_jac_mag_arr[type_jac_mag_arr > 0]
                                if len(type_jac_mag_nonzero) > 0:
                                    type_max = type_jac_mag_nonzero.max()
                                    type_mean = type_jac_mag_nonzero.mean()
                                    type_p95 = np.percentile(type_jac_mag_nonzero, 95)
                                    type_p99 = np.percentile(type_jac_mag_nonzero, 99)
                                    reporter.info(
                                        f"  {con_type} ({len(indices)} constraints): "
                                        f"max={type_max:.3e}, mean={type_mean:.3e}, "
                                        f"p95={type_p95:.3e}, p99={type_p99:.3e}",
                                    )
                                    # Warn if this type has extreme entries
                                    if type_max > 1e2:
                                        reporter.debug(
                                            f"    Warning: {con_type} has large max entry ({type_max:.3e} > 1e2)",
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
            if condition_number > 1e3:  # Target condition number (updated from 1e6)
                warn_msg = (
                    f"Scaled Jacobian has large condition number (max/min={condition_number:.3e} > 1e3). "
                    f"Consider more uniform scaling."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)
            elif condition_number > 1e2:  # Lower threshold for warning (updated from 1e4)
                warn_msg = (
                    f"Scaled Jacobian condition number is elevated (max/min={condition_number:.3e} > 1e4). "
                    f"May benefit from tighter scaling."
                )
                if reporter:
                    reporter.debug(warn_msg)
                else:
                    log.debug(warn_msg)

            # Check for very small elements that might cause numerical issues (overscaling)
            very_small = (jac_mag > 0) & (jac_mag < 1e-10)
            if very_small.sum() > 0:
                warn_msg = (
                    f"Scaled Jacobian has {very_small.sum()} very small elements (<1e-10). "
                    f"This indicates severe over-scaling and will be corrected."
                )
                if reporter:
                    reporter.warning(warn_msg)
                else:
                    log.warning(warn_msg)

            # Check for moderately small elements (underscaling)
            moderately_small = (jac_mag > 0) & (jac_mag < 1e-6) & (jac_mag >= 1e-10)
            if moderately_small.sum() > 0:
                debug_msg = (
                    f"Scaled Jacobian has {moderately_small.sum()} moderately small elements (<1e-6, >=1e-10). "
                    f"This may indicate moderate over-scaling."
                )
                if reporter:
                    reporter.debug(debug_msg)
                else:
                    log.debug(debug_msg)

            # Group-level diagnostics for over-scaling detection
            over_scaled_variable_groups: list[str] = []
            over_scaled_constraint_types: list[str] = []
            threshold = 1e-8  # Threshold for identifying over-scaled groups

            # Check variable groups (column norms)
            if meta and "variable_groups" in meta:
                variable_groups = meta["variable_groups"]
                for group_name, indices in variable_groups.items():
                    if len(indices) == 0:
                        continue
                    # Compute column norms for variables in this group
                    group_col_norms = []
                    for idx in indices:
                        if idx < jac_g0_scaled.shape[1]:
                            col_norm = np.linalg.norm(jac_g0_scaled[:, idx])
                            if col_norm > 0:
                                group_col_norms.append(col_norm)

                    if len(group_col_norms) > 0:
                        median_norm = np.median(group_col_norms)
                        if median_norm < threshold:
                            over_scaled_variable_groups.append(group_name)

            # Check constraint types (row norms)
            if meta and "constraint_groups" in meta:
                constraint_groups = meta["constraint_groups"]
                for con_type, indices in constraint_groups.items():
                    if len(indices) == 0:
                        continue
                    # Compute row norms for constraints of this type
                    group_row_norms = []
                    for idx in indices:
                        if idx < jac_g0_scaled.shape[0]:
                            row_norm = np.linalg.norm(jac_g0_scaled[idx, :])
                            if row_norm > 0:
                                group_row_norms.append(row_norm)

                    if len(group_row_norms) > 0:
                        median_norm = np.median(group_row_norms)
                        if median_norm < threshold:
                            over_scaled_constraint_types.append(con_type)

            return {
                "condition_number": condition_number,
                "jac_max": jac_max,
                "over_scaled_variable_groups": over_scaled_variable_groups,
                "over_scaled_constraint_types": over_scaled_constraint_types,
                "jac_mean": jac_mean,
                "jac_min": jac_min,
                "quality_score": quality_score,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
                "p99": p99,
                "n_overscaled": int(n_overscaled),
                "n_underscaled": int(n_underscaled),
                "overscaling_ratio": float(overscaling_ratio),
                "underscaling_ratio": float(underscaling_ratio),
            }
        # No constraint scaling available
        return default_metrics

    except Exception as e:
        if reporter:
            reporter.debug(f"Scaling verification failed: {e}")
        else:
            log.debug(f"Scaling verification failed: {e}")

    return default_metrics


def _relax_over_scaled_groups(
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    over_scaled_variable_groups: list[str],
    over_scaled_constraint_types: list[str],
    variable_groups: dict[str, list[int]] | None = None,
    constraint_groups: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Relax scale factors for over-scaled groups using geometric averaging.

    Uses geometric mean (sqrt) to move scale factors halfway back toward 1.0,
    which is appropriate for multi-decade scales. This selectively relaxes only
    problematic groups while preserving well-scaled ones.

    Args:
        scale: Current variable scaling factors
        scale_g: Current constraint scaling factors
        over_scaled_variable_groups: List of variable group names to relax
        over_scaled_constraint_types: List of constraint type names to relax
        variable_groups: Dict mapping group names to variable indices
        constraint_groups: Dict mapping constraint types to constraint indices

    Returns:
        Tuple of (relaxed_scale, relaxed_scale_g) arrays
    """
    relaxed_scale = scale.copy()
    relaxed_scale_g = scale_g.copy()

    # Relax variable groups
    if variable_groups is not None:
        for group_name in over_scaled_variable_groups:
            if group_name in variable_groups:
                indices = variable_groups[group_name]
                for idx in indices:
                    if 0 <= idx < len(relaxed_scale):
                        # Geometric mean with 1.0: sqrt(scale[idx] * 1.0)
                        relaxed_scale[idx] = np.sqrt(relaxed_scale[idx] * 1.0)

    # Relax constraint types
    if constraint_groups is not None:
        for con_type in over_scaled_constraint_types:
            if con_type in constraint_groups:
                indices = constraint_groups[con_type]
                for idx in indices:
                    if 0 <= idx < len(relaxed_scale_g):
                        # Geometric mean with 1.0: sqrt(scale_g[idx] * 1.0)
                        relaxed_scale_g[idx] = np.sqrt(relaxed_scale_g[idx] * 1.0)

    return relaxed_scale, relaxed_scale_g


def _identify_constraint_types(
    meta: dict[str, Any] | None,
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    jac_g0_arr: np.ndarray[Any, Any] | None = None,
    g0_arr: np.ndarray[Any, Any] | None = None,
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


def _map_constraint_type_to_category(con_type: str) -> str:
    """
    Map existing constraint types to category-based scaling categories.

    Args:
        con_type: Existing constraint type string (e.g., 'path_velocity', 'path_pressure')

    Returns:
        Category string: 'kinematic', 'thermodynamic', 'boundary', or 'continuity'
    """
    # Kinematic: position, velocity, acceleration constraints
    if con_type in {"path_velocity", "path_constraint"}:
        return "kinematic"

    # Thermodynamic: temperature, pressure, energy constraints
    if con_type in {"path_pressure", "combustion"}:
        return "thermodynamic"

    # Boundary: boundary conditions, periodicity
    if con_type in {"periodicity", "path_clearance"}:
        return "boundary"

    # Continuity: continuity constraints, collocation residuals
    if con_type in {"continuity", "collocation_residuals"}:
        return "continuity"

    # Default: treat unknown types as kinematic (most common)
    return "kinematic"


def _compute_scaled_jacobian(
    nlp: Any,
    x0: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None]:
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

        # DIAGNOSTIC: Check for NaN in inputs before scaling
        if np.any(np.isnan(jac_g0_arr)):
            import logging

            log = logging.getLogger(__name__)
            log.warning("NaN detected in unscaled Jacobian in _compute_scaled_jacobian")
            jac_g0_arr = np.nan_to_num(jac_g0_arr, nan=0.0, posinf=1e10, neginf=-1e10)

        if scale_g is not None and len(scale_g) > 0:
            if np.any(np.isnan(scale_g)):
                import logging

                log = logging.getLogger(__name__)
                log.warning("NaN detected in scale_g in _compute_scaled_jacobian")
                scale_g = np.nan_to_num(scale_g, nan=1.0, posinf=1e8, neginf=1e-8)

        if scale is not None and len(scale) > 0:
            if np.any(np.isnan(scale)):
                import logging

                log = logging.getLogger(__name__)
                log.warning("NaN detected in scale in _compute_scaled_jacobian")
                scale = np.nan_to_num(scale, nan=1.0, posinf=1e8, neginf=1e-8)

        # Compute scaled Jacobian: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
        jac_g0_scaled = np.zeros_like(jac_g0_arr)
        for i in range(min(jac_g0_arr.shape[0], len(scale_g))):
            for j in range(min(jac_g0_arr.shape[1], len(scale))):
                if scale[j] > 1e-10 and scale_g[i] > 1e-10:
                    jac_g0_scaled[i, j] = scale_g[i] * jac_g0_arr[i, j] / scale[j]
                else:
                    jac_g0_scaled[i, j] = 0.0

        # DIAGNOSTIC: Check for NaN in output
        if np.any(np.isnan(jac_g0_scaled)):
            import logging

            log = logging.getLogger(__name__)
            log.warning("NaN detected in scaled Jacobian output in _compute_scaled_jacobian")
            jac_g0_scaled = np.nan_to_num(jac_g0_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        return jac_g0_arr, jac_g0_scaled
    except Exception:
        return None, None


def _equilibrate_jacobian_iterative(
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any],
    n_iterations: int = 3,
    target: float = 1.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Iteratively equilibrate Jacobian by balancing row and column norms.

    This function performs iterative Jacobian equilibration to improve numerical
    conditioning. It alternates between scaling constraints (rows) and variables
    (columns) to balance the scaled Jacobian entries.

    Args:
        scale: Current variable scaling factors
        scale_g: Current constraint scaling factors
        jac_g0_arr: Unscaled Jacobian matrix
        n_iterations: Number of equilibration iterations (default: 3)
        target: Target magnitude for row/column norms (default: 1.0)

    Returns:
        Tuple of (equilibrated_scale, equilibrated_scale_g)
    """
    if jac_g0_arr is None or jac_g0_arr.size == 0:
        return scale.copy(), scale_g.copy()

    equilibrated_scale = scale.copy()
    equilibrated_scale_g = scale_g.copy()

    for iteration in range(n_iterations):
        # Compute scaled Jacobian: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
        jac_scaled = np.zeros_like(jac_g0_arr)
        for i in range(min(jac_g0_arr.shape[0], len(equilibrated_scale_g))):
            for j in range(min(jac_g0_arr.shape[1], len(equilibrated_scale))):
                if equilibrated_scale[j] > 1e-10 and equilibrated_scale_g[i] > 1e-10:
                    jac_scaled[i, j] = (
                        equilibrated_scale_g[i] * jac_g0_arr[i, j] / equilibrated_scale[j]
                    )
                else:
                    jac_scaled[i, j] = 0.0

        # Check for NaN/Inf values and replace with 0
        jac_scaled = np.nan_to_num(jac_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        # Compute row norms (constraint scaling): max(|J_scaled[i,:]|)
        row_norms = np.max(np.abs(jac_scaled), axis=1)
        # Replace any NaN/Inf with 0
        row_norms = np.nan_to_num(row_norms, nan=0.0, posinf=1e10, neginf=-1e10)

        # Compute column norms (variable scaling): max(|J_scaled[:,j]|)
        col_norms = np.max(np.abs(jac_scaled), axis=0)
        # Replace any NaN/Inf with 0
        col_norms = np.nan_to_num(col_norms, nan=0.0, posinf=1e10, neginf=-1e10)

        # Equilibrate rows (constraints)
        # Update scale_g to bring row norms to target
        for i in range(len(equilibrated_scale_g)):
            if row_norms[i] > 1e-10:
                adjustment = target / max(row_norms[i], 1e-10)
                equilibrated_scale_g[i] = equilibrated_scale_g[i] * adjustment
                # Clip to reasonable bounds
                equilibrated_scale_g[i] = np.clip(equilibrated_scale_g[i], 1e-8, 1e8)
            # If row_norm is zero or very small, leave scale_g unchanged

        # Recompute scaled Jacobian with updated constraint scales
        jac_scaled = np.zeros_like(jac_g0_arr)
        for i in range(min(jac_g0_arr.shape[0], len(equilibrated_scale_g))):
            for j in range(min(jac_g0_arr.shape[1], len(equilibrated_scale))):
                if equilibrated_scale[j] > 1e-10 and equilibrated_scale_g[i] > 1e-10:
                    jac_scaled[i, j] = (
                        equilibrated_scale_g[i] * jac_g0_arr[i, j] / equilibrated_scale[j]
                    )
                else:
                    jac_scaled[i, j] = 0.0

        # Check for NaN/Inf values and replace with 0
        jac_scaled = np.nan_to_num(jac_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        # Recompute column norms with updated constraint scales
        col_norms = np.max(np.abs(jac_scaled), axis=0)
        # Replace any NaN/Inf with 0
        col_norms = np.nan_to_num(col_norms, nan=0.0, posinf=1e10, neginf=-1e10)

        # Equilibrate columns (variables)
        # Update scale to bring column norms to target
        for j in range(len(equilibrated_scale)):
            if col_norms[j] > 1e-10:
                adjustment = target / max(col_norms[j], 1e-10)
                equilibrated_scale[j] = equilibrated_scale[j] * adjustment
                # Clip to reasonable bounds
                equilibrated_scale[j] = np.clip(equilibrated_scale[j], 1e-8, 1e8)
            # If col_norm is zero or very small, leave scale unchanged

    return equilibrated_scale, equilibrated_scale_g


def _diagnose_nan_in_jacobian(
    nlp: Any,
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

    diagnostics = {
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


def _evaluate_nlp_at_x0(
    nlp: Any,
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

    Returns:
        Tuple of (g0_arr, f0_val, jac_g0_arr, grad_f0_arr):
        - g0_arr: Constraint values at x0 (array)
        - f0_val: Objective value at x0 (scalar)
        - jac_g0_arr: Constraint Jacobian at x0 (array, or None if unavailable)
        - grad_f0_arr: Objective gradient at x0 (array, or None if unavailable)
    """
    import casadi as ca

    g0_arr = np.array([])
    f0_val = 0.0
    jac_g0_arr = None
    grad_f0_arr = None

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
                                meta_for_diag = nlp.meta
                            elif isinstance(nlp, dict) and "meta" in nlp:
                                meta_for_diag = nlp["meta"]

                        # Diagnose the NaN location
                        try:
                            from campro.logging import get_logger

                            log_diag = get_logger(__name__)
                            diag = _diagnose_nan_in_jacobian(
                                nlp, x0, first_row, first_col, meta_for_diag
                            )
                            log_diag.warning(
                                f"  Constraint type: {diag.get('constraint_type', 'unknown')}, "
                                f"Variable group: {diag.get('variable_group', 'unknown')}"
                            )
                            if diag.get("constraint_index_in_group") is not None:
                                log_diag.warning(
                                    f"  Constraint index in group: {diag.get('constraint_index_in_group')}"
                                )
                            if diag.get("variable_index_in_group") is not None:
                                log_diag.warning(
                                    f"  Variable index in group: {diag.get('variable_index_in_group')}"
                                )
                            if diag.get("constraint_value") is not None:
                                log_diag.warning(
                                    f"  Constraint value: {diag.get('constraint_value')}"
                                )
                            if diag.get("variable_value") is not None:
                                log_diag.warning(f"  Variable value: {diag.get('variable_value')}")
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


def _compute_unified_constraint_magnitudes(
    g0_arr: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any] | None,
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """
    Compute unified constraint magnitudes combining actual values, Jacobian sensitivity, and bounds.

    For each constraint i, computes:
    magnitude_i = max(|g_i(x0)|, ||J_i||_scaled, |bound_i|)

    This combines:
    - Actual constraint value at x0
    - Jacobian row sensitivity (scaled by variable scales)
    - Bound magnitude

    Args:
        g0_arr: Constraint values at x0
        jac_g0_arr: Constraint Jacobian at x0 (can be None)
        lbg: Lower constraint bounds (can be None)
        ubg: Upper constraint bounds (can be None)
        scale: Variable scaling factors

    Returns:
        Array of unified magnitudes (one per constraint)
    """
    n_cons = len(g0_arr)
    constraint_magnitudes = np.zeros(n_cons)

    for i in range(n_cons):
        # Actual constraint value
        g_val = abs(g0_arr[i]) if i < len(g0_arr) and np.isfinite(g0_arr[i]) else 0.0

        # Jacobian row sensitivity (scaled by variable scales)
        jac_row_norm = 0.0
        if jac_g0_arr is not None and i < jac_g0_arr.shape[0]:
            row_norm_sq = 0.0
            for j in range(min(jac_g0_arr.shape[1], len(scale))):
                if scale[j] > 1e-10:
                    row_norm_sq += (jac_g0_arr[i, j] / scale[j]) ** 2
            jac_row_norm = np.sqrt(row_norm_sq)

        # Bound magnitude
        bound_mag = 0.0
        if lbg is not None and i < len(lbg):
            lb = lbg[i]
            if np.isfinite(lb):
                bound_mag = max(bound_mag, abs(lb))
        if ubg is not None and i < len(ubg):
            ub = ubg[i]
            if np.isfinite(ub):
                bound_mag = max(bound_mag, abs(ub))

        # Unified magnitude: max of all three
        # Use 1e-12 floor to allow scaling of very small residuals (was 1e-6)
        constraint_magnitudes[i] = max(g_val, jac_row_norm, bound_mag, 1e-12)

    return constraint_magnitudes


def _analyze_magnitude_distribution(
    magnitudes: np.ndarray[Any, Any],
    aggressive: bool = False,
) -> dict[str, Any]:
    """
    Analyze magnitude distribution using robust statistics without distribution assumptions.

    Computes percentiles and IQR to identify center (median) and detect outliers.
    No assumptions about distribution shape (Poisson, Gaussian, etc.).

    Args:
        magnitudes: Array of constraint magnitudes
        aggressive: If True, use 3xIQR for outlier detection instead of 1.5xIQR.
                   More conservative, only flags truly extreme outliers.

    Returns:
        Dictionary with:
        - median: Median (p50) - the center point
        - iqr: Interquartile Range (p75 - p25)
        - outlier_mask: Boolean array indicating outliers (using IQR method)
        - percentiles: Dict with p5, p25, p50, p75, p95
    """
    magnitudes_nonzero = magnitudes[magnitudes > 1e-20]

    if len(magnitudes_nonzero) == 0:
        # Fallback if no valid magnitudes
        return {
            "median": 1.0,
            "iqr": 1.0,
            "outlier_mask": np.zeros(len(magnitudes), dtype=bool),
            "percentiles": {"p5": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0, "p95": 1.0},
        }

    # Compute robust percentiles
    p5 = np.percentile(magnitudes_nonzero, 5)
    p25 = np.percentile(magnitudes_nonzero, 25)
    p50 = np.percentile(magnitudes_nonzero, 50)  # Median (center)
    p75 = np.percentile(magnitudes_nonzero, 75)
    p95 = np.percentile(magnitudes_nonzero, 95)

    # Compute IQR for outlier detection
    iqr = p75 - p25

    # Detect outliers using IQR method (no distribution assumptions)
    # Use different IQR multiplier based on aggressiveness
    # - Standard (1.5xIQR): catches ~0.7% outliers
    # - Aggressive (3xIQR): catches ~0.003% outliers (only extreme values)
    iqr_multiplier = 3.0 if aggressive else 1.5
    lower_bound = p25 - iqr_multiplier * iqr
    upper_bound = p75 + iqr_multiplier * iqr

    # Handle edge case where IQR is very small
    if iqr < 1e-10:
        # If IQR is too small, use wider bounds based on percentiles
        lower_bound = p5
        upper_bound = p95

    outlier_mask = (magnitudes < lower_bound) | (magnitudes > upper_bound)

    return {
        "median": p50,
        "iqr": iqr,
        "outlier_mask": outlier_mask,
        "percentiles": {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
    }


def _normalize_to_median_center(
    magnitudes: np.ndarray[Any, Any],
    median: float,
    iqr: float,
    outlier_mask: np.ndarray[Any, Any],
    scale_bounds: tuple[float, float] = (1e-8, 1e8),
) -> np.ndarray[Any, Any]:
    """
    Normalize all magnitudes to center at median.

    Outliers are handled separately to prevent them from skewing the center.
    All normal constraints are normalized so their scaled magnitudes center at median.

    Args:
        magnitudes: Array of constraint magnitudes
        median: Median (center point) from distribution analysis
        iqr: Interquartile Range from distribution analysis
        outlier_mask: Boolean array indicating outliers
        scale_bounds: (min, max) bounds for outlier scale factors.
                     Tighter bounds (e.g., 1e-6, 1e6) prevent extreme multipliers.

    Returns:
        Array of scale factors (one per constraint)
    """
    scales = np.ones(len(magnitudes))
    target_center = 1.0  # Normalized center

    for i in range(len(magnitudes)):
        mag = magnitudes[i]

        if outlier_mask[i]:
            # Outlier: normalize but bound to prevent extreme scales
            # Still normalize to center, but limit scale factor
            if mag > 1e-10:
                scale = target_center / mag
                # Bound outlier scales using provided bounds
                scales[i] = np.clip(scale, scale_bounds[0], scale_bounds[1])
            else:
                scales[i] = 1.0
        # Normal constraint: normalize to center
        # Scale so that scaled magnitude = median (target_center)
        elif mag > 1e-20:
            scales[i] = target_center / mag
        else:
            scales[i] = 1.0

    return scales


def _cluster_by_magnitude(
    magnitudes: np.ndarray[Any, Any],
) -> dict[int, list[int]]:
    """
    Cluster magnitudes into log10 bins.

    Args:
        magnitudes: Array of magnitudes

    Returns:
        Dictionary mapping log10 bin index to list of indices
    """
    clusters: dict[int, list[int]] = {}

    # Handle zeros or negative values (shouldn't happen for magnitudes, but safety first)
    safe_mags = np.maximum(magnitudes, 1e-20)

    # Compute log10 magnitudes
    log_mags = np.log10(safe_mags)

    # Bin into integers (floor)
    # e.g., 0.05 -> -1.3 -> -2 (bin -2: [1e-2, 1e-1])
    # e.g., 50 -> 1.7 -> 1 (bin 1: [1e1, 1e2])
    bins = np.floor(log_mags).astype(int)

    # Group indices
    for i, bin_idx in enumerate(bins):
        if bin_idx not in clusters:
            clusters[bin_idx] = []
        clusters[bin_idx].append(i)

    return clusters


def _compute_constraint_scaling_by_type(
    constraint_types: dict[int, str],
    nlp: Any,
    x0: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any] | None = None,
    g0_arr: np.ndarray[Any, Any] | None = None,
    current_scale_g: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
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
        return np.array([])  # type: ignore[return-value]

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
    jac_row_max_entries = (
        None  # Max absolute entry per row (after variable scaling, before constraint scaling)
    )
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
                        row_norm_sq += scaled_entry**2
                        row_max = max(row_max, scaled_entry)
                jac_row_norms[i] = np.sqrt(row_norm_sq)
                jac_row_max_entries[i] = row_max

                # DIAGNOSTIC: Validate computed values
                if not np.isfinite(jac_row_norms[i]):
                    log.warning(
                        f"Non-finite jac_row_norms[{i}]: {jac_row_norms[i]}, replacing with 0.0"
                    )
                    jac_row_norms[i] = 0.0
                if not np.isfinite(jac_row_max_entries[i]):
                    log.warning(
                        f"Non-finite jac_row_max_entries[{i}]: {jac_row_max_entries[i]}, replacing with 0.0"
                    )
                    jac_row_max_entries[i] = 0.0
        else:
            # Without variable scales, use unscaled row norms and max entries
            jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
            jac_row_max_entries = np.abs(jac_g0_arr).max(axis=1)

            # DIAGNOSTIC: Check for NaN/Inf in computed norms
            if np.any(np.isnan(jac_row_norms)) or np.any(np.isinf(jac_row_norms)):
                log.warning(
                    f"NaN/Inf in jac_row_norms: NaN={np.sum(np.isnan(jac_row_norms))}, Inf={np.sum(np.isinf(jac_row_norms))}"
                )
                jac_row_norms = np.nan_to_num(jac_row_norms, nan=0.0, posinf=1e10, neginf=-1e10)

            if np.any(np.isnan(jac_row_max_entries)) or np.any(np.isinf(jac_row_max_entries)):
                log.warning(
                    f"NaN/Inf in jac_row_max_entries: NaN={np.sum(np.isnan(jac_row_max_entries))}, Inf={np.sum(np.isinf(jac_row_max_entries))}"
                )
                jac_row_max_entries = np.nan_to_num(
                    jac_row_max_entries, nan=0.0, posinf=1e10, neginf=-1e10
                )

    # Process each constraint type with specialized scaling

    # Store jac_sensitivity values for verification step
    stored_jac_sensitivity = {}

    # Check if we need to force recalculation due to poor current scaling
    if (
        current_scale_g is not None
        and len(current_scale_g) == n_cons
        and jac_g0_arr is not None
        and scale is not None
    ):
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
            log.debug(
                f"Current scaling produces large entries (est. max={max_scaled_entry_estimate:.3e} > 1e2), forcing recalculation"
            )
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
                g0_val = g0_arr[i]
                g0_mag = float(abs(g0_val)) if np.isfinite(g0_val) else 0.0

            # Get Jacobian row norm and max entry if available
            jac_norm = 0.0
            jac_max_entry = 0.0
            if jac_row_norms is not None and i < len(jac_row_norms):
                jac_norm = jac_row_norms[i]
                # DIAGNOSTIC: Check for NaN/Inf
                if not np.isfinite(jac_norm):
                    log.warning(f"Non-finite jac_norm[{i}]: {jac_norm}, replacing with 0.0")
                    jac_norm = 0.0
            if jac_row_max_entries is not None and i < len(jac_row_max_entries):
                jac_max_entry = jac_row_max_entries[i]
                # DIAGNOSTIC: Check for NaN/Inf
                if not np.isfinite(jac_max_entry):
                    log.warning(
                        f"Non-finite jac_max_entry[{i}]: {jac_max_entry}, replacing with 0.0"
                    )
                    jac_max_entry = 0.0
            # Use max entry for aggressive scaling decisions (more accurate than norm)
            # Fall back to norm if max entry not available
            jac_sensitivity = max(jac_max_entry, jac_norm) if jac_max_entry > 0 else jac_norm
            # DIAGNOSTIC: Validate jac_sensitivity
            if not np.isfinite(jac_sensitivity):
                log.warning(
                    f"Non-finite jac_sensitivity[{i}]: {jac_sensitivity}, replacing with 0.0"
                )
                jac_sensitivity = 0.0
            # Store for verification step
            stored_jac_sensitivity[i] = jac_sensitivity

            # Compute magnitude from bounds
            if lb == -np.inf and ub == np.inf:
                magnitude = max(g0_mag, jac_norm) if jac_norm > 0 else g0_mag
            elif lb == -np.inf:
                magnitude = max(float(abs(ub)), g0_mag, jac_norm)
            elif ub == np.inf:
                magnitude = max(float(abs(lb)), g0_mag, jac_norm)
            else:
                magnitude = max(float(abs(lb)), float(abs(ub)), g0_mag, jac_norm)

            # DIAGNOSTIC: Validate magnitude
            if not np.isfinite(magnitude):
                log.warning(
                    f"Non-finite magnitude[{i}]: {magnitude}, lb={lb}, ub={ub}, g0_mag={g0_mag}, jac_norm={jac_norm}"
                )
                magnitude = max(
                    float(abs(lb)) if np.isfinite(lb) else 1.0,
                    float(abs(ub)) if np.isfinite(ub) else 1.0,
                    g0_mag if np.isfinite(g0_mag) else 1.0,
                )

            # Map constraint type to category and get target magnitude
            category = _map_constraint_type_to_category(con_type)
            target_magnitude = CONSTRAINT_TYPE_TARGETS.get(category, 1.0)

            # Apply category-based scaling with type-specific handling for extreme cases
            # For extreme Jacobian entries (jac_sensitivity > 1e6), use aggressive scaling
            # Otherwise, scale to achieve target magnitude for the category
            if jac_sensitivity > 1e6:
                # Very extreme Jacobian entries - need very aggressive scaling
                # Target O(1) max entry for extreme cases regardless of category
                target_max_entry = 1e0
                scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                min_scale_needed = target_max_entry / jac_sensitivity
                scale_g[i] = np.clip(scale_g[i], max(min_scale_needed * 0.1, 1e-10), 1e2)
                if i < 5:  # Log first few for debugging
                    log.debug(
                        f"  {con_type}[{i}] (category={category}): extreme jac_sensitivity={jac_sensitivity:.3e}, "
                        f"scale={scale_g[i]:.3e}, expected_scaled_max={scale_g[i] * jac_sensitivity:.3e}"
                    )
            elif jac_sensitivity > 1e2:
                # Large Jacobian sensitivity - scale to target magnitude
                # Use category target but ensure max entry is reasonable
                effective_magnitude = max(magnitude, jac_sensitivity)
                scale_g[i] = target_magnitude / max(effective_magnitude, 1e-10)
                scale_g[i] = np.clip(scale_g[i], 1e-8, 1e2)
            elif magnitude > 1e-10 or jac_sensitivity > 1e-10:
                # Normal scaling: scale to achieve target magnitude for category
                effective_magnitude = (
                    max(magnitude, jac_sensitivity) if jac_sensitivity > 0 else magnitude
                )
                # Ensure we don't divide by zero or create NaN
                if effective_magnitude > 1e-10:
                    scale_g[i] = target_magnitude / effective_magnitude
                else:
                    scale_g[i] = 1.0
                # Clip based on category: tighter bounds for boundary, wider for thermodynamic
                if category == "boundary":
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
                elif category == "thermodynamic":
                    scale_g[i] = np.clip(scale_g[i], 1e-4, 1e2)
                else:
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
            # Small magnitude - keep existing scaling or use default
            elif scale_g[i] <= 1e-10:
                scale_g[i] = 1.0

            # DIAGNOSTIC: Validate computed scale_g[i] is finite
            if not np.isfinite(scale_g[i]):
                log.warning(
                    f"Non-finite scale_g[{i}] computed: {scale_g[i]}, category={category}, "
                    f"magnitude={magnitude}, jac_sensitivity={jac_sensitivity}, target={target_magnitude}"
                )
                scale_g[i] = 1.0  # Reset to safe default

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
                            f"applied_safety_factor={safety_factor:.3e}, final_scale={scale_g[i]:.3e}",
                        )

    # Overscaling detection and correction: Prevent very small scaled Jacobian entries
    # Very small entries (<1e-10) indicate overscaling and cause numerical precision issues
    # Balance: target max entry O(1-10), min entry >= 1e-6 (conservative threshold)
    min_scaled_entry_threshold = 1e-6  # Minimum acceptable scaled Jacobian entry (conservative)
    critical_overscaling_threshold = (
        1e-10  # Critical threshold - entries below this indicate severe overscaling
    )
    max_scaled_entry_target = 1e1  # Target max entry (O(10))

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

                    # Check for critical overscaling first (entries <1e-10)
                    if row_min < critical_overscaling_threshold:
                        # Critical overscaling: apply aggressive correction
                        # Target: bring min entry to at least 1e-6
                        target_min = min_scaled_entry_threshold
                        adjustment_factor = target_min / max(row_min, 1e-15)

                        # Check if adjustment would create too large max entry
                        new_max = row_max * adjustment_factor
                        if (
                            new_max <= max_scaled_entry_target * 50
                        ):  # Allow more aggressive correction for critical cases
                            # Apply aggressive adjustment
                            old_scale = scale_g[i]
                            scale_g[i] = scale_g[i] * adjustment_factor
                            n_corrected += 1

                            if i < 5:  # Log first few for debugging
                                con_type = constraint_types.get(i, "path_constraint")
                                log.debug(
                                    f"  Critical overscaling correction[{i}]: {con_type}, "
                                    f"row_min={row_min:.3e}, row_max={row_max:.3e}, "
                                    f"adjustment={adjustment_factor:.3e}, "
                                    f"old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}, "
                                    f"new_max={new_max:.3e}",
                                )
                        else:
                            # Compromise: use geometric mean to balance min and max
                            # Target: bring min to threshold while keeping max reasonable
                            compromise_factor = (target_min / max(row_min, 1e-15)) ** 0.7
                            if compromise_factor > 1.0:
                                old_scale = scale_g[i]
                                scale_g[i] = scale_g[i] * compromise_factor
                                n_corrected += 1

                                if i < 5:  # Log first few for debugging
                                    con_type = constraint_types.get(i, "path_constraint")
                                    log.debug(
                                        f"  Critical overscaling compromise[{i}]: {con_type}, "
                                        f"row_min={row_min:.3e}, row_max={row_max:.3e}, "
                                        f"compromise_factor={compromise_factor:.3e}, "
                                        f"old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}",
                                    )
                    # Check if row has overscaling (min entry too small but not critical)
                    elif row_min < min_scaled_entry_threshold:
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
                                    f"new_max={new_max:.3e}",
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
                                        f"old_scale={old_scale:.3e}, new_scale={scale_g[i]:.3e}",
                                    )

            # Also check for column-based overscaling (variables with very small Jacobian entries)
            # This can happen when variable scales are too large
            n_var_corrected = 0
            for j in range(min(jac_g0_scaled_check.shape[1], len(scale))):
                # Get column's scaled entries
                col_entries = jac_mag_check[:, j]
                col_nonzero = col_entries[col_entries > 0]

                if len(col_nonzero) > 0:
                    col_min = col_nonzero.min()
                    col_max = col_nonzero.max()

                    # Check for critical overscaling in column
                    if col_min < critical_overscaling_threshold:
                        # Variable scale is too large, need to reduce it
                        # Reducing scale[j] will increase scaled Jacobian entries
                        target_min = min_scaled_entry_threshold
                        reduction_factor = col_min / max(target_min, 1e-15)
                        # Clamp reduction to reasonable range
                        reduction_factor = np.clip(
                            reduction_factor, 0.1, 0.9
                        )  # Reduce scale by 10-90%

                        old_scale_j = scale[j]
                        scale[j] = scale[j] * reduction_factor
                        n_var_corrected += 1

                        if j < 5:  # Log first few for debugging
                            log.debug(
                                f"  Column overscaling correction[{j}]: "
                                f"col_min={col_min:.3e}, col_max={col_max:.3e}, "
                                f"reduction_factor={reduction_factor:.3e}, "
                                f"old_scale={old_scale_j:.3e}, new_scale={scale[j]:.3e}",
                            )
                    elif col_min < min_scaled_entry_threshold:
                        # Moderate overscaling: reduce variable scale less aggressively
                        reduction_factor = (col_min / max(min_scaled_entry_threshold, 1e-12)) ** 0.5
                        reduction_factor = np.clip(reduction_factor, 0.3, 0.9)  # Reduce by 10-70%

                        old_scale_j = scale[j]
                        scale[j] = scale[j] * reduction_factor
                        n_var_corrected += 1

                        if j < 5:  # Log first few for debugging
                            log.debug(
                                f"  Column overscaling correction[{j}]: "
                                f"col_min={col_min:.3e}, col_max={col_max:.3e}, "
                                f"reduction_factor={reduction_factor:.3e}, "
                                f"old_scale={old_scale_j:.3e}, new_scale={scale[j]:.3e}",
                            )

            if n_corrected > 0 or n_var_corrected > 0:
                log.debug(
                    f"Overscaling correction: adjusted {n_corrected} constraints and {n_var_corrected} variables "
                    f"to prevent very small entries",
                )

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
            upper_bound_extreme = 2.0  # 1e2
            if scale_g_log[i] < lower_bound_extreme:
                scale_g[i] = 10.0**lower_bound_extreme
            elif scale_g_log[i] > upper_bound_extreme:
                scale_g[i] = 10.0**upper_bound_extreme
            # Otherwise, preserve the aggressive scaling (don't clip)
        else:
            # For normal constraint types, use percentile-based clipping
            lower_bound = max(median_log - 2.0 * sqrt_10_log, -3.0)  # Allow down to 1e-3
            upper_bound = min(median_log + 2.0 * sqrt_10_log, 2.0)  # Allow up to 1e2
            if scale_g_log[i] < lower_bound:
                scale_g[i] = max(10.0**lower_bound, scale_g[i] * 0.1)
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = min(10.0**upper_bound, scale_g[i] * 10.0)

    return scale_g


def _try_scaling_strategy(
    strategy_name: str,
    nlp: Any,
    x0: np.ndarray[Any, Any],
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any],
    jac_g0_scaled: np.ndarray[Any, Any],
    variable_groups: dict[str, list[int]] | None,
    constraint_types: dict[int, str] | None = None,
    meta: dict[str, Any] | None = None,
    g0_arr: np.ndarray[Any, Any] | None = None,
    target_max_entry: float = 1e2,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
                new_scale[i] = np.clip(
                    new_scale[i],
                    scale_median / tight_factor,
                    scale_median * tight_factor,
                )

        if len(new_scale_g) > 0:
            scale_g_median = np.median(new_scale_g[new_scale_g > 1e-10])
            for i in range(len(new_scale_g)):
                if new_scale_g[i] > 1e-10:
                    new_scale_g[i] = np.clip(
                        new_scale_g[i],
                        scale_g_median / tight_factor,
                        scale_g_median * tight_factor,
                    )

    elif strategy_name == "row_max_scaling":
        # Strategy 2: Scale constraint rows based on max entry per row
        # Target: max scaled entry per row should be <= target_max_entry
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first (min entry too small)
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    # Limit increase to prevent creating new max entry issues
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > target_max_entry:
                    # Reduce scale_g[i] to bring max entry down to target
                    reduction_factor = target_max_entry / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (target_max_entry / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
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
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first (min entry too small)
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    # Limit reduction to reasonable range
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > target_max_entry:
                    # Increase scale[j] to bring max entry down to target
                    # But be conservative - don't increase too much at once
                    increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x increase
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / target_max_entry) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    # More conservative clamping to prevent extreme variable scaling
                    if col_max > 1e10:
                        # Extra aggressive for extreme entries, but still bounded
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

    elif strategy_name == "extreme_entry_targeting":
        # Strategy 4: Aggressively target extreme entries (>1e2)
        # Find rows and columns with extreme entries and scale them down aggressively
        # Use adaptive threshold based on current max entry
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        global_max = jac_mag.max() if jac_g0_scaled.size > 0 else 0.0
        if global_max > 1e10:
            # For very extreme entries, use more aggressive threshold
            extreme_threshold = 1e1  # Target O(10) for extreme cases
        else:
            extreme_threshold = 1e2  # Target O(100) for moderate cases

        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > extreme_threshold:
                    # Aggressively reduce scale_g[i] to bring max entry to threshold
                    reduction_factor = extreme_threshold / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (extreme_threshold / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
                    # Allow very aggressive scaling for extreme entries
                    if row_max > 1e10:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                    else:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-6, 1e3)

        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > extreme_threshold:
                    # Aggressively increase scale[j] to bring max entry to threshold
                    # But cap the increase to prevent over-scaling variables
                    increase_factor = min(col_max / extreme_threshold, 100.0)  # Cap at 100x
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / extreme_threshold) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    # Allow aggressive scaling but with tighter bounds
                    if col_max > 1e10:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

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
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        # First apply row scaling
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > target_max_entry:
                    reduction_factor = target_max_entry / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (target_max_entry / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
                    if row_max > 1e10:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                    else:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)

        # Then apply column scaling (more conservative to avoid over-scaling)
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > target_max_entry:
                    # Cap increase factor to prevent extreme variable scaling
                    increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / target_max_entry) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    if col_max > 1e10:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

    elif strategy_name == "constraint_type_aware":
        # Strategy: Constraint-type-aware scaling
        # Use specialized scaling strategies per constraint type
        import logging

        log = logging.getLogger(__name__)

        if constraint_types is None or len(constraint_types) == 0:
            # Fall back to standard scaling if types unavailable
            log.debug(
                "Constraint-type-aware: No constraint types available, falling back to standard scaling"
            )
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
                    log.debug(
                        f"Constraint-type-aware: Computed constraint values, shape={g0_arr.shape}"
                    )
            except Exception as e:
                log.debug(f"Constraint-type-aware: Failed to compute constraint values: {e}")
                g0_arr = None

        # Apply constraint-type-aware scaling
        # Pass current scale_g so it can refine existing scaling rather than starting from scratch
        log.debug(
            f"Constraint-type-aware: Computing scaling with current scale_g range=[{scale_g.min():.3e}, {scale_g.max():.3e}]"
        )
        try:
            new_scale_g = _compute_constraint_scaling_by_type(
                constraint_types,
                nlp,
                x0,
                lbg,
                ubg,
                new_scale,
                jac_g0_arr=jac_g0_arr,
                g0_arr=g0_arr,
                current_scale_g=scale_g,
            )
            log.debug(
                f"Constraint-type-aware: Computed new scale_g, range=[{new_scale_g.min():.3e}, {new_scale_g.max():.3e}]"
            )
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
                    scale_diff = abs(new_scale_g[idx] - scale_g[idx])
                    scale_ref = float(max(abs(scale_g[idx]), 1.0))
                    if scale_diff > 1e-6 * scale_ref:
                        n_modified += 1
                        if con_type not in modified_types:
                            modified_types[con_type] = {"count": 0, "max_change": 0.0}
                        modified_types[con_type]["count"] += 1
                        change_ratio = abs(new_scale_g[idx] / (scale_g[idx] + 1e-10))
                        modified_types[con_type]["max_change"] = max(
                            modified_types[con_type]["max_change"], change_ratio
                        )

            log.debug(f"Constraint-type-aware: Modified {n_modified} constraints")
            for con_type, stats in modified_types.items():
                log.debug(
                    f"  {con_type}: {stats['count']} constraints, max_change_ratio={stats['max_change']:.3e}"
                )
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

    elif strategy_name == "jacobian_equilibration":
        # Strategy: Iterative Jacobian equilibration
        # Balance row and column norms to improve numerical conditioning
        if jac_g0_arr is not None and jac_g0_arr.size > 0:
            new_scale, new_scale_g = _equilibrate_jacobian_iterative(
                new_scale,
                new_scale_g,
                jac_g0_arr,
                n_iterations=3,
                target=1.0,
            )

    return new_scale, new_scale_g


def _normalize_scales_to_center(
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    scale_f: float,
    target_center: float = 1.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """
    Normalize all scales to maintain center point and 10^1 ratio constraint.

    Ensures:
    - Median of all scales is at target_center
    - All scales within 10^1 ratio of center
    - Preserves relative importance of constraints

    Args:
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        scale_f: Objective scaling factor
        target_center: Target center point (default 1.0)

    Returns:
        Tuple of (normalized_scale, normalized_scale_g, normalized_scale_f)
    """
    # Compute median scales (robust center)
    scale_median = np.median(scale[scale > 1e-10]) if (scale > 1e-10).any() else target_center
    scale_g_median = (
        np.median(scale_g[scale_g > 1e-10])
        if len(scale_g) > 0 and (scale_g > 1e-10).any()
        else target_center
    )

    # Find overall center (median of all scales)
    all_scales = np.concatenate(
        [
            scale[scale > 1e-10],
            scale_g[scale_g > 1e-10],
            [abs(scale_f)] if abs(scale_f) > 1e-10 else [],
        ]
    )
    overall_median = np.median(all_scales) if len(all_scales) > 0 else target_center

    # Normalize to target_center
    if overall_median > 1e-10:
        center_ratio = target_center / overall_median
        scale = scale * center_ratio
        scale_g = scale_g * center_ratio
        scale_f = scale_f * center_ratio

    # Apply relaxed ratio constraint around center
    # We use a very loose constraint (1e12) to allow for physical scale differences
    # (e.g. pressure ~1e5 vs valve area ~1e-4 requires ~1e9 ratio in scales)
    # The previous 10.0 ratio was too restrictive for multi-physics problems
    max_ratio = 1e12
    sqrt_ratio = np.sqrt(max_ratio)
    lower_bound = target_center / sqrt_ratio
    upper_bound = target_center * sqrt_ratio

    # Clamp variable scales
    for i in range(len(scale)):
        if scale[i] > 1e-10:
            scale[i] = np.clip(scale[i], lower_bound, upper_bound)

    # Clamp constraint scales
    for i in range(len(scale_g)):
        if scale_g[i] > 1e-10:
            scale_g[i] = np.clip(scale_g[i], lower_bound, upper_bound)

    # Clamp objective scale
    if abs(scale_f) > 1e-10:
        scale_f = np.clip(scale_f, lower_bound, upper_bound)

    return scale, scale_g, scale_f


def _compute_unified_data_driven_scaling(
    nlp: Any,
    x0: np.ndarray[Any, Any],
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    variable_groups: dict[str, list[int]] | None = None,
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, dict[str, Any]]:
    """
    Unified data-driven scaling that analyzes actual constraint/objective distributions,
    finds the median from the data, and normalizes all constraints to that center point.

    Strategy:
    1. Evaluate all constraints g(x0), objective f(x0), constraint Jacobian J_g, and objective gradient grad f at once
    2. Compute variable scaling (simplified - unit-based + bounds)
    3. Compute unified constraint magnitudes combining values, Jacobian sensitivity, and bounds
    4. Analyze distribution using robust statistics (median, IQR, percentiles) - no distribution assumptions
    5. Find median from actual data
    6. Normalize all constraints to median center with outlier handling
    7. Normalize objective to same center
    8. Final normalization to maintain 10^1 ratio constraint

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess (unscaled)
        lbx: Lower variable bounds
        ubx: Upper variable bounds
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        variable_groups: Variable group mapping (optional)
        meta: Problem metadata (optional)
        reporter: Optional reporter for logging

    Returns:
        Tuple of (scale, scale_g, scale_f, quality_metrics):
        - scale: Variable scaling factors
        - scale_g: Constraint scaling factors
        - scale_f: Objective scaling factor
        - quality_metrics: Dictionary with quality metrics (condition_number, quality_score, etc.)
    """
    if reporter:
        reporter.info(
            "Computing unified data-driven scaling (analyzing distribution, normalizing to median center)..."
        )

    # Step 1: Evaluate everything at once
    g0_arr, f0_val, jac_g0_arr, grad_f0_arr = _evaluate_nlp_at_x0(nlp, x0, meta=meta)

    if len(g0_arr) == 0:
        # No constraints - use simple variable scaling only
        scale = _compute_variable_scaling(lbx, ubx, x0=x0, variable_groups=variable_groups)
        scale_g = np.array([])
        scale_f = 1.0
        quality_metrics = {"condition_number": 0.0, "quality_score": 0.0}
        return scale, scale_g, scale_f, quality_metrics

    # Step 2: Compute variable scaling (simplified - unit-based + bounds, no Jacobian refinement)
    # Get unscaled Jacobian for variable scaling if needed (but don't use for refinement)
    scale = _compute_variable_scaling(
        lbx,
        ubx,
        x0=x0,
        variable_groups=variable_groups,
        jac_g0_arr=None,
    )

    # Step 3: Compute unified constraint magnitudes
    constraint_magnitudes = _compute_unified_constraint_magnitudes(
        g0_arr,
        jac_g0_arr,
        lbg,
        ubg,
        scale,
    )

    # Step 4 & 5: Analyze distribution and normalize (Per-Group Strategy)
    scale_g = np.ones(len(constraint_magnitudes))
    if reporter:
        reporter.info(
            f"DEBUG: constraint_magnitudes min={constraint_magnitudes.min():.3e}, max={constraint_magnitudes.max():.3e}, mean={constraint_magnitudes.mean():.3e}"
        )

    # Track global stats for logging/fallback
    global_dist = _analyze_magnitude_distribution(constraint_magnitudes)
    median = global_dist["median"]
    iqr = global_dist["iqr"]
    outlier_mask = global_dist["outlier_mask"]
    percentiles = global_dist["percentiles"]

    if reporter:
        reporter.info("Using magnitude-based clustering scaling strategy...")

    # Betts-Style Scaling for Collocation Residuals
    # ---------------------------------------------
    # Calculate group scales from variable scales
    # Map: constraint_group -> list of variable names to combine

    # Helper to get scale for a variable name safely
    def _combine_scales(scales_list):
        valid_scales = [s for s in scales_list if s is not None and np.isfinite(s) and s > 1e-10]
        if valid_scales:
            return float(np.median(valid_scales))
        return 1.0

    def get_var_scale(name):
        # Find index of variable group
        if variable_groups and name in variable_groups:
            indices = variable_groups[name]
            if indices:
                # Use median of variable scales in this group
                return float(np.median(scale[indices]))
        return None

    # Compute scales for each semantic group
    # Note: Variable groups are named "positions", "velocities", etc. in nlp.py
    scale_positions = _combine_scales([get_var_scale("positions")])
    scale_velocities = _combine_scales([get_var_scale("velocities")])
    scale_densities = _combine_scales([get_var_scale("densities")])
    scale_temperatures = _combine_scales([get_var_scale("temperatures")])
    scale_fractions = _combine_scales([get_var_scale("scavenging_fractions")])

    # Integrals combine multiple variable groups
    scale_integrals = _combine_scales(
        [
            get_var_scale("scavenging_masses"),
            get_var_scale("scavenging_area_integrals"),
            get_var_scale("scavenging_time_moments"),
        ]
    )

    # Track which indices have been scaled by this method so we exclude them from clustering
    betts_scaled_indices = set()

    # Apply to constraint groups
    if meta and "constraint_groups" in meta:
        c_groups = meta["constraint_groups"]

        # Map group name to scale value
        betts_groups = {
            "collocation_residuals_positions": scale_positions,
            "collocation_residuals_velocities": scale_velocities,
            "collocation_residuals_densities": scale_densities,
            "collocation_residuals_temperatures": scale_temperatures,
            "collocation_residuals_fractions": scale_fractions,
            "collocation_residuals_integrals": scale_integrals,
        }

        for group_name, group_scale in betts_groups.items():
            if group_name in c_groups:
                indices = c_groups[group_name]
                if indices:
                    idx_array = np.array(indices)
                    scale_g[idx_array] = group_scale
                    betts_scaled_indices.update(indices)

                    if reporter:
                        reporter.info(
                            f"  Betts-Style Group {group_name}: scale={group_scale:.1e} "
                            f"(derived from variables)"
                        )

    # Cluster remaining constraints by magnitude
    # Filter out indices that were already scaled
    remaining_indices = [
        i for i in range(len(constraint_magnitudes)) if i not in betts_scaled_indices
    ]

    if not remaining_indices:
        return scale, scale_g, scale_f, {"condition_number": 0.0, "quality_score": 1.0}

    remaining_magnitudes = constraint_magnitudes[remaining_indices]

    # We need to map back from remaining_magnitudes index to original index
    # Create a map: local_idx -> global_idx
    local_to_global = {i: global_idx for i, global_idx in enumerate(remaining_indices)}

    clusters = _cluster_by_magnitude(remaining_magnitudes)

    # Sort bins for consistent processing
    sorted_bins = sorted(clusters.keys())

    for bin_idx in sorted_bins:
        local_indices = clusters[bin_idx]
        if not local_indices:
            continue

        # Map back to global indices
        global_indices = [local_to_global[i] for i in local_indices]
        idx_array = np.array(global_indices)

        # Use original magnitudes for analysis
        group_mags = constraint_magnitudes[idx_array]

        # Determine aggressiveness based on magnitude
        # Extreme bins: < -6 (1e-6) or > 6 (1e6)
        is_extreme = bin_idx < -6 or bin_idx > 6

        # Analyze distribution for this cluster
        # Use aggressive detection for extreme bins to avoid over-scaling outliers
        group_dist = _analyze_magnitude_distribution(group_mags, aggressive=is_extreme)
        group_median = group_dist["median"]
        group_iqr = group_dist["iqr"]
        group_outliers = group_dist["outlier_mask"]

        # Determine scale bounds
        # Standard: [1e-8, 1e8]
        # Extreme: [1e-4, 1e8] (tighter lower bound to prevent extreme multipliers)
        scale_bounds = (1e-4, 1e8) if is_extreme else (1e-8, 1e8)

        # Compute scales
        group_scales = _normalize_to_median_center(
            group_mags,
            group_median,
            group_iqr,
            group_outliers,
            scale_bounds=scale_bounds,
        )

        # Apply scales
        scale_g[idx_array] = group_scales

        # Logging with semantic tags
        if reporter:
            # Identify semantic groups in this cluster
            tags = set()
            if meta and "constraint_groups" in meta:
                for group_name, group_indices in meta["constraint_groups"].items():
                    # Skip the Betts-scaled groups in this check to reduce noise
                    if group_name.startswith("collocation_residuals_"):
                        continue

                    group_set = set(group_indices)
                    cluster_set = set(global_indices)
                    if not group_set.isdisjoint(cluster_set):
                        tags.add(group_name)

            tag_str = ", ".join(sorted(tags)) if tags else "unnamed"
            magnitude_label = f"1e{bin_idx}"
            reporter.info(
                f"  Cluster {magnitude_label:>5} ({len(global_indices):3d} items): "
                f"median={group_median:.1e}, scale_mean={group_scales.mean():.1e} "
                f"[{'AGGRESSIVE' if is_extreme else 'STANDARD'}] "
                f"Tags: {tag_str}"
            )

    # Log distribution analysis (Global)
    if reporter:
        n_outliers = int(outlier_mask.sum())
        reporter.info(
            f"Global distribution: median={median:.3e}, IQR={iqr:.3e}, "
            f"outliers={n_outliers}/{len(constraint_magnitudes)} ({100.0 * n_outliers / len(constraint_magnitudes):.1f}%), "
            f"percentiles: p5={percentiles['p5']:.3e}, p25={percentiles['p25']:.3e}, "
            f"p50={percentiles['p50']:.3e}, p75={percentiles['p75']:.3e}, p95={percentiles['p95']:.3e}",
        )

    # Step 6: Compute objective scaling (normalize to same center as constraints)
    obj_magnitude = max(abs(f0_val), 1.0)
    if grad_f0_arr is not None:
        # Include gradient norm in objective magnitude
        grad_norm = np.linalg.norm(grad_f0_arr / np.maximum(scale, 1e-10))
        obj_magnitude = max(obj_magnitude, grad_norm)

    # Normalize objective to same center as constraints
    target_center = 1.0
    if obj_magnitude > 1e-10:
        scale_f = median / obj_magnitude
    else:
        scale_f = 1.0

    # Step 7: Adaptive global normalization
    # Only apply if per-group scaling hasn't already achieved good conditioning
    # Check if scales are already well-centered (median close to 1.0)
    all_scales = np.concatenate(
        [
            scale[scale > 1e-10],
            scale_g[scale_g > 1e-10],
            [abs(scale_f)] if abs(scale_f) > 1e-10 else [],
        ]
    )
    current_median = np.median(all_scales) if len(all_scales) > 0 else 1.0

    # Check if we need global normalization
    # Skip if median is already close to target (within 10x) and spread is reasonable
    median_ratio = current_median / target_center
    needs_normalization = not (0.1 <= median_ratio <= 10.0)

    if needs_normalization:
        if reporter:
            reporter.info(
                f"Applying group-aware global normalization (current median={current_median:.3e}, "
                f"target={target_center:.3e})"
            )
        # Apply group-aware normalization that preserves relative group structure
        scale, scale_g, scale_f = _normalize_scales_to_center(
            scale, scale_g, scale_f, target_center=target_center
        )
    elif reporter:
        reporter.info(
            f"Skipping global normalization (per-group scaling already good: "
            f"median={current_median:.3e}, target={target_center:.3e})"
        )

    # TARGETED FIX: Energy Constraint Scaling
    # The Energy collocation constraints are often dominated by large residuals (due to units),
    # leading to small scale_g, which makes the Jacobian row vanish.
    # We force scaling based on the Energy variable magnitude to ensure J ~ 1/dt.
    if (
        meta
        and "constraint_groups" in meta
        and variable_groups
        and "temperatures" in variable_groups
    ):
        # Note: 'temperatures' group contains Energy variables (E) as per nlp.py
        e_indices = variable_groups["temperatures"]

        if e_indices:
            # Get median magnitude of Energy variables from x0 (unscaled)
            # We use x0 because 'scale' array might be clamped or influenced by bounds,
            # whereas we want to normalize the physical residual (which depends on x0 magnitude).
            e_vals = np.abs(x0[e_indices])
            mag_E = np.median(e_vals) if len(e_vals) > 0 else 1.0

            # Target scale_g = 1 / mag_E
            # This normalizes the constraint residual (roughly proportional to E) to ~1.
            target_scale_g = 1.0 / max(mag_E, 1e-3)

            # Apply to all Energy-related groups (collocation, continuity, boundary)
            count = 0
            for group_name, indices in meta["constraint_groups"].items():
                if (
                    group_name.startswith("collocation_E")
                    or group_name.startswith("continuity_E")
                    or group_name.startswith("boundary_E")
                    or group_name.startswith("boundary_initial_E")
                    or group_name.startswith("boundary_final_E")
                ):
                    scale_g[indices] = target_scale_g
                    count += len(indices)

            if reporter and count > 0:
                reporter.info(
                    f"Applied targeted Energy scaling to {count} constraints (collocation/continuity/boundary). "
                    f"Override Scale: {target_scale_g:.2e} (based on Var Mag: {mag_E:.2e})"
                )

    # Step 8: Verify scaling quality
    quality_metrics = _verify_scaling_quality(
        nlp,
        x0,
        scale,
        scale_g,
        lbg,
        ubg,
        reporter=reporter,
        meta=meta,
    )

    # Add distribution analysis to quality metrics
    quality_metrics["distribution_analysis"] = {
        "median": median,
        "iqr": iqr,
        "n_outliers": int(outlier_mask.sum()),
        "outlier_ratio": float(outlier_mask.sum() / len(constraint_magnitudes))
        if len(constraint_magnitudes) > 0
        else 0.0,
        "percentiles": percentiles,
    }

    # SAFETY CLAMP: Ensure no constraint is scaled effectively to zero
    # "Small elements" in Jacobian often come from tiny scale_g
    # We enforce a floor.
    if reporter:
        n_below_floor = np.sum(scale_g < 1e-4)
        if n_below_floor > 0:
            reporter.info(f"Clamping {n_below_floor} constraint scale factors to 1e-4 floor.")

    scale_g = np.maximum(scale_g, 1e-4)
    # Also clamp upper bound to avoid exploding gradients
    scale_g = np.minimum(scale_g, 1e5)

    if reporter:
        condition_number = quality_metrics.get("condition_number", np.inf)
        quality_score = quality_metrics.get("quality_score", 0.0)
        reporter.info(
            f"Unified scaling complete: condition_number={condition_number:.3e}, "
            f"quality_score={quality_score:.3f}, "
            f"scaled_constraints_center={median:.3e}",
        )

    return scale, scale_g, scale_f, quality_metrics


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
    key_data: dict[str, Any] = {
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
) -> tuple[np.ndarray[Any, Any] | None, np.ndarray[Any, Any] | None, dict[str, Any] | None]:
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
        with open(cache_path) as f:
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
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    quality: dict[str, Any],
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
            with open(cache_path) as f:
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
    x0: np.ndarray[Any, Any],
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    variable_groups: dict[str, list[int]] | None,
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
    max_iterations: int = 5,
    target_condition_number: float = 1e3,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
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
        target_condition_number: Target condition number (default 1e3)

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
    cached_scale, cached_scale_g, cached_quality = _load_scaling_cache(
        cache_key, n_vars, n_constraints
    )

    if cached_scale is not None and cached_scale_g is not None:
        # Verify cached scaling quality
        cached_quality_check = _verify_scaling_quality(
            nlp,
            x0,
            cached_scale,
            cached_scale_g,
            lbg,
            ubg,
            reporter=None,
            meta=meta,
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
                    f"quality_score={cached_score:.3f} >= {min_quality_score:.3f}",
                )
            return cached_scale, cached_scale_g, cached_quality_check
            # Use cached scaling as starting point for iteration
            if reporter:
                condition_msg = (
                    f"condition_number={cached_condition:.3e} > {target_condition_number:.3e}"
                    if cached_condition > target_condition_number
                    else ""
                )
                quality_msg = (
                    f"quality_score={cached_score:.3f} < {min_quality_score:.3f}"
                    if cached_score < min_quality_score
                    else ""
                )
                reason = " or ".join(filter(None, [condition_msg, quality_msg]))
                reporter.info(
                    f"Using cached scaling as starting point: condition_number={cached_condition:.3e}, "
                    f"quality_score={cached_score:.3f} ({reason}), will iterate to improve",
                )
            # Start with cached scaling instead of computing initial scaling
            scale = cached_scale.copy()
            scale_g = cached_scale_g.copy()
            quality = cached_quality_check
            condition_number = cached_condition
            initial_condition_number = condition_number
            # Skip initial scaling computation and go straight to iteration
            skip_initial_scaling = True

    # Get unscaled Jacobian early for use in variable scaling
    jac_g0_arr_initial = None
    try:
        jac_g0_arr_initial, _ = _compute_scaled_jacobian(
            nlp, x0, np.ones(len(x0)), np.ones(len(lbg) if lbg is not None else 0)
        )
    except Exception:
        pass  # Jacobian unavailable, will use fallback

    if not skip_initial_scaling:
        # Compute initial scaling with Jacobian if available
        scale = _compute_variable_scaling(
            lbx,
            ubx,
            x0=x0,
            variable_groups=variable_groups,
            jac_g0_arr=jac_g0_arr_initial,
        )
        try:
            scale_g = _compute_constraint_scaling_from_evaluation(nlp, x0, lbg, ubg, scale=scale)
        except Exception:
            scale_g = _compute_constraint_scaling(lbg, ubg)

        # Check initial quality
        quality = _verify_scaling_quality(
            nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta
        )
        condition_number = quality.get("condition_number", np.inf)
        initial_condition_number = condition_number  # Save for comparison at end

        if reporter:
            reporter.info(
                f"Initial scaling quality: condition_number={condition_number:.3e}, quality_score={quality.get('quality_score', 0.0):.3f}"
            )

        # If condition number is acceptable, save and return initial scaling
        if condition_number <= target_condition_number:
            _save_scaling_cache(cache_key, scale, scale_g, quality)
            return scale, scale_g, quality

    # Get unscaled and scaled Jacobian for strategy evaluation
    jac_g0_arr, jac_g0_scaled = _compute_scaled_jacobian(nlp, x0, scale, scale_g)
    if jac_g0_arr is None or jac_g0_scaled is None:
        # Can't compute Jacobian, fall back to simple tightening
        if reporter:
            reporter.warning(
                "Cannot compute Jacobian for refinement, using simple ratio tightening"
            )
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
                "jacobian_equilibration",  # Iterative Jacobian equilibration
                "combined_row_column",  # Combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",  # Scale rows with large max entries
                "column_max_scaling",  # Scale columns with large max entries
                "percentile_based",  # Scale based on percentiles
            ]
        else:
            strategies = [
                "jacobian_equilibration",  # Iterative Jacobian equilibration
                "combined_row_column",  # Most aggressive - combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",  # Scale rows with large max entries
                "column_max_scaling",  # Scale columns with large max entries
                "percentile_based",  # Scale based on percentiles
            ]
    # For moderately ill-conditioned problems, include conservative strategies
    elif has_constraint_groups:
        strategies = [
            "constraint_type_aware",  # Most targeted - use constraint type information
            "jacobian_equilibration",  # Iterative Jacobian equilibration
            "combined_row_column",  # Combine row and column scaling
            "extreme_entry_targeting",  # Aggressively target extreme entries
            "row_max_scaling",  # Scale rows with large max entries
            "column_max_scaling",  # Scale columns with large max entries
            "percentile_based",  # Scale based on percentiles
            "tighten_ratios",  # Conservative - tighten ratios
        ]
    else:
        strategies = [
            "jacobian_equilibration",  # Iterative Jacobian equilibration
            "combined_row_column",  # Most aggressive - combine row and column scaling
            "extreme_entry_targeting",  # Aggressively target extreme entries
            "row_max_scaling",  # Scale rows with large max entries
            "column_max_scaling",  # Scale columns with large max entries
            "percentile_based",  # Scale based on percentiles
            "tighten_ratios",  # Conservative - tighten ratios
        ]

    # Track previous iteration's condition number for improvement detection
    prev_condition_number = condition_number

    # Track relaxed groups to prevent oscillations (cache relaxation decisions)
    relaxed_groups_cache: dict[str, int] = {}  # Maps group name to iteration when relaxed
    skip_iterations = 3  # Skip re-tightening for 3 iterations after relaxation

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
                    reporter.debug(
                        f"  Input scales: scale range=[{scale.min():.3e}, {scale.max():.3e}], scale_g range=[{scale_g.min():.3e}, {scale_g.max():.3e}]"
                    )
                test_scale, test_scale_g = _try_scaling_strategy(
                    strategy,
                    nlp,
                    x0,
                    lbx,
                    ubx,
                    lbg,
                    ubg,
                    scale,
                    scale_g,
                    jac_g0_arr,
                    jac_g0_scaled,
                    variable_groups,
                    constraint_types=constraint_types if has_constraint_groups else None,
                    meta=meta,
                    g0_arr=g0_arr,
                    target_max_entry=1e2,
                )
                if reporter:
                    reporter.debug(
                        f"  Output scales: scale range=[{test_scale.min():.3e}, {test_scale.max():.3e}], scale_g range=[{test_scale_g.min():.3e}, {test_scale_g.max():.3e}]"
                    )
                    scale_changed = not np.allclose(test_scale, scale, rtol=1e-6)
                    scale_g_changed = not np.allclose(test_scale_g, scale_g, rtol=1e-6)
                    reporter.debug(
                        f"  Strategy modified scales: scale={scale_changed}, scale_g={scale_g_changed}"
                    )

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
                            nlp,
                            x0,
                            lbg,
                            ubg,
                            scale=test_scale,
                        )
                    except Exception:
                        test_scale_g = _compute_constraint_scaling(lbg, ubg)
                elif scale_was_modified and scale_g_was_modified:
                    # Both were modified - need to recompute constraint scaling but preserve
                    # the aggressive adjustments from the strategy
                    old_scale_g = scale_g.copy()
                    try:
                        new_base_scale_g = _compute_constraint_scaling_from_evaluation(
                            nlp,
                            x0,
                            lbg,
                            ubg,
                            scale=test_scale,
                        )
                        # Preserve relative adjustments: apply the ratio of old strategy-modified
                        # scale_g to the new base scale_g
                        if (
                            len(old_scale_g) > 0
                            and len(new_base_scale_g) > 0
                            and len(test_scale_g) > 0
                        ):
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
                    nlp,
                    x0,
                    test_scale,
                    test_scale_g,
                )
                if test_jac_g0_scaled is not None:
                    # Use the new scaled Jacobian for quality evaluation
                    test_jac_mag = np.abs(test_jac_g0_scaled)
                    test_jac_max = test_jac_mag.max()
                    test_jac_min = (
                        test_jac_mag[test_jac_mag > 0].min() if (test_jac_mag > 0).any() else 1e-10
                    )
                    test_condition_number = (
                        test_jac_max / test_jac_min if test_jac_min > 0 else np.inf
                    )
                    # Compute quality score
                    condition_score = 1.0 / (1.0 + np.log10(max(test_condition_number / 1e3, 1.0)))
                    max_score = 1.0 / (1.0 + np.log10(max(test_jac_max / 1e2, 1.0)))
                    test_quality_score = 0.5 * condition_score + 0.5 * max_score
                else:
                    # Fallback to full quality evaluation
                    test_quality = _verify_scaling_quality(
                        nlp,
                        x0,
                        test_scale,
                        test_scale_g,
                        lbg,
                        ubg,
                        reporter=None,
                        meta=meta,
                    )
                    test_condition_number = test_quality.get("condition_number", np.inf)
                    test_quality_score = test_quality.get("quality_score", 0.0)

                # Log evaluation for all strategies
                if reporter:
                    # Format values safely (handle inf/nan)
                    jac_max_str = (
                        f"{test_jac_max:.3e}"
                        if (test_jac_g0_scaled is not None and np.isfinite(test_jac_max))
                        else ("N/A" if test_jac_g0_scaled is None else str(test_jac_max))
                    )
                    jac_min_str = (
                        f"{test_jac_min:.3e}"
                        if (test_jac_g0_scaled is not None and np.isfinite(test_jac_min))
                        else ("N/A" if test_jac_g0_scaled is None else str(test_jac_min))
                    )
                    condition_str = (
                        f"{test_condition_number:.3e}"
                        if np.isfinite(test_condition_number)
                        else str(test_condition_number)
                    )
                    reporter.debug(
                        f"Strategy '{strategy}' evaluation: condition_number={condition_str}, "
                        f"quality_score={test_quality_score:.3f}, "
                        f"jac_max={jac_max_str}, "
                        f"jac_min={jac_min_str}",
                    )
                    # Log per-constraint-type statistics for constraint-type-aware
                    if (
                        strategy == "constraint_type_aware"
                        and meta
                        and "constraint_groups" in meta
                        and test_jac_g0_scaled is not None
                    ):
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
                                type_max_str = (
                                    f"{type_max:.3e}" if np.isfinite(type_max) else str(type_max)
                                )
                                type_mean_str = (
                                    f"{type_mean:.3e}" if np.isfinite(type_mean) else str(type_mean)
                                )
                                reporter.debug(
                                    f"  {con_type} ({len(indices)} constraints): "
                                    f"max_entry={type_max_str}, mean_max_entry={type_mean_str}",
                                )

                # Check if this is better (lower condition number or higher quality score)
                # Handle inf/nan comparisons safely
                condition_better = (
                    np.isfinite(test_condition_number)
                    and np.isfinite(best_condition_number)
                    and test_condition_number < best_condition_number
                ) or (np.isfinite(test_condition_number) and not np.isfinite(best_condition_number))
                quality_better = (
                    np.isfinite(test_condition_number)
                    and np.isfinite(best_condition_number)
                    and test_condition_number == best_condition_number
                    and test_quality_score > best_quality_score
                )
                is_better = condition_better or quality_better

                if reporter:
                    reporter.debug(
                        f"Strategy '{strategy}' comparison: "
                        f"condition_better={condition_better}, "
                        f"quality_better={quality_better}, "
                        f"is_best={is_better}",
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
        quality = _verify_scaling_quality(
            nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta
        )
        condition_number = quality.get("condition_number", np.inf)

        # Post-processing: Detect and correct overscaling after each iteration
        # This ensures we don't accumulate overscaling issues across iterations
        # BUT: be conservative to avoid creating extreme scales that worsen condition number
        n_overscaled = quality.get("n_overscaled", 0)
        n_underscaled = quality.get("n_underscaled", 0)
        overscaling_ratio = quality.get("overscaling_ratio", 0.0)

        # Store original condition number to check if correction makes things worse
        original_condition_number = condition_number

        if n_overscaled > 0 or overscaling_ratio > 0.01:  # More than 1% overscaled entries
            if reporter:
                reporter.info(
                    f"Post-processing: Detected {n_overscaled} overscaled entries ({overscaling_ratio * 100:.1f}%), "
                    f"applying conservative corrective scaling",
                )

            # Apply overscaling correction using the same logic as in _compute_constraint_scaling_by_type
            # Recompute scaled Jacobian to check current state
            _, jac_g0_scaled_check = _compute_scaled_jacobian(nlp, x0, scale, scale_g)

            if jac_g0_scaled_check is not None:
                jac_mag_check = np.abs(jac_g0_scaled_check)
                min_scaled_entry_threshold = 1e-6
                critical_overscaling_threshold = 1e-10
                n_corrected = 0

                # Store original scales for rollback if needed
                original_scale_g = scale_g.copy()
                original_scale = scale.copy()

                # Conservative bounds on scale factors to prevent extreme values
                max_scale_g = 1e3  # Maximum constraint scale factor
                min_scale_g = 1e-6  # Minimum constraint scale factor
                max_scale = 1e3  # Maximum variable scale factor
                min_scale = 1e-4  # Minimum variable scale factor

                # Correct constraint scaling for overscaled rows (conservative)
                for i in range(min(jac_g0_scaled_check.shape[0], len(scale_g))):
                    row_entries = jac_mag_check[i, :]
                    row_nonzero = row_entries[row_entries > 0]

                    if len(row_nonzero) > 0:
                        row_min = row_nonzero.min()
                        row_max = row_nonzero.max()

                        # Check for critical overscaling
                        if row_min < critical_overscaling_threshold:
                            target_min = min_scaled_entry_threshold
                            # Limit adjustment factor to prevent extreme scales
                            max_adjustment = 1e4  # Cap at 10000x increase
                            adjustment_factor = min(
                                target_min / max(row_min, 1e-15), max_adjustment
                            )
                            new_max = row_max * adjustment_factor

                            # Only apply if new_max is reasonable AND new scale_g is within bounds
                            if new_max <= 1e2 and adjustment_factor > 1.0:
                                new_scale_g_i = scale_g[i] * adjustment_factor
                                # Clamp to reasonable bounds
                                new_scale_g_i = np.clip(new_scale_g_i, min_scale_g, max_scale_g)
                                if new_scale_g_i != scale_g[i]:  # Only update if changed
                                    scale_g[i] = new_scale_g_i
                                    n_corrected += 1
                        elif row_min < min_scaled_entry_threshold:
                            # Moderate overscaling - be more conservative
                            adjustment_factor = min_scaled_entry_threshold / max(row_min, 1e-12)
                            # Limit to smaller adjustment
                            adjustment_factor = min(adjustment_factor, 1e2)  # Cap at 100x
                            new_max = row_max * adjustment_factor

                            if new_max <= 1e2 and adjustment_factor > 1.0:
                                new_scale_g_i = scale_g[i] * adjustment_factor
                                new_scale_g_i = np.clip(new_scale_g_i, min_scale_g, max_scale_g)
                                if new_scale_g_i != scale_g[i]:
                                    scale_g[i] = new_scale_g_i
                                    n_corrected += 1

                # Correct variable scaling for overscaled columns (conservative)
                n_var_corrected = 0
                for j in range(min(jac_g0_scaled_check.shape[1], len(scale))):
                    col_entries = jac_mag_check[:, j]
                    col_nonzero = col_entries[col_entries > 0]

                    if len(col_nonzero) > 0:
                        col_min = col_nonzero.min()

                        if col_min < critical_overscaling_threshold:
                            # Reduce variable scale to increase min entry
                            target_min = min_scaled_entry_threshold
                            reduction_factor = col_min / max(target_min, 1e-15)
                            # More conservative reduction
                            reduction_factor = np.clip(
                                reduction_factor, 0.2, 0.8
                            )  # Reduce by 20-80%
                            new_scale_j = scale[j] * reduction_factor
                            new_scale_j = np.clip(new_scale_j, min_scale, max_scale)
                            if new_scale_j != scale[j]:
                                scale[j] = new_scale_j
                                n_var_corrected += 1
                        elif col_min < min_scaled_entry_threshold:
                            # Moderate overscaling - be conservative
                            reduction_factor = (
                                col_min / max(min_scaled_entry_threshold, 1e-12)
                            ) ** 0.5
                            reduction_factor = np.clip(
                                reduction_factor, 0.5, 0.9
                            )  # Reduce by 10-50%
                            new_scale_j = scale[j] * reduction_factor
                            new_scale_j = np.clip(new_scale_j, min_scale, max_scale)
                            if new_scale_j != scale[j]:
                                scale[j] = new_scale_j
                                n_var_corrected += 1

                if n_corrected > 0 or n_var_corrected > 0:
                    # Re-verify quality after correction
                    quality_after = _verify_scaling_quality(
                        nlp, x0, scale, scale_g, lbg, ubg, reporter=None, meta=meta
                    )
                    condition_number_after = quality_after.get("condition_number", np.inf)

                    # Only keep correction if it improves or doesn't significantly worsen condition number
                    # Allow up to 2x worse condition number to fix overscaling
                    if condition_number_after <= original_condition_number * 2.0:
                        if reporter:
                            reporter.info(
                                f"Post-processing: Corrected {n_corrected} constraints and {n_var_corrected} variables "
                                f"for overscaling (condition: {original_condition_number:.3e} -> {condition_number_after:.3e})",
                            )
                        quality = quality_after
                        condition_number = condition_number_after
                    else:
                        # Rollback if correction made things worse
                        if reporter:
                            reporter.info(
                                f"Post-processing: Correction worsened condition number "
                                f"({original_condition_number:.3e} -> {condition_number_after:.3e}), rolling back",
                            )
                        scale_g = original_scale_g
                        scale = original_scale
                        condition_number = original_condition_number

        if reporter:
            strategy_msg = f" (best: {best_strategy})" if best_strategy else ""
            improvement = prev_condition_number / condition_number if condition_number > 0 else 1.0
            reporter.info(
                f"Iteration {iteration + 1} quality: condition_number={condition_number:.3e}, "
                f"quality_score={quality.get('quality_score', 0.0):.3f}, "
                f"improvement={improvement:.2f}x{strategy_msg}",
            )

        # Check for over-scaled groups and relax if needed (prevent oscillations)
        over_scaled_variable_groups_raw = quality.get("over_scaled_variable_groups", [])
        over_scaled_constraint_types_raw = quality.get("over_scaled_constraint_types", [])
        over_scaled_variable_groups: list[str] = (
            over_scaled_variable_groups_raw
            if isinstance(over_scaled_variable_groups_raw, list)
            else []
        )
        over_scaled_constraint_types: list[str] = (
            over_scaled_constraint_types_raw
            if isinstance(over_scaled_constraint_types_raw, list)
            else []
        )

        if (over_scaled_variable_groups or over_scaled_constraint_types) and meta:
            # Check if groups were recently relaxed (skip if within skip_iterations)
            should_relax = False
            groups_to_relax: list[tuple[str, str]] = []

            variable_groups = meta.get("variable_groups", {})
            constraint_groups = meta.get("constraint_groups", {})

            for group_name in over_scaled_variable_groups:
                if (
                    group_name not in relaxed_groups_cache
                    or (iteration - relaxed_groups_cache[group_name]) >= skip_iterations
                ):
                    should_relax = True
                    groups_to_relax.append(("variable", group_name))

            for con_type in over_scaled_constraint_types:
                if (
                    con_type not in relaxed_groups_cache
                    or (iteration - relaxed_groups_cache[con_type]) >= skip_iterations
                ):
                    should_relax = True
                    groups_to_relax.append(("constraint", con_type))

            if should_relax:
                # Relax over-scaled groups
                relaxed_scale, relaxed_scale_g = _relax_over_scaled_groups(
                    scale,
                    scale_g,
                    over_scaled_variable_groups,
                    over_scaled_constraint_types,
                    variable_groups,
                    constraint_groups,
                )

                # Update scales
                scale = relaxed_scale
                scale_g = relaxed_scale_g

                # Cache relaxation decisions
                for group_type, group_name in groups_to_relax:
                    relaxed_groups_cache[group_name] = iteration

                # Re-evaluate quality after relaxation
                quality = _verify_scaling_quality(
                    nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta
                )
                condition_number = quality.get("condition_number", np.inf)

                if reporter:
                    reporter.info(
                        f"Relaxed over-scaled groups: variables={over_scaled_variable_groups}, "
                        f"constraints={over_scaled_constraint_types}. "
                        f"New condition_number={condition_number:.3e}",
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
        improvement_ratio = (
            prev_condition_number / condition_number if condition_number > 0 else 1.0
        )
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
            f"Final condition_number={condition_number:.3e} (target < {target_condition_number:.3e})",
        )

    # Save best scaling found (even if not perfect)
    # Only save if it's better than what we started with
    if condition_number < initial_condition_number:
        _save_scaling_cache(cache_key, scale, scale_g, quality)

    return scale, scale_g, quality


def _compute_objective_scaling(
    nlp: Any,
    x0: np.ndarray[Any, Any],
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
            f0_val = float(f0) if hasattr(f0, "__float__") else float(np.array(f0).item())
        except Exception as e:
            log.warning(
                f"Failed to evaluate objective at initial guess: {e}, skipping objective scaling"
            )
            return 1.0

        # Compute scale factor: handle very large objective values (>1e6) more aggressively
        magnitude = max(abs(f0_val), 1.0)

        if magnitude > 1e6:
            # Very large objective: use more aggressive scaling to target O(1e-6) scaled objective
            # This ensures scaled objective is well below 1e6
            scale_f = 1e-6 / magnitude
            log.debug(
                f"Objective scaling (large value): f0={f0_val:.6e}, magnitude={magnitude:.6e}, scale_f={scale_f:.6e}"
            )
        else:
            # Normal scaling: target O(1) scaled objective
            scale_f = 1.0 / magnitude
            log.debug(f"Objective scaling: f0={f0_val:.6e}, scale_f={scale_f:.6e}")

        # Clamp to reasonable range to prevent extreme scaling factors
        # Minimum scale: 1e-10 (very small objectives get minimal scaling)
        # Maximum scale: 1e6 (very large objectives get aggressive scaling)
        scale_f = np.clip(scale_f, 1e-10, 1e6)

        # Validate: check if scaled objective would be reasonable
        scaled_obj_estimate = abs(f0_val) * scale_f
        if scaled_obj_estimate > 1e6:
            # If scaled objective would still be >1e6, apply additional reduction
            additional_reduction = 1e6 / scaled_obj_estimate
            scale_f = scale_f * additional_reduction
            log.debug(
                f"Objective scaling: applied additional reduction {additional_reduction:.6e} to ensure scaled objective <1e6"
            )

        log.debug(
            f"Objective scaling final: f0={f0_val:.6e}, scale_f={scale_f:.6e}, estimated_scaled_obj={abs(f0_val) * scale_f:.6e}"
        )
        return scale_f

    except Exception as e:
        log.warning(f"Error in objective scaling: {e}, skipping objective scaling")
        return 1.0


def _compute_constraint_scaling_from_evaluation(
    nlp: Any,
    x0: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
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
            log.warning(
                f"Failed to evaluate constraints at initial guess: {e}, using bounds-based scaling"
            )
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
                ref_jac_norm_val = np.median(non_zero_norms)
                ref_jac_norm = float(ref_jac_norm_val)
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
                ref_magnitude_val = np.median(non_zero)
                ref_magnitude = float(ref_magnitude_val)
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
                    magnitude = float(abs(g_val)) if np.isfinite(g_val) else 1.0
                elif lb == -np.inf:
                    magnitude = float(abs(ub))
                elif ub == np.inf:
                    magnitude = float(abs(lb))
                else:
                    magnitude = max(float(abs(lb)), float(abs(ub)))

                # Incorporate actual constraint value (but don't let tiny values dominate)
                if np.isfinite(g_val):
                    # Use max of bounds and value, but cap value influence
                    g_val_abs = float(abs(g_val))
                    bounds_max = max(float(abs(lb)), float(abs(ub)))
                    magnitude = max(magnitude, min(g_val_abs, bounds_max * 10.0))
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
                        magnitude = max(
                            magnitude, jac_norm_i / 1e3
                        )  # More aggressive normalization

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
            f"n_equality={equality_mask.sum()}/{n_cons}",
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
        upper_bound = min(median_log + sqrt_10_log, 2.0)  # At most 1e2

        n_clipped = 0
        for i in range(n_cons):
            if scale_g_log[i] < lower_bound:
                scale_g[i] = 10.0**lower_bound
                n_clipped += 1
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = 10.0**upper_bound
                n_clipped += 1
        # Apply clamping strategy similar to variables: ensure constraint scales stay within [median/sqrt(10), median*sqrt(10)]
        # This gives 10¹ ratio (tighter than before)
        scale_g_median = np.median(scale_g[scale_g > 1e-10])
        if scale_g_median > 1e-10:
            sqrt_10 = np.sqrt(10.0)
            for i in range(n_cons):
                if scale_g[i] > 1e-10:
                    scale_g[i] = np.clip(
                        scale_g[i], scale_g_median / sqrt_10, scale_g_median * sqrt_10
                    )
        log.info(
            f"Constraint scaling (post-normalization): range=[{scale_g.min():.3e}, {scale_g.max():.3e}], "
            f"mean={scale_g.mean():.3e}, clipped={n_clipped}/{n_cons} outliers",
        )
        return scale_g

    except Exception as e:
        log.warning(
            f"Error in constraint evaluation-based scaling: {e}, using bounds-based scaling"
        )
        return _compute_constraint_scaling(lbg, ubg)  # type: ignore[return-value]


# _apply_problem_bounds extracted to setup.py


def solve_cycle_robust(params: dict[str, Any]) -> dict[str, Any]:
    """
    Solve OP engine cycle with robust IPOPT settings.

    This function uses more conservative IPOPT settings for difficult problems.

    Args:
        params: Problem parameters dictionary

    Returns:
        Solution object with optimization results
    """
    # Set robust problem type
    params_robust = params.copy()
    params_robust["problem_type"] = "robust"

    return solve_cycle(params_robust)


def solve_cycle_with_warm_start(
    params: dict[str, Any],
    x0: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """
    Solve OP engine cycle with warm start.

    Args:
        params: Problem parameters dictionary
        x0: Initial guess for optimization variables

    Returns:
        Solution object with optimization results
    """
    # Add warm start information
    params_warm = params.copy()
    params_warm["warm_start"] = {"x0": x0.tolist()}

    return solve_cycle(params_warm)


def solve_cycle_with_fuel_continuation(
    params: dict[str, Any],
    fuel_steps: list[float] | None = None,
    max_retries: int = 2,
) -> Solution:
    """
    Solve cycle using fuel-based continuation (homotopy).

    Gradually ramps fuel from 0 (motoring) to target load, using each
    solution as warm start for the next step. This provides a smooth path
    from the easy-to-solve motoring cycle to the stiff combustion problem.

    Args:
        params: Problem parameters dictionary
        fuel_steps: Fuel fractions to solve sequentially [0.0, 0.1, 0.5, 1.0] (default)
                   Each value is a fraction of the target fuel mass
        max_retries: Maximum retries per step with relaxed tolerance (default: 2)

    Returns:
        Solution object from final fuel level (target load)

    Example:
        >>> params = asdict(ConfigFactory.create_default_config())
        >>> params["combustion"]["fuel_mass_kg"] = 5e-6
        >>> result = solve_cycle_with_fuel_continuation(params)
        >>> assert result.success
    """
    # Default continuation schedule: motoring → 10% → 50% → 100%
    if fuel_steps is None:
        fuel_steps = [0.0, 0.1, 0.5, 1.0]

    # Extract target fuel mass
    combustion_cfg = params.get("combustion", {})
    target_fuel_mass = float(combustion_cfg.get("fuel_mass_kg", 5e-6))

    # Validate and normalize fuel steps
    fuel_steps = list(fuel_steps)  # Make a copy
    if fuel_steps[0] != 0.0:
        fuel_steps = [0.0] + fuel_steps
    if fuel_steps[-1] != 1.0:
        fuel_steps.append(1.0)

    log.info(
        f"Starting fuel continuation: {len(fuel_steps)} steps, "
        f"target_fuel={target_fuel_mass:.2e} kg, schedule={fuel_steps}"
    )

    current_solution = None

    for i, fuel_fraction in enumerate(fuel_steps):
        fuel_mass = target_fuel_mass * fuel_fraction
        step_name = "motoring" if fuel_fraction == 0.0 else f"{fuel_fraction * 100:.0f}% load"

        log.info(
            f"Continuation step {i + 1}/{len(fuel_steps)}: {step_name} (fuel={fuel_mass:.2e} kg)"
        )

        # Create problem for this fuel level
        params_step = params.copy()
        params_step["combustion"] = combustion_cfg.copy()
        params_step["combustion"]["fuel_mass_kg"] = fuel_mass

        # Use previous solution as warm start
        if current_solution is not None and current_solution.success:
            x_prev = current_solution.meta.get("optimization", {}).get("x_opt")
            if x_prev is not None:
                # Convert to list if numpy array
                x0_list = x_prev.tolist() if isinstance(x_prev, np.ndarray) else x_prev
                params_step["warm_start"] = {"x0": x0_list}
                log.debug(f"Using warm start from previous step (n_vars={len(x0_list)})")

        # Solve with retries
        result = _solve_with_retries(params_step, max_retries=max_retries, step_name=step_name)

        if not result.success:
            log.warning(f"Continuation failed at step {i + 1}/{len(fuel_steps)}: {step_name}")
            if current_solution is not None and current_solution.success:
                log.warning(f"Returning last successful solution (step {i})")
                # Mark as partial success in metadata
                current_solution.meta.setdefault("continuation", {})["partial"] = True
                current_solution.meta["continuation"]["final_step"] = i
                return current_solution
            else:
                log.error("No successful solution found in continuation")
                return result

        current_solution = result
        log.info(f"Step {i + 1}/{len(fuel_steps)} converged: {step_name}")

    log.info("Fuel continuation completed successfully")
    if current_solution is None:
        raise RuntimeError(f"Fuel continuation failed: step {step_name} returned no solution")

    # Mark as full continuation success
    current_solution.meta.setdefault("continuation", {})["complete"] = True
    current_solution.meta["continuation"]["steps"] = len(fuel_steps)
    return current_solution


def _solve_with_retries(
    params: dict[str, Any],
    max_retries: int = 2,
    step_name: str = "unknown",
) -> Solution:
    """
    Solve with automatic retry using progressively relaxed tolerance.

    If the solve fails, automatically retries with 10x relaxed tolerance
    on each attempt (e.g., 1e-6 → 1e-5 → 1e-4).

    Args:
        params: Problem parameters dictionary
        max_retries: Maximum number of retries (default: 2)
        step_name: Name of step for logging (default: "unknown")

    Returns:
        Solution object from first successful attempt
    """
    base_tol = params.get("solver", {}).get("ipopt", {}).get("ipopt.tol", 1e-6)

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Relax tolerance for retry
            tol = base_tol * (10**attempt)
            params_retry = params.copy()
            params_retry.setdefault("solver", {}).setdefault("ipopt", {})
            params_retry["solver"]["ipopt"]["ipopt.tol"] = tol
            log.info(f"Retry {attempt}/{max_retries} for {step_name} with relaxed tol={tol:.2e}")
            result = solve_cycle(params_retry)
        else:
            result = solve_cycle(params)

        if result["success"]:
            if attempt > 0:
                log.info(f"Converged on retry {attempt} with tol={tol:.2e}")
                # Mark retry in metadata
                result.setdefault("meta", {}).setdefault("retry", {})["attempt"] = attempt
                result["meta"]["retry"]["tolerance"] = tol
            # Convert to Solution
            return Solution(meta=result.get("meta", {}), data=result)

    log.warning(f"All {max_retries + 1} attempts failed for {step_name}")
    # Convert last result to Solution
    return Solution(meta=result.get("meta", {}), data=result)


def solve_cycle_with_refinement(
    params: dict[str, Any],
    refinement_strategy: str = "adaptive",
) -> dict[str, Any]:
    """
    Solve cycle with 0D to 1D refinement switching.

    Args:
        params: Problem parameters
        refinement_strategy: Refinement strategy ("adaptive", "fixed", "error_based")

    Returns:
        Solution dictionary
    """
    log.info(f"Starting cycle solve with {refinement_strategy} refinement strategy")

    # Initial 0D solve
    params_0d = params.copy()
    params_0d["model_type"] = "0d"
    params_0d["num"] = params_0d.get("num", {})
    params_0d["num"]["K"] = params_0d["num"].get("K_0d", 10)

    log.info("Solving with 0D model...")
    result_0d = solve_cycle(params_0d)

    if not result_0d["success"]:
        log.warning("0D solve failed, trying 1D directly")
        return solve_cycle(params)

    # Check if refinement is needed
    if refinement_strategy == "fixed":
        # Always refine to 1D
        refine = True
    elif refinement_strategy == "error_based":
        # Refine based on error estimates
        refine = _should_refine_error_based(result_0d, params)
    else:  # adaptive
        # Refine based on problem characteristics
        refine = _should_refine_adaptive(result_0d, params)

    if not refine:
        log.info("0D solution is sufficient, no refinement needed")
        return result_0d

    # Refine to 1D
    log.info("Refining to 1D model...")
    params_1d = params.copy()
    params_1d["model_type"] = "1d"
    params_1d["num"] = params_1d.get("num", {})
    params_1d["num"]["K"] = params_1d["num"].get("K_1d", 30)

    # Use 0D solution as warm start
    warm_start = _create_warm_start_from_0d(result_0d, params_1d)
    params_1d["warm_start"] = warm_start

    result_1d = solve_cycle(params_1d)

    if result_1d["success"]:
        log.info("1D refinement successful")
        return result_1d
    log.warning("1D refinement failed, returning 0D solution")
    return result_0d


def _should_refine_error_based(result_0d: dict[str, Any], params: dict[str, Any]) -> bool:
    """Determine if refinement is needed based on error estimates."""
    # Check convergence criteria
    if result_0d.get("kkt_error", float("inf")) > 1e-4:
        return True

    # Check objective function value
    f_opt = result_0d.get("f_opt", float("inf"))
    if f_opt > 1e6:  # High objective value might indicate poor solution
        return True

    # Check problem size
    num_intervals = params.get("num", {}).get("K", 10)
    if num_intervals < 20:  # Small problem might benefit from refinement
        return True

    return False


def _should_refine_adaptive(result_0d: dict[str, Any], params: dict[str, Any]) -> bool:
    """Determine if refinement is needed based on problem characteristics."""
    # Check problem complexity
    if params.get("complex_geometry", False):
        return True

    # Check if high accuracy is required
    if params.get("high_accuracy", False):
        return True

    # Check if 1D effects are important
    if params.get("1d_effects_important", False):
        return True

    # Check solution quality
    if result_0d.get("kkt_error", float("inf")) > 1e-5:
        return True

    return False


def _create_warm_start_from_0d(
    result_0d: dict[str, Any],
    params_1d: dict[str, Any],
) -> dict[str, Any]:
    """Create warm start for 1D solve from 0D solution."""
    if not result_0d["success"] or result_0d["x_opt"] is None:
        return {}

    x_0d = result_0d["x_opt"]
    if not isinstance(x_0d, (list, np.ndarray)):
        return {}
    # Convert to list[float] for interpolation functions
    if isinstance(x_0d, np.ndarray):
        x_0d_list: list[float] = x_0d.tolist()
    else:
        x_0d_list = x_0d
    n_0d = len(x_0d_list)
    num_intervals_1d = params_1d.get("num", {}).get("K", 30)
    poly_degree = params_1d.get("num", {}).get("C", 3)
    n_1d = num_intervals_1d * poly_degree * 6  # Assuming 6 variables per collocation point

    # Interpolate 0D solution to 1D grid
    if n_1d > n_0d:
        # Upsample using linear interpolation
        x_1d = _interpolate_solution(x_0d_list, n_1d)
    else:
        # Downsample using averaging
        x_1d = _downsample_solution(x_0d_list, n_1d)

    return {
        "x0": x_1d,
        "lambda0": result_0d.get("lambda_opt", []),
        "mu0": result_0d.get("mu_opt", []),
    }


def _interpolate_solution(x_0d: list[float], n_1d: int) -> list[float]:
    """Interpolate solution from 0D to 1D grid."""
    x_0d_array = np.array(x_0d)
    n_0d = len(x_0d_array)

    # Create interpolation points
    x_0d_indices = np.linspace(0, n_0d - 1, n_0d)
    x_1d_indices = np.linspace(0, n_0d - 1, n_1d)

    # Linear interpolation
    x_1d_array = np.interp(x_1d_indices, x_0d_indices, x_0d_array)

    return list(x_1d_array.tolist())


def _downsample_solution(x_0d: list[float], n_1d: int) -> list[float]:
    """Downsample solution from 0D to 1D grid."""
    x_0d_array = np.array(x_0d)
    n_0d = len(x_0d_array)

    # Average over groups
    group_size = n_0d // n_1d
    x_1d_list = []

    for i in range(n_1d):
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, n_0d)
        x_1d_list.append(np.mean(x_0d_array[start_idx:end_idx]))

    return [float(x) for x in x_1d_list]


def solve_cycle_adaptive(
    params: dict[str, Any],
    max_refinements: int = 3,
) -> dict[str, Any]:
    """
    Solve cycle with adaptive refinement strategy.

    Args:
        params: Problem parameters
        max_refinements: Maximum number of refinements

    Returns:
        Solution dictionary
    """
    log.info(f"Starting adaptive cycle solve with max {max_refinements} refinements")

    # Start with 0D model
    current_model = "0d"
    current_result: dict[str, Any] | None = None

    for refinement in range(max_refinements + 1):
        log.info(f"Refinement {refinement}: Solving with {current_model} model")

        # Set up problem for current model
        params_current = params.copy()
        params_current["model_type"] = current_model
        params_current["num"] = params_current.get("num", {})

        if current_model == "0d":
            params_current["num"]["K"] = params_current["num"].get("K_0d", 10)
        else:
            params_current["num"]["K"] = params_current["num"].get("K_1d", 30)

        # Use previous result as warm start
        if current_result is not None and current_result["success"]:
            warm_start = _create_warm_start_from_0d(current_result, params_current)
            params_current["warm_start"] = warm_start

        # Solve current model
        current_result = solve_cycle(params_current)

        if not current_result["success"]:
            log.warning(f"{current_model} solve failed at refinement {refinement}")
            if refinement == 0:
                return current_result
            # Return previous successful result
            break

        # Check if refinement is needed
        if refinement < max_refinements:
            if _should_refine_adaptive(current_result, params_current):
                current_model = "1d"
                log.info(f"Refining to 1D model for refinement {refinement + 1}")
            else:
                log.info("No further refinement needed")
                break
        else:
            log.info("Maximum refinements reached")
            break

    if current_result is None:
        # Return empty result if no solve was attempted
        return {"success": False, "message": "No solve attempted"}
    return current_result


def get_driver_function(driver_type: str = "standard") -> Any:
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


def _analyze_constraint_rank(
    nlp: dict[str, Any],
    x0: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    meta: dict[str, Any] | None = None,
    reporter: Any = None,
) -> None:
    """
    Analyze constraint Jacobian rank using SVD to identify redundant constraints.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        meta: Problem metadata (optional)
        reporter: Reporter for logging
    """
    if not reporter:
        return

    try:
        import numpy as np
        import casadi as ca

        # from scipy.sparse.linalg import svds # Unused
        import os  # Added for os.environ.get

        # from campro.logging import get_logger # This line was problematic, assuming it's not needed or was a typo for a comment
        # SVD Guard: Skip for large problems unless forced
        # For K=100 (6000 vars), SVD takes minutes.
        force_diagnostics = os.environ.get("FREE_PISTON_DIAGNOSTICS") == "1"
        is_large = len(x0) > 3000

        if is_large and not force_diagnostics:
            reporter.info(
                f"Skipping SVD audit for large problem ({len(x0)} variables). Set FREE_PISTON_DIAGNOSTICS=1 to force."
            )
            return

        reporter.info("Starting constraint rank audit (SVD)...")

        # Evaluate Jacobian at x0
        # jac_g(x) returns sparse matrix
        J_func = (
            nlp["jac_g_x"]
            if "jac_g_x" in nlp
            else ca.Function("J", [nlp["x"]], [ca.jacobian(nlp["g"], nlp["x"])])
        )
        J_val = J_func(x0)

        # Convert to dense or sparse matrix
        if hasattr(J_val, "full"):
            J = J_val.full()
        else:
            J = np.array(J_val)

        # Apply scaling: J_scaled = diag(scale_g) @ J @ diag(1/scale)
        # Efficient scaling using broadcasting
        # Rows scaled by scale_g, cols scaled by 1/scale
        J_scaled = J * scale_g[:, np.newaxis] / scale[np.newaxis, :]

        # Compute SVD
        # For large matrices, compute only smallest singular values?
        # But we want full spectrum to see the gap.
        # If matrix is huge, use randomized SVD or just top/bottom k
        m, n = J_scaled.shape
        k = min(m, n)

        if k > 1000:
            reporter.info(f"Large Jacobian ({m}x{n}), computing subset of singular values...")
            # Compute largest and smallest
            try:
                # Largest
                # svds(J_scaled, k=10, which="LM") # Unused
                # Smallest (shift-invert mode often needed for 0, but 'SM' might work)
                # For dense, just use standard svd
                u, s, _ = np.linalg.svd(J_scaled, full_matrices=False)
            except Exception:
                # Fallback to dense SVD
                u, s, _ = np.linalg.svd(J_scaled, full_matrices=False)
        else:
            u, s, _ = np.linalg.svd(J_scaled, full_matrices=False)

        # Analyze singular values
        s_max = s.max()
        s_min = s.min()
        cond = s_max / s_min if s_min > 1e-20 else float("inf")

        reporter.info(
            f"Jacobian SVD: sigma_max={s_max:.2e}, sigma_min={s_min:.2e}, cond={cond:.2e}"
        )

        # Identify near-zero singular values (redundant constraints)
        # Threshold: sigma < 1e-8 * sigma_max
        threshold = 1e-8 * s_max
        small_sv_indices = np.where(s < threshold)[0]

        if len(small_sv_indices) > 0:
            reporter.warning(
                f"Found {len(small_sv_indices)} near-zero singular values (< {threshold:.2e}). "
                "Constraints are likely redundant."
            )

            # Identify which constraints contribute to the null space
            # Look at left singular vectors (U) corresponding to small sigma
            # U columns are the directions in constraint space

            # Analyze the smallest singular value's vector
            idx_smallest = -1  # Last one is smallest in numpy svd
            u_smallest = u[:, idx_smallest]

            # Find constraints with large components in this vector
            # These are the ones involved in the dependency
            contrib_indices = np.where(np.abs(u_smallest) > 0.1)[0]

            reporter.info(
                f"Constraints involved in smallest singular value (sigma={s[-1]:.2e}): "
                f"{contrib_indices.tolist()}"
            )

            # Map indices to constraint groups if metadata available
            if meta:
                if "constraint_groups" in meta:
                    constraint_groups = meta["constraint_groups"]
                    reporter.info(
                        f"Mapping {len(contrib_indices)} indices to {len(constraint_groups)} groups..."
                    )

                    # Invert mapping: index -> group name
                    index_to_group = {}
                    for name, indices in constraint_groups.items():
                        # indices might be a slice or list
                        if isinstance(indices, slice):
                            rng = range(indices.start or 0, indices.stop, indices.step or 1)
                            for i in rng:
                                index_to_group[i] = name
                        elif isinstance(indices, (list, np.ndarray)):
                            for i in indices:
                                index_to_group[i] = name

                    # Map contributing indices to groups
                    involved_groups = set()
                    unmapped_indices = []
                    for idx in contrib_indices:
                        if idx in index_to_group:
                            involved_groups.add(index_to_group[idx])
                        else:
                            unmapped_indices.append(idx)

                    if involved_groups:
                        reporter.info(
                            f"Redundant constraints likely in groups: {sorted(list(involved_groups))}"
                        )

                    if unmapped_indices:
                        reporter.warning(
                            f"Could not map {len(unmapped_indices)} indices to groups: {unmapped_indices[:10]}..."
                        )
                else:
                    reporter.warning("Metadata provided but 'constraint_groups' key missing.")
            else:
                reporter.warning("No metadata provided for constraint mapping.")

        else:
            reporter.info("No obvious rank deficiency found (all sigma > 1e-8 * sigma_max).")

    except Exception as e:
        reporter.warning(f"Constraint rank audit failed: {e}")


def _combine_scales(scales: list[float | None]) -> float:
    """
    Combine multiple variable scales into a single robust group scale.

    Args:
        scales: List of scaling factors (can contain None)

    Returns:
        Median scale factor, clamped to [1e-6, 1e6]. Returns 1.0 if no valid scales.
    """
    # Filter out None values
    vals = [s for s in scales if s is not None]

    if not vals:
        return 1.0

    # Compute median
    vals.sort()
    m = vals[len(vals) // 2]

    # Clamp to reasonable range to prevent extreme scaling
    return max(min(m, 1e6), 1e-6)
