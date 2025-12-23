from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from campro.diagnostics.scaling import build_scaled_nlp, scale_bounds, scale_value, unscale_value

# from campro_unaligned.freepiston.zerod.cv import cv_residual
from campro.logging import get_logger
from campro.optimization.core.solution import Solution
from campro.optimization.initialization.manager import InitializationManager
from campro.optimization.initialization.setup import setup_optimization_bounds
from campro.optimization.io.save import save_json
from campro.optimization.nlp import build_collocation_nlp
from campro.optimization.nlp.diagnostics import summarize_ipopt_iterations
from campro.optimization.nlp.scaling import (
    analyze_constraint_rank,
    compute_unified_data_driven_scaling,
)
from campro.optimization.solvers.ipopt_options import (
    configure_ma27_memory,
    create_ipopt_options,
    get_available_hsl_solvers,
)
from campro.optimization.solvers.ipopt_solver import IPOPTSolver
from campro.utils import format_duration
from campro.utils.structured_reporter import StructuredReporter

# CEM integration for pre-optimization validation
try:
    from truthmaker.cem import CEMClient, ViolationSeverity

    CEM_AVAILABLE = True
except ImportError:
    CEM_AVAILABLE = False

log = get_logger(__name__)

_FALSEY = {"0", "false", "no", "off"}

# Constraint type category targets for scaling
# These targets represent the desired magnitude of scaled constraints per category
# Constraint targets moved to scaling.py


# HSL solver detection and MA27 memory configuration extracted to ipopt_options.py
# Backward-compatible aliases
_get_available_hsl_solvers = get_available_hsl_solvers
_configure_ma27_memory = configure_ma27_memory


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


# IPOPT iteration summarization extracted to nlp/diagnostics.py
# Backward-compatible alias
_summarize_ipopt_iterations = summarize_ipopt_iterations


# Scaling functions extracted to nlp/scaling.py
# _analyze_constraint_rank, _compute_constraint_scaling_by_type aliases removed


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


def solve_cycle(params: dict[str, Any]) -> Solution:
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

    # === CEM Pre-Validation ===
    # Early-exit if CEM reports FATAL violations to avoid wasted computation
    cem_enabled = params.get("cem", {}).get("enabled", CEM_AVAILABLE)
    if cem_enabled and CEM_AVAILABLE:
        try:
            with CEMClient(mock=True) as cem:
                # Get operating envelope for this configuration
                bore = params.get("geometry", {}).get("bore", 0.1)
                stroke = params.get("geometry", {}).get("stroke", 0.1)
                cr = params.get("geometry", {}).get("cr", 15.0)
                rpm = params.get("operating", {}).get("rpm", 3000.0)

                envelope = cem.get_thermo_envelope(bore, stroke, cr, rpm)
                if not envelope.feasible:
                    log.warning("CEM reports infeasible operating envelope")

                # Store envelope in params for bounds setup
                params["_cem_envelope"] = envelope
        except Exception as e:
            log.debug(f"CEM pre-validation skipped: {e}")

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
    # NOTE: Legacy cv_residual check removed - campro_unaligned module archived
    res = None

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
        ipopt_options = create_ipopt_options(ipopt_opts_dict, params)
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
        scale, scale_g, _, scaling_quality = compute_unified_data_driven_scaling(
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
            analyze_constraint_rank(nlp, x0, scale, scale_g, meta=meta, reporter=reporter)

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
                def get_var(name: str) -> np.ndarray | None:
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
    return solution


# _setup_optimization_bounds extracted to campro.optimization.initialization.setup


# Re-export solve variants for backward compatibility
from campro.optimization.solve_strategies import (
    get_driver_function,
    solve_cycle_adaptive,
    solve_cycle_robust,
    solve_cycle_with_fuel_continuation,
    solve_cycle_with_refinement,
    solve_cycle_with_warm_start,
)


def solve_cycle_orchestrated(
    params: dict[str, Any],
    budget: int = 1000,
    use_cem: bool = True,
    use_surrogate: bool = True,
    mock_simulation: bool = False,
    mock_solver: bool = False,
) -> Solution:
    """
    Solve optimization problem using CEM-gated orchestrator.

    This replaces the monolithic solve loop with a structured orchestration:
    CEM (Feasibility) -> Surrogate (Prediction) -> Solver (Refinement) -> Sim (Truth)

    Args:
        params: Problem parameters
        budget: Simulation budget
        use_cem: Whether to use CEM for constraints/generation
        use_surrogate: Whether to use ML surrogate
        mock_simulation: Whether to use mock simulation (for testing)
        mock_solver: Whether to use mock solver (for testing)

    Returns:
        Solution object
    """
    from campro.orchestration import OrchestrationConfig, Orchestrator
    from campro.orchestration.adapters import (
        CEMClientAdapter,
        EnsembleSurrogateAdapter,
        IPOPTSolverAdapter,
        MockSimulationAdapter,
        MockSurrogateAdapter,
        PhysicsSimulationAdapter,
        SimpleSolverAdapter,
    )

    log.info("Initializing CEM-gated orchestrator...")

    # Configure layers
    cem_adapter = CEMClientAdapter(mock=not use_cem)

    surrogate_adapter = None
    if use_surrogate:
        surrogate_adapter = EnsembleSurrogateAdapter.load("thermo")

    if surrogate_adapter is None:
        log.info("Using mock surrogate (model not found or disabled)")
        surrogate_adapter = MockSurrogateAdapter()

    if mock_solver:
        solver_adapter = SimpleSolverAdapter()
    else:
        solver_adapter = IPOPTSolverAdapter(base_params=params)

    if mock_simulation:
        sim_adapter = MockSimulationAdapter()
    else:
        sim_adapter = PhysicsSimulationAdapter()

    config = OrchestrationConfig(total_sim_budget=budget)

    orchestrator = Orchestrator(
        cem=cem_adapter,
        surrogate=surrogate_adapter,
        solver=solver_adapter,
        simulation=sim_adapter,
        config=config,
    )

    # Inject provenance into CEM adapter if supported
    if hasattr(cem_adapter, "set_provenance"):
        cem_adapter.set_provenance(orchestrator.provenance)

    # Run optimization
    result = orchestrator.optimize(params)

    # Return best solution
    if result.best_candidate:
        log.info(f"Orchestration complete. Best objective: {result.best_objective:.6f}")

        # If mocking, return mock solution immediately
        if mock_simulation or mock_solver:
            from campro.optimization.core.solution import Solution

            return Solution(
                meta={"optimization": {"success": True, "f_opt": result.best_objective}},
                data={"x": None, **result.best_candidate},
            )

        # Final solve to ensure we return a full Solution object with all diagnostics
        final_params = {**params, **result.best_candidate}
        return solve_cycle(final_params)
    else:
        log.warning("Orchestration failed to find solution, falling back to standard solve")
        return solve_cycle(params)


__all__ = [
    "get_driver_function",
    "solve_cycle",
    "solve_cycle_adaptive",
    "solve_cycle_orchestrated",
    "solve_cycle_robust",
    "solve_cycle_with_fuel_continuation",
    "solve_cycle_with_refinement",
    "solve_cycle_with_warm_start",
]
