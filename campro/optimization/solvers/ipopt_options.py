from __future__ import annotations

"""Central facility for constructing Ipopt/CasADi solver options.

This module converts internal :class:`campro.optimization.solvers.ipopt_solver.IPOPTOptions`
into the dictionary expected by CasADi's ``nlpsol`` and (optionally) writes an
``ipopt.opt`` file so Ipopt can read *run-time* parameters not recognised at
creation time.

Design goals
------------
1. Single source of truth for *all* Ipopt parameters used in the project.
2. Automatic handling of solver-specific defaults (MA27 vs MA57).
3. Optional emission of an ``ipopt.opt`` file at the path configured in
   :pydata:`campro.constants.IPOPT_OPT_PATH`.
4. No external side-effects unless ``emit_file=True``.
"""

import os  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Final  # noqa: E402

from campro.constants import IPOPT_OPT_PATH  # noqa: E402
from campro.logging import get_logger  # noqa: E402
from campro.optimization.solvers.ipopt_solver import IPOPTOptions, get_robust_ipopt_options

log = get_logger(__name__)

# -- Public API -------------------------------------------------------------


def build_casadi_options(
    ipopt_options: IPOPTOptions,  # type: ignore[name-defined]
    solver: str,
    *,
    emit_file: bool = True,
) -> dict[str, Any]:
    """Return CasADi options dict and optionally write *ipopt.opt* file.

    Parameters
    ----------
    ipopt_options
        Options dataclass from freepiston layer (or compatible subset).
    solver
        Selected linear solver (MA27, MA57, ...).
    emit_file
        When *True* (default) write ``ipopt.opt`` file alongside returning the
        options dict.  Existing files are overwritten.
    """

    opts: dict[str, Any] = {}

    # Map common dataclass fields to ipopt.* keys – prefer explicit list to
    # avoid leaking unwanted attributes.
    _DIRECT_MAP: Final = {
        "max_iter": "ipopt.max_iter",
        "max_cpu_time": "ipopt.max_cpu_time",
        "tol": "ipopt.tol",
        "acceptable_tol": "ipopt.acceptable_tol",
        "acceptable_iter": "ipopt.acceptable_iter",
        "mu_strategy": "ipopt.mu_strategy",
        "mu_init": "ipopt.mu_init",
        "mu_max": "ipopt.mu_max",
        "mu_min": "ipopt.mu_min",
        "line_search_method": "ipopt.line_search_method",
        "dual_inf_tol": "ipopt.dual_inf_tol",
        "compl_inf_tol": "ipopt.compl_inf_tol",
        "constr_viol_tol": "ipopt.constr_viol_tol",
        "print_level": "ipopt.print_level",
        "print_frequency_iter": "ipopt.print_frequency_iter",
        "print_frequency_time": "ipopt.print_frequency_time",
        "hessian_approximation": "ipopt.hessian_approximation",
        "limited_memory_max_history": "ipopt.limited_memory_max_history",
        "limited_memory_update_type": "ipopt.limited_memory_update_type",
    }

    data = asdict(ipopt_options)
    for field, key in _DIRECT_MAP.items():
        if field in data and data[field] is not None:
            opts[key] = data[field]

    # Note: linear_solver and hsllib are set by the IPOPT factory

    # Warm-start params (if enabled)
    if data.get("warm_start_init_point", "no") == "yes":
        opts["ipopt.warm_start_init_point"] = data["warm_start_init_point"]
        opts["ipopt.warm_start_bound_push"] = data.get("warm_start_bound_push", 1e-6)
        opts["ipopt.warm_start_mult_bound_push"] = data.get(
            "warm_start_mult_bound_push",
            1e-6,
        )

    # Add user-supplied linear_solver_options dict verbatim (prefixed keys)
    ls_opts = data.get("linear_solver_options") or {}
    for k, v in ls_opts.items():
        opts[f"ipopt.{k}"] = v

    # Robust defaults (only if not provided explicitly)
    opts.setdefault("ipopt.max_iter", 4000)
    opts.setdefault("ipopt.acceptable_tol", 1e-6)
    opts.setdefault("ipopt.acceptable_iter", 15)
    opts.setdefault("ipopt.mu_strategy", "adaptive")
    opts.setdefault("ipopt.nlp_scaling_method", "gradient-based")
    opts.setdefault("ipopt.nlp_scaling_max_gradient", 100.0)
    opts.setdefault("ipopt.bound_relax_factor", 1e-8)
    opts.setdefault("ipopt.print_user_options", "yes")

    # Emit option file with *run-time* parameters (Ipopt expects bare keys)
    if emit_file:
        _write_option_file(opts)
        opts["ipopt.option_file_name"] = IPOPT_OPT_PATH

    return opts


def create_ipopt_options(ipopt_opts_dict: dict[str, Any], params: dict[str, Any]) -> IPOPTOptions:
    """
    Create IPOPTOptions from dictionary and params.

    This acts as a factory for IPOPTOptions, applying logic to:
    1. Start with robust defaults
    2. Apply user overrides
    3. Configure scaling, linear solver, and barrier strategy based on problem size
    4. Handle legacy compatibility options

    Args:
        ipopt_opts_dict: Dictionary of IPOPT options (often from config)
        params: Full problem parameters dictionary (for context like problem size)

    Returns:
        Configured IPOPTOptions object
    """
    # Start with robust options as baseline
    options = get_robust_ipopt_options()

    # Get problem parameters
    num = params.get("num", {})
    # Unused but kept for consistency if needed later
    # num_intervals = int(num.get("K", 10))
    # poly_degree = int(num.get("C", 3))

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
        # Reset max_iter if it was the default small value
        if options.max_iter == 500:
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

    available_solvers = get_available_hsl_solvers()
    available_display = ", ".join(sorted(available_solvers)) if available_solvers else "unknown"
    if options.linear_solver_options is None:
        options.linear_solver_options = {}

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
        configure_ma27_memory(options, n_vars, n_constraints)
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

    # Ensure print_frequency_iter is 1 to show every iteration
    # if options.print_frequency_iter != 1:
    # options.print_frequency_iter = 1
    # log.debug("Enforced print_frequency_iter=1 for detailed trace")

    return options


# -- Private helpers --------------------------------------------------------


def _write_option_file(opts: dict[str, Any]) -> None:
    """Write ``ipopt.opt`` file from *opts*.

    Only ``ipopt.*`` keys are written; the prefix is stripped.
    """

    path = Path(IPOPT_OPT_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for key, value in sorted(opts.items()):
                if not key.startswith("ipopt."):
                    continue
                bare_key = key.partition("ipopt.")[2]
                fh.write(f"{bare_key} {value}\n")
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Failed to write Ipopt option file %s: %s", path, exc)
        raise


def get_available_hsl_solvers() -> set[str]:
    """Return the set of available HSL solvers (best-effort detection)."""
    try:
        from campro.environment.hsl_detector import clear_cache, detect_available_solvers
    except Exception:
        return set()

    try:
        # Clear cache to ensure we get fresh detection results
        clear_cache()
        # Use runtime detection to verify symbols actually exist in the HSL library
        solvers = detect_available_solvers(test_runtime=True) or []
    except Exception as exc:
        log.debug("HSL solver detection failed: %s", exc)
        return set()

    return {solver.lower() for solver in solvers}


def configure_ma27_memory(
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
