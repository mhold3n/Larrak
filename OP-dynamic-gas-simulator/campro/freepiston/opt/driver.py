from __future__ import annotations

from typing import Any, Dict

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
from campro.logging import get_logger

log = get_logger(__name__)


def solve_cycle(P: Dict[str, Any]) -> Dict[str, Any]:
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

    # Minimal residual evaluation at a nominal state (placeholder)
    mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
    gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5}
    res = cv_residual(mech, gas, {"geom": P.get("geom", {}), "flows": {}})

    # Build NLP
    try:
        nlp, meta = build_collocation_nlp(P)

        # Get solver options
        solver_opts = P.get("solver", {}).get("ipopt", {})
        warm_start = P.get("warm_start", {})

        # Create IPOPT solver
        ipopt_options = _create_ipopt_options(solver_opts, P)
        solver = IPOPTSolver(ipopt_options)

        # Set up initial guess and bounds
        x0, lbx, ubx, lbg, ubg, p = _setup_optimization_bounds(nlp, P, warm_start)

        # Solve optimization problem
        log.info("Starting IPOPT optimization...")
        result = solver.solve(nlp, x0, lbx, ubx, lbg, ubg, p)

        if result.success:
            log.info(f"Optimization successful: {result.message}")
            log.info(
                f"Iterations: {result.iterations}, CPU time: {result.cpu_time:.2f}s",
            )
            log.info(f"Objective value: {result.f_opt:.6e}")
            log.info(f"KKT error: {result.kkt_error:.2e}")
            log.info(f"Feasibility error: {result.feasibility_error:.2e}")
        else:
            log.warning(f"Optimization failed: {result.message}")
            log.warning(f"Status: {result.status}, Iterations: {result.iterations}")

        # Store results
        optimization_result = {
            "success": result.success,
            "x_opt": result.x_opt,
            "f_opt": result.f_opt,
            "iterations": result.iterations,
            "cpu_time": result.cpu_time,
            "message": result.message,
            "status": result.status,
            "kkt_error": result.kkt_error,
            "feasibility_error": result.feasibility_error,
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
                log.warning(f"Checkpoint save failed: {exc}")

    except Exception as e:
        log.error(f"Failed to build or solve NLP: {e!s}")
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
        }

    return Solution(
        meta={"grid": grid, "meta": meta, "optimization": optimization_result},
        data={"residual_sample": res, "nlp": nlp},
    )


def _create_ipopt_options(
    solver_opts: Dict[str, Any], P: Dict[str, Any],
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
        if hasattr(options, key):
            setattr(options, key, value)
        else:
            log.warning(f"Unknown IPOPT option: {key}")

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
        options.linear_solver = "ma57"
        options.hessian_approximation = "limited-memory"
        options.max_iter = 10000
        log.info(
            f"Large problem detected ({n_vars} vars, {n_constraints} constraints), using robust settings",
        )

    return options


def _setup_optimization_bounds(
    nlp: Any,
    P: Dict[str, Any],
    warm_start: Dict[str, Any],
) -> tuple:
    """Set up optimization bounds and initial guess."""
    if nlp is None:
        return None, None, None, None, None, None

    try:
        # Get problem dimensions
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
                x0 = np.zeros(n_vars)
        else:
            x0 = np.zeros(n_vars)

        # Set up parameters
        p = np.array([])  # No parameters for now

        # Apply problem-specific bounds
        _apply_problem_bounds(lbx, ubx, lbg, ubg, P)

        return x0, lbx, ubx, lbg, ubg, p

    except Exception as e:
        log.error(f"Failed to set up optimization bounds: {e!s}")
        return None, None, None, None, None, None


def _apply_problem_bounds(
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray,
    ubg: np.ndarray,
    P: Dict[str, Any],
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


def solve_cycle_robust(P: Dict[str, Any]) -> Dict[str, Any]:
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
    P: Dict[str, Any],
    x0: np.ndarray,
) -> Dict[str, Any]:
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
    P: Dict[str, Any],
    refinement_strategy: str = "adaptive",
) -> Dict[str, Any]:
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


def _should_refine_error_based(result_0d: Dict[str, Any], P: Dict[str, Any]) -> bool:
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


def _should_refine_adaptive(result_0d: Dict[str, Any], P: Dict[str, Any]) -> bool:
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
    result_0d: Dict[str, Any],
    P_1d: Dict[str, Any],
) -> Dict[str, Any]:
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
    P: Dict[str, Any],
    max_refinements: int = 3,
) -> Dict[str, Any]:
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
