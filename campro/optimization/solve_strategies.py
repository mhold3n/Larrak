"""
Solve cycle strategy variants for OP engine optimization.

This module provides alternative solve strategies for different use cases:
- Robust solving with conservative settings
- Warm start from previous solution
- Fuel-based continuation (homotopy)
- 0D→1D refinement
- Adaptive refinement

All strategies delegate to the core `solve_cycle` function in driver.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from campro.logging import get_logger
from campro.optimization.core.solution import Solution
from campro.optimization.driver import solve_cycle

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


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
    tol = base_tol  # Initialize for scope

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
    if not result_0d["success"] or result_0d.get("x_opt") is None:
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
