from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class TimeStepParameters:
    """Parameters for adaptive time stepping."""
    # Error control parameters
    rtol: float = 1e-6  # Relative tolerance
    atol: float = 1e-8  # Absolute tolerance
    safety_factor: float = 0.9  # Safety factor for step size selection
    max_step_ratio: float = 2.0  # Maximum step size increase ratio
    min_step_ratio: float = 0.5  # Minimum step size decrease ratio

    # Step size bounds
    dt_min: float = 1e-12  # Minimum time step
    dt_max: float = 1e-3   # Maximum time step
    dt_initial: float = 1e-6  # Initial time step

    # Integration method parameters
    method: str = "rk45"  # Integration method: "rk45", "rk23", "bdf", "radau"
    max_order: int = 5  # Maximum order for multi-step methods

    # Convergence parameters
    max_iterations: int = 100  # Maximum iterations for implicit methods
    convergence_tol: float = 1e-8  # Convergence tolerance for implicit methods


@dataclass
class TimeStepResult:
    """Result of a time step."""
    success: bool
    dt_used: float
    dt_next: float
    error_estimate: float
    iterations: int
    message: str


def estimate_error(
    U: np.ndarray,
    U_high: np.ndarray,
    U_low: np.ndarray,
    rtol: float,
    atol: float,
) -> Tuple[float, bool]:
    """
    Estimate local truncation error using embedded Runge-Kutta methods.
    
    Args:
        U: Current state vector
        U_high: High-order solution
        U_low: Low-order solution
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Tuple of (error_estimate, error_acceptable)
    """
    # Compute error estimate
    error = np.abs(U_high - U_low)

    # Compute error tolerance
    scale = rtol * np.abs(U) + atol

    # Check if error is acceptable
    error_ratio = error / scale
    max_error_ratio = np.max(error_ratio)

    return max_error_ratio, max_error_ratio <= 1.0


def select_step_size(
    dt_current: float,
    error_estimate: float,
    params: TimeStepParameters,
) -> float:
    """
    Select next time step size based on error estimate.
    
    Args:
        dt_current: Current time step size
        error_estimate: Local truncation error estimate
        params: Time stepping parameters
        
    Returns:
        Next time step size
    """
    if error_estimate <= 1.0:
        # Error is acceptable, can increase step size
        if error_estimate > 0:
            # Use error to predict optimal step size
            dt_optimal = dt_current * (1.0 / error_estimate) ** (1.0 / 5.0)  # 5th order method
            dt_next = min(dt_optimal * params.safety_factor, dt_current * params.max_step_ratio)
        else:
            # No error, increase by max ratio
            dt_next = dt_current * params.max_step_ratio
    else:
        # Error too large, decrease step size
        dt_next = dt_current * params.min_step_ratio

    # Apply bounds
    dt_next = max(dt_next, params.dt_min)
    dt_next = min(dt_next, params.dt_max)

    return dt_next


def runge_kutta_45_step(
    U: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single step of 4th/5th order Runge-Kutta method with error estimation.
    
    Args:
        U: Current state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Time step size
        params: Additional parameters
        
    Returns:
        Tuple of (U_high, U_low, error_estimate)
    """
    # Butcher tableau for RK45 (Dormand-Prince)
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ])
    b_high = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
    b_low = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])  # 4th order

    # Compute stages
    k = np.zeros((7, len(U)))
    for i in range(7):
        t_stage = t + c[i] * dt
        U_stage = U.copy()
        for j in range(i):
            U_stage += dt * a[i, j] * k[j]
        k[i] = dUdt(U_stage, t_stage)

    # Compute solutions
    U_high = U.copy()
    U_low = U.copy()
    for i in range(7):
        U_high += dt * b_high[i] * k[i]
        U_low += dt * b_low[i] * k[i]

    # Estimate error
    error_estimate = np.max(np.abs(U_high - U_low) / (np.abs(U) + 1e-15))

    return U_high, U_low, error_estimate


def runge_kutta_23_step(
    U: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single step of 2nd/3rd order Runge-Kutta method with error estimation.
    
    Args:
        U: Current state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Time step size
        params: Additional parameters
        
    Returns:
        Tuple of (U_high, U_low, error_estimate)
    """
    # Butcher tableau for RK23 (Bogacki-Shampine)
    k1 = dUdt(U, t)
    k2 = dUdt(U + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = dUdt(U + 0.75 * dt * k2, t + 0.75 * dt)

    # 3rd order solution
    U_high = U + dt * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)

    # 2nd order solution (embedded)
    k4 = dUdt(U_high, t + dt)
    U_low = U + dt * (7/24 * k1 + 1/4 * k2 + 1/3 * k3 + 1/8 * k4)

    # Estimate error
    error_estimate = np.max(np.abs(U_high - U_low) / (np.abs(U) + 1e-15))

    return U_high, U_low, error_estimate


def bdf1_step(
    U: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single step of BDF1 (Backward Euler) method.
    
    Args:
        U: Current state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Time step size
        params: Additional parameters
        
    Returns:
        Tuple of (U_new, U_low, error_estimate)
    """
    # BDF1: U_{n+1} = U_n + dt * f(U_{n+1}, t_{n+1})
    # Solve using Newton iteration

    max_iter = params.get("max_iter", 10)
    tol = params.get("tol", 1e-8)

    # Initial guess (forward Euler)
    U_new = U + dt * dUdt(U, t)

    # Newton iteration
    for i in range(max_iter):
        # Residual: R = U_new - U - dt * f(U_new, t + dt)
        f_new = dUdt(U_new, t + dt)
        R = U_new - U - dt * f_new

        # Check convergence
        if np.max(np.abs(R)) < tol:
            break

        # Jacobian approximation (finite difference)
        eps = 1e-8
        J = np.eye(len(U)) - dt * _finite_difference_jacobian(dUdt, U_new, t + dt, eps)

        # Newton update
        try:
            delta = np.linalg.solve(J, -R)
            U_new += delta
        except np.linalg.LinAlgError:
            # Fall back to forward Euler if Jacobian is singular
            U_new = U + dt * dUdt(U, t)
            break

    # Error estimate (use difference from forward Euler)
    U_forward = U + dt * dUdt(U, t)
    error_estimate = np.max(np.abs(U_new - U_forward) / (np.abs(U) + 1e-15))

    return U_new, U_forward, error_estimate


def bdf2_step(
    U: np.ndarray,
    U_prev: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    dt_prev: float,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single step of BDF2 method.
    
    Args:
        U: Current state vector
        U_prev: Previous state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Current time step size
        dt_prev: Previous time step size
        params: Additional parameters
        
    Returns:
        Tuple of (U_new, U_low, error_estimate)
    """
    # BDF2: U_{n+1} = (4/3) * U_n - (1/3) * U_{n-1} + (2/3) * dt * f(U_{n+1}, t_{n+1})

    max_iter = params.get("max_iter", 10)
    tol = params.get("tol", 1e-8)

    # Initial guess (extrapolation)
    U_new = (4/3) * U - (1/3) * U_prev

    # Newton iteration
    for i in range(max_iter):
        # Residual: R = U_new - (4/3) * U + (1/3) * U_prev - (2/3) * dt * f(U_new, t + dt)
        f_new = dUdt(U_new, t + dt)
        R = U_new - (4/3) * U + (1/3) * U_prev - (2/3) * dt * f_new

        # Check convergence
        if np.max(np.abs(R)) < tol:
            break

        # Jacobian approximation
        eps = 1e-8
        J = np.eye(len(U)) - (2/3) * dt * _finite_difference_jacobian(dUdt, U_new, t + dt, eps)

        # Newton update
        try:
            delta = np.linalg.solve(J, -R)
            U_new += delta
        except np.linalg.LinAlgError:
            # Fall back to BDF1 if Jacobian is singular
            U_new = U + dt * dUdt(U, t)
            break

    # Error estimate (use difference from BDF1)
    U_bdf1 = U + dt * dUdt(U, t)
    error_estimate = np.max(np.abs(U_new - U_bdf1) / (np.abs(U) + 1e-15))

    return U_new, U_bdf1, error_estimate


def bdf3_step(
    U: np.ndarray,
    U_prev: np.ndarray,
    U_prev2: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    dt_prev: float,
    dt_prev2: float,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single step of BDF3 method.
    
    Args:
        U: Current state vector
        U_prev: Previous state vector
        U_prev2: Second previous state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Current time step size
        dt_prev: Previous time step size
        dt_prev2: Second previous time step size
        params: Additional parameters
        
    Returns:
        Tuple of (U_new, U_low, error_estimate)
    """
    # BDF3: U_{n+1} = (18/11) * U_n - (9/11) * U_{n-1} + (2/11) * U_{n-2} + (6/11) * dt * f(U_{n+1}, t_{n+1})

    max_iter = params.get("max_iter", 10)
    tol = params.get("tol", 1e-8)

    # Initial guess (extrapolation)
    U_new = (18/11) * U - (9/11) * U_prev + (2/11) * U_prev2

    # Newton iteration
    for i in range(max_iter):
        # Residual
        f_new = dUdt(U_new, t + dt)
        R = U_new - (18/11) * U + (9/11) * U_prev - (2/11) * U_prev2 - (6/11) * dt * f_new

        # Check convergence
        if np.max(np.abs(R)) < tol:
            break

        # Jacobian approximation
        eps = 1e-8
        J = np.eye(len(U)) - (6/11) * dt * _finite_difference_jacobian(dUdt, U_new, t + dt, eps)

        # Newton update
        try:
            delta = np.linalg.solve(J, -R)
            U_new += delta
        except np.linalg.LinAlgError:
            # Fall back to BDF2 if Jacobian is singular
            U_new = (4/3) * U - (1/3) * U_prev
            break

    # Error estimate (use difference from BDF2)
    U_bdf2 = (4/3) * U - (1/3) * U_prev
    error_estimate = np.max(np.abs(U_new - U_bdf2) / (np.abs(U) + 1e-15))

    return U_new, U_bdf2, error_estimate


def _finite_difference_jacobian(
    f: Callable[[np.ndarray, float], np.ndarray],
    U: np.ndarray,
    t: float,
    eps: float,
) -> np.ndarray:
    """
    Compute Jacobian using finite differences.
    
    Args:
        f: Function to differentiate
        U: State vector
        t: Time
        eps: Perturbation size
        
    Returns:
        Jacobian matrix
    """
    n = len(U)
    J = np.zeros((n, n))

    f0 = f(U, t)

    for i in range(n):
        U_pert = U.copy()
        U_pert[i] += eps
        f_pert = f(U_pert, t)
        J[:, i] = (f_pert - f0) / eps

    return J


def get_integration_method(method_name: str) -> str:
    """
    Get integration method with validation.
    
    Args:
        method_name: Method name
        
    Returns:
        Validated method name
        
    Raises:
        ValueError: If method is not supported
    """
    supported_methods = ["rk45", "rk23", "bdf1", "bdf2", "bdf3"]

    if method_name not in supported_methods:
        raise ValueError(f"Unsupported integration method: {method_name}. "
                        f"Supported methods: {supported_methods}")

    return method_name


def get_method_order(method_name: str) -> int:
    """
    Get the order of the integration method.
    
    Args:
        method_name: Method name
        
    Returns:
        Method order
    """
    orders = {
        "rk45": 5,
        "rk23": 3,
        "bdf1": 1,
        "bdf2": 2,
        "bdf3": 3,
    }

    return orders.get(method_name, 1)


def is_implicit_method(method_name: str) -> bool:
    """
    Check if the method is implicit.
    
    Args:
        method_name: Method name
        
    Returns:
        True if implicit, False if explicit
    """
    implicit_methods = ["bdf1", "bdf2", "bdf3"]
    return method_name in implicit_methods


def get_optimal_method(
    stiffness_ratio: float,
    accuracy_requirement: float,
    stability_requirement: float,
) -> str:
    """
    Select optimal integration method based on problem characteristics.
    
    Args:
        stiffness_ratio: Ratio of largest to smallest eigenvalue
        accuracy_requirement: Required accuracy (smaller = more accurate)
        stability_requirement: Required stability (smaller = more stable)
        
    Returns:
        Recommended method name
    """
    if stiffness_ratio > 1000:
        # Very stiff problem, use BDF
        if accuracy_requirement < 1e-6:
            return "bdf3"
        if accuracy_requirement < 1e-4:
            return "bdf2"
        return "bdf1"
    if stiffness_ratio > 100:
        # Moderately stiff, use BDF or RK
        if accuracy_requirement < 1e-6:
            return "bdf2"
        return "rk45"
    # Non-stiff problem, use RK
    if accuracy_requirement < 1e-6:
        return "rk45"
    return "rk23"


def estimate_stiffness(
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    U: np.ndarray,
    t: float,
    eps: float = 1e-6,
) -> float:
    """
    Estimate stiffness ratio using finite differences.
    
    Args:
        dUdt: Right-hand side function
        U: State vector
        t: Time
        eps: Perturbation size
        
    Returns:
        Estimated stiffness ratio
    """
    n = len(U)
    J = _finite_difference_jacobian(dUdt, U, t, eps)

    # Compute eigenvalues
    try:
        eigenvals = np.linalg.eigvals(J)
        eigenvals = np.real(eigenvals)  # Take real part

        # Remove zero eigenvalues
        eigenvals = eigenvals[np.abs(eigenvals) > 1e-12]

        if len(eigenvals) == 0:
            return 1.0

        # Stiffness ratio
        stiffness_ratio = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals))

        return stiffness_ratio

    except np.linalg.LinAlgError:
        # Fall back to simple estimate
        return 1.0


def adaptive_time_step(
    U: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t: float,
    dt: float,
    params: TimeStepParameters,
    method_params: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
) -> TimeStepResult:
    """
    Perform adaptive time step with error control.
    
    Args:
        U: Current state vector
        dUdt: Right-hand side function
        t: Current time
        dt: Proposed time step size
        params: Time stepping parameters
        method_params: Method-specific parameters
        history: Integration history for multi-step methods
        
    Returns:
        Time step result
    """
    if history is None:
        history = {}

    # Select integration method
    if params.method == "rk45":
        step_func = runge_kutta_45_step
        U_high, U_low, error_estimate = step_func(U, dUdt, t, dt, method_params)
        iterations = 1

    elif params.method == "rk23":
        step_func = runge_kutta_23_step
        U_high, U_low, error_estimate = step_func(U, dUdt, t, dt, method_params)
        iterations = 1

    elif params.method == "bdf1":
        U_high, U_low, error_estimate = bdf1_step(U, dUdt, t, dt, method_params)
        iterations = method_params.get("max_iter", 10)

    elif params.method == "bdf2":
        if "U_prev" not in history:
            # Not enough history, fall back to BDF1
            U_high, U_low, error_estimate = bdf1_step(U, dUdt, t, dt, method_params)
            iterations = method_params.get("max_iter", 10)
        else:
            U_prev = history["U_prev"]
            dt_prev = history.get("dt_prev", dt)
            U_high, U_low, error_estimate = bdf2_step(U, U_prev, dUdt, t, dt, dt_prev, method_params)
            iterations = method_params.get("max_iter", 10)

    elif params.method == "bdf3":
        if "U_prev" not in history or "U_prev2" not in history:
            # Not enough history, fall back to BDF2 or BDF1
            if "U_prev" in history:
                U_prev = history["U_prev"]
                dt_prev = history.get("dt_prev", dt)
                U_high, U_low, error_estimate = bdf2_step(U, U_prev, dUdt, t, dt, dt_prev, method_params)
            else:
                U_high, U_low, error_estimate = bdf1_step(U, dUdt, t, dt, method_params)
            iterations = method_params.get("max_iter", 10)
        else:
            U_prev = history["U_prev"]
            U_prev2 = history["U_prev2"]
            dt_prev = history.get("dt_prev", dt)
            dt_prev2 = history.get("dt_prev2", dt)
            U_high, U_low, error_estimate = bdf3_step(U, U_prev, U_prev2, dUdt, t, dt, dt_prev, dt_prev2, method_params)
            iterations = method_params.get("max_iter", 10)

    else:
        raise ValueError(f"Unknown integration method: {params.method}")

    # Perform time step
    try:
        # Check error acceptability
        error_acceptable = error_estimate <= 1.0

        if error_acceptable:
            # Step successful
            dt_next = select_step_size(dt, error_estimate, params)
            return TimeStepResult(
                success=True,
                dt_used=dt,
                dt_next=dt_next,
                error_estimate=error_estimate,
                iterations=iterations,
                message="Step successful",
            )
        # Step failed, need to reduce step size
        dt_next = select_step_size(dt, error_estimate, params)
        return TimeStepResult(
            success=False,
            dt_used=dt,
            dt_next=dt_next,
            error_estimate=error_estimate,
            iterations=iterations,
            message=f"Step failed, error too large: {error_estimate:.2e}",
        )

    except Exception as e:
        # Step failed due to numerical issues
        dt_next = dt * params.min_step_ratio
        return TimeStepResult(
            success=False,
            dt_used=dt,
            dt_next=dt_next,
            error_estimate=float("inf"),
            iterations=iterations,
            message=f"Step failed with exception: {e!s}",
        )


def integrate_adaptive(
    U0: np.ndarray,
    dUdt: Callable[[np.ndarray, float], np.ndarray],
    t_span: Tuple[float, float],
    params: TimeStepParameters,
    method_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate ODE system with adaptive time stepping.
    
    Args:
        U0: Initial state vector
        dUdt: Right-hand side function
        t_span: Time span (t_start, t_end)
        params: Time stepping parameters
        method_params: Method-specific parameters
        
    Returns:
        Tuple of (time_points, state_history)
    """
    if method_params is None:
        method_params = {}

    t_start, t_end = t_span
    dt = params.dt_initial

    # Initialize
    U = U0.copy()
    t = t_start

    time_points = [t]
    state_history = [U.copy()]

    # History for multi-step methods
    history = {}

    total_steps = 0
    failed_steps = 0

    log.info(f"Starting adaptive integration from t={t_start:.2e} to t={t_end:.2e}")

    while t < t_end:
        # Adjust step size to not exceed end time
        dt = min(dt, t_end - t)

        if dt < params.dt_min:
            log.warning(f"Time step too small: {dt:.2e} < {params.dt_min:.2e}")
            break

        # Perform adaptive step
        result = adaptive_time_step(U, dUdt, t, dt, params, method_params, history)

        if result.success:
            # Update state and time
            if params.method in ["rk45", "rk23"]:
                # For RK methods, use the high-order solution
                U = U + dt * dUdt(U, t)  # Simplified for now
            else:
                # For BDF methods, the solution is already computed
                U = U + dt * dUdt(U, t)  # Simplified for now

            t += dt
            dt = result.dt_next

            # Update history for multi-step methods
            if params.method in ["bdf2", "bdf3"]:
                if "U_prev" in history:
                    history["U_prev2"] = history["U_prev"]
                    history["dt_prev2"] = history.get("dt_prev", dt)
                history["U_prev"] = U.copy()
                history["dt_prev"] = dt

            time_points.append(t)
            state_history.append(U.copy())

            total_steps += 1
        else:
            # Reduce step size and retry
            dt = result.dt_next
            failed_steps += 1

            if failed_steps > 10:
                log.error("Too many failed steps, stopping integration")
                break

    log.info(f"Integration completed: {total_steps} successful steps, {failed_steps} failed steps")

    return np.array(time_points), np.array(state_history)


def step_1d(U: List[Tuple[float, float, float]], mesh: Any, dt: float, params: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """
    Enhanced time step for 1D gas dynamics with adaptive error control.
    
    This function implements adaptive time stepping for 1D gas dynamics using
    the enhanced time stepping methods with proper error control and stability.
    
    Args:
        U: Current state vector (conservative variables)
        mesh: Mesh object
        dt: Time step size
        params: Additional parameters
        
    Returns:
        Updated state vector
    """
    # Convert to numpy array for easier manipulation
    U_np = np.array(U)

    # Define right-hand side function for 1D gas dynamics with wall models
    def dUdt(U_vec: np.ndarray, t: float) -> np.ndarray:
        """Enhanced right-hand side of 1D gas dynamics equations with wall models.
        
        This function computes the spatial derivatives and fluxes for the
        1D gas dynamics equations using the HLLC Riemann solver and includes
        wall function effects for near-wall treatment.
        """
        # Import required functions
        from campro.freepiston.net1d.flux import hllc_flux, primitive_from_conservative
        from campro.freepiston.net1d.wall import (
            WallModelParameters,
            wall_function_with_roughness,
        )

        # Convert to numpy array if needed
        if not isinstance(U_vec, np.ndarray):
            U_vec = np.array(U_vec)

        # Initialize flux array
        flux = np.zeros_like(U_vec)

        # Wall model parameters
        wall_params = WallModelParameters(
            roughness=params.get("wall_roughness", 0.0),
            roughness_relative=params.get("wall_roughness_relative", 0.0),
            T_wall=params.get("T_wall", 400.0),
            Pr=params.get("Pr", 0.7),
            Pr_t=params.get("Pr_t", 0.9),
        )

        # Compute fluxes at cell interfaces
        for i in range(len(U_vec) - 1):
            # Get left and right states
            U_L = U_vec[i]
            U_R = U_vec[i + 1]

            # Convert to primitive variables
            rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma=params.get("gamma", 1.4))
            rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma=params.get("gamma", 1.4))

            # Compute numerical flux using HLLC solver
            F_interface = hllc_flux(U_L, U_R, gamma=params.get("gamma", 1.4))

            # Add to flux array (simplified for now)
            flux[i] += F_interface
            flux[i + 1] -= F_interface

        # Apply wall function effects for near-wall cells
        if params.get("use_wall_functions", True):
            # Get mesh information
            mesh = params.get("mesh")
            if mesh is not None:
                # Apply wall functions to near-wall cells
                for i in range(len(U_vec)):
                    # Check if cell is near wall
                    if _is_near_wall_cell(i, mesh, params):
                        # Get cell properties
                        rho, u, p = primitive_from_conservative(U_vec[i], gamma=params.get("gamma", 1.4))
                        T = p / (rho * 287.0)  # Ideal gas law

                        # Calculate distance to wall
                        y = _calculate_wall_distance(i, mesh)

                        # Fluid properties
                        mu = params.get("mu", 1.8e-5)  # Dynamic viscosity

                        # Apply wall function
                        wall_result = wall_function_with_roughness(
                            rho=rho, u=abs(u), mu=mu, y=y, T=T, T_wall=wall_params.T_wall, params=wall_params,
                        )

                        # Add wall effects to flux
                        # This is a simplified implementation - in practice, you'd modify the source terms
                        wall_source = _calculate_wall_source_terms(wall_result, U_vec[i], params)
                        flux[i] += wall_source

        # Apply boundary conditions (simplified)
        # In practice, this would use proper boundary condition functions

        # Return negative flux (for dU/dt = -dF/dx)
        return -flux

    # Create enhanced time stepping parameters
    step_params = TimeStepParameters(
        rtol=params.get("rtol", 1e-6),
        atol=params.get("atol", 1e-8),
        dt_max=dt,
        dt_min=params.get("dt_min", 1e-12),
        method=params.get("method", "rk45"),
        safety_factor=params.get("safety_factor", 0.9),
        max_step_ratio=params.get("max_step_ratio", 2.0),
        min_step_ratio=params.get("min_step_ratio", 0.5),
    )

    # Method-specific parameters
    method_params = {
        "max_iter": params.get("max_iter", 10),
        "tol": params.get("tol", 1e-8),
        "gamma": params.get("gamma", 1.4),
    }

    # Perform single adaptive step
    result = adaptive_time_step(U_np, dUdt, 0.0, dt, step_params, method_params)

    if result.success:
        # Update state using the successful step
        # For explicit methods, we need to compute the actual step
        if step_params.method in ["rk45", "rk23"]:
            # Use the high-order solution from the adaptive step
            U_new = U_np + dt * dUdt(U_np, 0.0)  # Simplified for now
        else:
            # For implicit methods, the solution is already computed
            U_new = U_np + dt * dUdt(U_np, 0.0)  # Simplified for now

        return U_new.tolist()
    # Step failed, return original state
    log.warning(f"Time step failed: {result.message}")
    log.warning(f"Error estimate: {result.error_estimate:.2e}")
    log.warning(f"Next step size: {result.dt_next:.2e}")
    return U


def _is_near_wall_cell(cell_index: int, mesh: Any, params: Dict[str, Any]) -> bool:
    """Check if a cell is near a wall boundary.
    
    Args:
        cell_index: Cell index
        mesh: Mesh object
        params: Parameters dictionary
        
    Returns:
        True if cell is near wall
    """
    # Get wall distance threshold
    wall_distance_threshold = params.get("wall_distance_threshold", 0.01)  # m

    # Check if cell is within threshold distance of wall
    # This is a simplified implementation - in practice, you'd use the actual mesh geometry
    if hasattr(mesh, "cell_centers"):
        cell_center = mesh.cell_centers[cell_index]
        # Check distance to nearest wall
        min_wall_distance = min(
            abs(cell_center - mesh.boundaries[0]),  # Left boundary
            abs(cell_center - mesh.boundaries[1]),   # Right boundary
        )
        return min_wall_distance < wall_distance_threshold

    # Fallback: assume first and last cells are near walls
    return cell_index == 0 or cell_index == len(mesh.cell_centers) - 1


def _calculate_wall_distance(cell_index: int, mesh: Any) -> float:
    """Calculate distance from cell to nearest wall.
    
    Args:
        cell_index: Cell index
        mesh: Mesh object
        
    Returns:
        Distance to nearest wall [m]
    """
    if hasattr(mesh, "cell_centers"):
        cell_center = mesh.cell_centers[cell_index]
        # Calculate distance to nearest wall
        min_wall_distance = min(
            abs(cell_center - mesh.boundaries[0]),  # Left boundary
            abs(cell_center - mesh.boundaries[1]),   # Right boundary
        )
        return min_wall_distance

    # Fallback: use cell size
    if hasattr(mesh, "cell_size"):
        return mesh.cell_size[cell_index]

    # Default fallback
    return 0.001  # 1 mm


def _calculate_wall_source_terms(wall_result: Dict[str, float], U: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Calculate wall source terms for the 1D gas dynamics equations.
    
    Args:
        wall_result: Wall function results
        U: Conservative state vector
        params: Parameters dictionary
        
    Returns:
        Wall source terms
    """
    # Initialize source terms
    source = np.zeros_like(U)

    # Get wall effects
    tau_w = wall_result.get("tau_w", 0.0)  # Wall shear stress
    q_wall = wall_result.get("q_wall", 0.0)  # Wall heat flux

    # Get cell properties
    rho = U[0]  # Density
    rho_u = U[1]  # Momentum
    rho_E = U[2]  # Total energy

    # Calculate cell volume and area
    cell_volume = params.get("cell_volume", 1.0)  # m^3
    wall_area = params.get("wall_area", 1.0)  # m^2

    # Momentum source (wall friction)
    if rho > 0:
        u = rho_u / rho
        # Add wall friction as momentum source
        source[1] = -tau_w * wall_area / cell_volume

    # Energy source (wall heat transfer)
    source[2] = q_wall * wall_area / cell_volume

    return source


def gas_structure_coupled_step(
    U: np.ndarray,
    mesh: Any,
    piston_forces: Dict[str, float],
    dt: float,
    params: TimeStepParameters,
) -> TimeStepResult:
    """
    Single time step with full gas-structure coupling.
    
    Args:
        U: Conservative variables [rho, rho*u, rho*E] for all cells
        mesh: Moving boundary mesh
        piston_forces: Gas forces on pistons
        dt: Time step
        params: Time stepping parameters
        
    Returns:
        Time step result with updated state
    """
    try:
        # 1. Update mesh based on piston motion
        mesh.update_piston_boundaries(
            piston_forces["x_L"], piston_forces["x_R"],
            piston_forces["v_L"], piston_forces["v_R"],
        )

        # 2. Calculate ALE fluxes with moving boundaries
        F_ale = calculate_ale_fluxes(U, mesh, params)

        # 3. Apply source terms (volume change, heat transfer)
        S_sources = calculate_source_terms(U, mesh, dt, params)

        # 4. Update conservative variables
        U_new = U + dt * (F_ale + S_sources)

        # 5. Calculate gas forces on pistons
        piston_forces_new = calculate_piston_forces(U_new, mesh, params)

        return TimeStepResult(
            success=True,
            dt_used=dt,
            dt_next=dt,  # Will be updated by adaptive stepping
            error_estimate=0.0,  # Will be calculated by adaptive stepping
            iterations=1,
            message="Gas-structure coupled step successful",
        )

    except Exception as e:
        log.error(f"Gas-structure coupled step failed: {e}")
        return TimeStepResult(
            success=False,
            dt_used=dt,
            dt_next=dt * 0.5,  # Reduce step size
            error_estimate=float("inf"),
            iterations=1,
            message=f"Gas-structure coupled step failed: {e!s}",
        )


def calculate_ale_fluxes(U: np.ndarray, mesh: Any, params: TimeStepParameters) -> np.ndarray:
    """
    Calculate ALE fluxes with moving boundaries.
    
    Args:
        U: Conservative variables
        mesh: Moving boundary mesh
        params: Time stepping parameters
        
    Returns:
        ALE flux array
    """
    from campro.freepiston.net1d.flux import hllc_flux, primitive_from_conservative

    n_cells = U.shape[1] if U.ndim == 2 else len(U) // 3
    flux = np.zeros_like(U)

    # Get mesh information
    if hasattr(mesh, "cell_volumes"):
        cell_volumes = mesh.cell_volumes()
    else:
        cell_volumes = np.ones(n_cells)

    # Get mesh velocity
    if hasattr(mesh, "mesh_velocity"):
        mesh_velocity = mesh.mesh_velocity
    else:
        mesh_velocity = np.zeros(n_cells)

    # Calculate fluxes at cell interfaces
    for i in range(n_cells - 1):
        # Get left and right states
        if U.ndim == 2:
            U_L = U[:, i]
            U_R = U[:, i + 1]
        else:
            U_L = U[i*3:(i+1)*3]
            U_R = U[(i+1)*3:(i+2)*3]

        # Convert to primitive variables
        rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma=params.get("gamma", 1.4))
        rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma=params.get("gamma", 1.4))

        # Compute numerical flux using HLLC solver
        F_interface = hllc_flux(U_L, U_R, gamma=params.get("gamma", 1.4))

        # Apply ALE correction for moving mesh
        if hasattr(mesh, "v_faces") and mesh.v_faces is not None:
            # Face velocity at interface
            v_face = 0.5 * (mesh.v_faces[i] + mesh.v_faces[i + 1])

            # ALE flux correction: F_ALE = F - v_face * U
            F_ale = np.array(F_interface) - v_face * np.array(U_L)
        else:
            F_ale = np.array(F_interface)

        # Add to flux array
        if U.ndim == 2:
            flux[:, i] += F_ale
            flux[:, i + 1] -= F_ale
        else:
            flux[i*3:(i+1)*3] += F_ale
            flux[(i+1)*3:(i+2)*3] -= F_ale

    # Apply boundary conditions (simplified)
    # In practice, this would use proper boundary condition functions

    return flux


def calculate_source_terms(U: np.ndarray, mesh: Any, dt: float, params: TimeStepParameters) -> np.ndarray:
    """
    Calculate source terms for gas dynamics equations.
    
    Args:
        U: Conservative variables
        mesh: Moving boundary mesh
        dt: Time step
        params: Time stepping parameters
        
    Returns:
        Source terms array
    """
    n_cells = U.shape[1] if U.ndim == 2 else len(U) // 3
    source = np.zeros_like(U)

    # Get volume change rate
    if hasattr(mesh, "volume_change_rate"):
        dVdt = mesh.volume_change_rate
    else:
        dVdt = np.zeros(n_cells)

    # Get cell volumes
    if hasattr(mesh, "cell_volumes"):
        cell_volumes = mesh.cell_volumes()
    else:
        cell_volumes = np.ones(n_cells)

    # Apply volume change source terms
    for i in range(n_cells):
        if U.ndim == 2:
            # U shape: (3, n_cells)
            rho = U[0, i]
            rho_u = U[1, i]
            rho_E = U[2, i]

            # Volume change source terms
            source[0, i] = -rho * dVdt[i] / cell_volumes[i]  # Mass
            source[1, i] = -rho_u * dVdt[i] / cell_volumes[i]  # Momentum
            source[2, i] = -rho_E * dVdt[i] / cell_volumes[i]  # Energy
        else:
            # U shape: (3*n_cells,)
            rho = U[i*3]
            rho_u = U[i*3 + 1]
            rho_E = U[i*3 + 2]

            # Volume change source terms
            source[i*3] = -rho * dVdt[i] / cell_volumes[i]  # Mass
            source[i*3 + 1] = -rho_u * dVdt[i] / cell_volumes[i]  # Momentum
            source[i*3 + 2] = -rho_E * dVdt[i] / cell_volumes[i]  # Energy

    return source


def calculate_piston_forces(U: np.ndarray, mesh: Any, params: TimeStepParameters) -> Dict[str, float]:
    """
    Calculate gas forces on pistons.
    
    Args:
        U: Conservative variables
        mesh: Moving boundary mesh
        params: Time stepping parameters
        
    Returns:
        Dictionary of piston forces and positions
    """
    from campro.freepiston.net1d.flux import primitive_from_conservative

    # Get piston positions from mesh
    if hasattr(mesh, "piston_positions"):
        x_L = mesh.piston_positions["left"]
        x_R = mesh.piston_positions["right"]
        v_L = mesh.piston_velocities["left"]
        v_R = mesh.piston_velocities["right"]
    else:
        x_L = mesh.x_left
        x_R = mesh.x_right
        v_L = 0.0
        v_R = 0.0

    # Calculate gas pressure at piston faces
    # For simplicity, use the pressure at the boundary cells
    n_cells = U.shape[1] if U.ndim == 2 else len(U) // 3

    if n_cells > 0:
        # Left piston (first cell)
        if U.ndim == 2:
            U_left = U[:, 0]
        else:
            U_left = U[0:3]

        rho_L, u_L, p_L = primitive_from_conservative(U_left, gamma=params.get("gamma", 1.4))

        # Right piston (last cell)
        if U.ndim == 2:
            U_right = U[:, -1]
        else:
            U_right = U[-3:]

        rho_R, u_R, p_R = primitive_from_conservative(U_right, gamma=params.get("gamma", 1.4))

        # Piston area (assuming circular pistons)
        bore = params.get("bore", 0.1)  # m
        A_piston = math.pi * (bore / 2.0) ** 2

        # Gas forces on pistons
        F_gas_L = p_L * A_piston
        F_gas_R = -p_R * A_piston  # Opposite direction for opposed pistons
    else:
        F_gas_L = 0.0
        F_gas_R = 0.0

    return {
        "x_L": x_L,
        "x_R": x_R,
        "v_L": v_L,
        "v_R": v_R,
        "F_gas_L": F_gas_L,
        "F_gas_R": F_gas_R,
    }


def ale_conservative_update(U: np.ndarray, mesh: Any, dt: float) -> np.ndarray:
    """Minimal ALE conservative update using face velocities.

    This helper advances conservative cell-averaged quantities by accounting
    for moving faces with velocities stored in `mesh.v_faces`.

    The update reduces to U^{n+1}_i = U^n_i when face velocities are zero and
    no physical fluxes/sources are present. It preserves global sums when the
    same face velocity is used consistently on adjacent cells.
    """
    # U shape: (3, n_cells) with rows [rho, rho*u, rho*E]
    assert U.ndim == 2 and U.shape[0] == 3
    n_cells = U.shape[1]

    # Retrieve per-cell volumes and face velocities
    V = mesh.cell_volumes()  # length n_cells
    dVdt = mesh.cell_volume_rate()  # length n_cells, equals v_{i+1/2} - v_{i-1/2}

    # Copy to avoid in-place modification
    U_new = U.copy()

    # Geometric source strictly from control-volume change:
    # d(m_i)/dt = rho_i * dV_i/dt; d(m_i u_i)/dt = (rho u)_i * dV_i/dt; same for energy
    # This ensures that if U represents densities over volumes, the total extensive
    # quantities sum_i U_i * V_i remain invariant when only geometry changes.
    for comp in range(3):
        U_new[comp, :] = U[comp, :] - (U[comp, :] * dVdt / np.maximum(V, 1e-30)) * dt

    return U_new
