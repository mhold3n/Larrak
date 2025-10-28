from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from campro.freepiston.core.states import MechState
from campro.freepiston.zerod.cv import cv_residual, volume_from_pistons
from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class ValidationParameters:
    """Parameters for cross-fidelity validation."""

    # Simulation parameters
    t_end: float = 1.0  # End time for simulation
    dt_0d: float = 1e-4  # Time step for 0D model
    dt_1d: float = 1e-5  # Time step for 1D model

    # Tolerance parameters
    relative_tolerance: float = 1e-3  # Relative tolerance for comparison
    absolute_tolerance: float = 1e-6  # Absolute tolerance for comparison

    # Validation metrics
    validate_mass_conservation: bool = True
    validate_energy_conservation: bool = True
    validate_pressure: bool = True
    validate_temperature: bool = True
    validate_piston_dynamics: bool = True

    # Output parameters
    output_frequency: int = 100  # Output every N time steps
    save_solution_history: bool = True


@dataclass
class ValidationResult:
    """Result of cross-fidelity validation."""

    success: bool
    max_relative_error: float
    max_absolute_error: float
    error_metrics: dict[str, float]
    solution_comparison: dict[str, np.ndarray]
    validation_message: str

    # Timing information
    cpu_time_0d: float
    cpu_time_1d: float

    # Solution quality
    mass_conservation_error: float
    energy_conservation_error: float


@dataclass
class ModelComparison:
    """Comparison between 0D and 1D model results."""

    time_points: np.ndarray
    pressure_0d: np.ndarray
    pressure_1d: np.ndarray
    temperature_0d: np.ndarray
    temperature_1d: np.ndarray
    density_0d: np.ndarray
    density_1d: np.ndarray
    piston_position_0d: np.ndarray
    piston_position_1d: np.ndarray
    piston_velocity_0d: np.ndarray
    piston_velocity_1d: np.ndarray


def cross_fidelity_validation(
    problem_params: dict[str, Any],
    validation_params: ValidationParameters | None = None,
) -> ValidationResult:
    """
    Perform cross-fidelity validation between 0D and 1D models.

    This function runs both 0D and 1D simulations for the same problem
    and compares the results to validate the models against each other.

    Args:
        problem_params: Problem parameters for both models
        validation_params: Validation parameters

    Returns:
        Validation result with comparison metrics
    """
    if validation_params is None:
        validation_params = ValidationParameters()

    log.info("Starting cross-fidelity validation between 0D and 1D models")

    # Run 0D simulation
    log.info("Running 0D simulation...")
    start_time = time.time()
    result_0d = _run_0d_simulation(problem_params, validation_params)
    cpu_time_0d = time.time() - start_time

    # Run 1D simulation
    log.info("Running 1D simulation...")
    start_time = time.time()
    result_1d = _run_1d_simulation(problem_params, validation_params)
    cpu_time_1d = time.time() - start_time

    # Compare results
    log.info("Comparing results...")
    comparison = _compare_results(result_0d, result_1d, validation_params)

    # Compute validation metrics
    error_metrics = _compute_error_metrics(comparison, validation_params)

    # Check validation success
    success = _check_validation_success(error_metrics, validation_params)

    # Create validation result
    validation_result = ValidationResult(
        success=success,
        max_relative_error=max(error_metrics.values()) if error_metrics else 0.0,
        max_absolute_error=max(abs(v) for v in error_metrics.values())
        if error_metrics
        else 0.0,
        error_metrics=error_metrics,
        solution_comparison=comparison.__dict__,
        validation_message=_create_validation_message(success, error_metrics),
        cpu_time_0d=cpu_time_0d,
        cpu_time_1d=cpu_time_1d,
        mass_conservation_error=error_metrics.get("mass_conservation", 0.0),
        energy_conservation_error=error_metrics.get("energy_conservation", 0.0),
    )

    log.info(
        f"Cross-fidelity validation completed: {validation_result.validation_message}",
    )

    return validation_result


def _run_0d_simulation(
    problem_params: dict[str, Any],
    validation_params: ValidationParameters,
) -> dict[str, np.ndarray]:
    """Run 0D simulation."""
    # Extract problem parameters
    geom = problem_params.get("geom", {})
    initial_conditions = problem_params.get("initial_conditions", {})

    # Set up initial conditions
    x_L_0 = initial_conditions.get("x_L", 0.05)
    v_L_0 = initial_conditions.get("v_L", 0.0)
    x_R_0 = initial_conditions.get("x_R", 0.15)
    v_R_0 = initial_conditions.get("v_R", 0.0)
    rho_0 = initial_conditions.get("rho", 1.2)
    T_0 = initial_conditions.get("T", 1000.0)
    p_0 = initial_conditions.get("p", 1.0e5)

    # Create initial state
    mech_0 = MechState(x_L=x_L_0, v_L=v_L_0, x_R=x_R_0, v_R=v_R_0)
    gas_0 = {"rho": rho_0, "T": T_0, "p": p_0, "E": p_0 / ((1.4 - 1) * rho_0)}

    # Set up time integration
    t_span = (0.0, validation_params.t_end)
    dt = validation_params.dt_0d
    n_steps = int((t_span[1] - t_span[0]) / dt)

    # Initialize solution arrays
    time_points = np.linspace(t_span[0], t_span[1], n_steps + 1)
    pressure = np.zeros(n_steps + 1)
    temperature = np.zeros(n_steps + 1)
    density = np.zeros(n_steps + 1)
    piston_position = np.zeros(n_steps + 1)
    piston_velocity = np.zeros(n_steps + 1)

    # Store initial conditions
    pressure[0] = p_0
    temperature[0] = T_0
    density[0] = rho_0
    piston_position[0] = x_R_0 - x_L_0
    piston_velocity[0] = v_R_0 - v_L_0

    # Time integration
    mech = mech_0
    gas = gas_0.copy()

    for i in range(1, n_steps + 1):
        # Compute residuals
        residuals = cv_residual(mech, gas, problem_params)

        # Update state (simple forward Euler)
        # Mass balance: dm/dt = mdot_in - mdot_ex
        dm_dt = residuals["dm_dt"]
        dU_dt = residuals["dU_dt"]

        # Update gas state
        V = volume_from_pistons(
            B=geom.get("B", 0.1),
            Vc=geom.get("Vc", 1e-5),
            x_L=mech.x_L,
            x_R=mech.x_R,
        )

        m = gas["rho"] * V
        U = m * gas["E"]

        # Update mass and energy
        m_new = m + dt * dm_dt
        U_new = U + dt * dU_dt

        # Update gas properties
        if m_new > 0:
            gas["rho"] = m_new / V
            gas["E"] = U_new / m_new
            gas["p"] = (1.4 - 1) * gas["rho"] * gas["E"]
            gas["T"] = gas["p"] / (gas["rho"] * 287.0)  # Ideal gas law

        # Update piston positions (simple dynamics)
        # This is a placeholder - in practice, you'd solve the full piston dynamics
        mech.x_L += dt * mech.v_L
        mech.x_R += dt * mech.v_R

        # Store solution
        pressure[i] = gas["p"]
        temperature[i] = gas["T"]
        density[i] = gas["rho"]
        piston_position[i] = mech.x_R - mech.x_L
        piston_velocity[i] = mech.v_R - mech.v_L

    return {
        "time_points": time_points,
        "pressure": pressure,
        "temperature": temperature,
        "density": density,
        "piston_position": piston_position,
        "piston_velocity": piston_velocity,
    }


def _run_1d_simulation(
    problem_params: dict[str, Any],
    validation_params: ValidationParameters,
) -> dict[str, np.ndarray]:
    """Run 1D simulation."""
    # Extract problem parameters
    geom = problem_params.get("geom", {})
    initial_conditions = problem_params.get("initial_conditions", {})

    # Set up initial conditions
    x_L_0 = initial_conditions.get("x_L", 0.05)
    v_L_0 = initial_conditions.get("v_L", 0.0)
    x_R_0 = initial_conditions.get("x_R", 0.15)
    v_R_0 = initial_conditions.get("v_R", 0.0)
    rho_0 = initial_conditions.get("rho", 1.2)
    T_0 = initial_conditions.get("T", 1000.0)
    p_0 = initial_conditions.get("p", 1.0e5)

    # Set up 1D mesh
    n_cells = problem_params.get("n_cells", 50)
    if not isinstance(n_cells, int) or n_cells <= 0:
        n_cells = 10
    length = x_R_0 - x_L_0
    if length <= 0:
        length = abs(length) if length != 0 else 1e-3
    dx = length / n_cells

    # Initialize 1D state
    U = np.zeros((n_cells, 3))  # [rho, rho*u, rho*E]
    rho_0 = max(rho_0, 1e-9)
    p_0 = max(p_0, 1e-9)
    for i in range(n_cells):
        U[i, 0] = rho_0  # density
        U[i, 1] = 0.0  # momentum (zero velocity)
        E_spec = p_0 / max((1.4 - 1.0) * rho_0, 1e-12)
        U[i, 2] = rho_0 * E_spec

    # Set up time integration
    t_span = (0.0, validation_params.t_end)
    dt = validation_params.dt_1d
    n_steps = int((t_span[1] - t_span[0]) / dt)

    # Initialize solution arrays
    time_points = np.linspace(t_span[0], t_span[1], n_steps + 1)
    pressure = np.zeros(n_steps + 1)
    temperature = np.zeros(n_steps + 1)
    density = np.zeros(n_steps + 1)
    piston_position = np.zeros(n_steps + 1)
    piston_velocity = np.zeros(n_steps + 1)

    # Store initial conditions
    pressure[0] = p_0
    temperature[0] = T_0
    density[0] = rho_0
    piston_position[0] = x_R_0 - x_L_0
    piston_velocity[0] = v_R_0 - v_L_0

    # Time integration
    for i in range(1, n_steps + 1):
        # Simple 1D time step (placeholder)
        # In practice, this would use the full 1D gas dynamics solver
        U = _simple_1d_time_step(U, dt, dx, problem_params)

        # Compute average quantities
        rho_avg = np.mean(U[:, 0])
        u_avg = np.mean(U[:, 1] / U[:, 0])
        E_avg = np.mean(U[:, 2] / U[:, 0])

        # Compute pressure and temperature
        p_avg = (1.4 - 1) * rho_avg * E_avg
        T_avg = p_avg / (rho_avg * 287.0)

        # Update piston positions (simple dynamics)
        x_L = x_L_0 + i * dt * v_L_0
        x_R = x_R_0 + i * dt * v_R_0

        # Store solution
        pressure[i] = p_avg
        temperature[i] = T_avg
        density[i] = rho_avg
        piston_position[i] = x_R - x_L
        piston_velocity[i] = v_R_0 - v_L_0

    return {
        "time_points": time_points,
        "pressure": pressure,
        "temperature": temperature,
        "density": density,
        "piston_position": piston_position,
        "piston_velocity": piston_velocity,
    }


def _simple_1d_time_step(
    U: np.ndarray,
    dt: float,
    dx: float,
    problem_params: dict[str, Any],
) -> np.ndarray:
    """Simple 1D time step (placeholder)."""
    # This is a placeholder implementation
    # In practice, this would use the full 1D gas dynamics solver with HLLC flux

    n_cells = U.shape[0]
    U_new = U.copy()

    # Simple diffusion to prevent numerical instability
    for i in range(1, n_cells - 1):
        for j in range(3):
            U_new[i, j] = U[i, j] + 0.1 * dt / dx**2 * (
                U[i + 1, j] - 2 * U[i, j] + U[i - 1, j]
            )

    return U_new


def _compare_results(
    result_0d: dict[str, np.ndarray],
    result_1d: dict[str, np.ndarray],
    validation_params: ValidationParameters,
) -> ModelComparison:
    """Compare 0D and 1D results."""
    # Interpolate to common time grid
    time_0d = result_0d["time_points"]
    time_1d = result_1d["time_points"]

    # Use finer time grid for comparison
    time_common = np.linspace(0, validation_params.t_end, 1000)

    # Interpolate solutions to common time grid
    pressure_0d = np.interp(time_common, time_0d, result_0d["pressure"])
    pressure_1d = np.interp(time_common, time_1d, result_1d["pressure"])
    temperature_0d = np.interp(time_common, time_0d, result_0d["temperature"])
    temperature_1d = np.interp(time_common, time_1d, result_1d["temperature"])
    density_0d = np.interp(time_common, time_0d, result_0d["density"])
    density_1d = np.interp(time_common, time_1d, result_1d["density"])
    piston_position_0d = np.interp(time_common, time_0d, result_0d["piston_position"])
    piston_position_1d = np.interp(time_common, time_1d, result_1d["piston_position"])
    piston_velocity_0d = np.interp(time_common, time_0d, result_0d["piston_velocity"])
    piston_velocity_1d = np.interp(time_common, time_1d, result_1d["piston_velocity"])

    return ModelComparison(
        time_points=time_common,
        pressure_0d=pressure_0d,
        pressure_1d=pressure_1d,
        temperature_0d=temperature_0d,
        temperature_1d=temperature_1d,
        density_0d=density_0d,
        density_1d=density_1d,
        piston_position_0d=piston_position_0d,
        piston_position_1d=piston_position_1d,
        piston_velocity_0d=piston_velocity_0d,
        piston_velocity_1d=piston_velocity_1d,
    )


def _compute_error_metrics(
    comparison: ModelComparison,
    validation_params: ValidationParameters,
) -> dict[str, float]:
    """Compute error metrics between 0D and 1D results."""
    error_metrics = {}

    # Pressure error
    if validation_params.validate_pressure:
        pressure_error = np.abs(comparison.pressure_1d - comparison.pressure_0d)
        pressure_relative_error = pressure_error / (
            np.abs(comparison.pressure_0d) + 1e-12
        )
        error_metrics["pressure_relative"] = np.max(pressure_relative_error)
        error_metrics["pressure_absolute"] = np.max(pressure_error)

    # Temperature error
    if validation_params.validate_temperature:
        temperature_error = np.abs(
            comparison.temperature_1d - comparison.temperature_0d,
        )
        temperature_relative_error = temperature_error / (
            np.abs(comparison.temperature_0d) + 1e-12
        )
        error_metrics["temperature_relative"] = np.max(temperature_relative_error)
        error_metrics["temperature_absolute"] = np.max(temperature_error)

    # Density error
    density_error = np.abs(comparison.density_1d - comparison.density_0d)
    density_relative_error = density_error / (np.abs(comparison.density_0d) + 1e-12)
    error_metrics["density_relative"] = np.max(density_relative_error)
    error_metrics["density_absolute"] = np.max(density_error)

    # Piston dynamics error
    if validation_params.validate_piston_dynamics:
        position_error = np.abs(
            comparison.piston_position_1d - comparison.piston_position_0d,
        )
        position_relative_error = position_error / (
            np.abs(comparison.piston_position_0d) + 1e-12
        )
        error_metrics["piston_position_relative"] = np.max(position_relative_error)
        error_metrics["piston_position_absolute"] = np.max(position_error)

        velocity_error = np.abs(
            comparison.piston_velocity_1d - comparison.piston_velocity_0d,
        )
        velocity_relative_error = velocity_error / (
            np.abs(comparison.piston_velocity_0d) + 1e-12
        )
        error_metrics["piston_velocity_relative"] = np.max(velocity_relative_error)
        error_metrics["piston_velocity_absolute"] = np.max(velocity_error)

    # Mass conservation error (simplified)
    if validation_params.validate_mass_conservation:
        mass_0d = comparison.density_0d * comparison.piston_position_0d
        mass_1d = comparison.density_1d * comparison.piston_position_1d
        mass_error = np.abs(mass_1d - mass_0d)
        mass_relative_error = mass_error / (np.abs(mass_0d) + 1e-12)
        error_metrics["mass_conservation"] = np.max(mass_relative_error)

    # Energy conservation error (simplified)
    if validation_params.validate_energy_conservation:
        energy_0d = comparison.pressure_0d * comparison.piston_position_0d / (1.4 - 1)
        energy_1d = comparison.pressure_1d * comparison.piston_position_1d / (1.4 - 1)
        energy_error = np.abs(energy_1d - energy_0d)
        energy_relative_error = energy_error / (np.abs(energy_0d) + 1e-12)
        error_metrics["energy_conservation"] = np.max(energy_relative_error)

    return error_metrics


def _check_validation_success(
    error_metrics: dict[str, float],
    validation_params: ValidationParameters,
) -> bool:
    """Check if validation is successful based on error metrics."""
    for metric_name, error_value in error_metrics.items():
        if "relative" in metric_name:
            if error_value > validation_params.relative_tolerance:
                return False
        elif "absolute" in metric_name:
            if error_value > validation_params.absolute_tolerance:
                return False
        # For other metrics, use relative tolerance
        elif error_value > validation_params.relative_tolerance:
            return False

    return True


def _create_validation_message(
    success: bool,
    error_metrics: dict[str, float],
) -> str:
    """Create validation message."""
    if success:
        max_error = max(error_metrics.values()) if error_metrics else 0.0
        return f"Validation PASSED: Maximum relative error = {max_error:.2e}"
    failed_metrics = [
        name for name, error in error_metrics.items() if error > 1e-3
    ]  # Threshold for failure
    return f"Validation FAILED: Failed metrics = {failed_metrics}"


def create_validation_report(
    validation_result: ValidationResult,
    output_file: str | None = None,
) -> str:
    """
    Create a detailed validation report.

    Args:
        validation_result: Validation result
        output_file: Optional output file path

    Returns:
        Report text
    """
    report = []
    report.append("=" * 80)
    report.append("CROSS-FIDELITY VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("SUMMARY:")
    report.append(f"  Success: {validation_result.success}")
    report.append(f"  Message: {validation_result.validation_message}")
    report.append(f"  Max Relative Error: {validation_result.max_relative_error:.2e}")
    report.append(f"  Max Absolute Error: {validation_result.max_absolute_error:.2e}")
    report.append("")

    # Timing
    report.append("TIMING:")
    report.append(f"  0D Simulation: {validation_result.cpu_time_0d:.1f} seconds")
    report.append(f"  1D Simulation: {validation_result.cpu_time_1d:.1f} seconds")
    report.append(
        f"  Speedup: {validation_result.cpu_time_1d / validation_result.cpu_time_0d:.2f}x",
    )
    report.append("")

    # Error metrics
    report.append("ERROR METRICS:")
    for metric_name, error_value in validation_result.error_metrics.items():
        report.append(f"  {metric_name}: {error_value:.2e}")
    report.append("")

    # Conservation
    report.append("CONSERVATION:")
    report.append(
        f"  Mass Conservation Error: {validation_result.mass_conservation_error:.2e}",
    )
    report.append(
        f"  Energy Conservation Error: {validation_result.energy_conservation_error:.2e}",
    )
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        log.info(f"Validation report saved to {output_file}")

    return report_text


def run_validation_suite() -> list[ValidationResult]:
    """
    Run a suite of validation tests.

    Returns:
        List of validation results
    """
    log.info("Running cross-fidelity validation suite")

    validation_results = []

    # Test case 1: Simple compression
    test_case_1 = {
        "geom": {"B": 0.1, "Vc": 1e-5},
        "initial_conditions": {
            "x_L": 0.05,
            "v_L": 0.0,
            "x_R": 0.15,
            "v_R": 0.0,
            "rho": 1.2,
            "T": 1000.0,
            "p": 1.0e5,
        },
        "n_cells": 50,
    }

    validation_params_1 = ValidationParameters(
        t_end=0.1,
        relative_tolerance=1e-2,
        validate_pressure=True,
        validate_temperature=True,
        validate_piston_dynamics=True,
    )

    result_1 = cross_fidelity_validation(test_case_1, validation_params_1)
    validation_results.append(result_1)

    # Test case 2: Rapid compression
    test_case_2 = {
        "geom": {"B": 0.1, "Vc": 1e-5},
        "initial_conditions": {
            "x_L": 0.05,
            "v_L": 0.0,
            "x_R": 0.15,
            "v_R": 1.0,  # Moving piston
            "rho": 1.2,
            "T": 1000.0,
            "p": 1.0e5,
        },
        "n_cells": 100,
    }

    validation_params_2 = ValidationParameters(
        t_end=0.05,
        relative_tolerance=5e-2,
        validate_pressure=True,
        validate_temperature=True,
        validate_piston_dynamics=True,
    )

    result_2 = cross_fidelity_validation(test_case_2, validation_params_2)
    validation_results.append(result_2)

    # Test case 3: High pressure
    test_case_3 = {
        "geom": {"B": 0.1, "Vc": 1e-5},
        "initial_conditions": {
            "x_L": 0.05,
            "v_L": 0.0,
            "x_R": 0.15,
            "v_R": 0.0,
            "rho": 5.0,
            "T": 2000.0,
            "p": 1.0e6,
        },
        "n_cells": 75,
    }

    validation_params_3 = ValidationParameters(
        t_end=0.1,
        relative_tolerance=1e-2,
        validate_pressure=True,
        validate_temperature=True,
        validate_piston_dynamics=True,
    )

    result_3 = cross_fidelity_validation(test_case_3, validation_params_3)
    validation_results.append(result_3)

    log.info(f"Validation suite completed: {len(validation_results)} test cases")

    return validation_results
