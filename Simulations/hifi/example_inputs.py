"""Example SimulationInput configurations for testing hifi adapters.

Provides factory functions to create realistic engine simulation inputs
with geometry, operating conditions, and cycle-resolved boundary conditions.
"""

from typing import Optional

import numpy as np

from Simulations.common.io_schema import (
    BoundaryConditions,
    GeometryConfig,
    OperatingPoint,
    SimulationInput,
    SimulationOutput,
)


def create_baseline_geometry(
    bore_mm: float = 85.0,
    stroke_mm: float = 90.0,
    compression_ratio: float = 12.5,
) -> GeometryConfig:
    """
    Create baseline engine geometry.

    Default: 85mm bore x 90mm stroke, ~500cc single cylinder
    """
    return GeometryConfig(
        bore=bore_mm / 1000,
        stroke=stroke_mm / 1000,
        conrod=stroke_mm / 1000 * 1.7,  # Typical conrod/stroke ratio
        compression_ratio=compression_ratio,
        liner_thickness=0.005,
        piston_clearance=50e-6,
    )


def create_operating_point(
    rpm: float = 3000.0,
    load_fraction: float = 1.0,  # 0-1, affects intake pressure
    lambda_val: float = 1.0,
) -> OperatingPoint:
    """
    Create operating point for steady-state conditions.

    Args:
        rpm: Engine speed
        load_fraction: 0=idle, 1=WOT
        lambda_val: Air/fuel ratio (1.0 = stoichiometric)
    """
    # Intake pressure: 30kPa (idle) to 100kPa (WOT)
    p_intake = 30000 + load_fraction * 70000

    return OperatingPoint(
        rpm=rpm,
        lambda_val=lambda_val,
        p_intake=p_intake,
        T_intake=300.0 + load_fraction * 20,  # Warmer at high load
        T_coolant=363.0,  # 90°C
        T_oil=373.0,  # 100°C
    )


def create_boundary_conditions(
    geometry: GeometryConfig,
    operating_point: OperatingPoint,
    n_points: int = 720,  # 0.5° resolution
) -> BoundaryConditions:
    """
    Generate realistic cycle-resolved boundary conditions.

    Creates pressure and temperature traces for a complete 720° cycle
    with motoring + combustion pressure rise.
    """
    # Crank angle from -360 to 360 (exhaust TDC to compression TDC)
    ca = np.linspace(-360, 360, n_points)

    # Compute volume ratio for motoring pressure
    bore = geometry.bore
    stroke = geometry.stroke
    cr = geometry.compression_ratio

    # Piston position (normalized)
    theta_rad = np.radians(ca)
    # Simplified kinematics
    x_norm = 0.5 * (1 - np.cos(theta_rad))  # 0 at TDC, 1 at BDC

    # Volume ratio: V/Vc where Vc = clearance volume
    v_ratio = 1 + (cr - 1) * x_norm

    # Motoring pressure (polytropic)
    p_intake = operating_point.p_intake
    gamma = 1.35  # Polytropic exponent
    p_motor = p_intake * (1 / v_ratio) ** gamma

    # Add combustion pressure rise (Wiebe-like bump near TDC)
    ca_combustion = ca + 360  # Shift so 0 = firing TDC
    combustion_center = 10  # deg after TDC
    combustion_width = 40  # deg

    # Gaussian-like combustion pressure addition
    x_burn = (ca_combustion - combustion_center) / combustion_width
    burn_fraction = np.exp(-0.5 * x_burn**2) * np.heaviside(ca_combustion + 20, 0.5)

    # Peak pressure: ~80 bar at full load
    p_max_combustion = 80e5 * (operating_point.p_intake / 100000)
    p_gas = p_motor + burn_fraction * (p_max_combustion - p_motor.max())
    p_gas = np.maximum(p_gas, p_intake * 0.5)  # Floor at 50% intake

    # Temperature (ideal gas relation)
    T_intake = operating_point.T_intake
    T_gas = T_intake * (p_gas / p_intake) ** ((gamma - 1) / gamma)
    T_gas = np.clip(T_gas, T_intake, 2500)  # Cap at realistic combustion temp

    # Woschni HTC (simplified)
    C1 = 2.28  # Woschni constant
    mean_piston_speed = 2 * stroke * operating_point.rpm / 60
    htc = C1 * (bore**-0.2) * (p_gas / 1e5) ** 0.8 * T_gas**-0.53 * mean_piston_speed**0.8
    htc = np.clip(htc, 100, 5000)  # W/m²K bounds

    # Piston speed
    omega = operating_point.rpm * 2 * np.pi / 60
    piston_speed = 0.5 * stroke * omega * np.sin(theta_rad)

    return BoundaryConditions(
        crank_angle=ca.tolist(),
        pressure_gas=p_gas.tolist(),
        temperature_gas=T_gas.tolist(),
        heat_transfer_coeff=htc.tolist(),
        piston_speed=piston_speed.tolist(),
    )


def create_simulation_input(
    run_id: str = "test_run_001",
    bore_mm: float = 85.0,
    stroke_mm: float = 90.0,
    rpm: float = 3000.0,
    load_fraction: float = 1.0,
    compression_ratio: float = 12.5,
    solver_settings: dict | None = None,
) -> SimulationInput:
    """
    Create complete SimulationInput for adapter testing.

    Args:
        run_id: Unique identifier for this run
        bore_mm: Cylinder bore in mm
        stroke_mm: Piston stroke in mm
        rpm: Engine speed
        load_fraction: 0-1 load (affects intake pressure)
        compression_ratio: Geometric CR
        solver_settings: Optional solver-specific settings

    Returns:
        Complete SimulationInput ready for adapter.load_input()
    """
    geometry = create_baseline_geometry(bore_mm, stroke_mm, compression_ratio)
    operating_point = create_operating_point(rpm, load_fraction)
    boundary_conditions = create_boundary_conditions(geometry, operating_point)

    return SimulationInput(
        run_id=run_id,
        solver_settings=solver_settings or {},
        geometry=geometry,
        operating_point=operating_point,
        boundary_conditions=boundary_conditions,
    )


# Preset configurations for common test scenarios
def wot_high_speed() -> SimulationInput:
    """Wide Open Throttle at 6000 RPM - thermal stress test."""
    return create_simulation_input(
        run_id="wot_6000rpm",
        rpm=6000.0,
        load_fraction=1.0,
    )


def part_load_cruise() -> SimulationInput:
    """Part load cruise at 2500 RPM - efficiency test."""
    return create_simulation_input(
        run_id="cruise_2500rpm",
        rpm=2500.0,
        load_fraction=0.5,
    )


def idle_condition() -> SimulationInput:
    """Idle at 800 RPM - low thermal load."""
    return create_simulation_input(
        run_id="idle_800rpm",
        rpm=800.0,
        load_fraction=0.1,
    )


def high_compression() -> SimulationInput:
    """High compression ratio for knock study."""
    return create_simulation_input(
        run_id="high_cr_study",
        rpm=3000.0,
        load_fraction=1.0,
        compression_ratio=14.0,
    )


if __name__ == "__main__":
    # Demo: create and print simulation input
    sim_input = create_simulation_input()

    print("SimulationInput created:")
    print(f"  Run ID: {sim_input.run_id}")
    print(
        f"  Geometry: {sim_input.geometry.bore * 1000:.1f}mm x {sim_input.geometry.stroke * 1000:.1f}mm"
    )
    print(
        f"  Operating: {sim_input.operating_point.rpm:.0f} RPM, {sim_input.operating_point.p_intake / 1e5:.2f} bar"
    )
    print(f"  BC points: {len(sim_input.boundary_conditions.crank_angle)}")
    print(f"  P_max: {max(sim_input.boundary_conditions.pressure_gas) / 1e5:.1f} bar")
    print(f"  T_max: {max(sim_input.boundary_conditions.temperature_gas):.0f} K")
