"""
Physics utility functions for free-piston engine optimization.
"""



def estimate_cycle_time_from_physics(
    stroke: float,
    fuel_mass: float,
    afr: float = 14.7,
    bore: float = 0.1,
) -> float:
    """
    Estimate a reasonable cycle time based on physics constraints.

    Uses heuristics based on mean piston speed and combustion duration scales.

    Args:
        stroke: Piston stroke [m]
        fuel_mass: Fuel mass per cycle [kg]
        afr: Air-fuel ratio
        bore: Cylinder bore [m] (optional, for scaling)

    Returns:
        Estimated cycle time [s]
    """
    # Table of base cycle times based on stroke length
    # Stroke (m) -> Base RPM -> Base Cycle Time (s)
    # < 0.025m (25mm) -> 9000 RPM
    # < 0.045m (45mm) -> 6000 RPM
    # < 0.080m (80mm) -> 4000 RPM
    # >= 0.080m       -> 2000 RPM

    if stroke < 0.025:
        base_rpm = 9000.0
    elif stroke < 0.045:
        base_rpm = 6000.0
    elif stroke < 0.080:
        base_rpm = 4000.0
    else:
        base_rpm = 2000.0

    base_cycle_time = 60.0 / base_rpm

    # Adjust for Air-Fuel Ratio (AFR)
    # Combustion speed depends on equivalence ratio (phi).
    # Peak flame speed is typically around phi ~ 1.1 (slightly rich).
    # Leaner or richer mixtures burn slower, requiring a longer cycle time (lower RPM).

    phi = 14.7 / max(1.0, afr)  # Equivalence ratio

    # Simple penalty factor for off-peak combustion speed
    # If phi is 1.1, factor is 1.0.
    # As phi deviates, factor increases (slowing down the cycle).
    # e.g. phi=0.6 (lean) -> |0.6 - 1.1| = 0.5 -> factor = 1.25
    combustion_time_factor = 1.0 + 0.5 * abs(phi - 1.1)

    estimated_time = base_cycle_time * combustion_time_factor

    # Clamp to reasonable bounds to prevent numerical instability
    # Max RPM ~15000 -> Min time ~0.004s
    # Min RPM ~600 -> Max time ~0.1s
    t_est = max(0.004, min(0.1, estimated_time))

    return t_est
