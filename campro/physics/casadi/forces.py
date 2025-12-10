"""
CasADi-based piston force calculations.

This module provides force computation from gas pressure and bore geometry,
ported to CasADi MX for use in symbolic optimization.
"""

import casadi as ca

from campro.logging import get_logger

log = get_logger(__name__)


def create_piston_force_simple() -> ca.Function:
    """
    Create CasADi function for simple piston force calculation.

    Computes piston force from gas pressure and bore diameter using
    F = pressure * area, where area = π * (bore/2)².

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (pressure, bore) -> F

        Inputs:
            pressure : MX(n,1) - Gas pressure (Pa)
            bore : MX - Cylinder bore diameter (mm)

        Outputs:
            F : MX(n,1) - Piston force (N)

    Notes
    -----
    - Based on `campro.optimization.core/piston.py:93-95`
    - Units: pressure (Pa), bore (mm) → force (N)
    - Area conversion: mm² to m² via 1e-6 factor

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.forces import create_piston_force_simple
    >>>
    >>> force_fn = create_piston_force_simple()
    >>> pressure = ca.DM([1e5, 2e5, 3e5])  # Pa
    >>> bore = 100.0  # mm
    >>> F = force_fn(pressure, bore)
    """
    # Define symbolic inputs
    pressure = ca.MX.sym("pressure", 10, 1)  # Fixed size vector
    bore = ca.MX.sym("bore")

    # Compute piston area in mm²
    area_mm2 = ca.pi * (bore / 2.0) ** 2

    # Convert area from mm² to m²
    area_m2 = area_mm2 * 1e-6

    # Compute force: F = pressure * area
    F = pressure * area_m2

    # Create function
    force_fn = ca.Function(
        "piston_force_simple",
        [pressure, bore],
        [F],
        ["pressure", "bore"],
        ["F"],
    )

    log.info("Created piston_force_simple CasADi function")
    return force_fn
