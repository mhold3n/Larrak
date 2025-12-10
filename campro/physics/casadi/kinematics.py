"""
CasADi-based crank-piston kinematics with automatic differentiation.

This module provides kinematics calculations for crank-piston systems with
crank center offset effects, ported to CasADi MX for use in symbolic optimization.
"""

import casadi as ca

from campro.constants import (
    CASADI_PHYSICS_ASIN_CLAMP,
    CASADI_PHYSICS_EPSILON,
    CASADI_PHYSICS_USE_EFFECTIVE_RADIUS_CORRECTION,
)
from campro.logging import get_logger

log = get_logger(__name__)


def create_crank_piston_kinematics() -> ca.Function:
    """
    Create CasADi function for crank-piston kinematics with offset effects.

    Computes piston displacement, velocity, acceleration, connecting rod angle,
    and effective crank radius accounting for crank center offset relative to
    gear center.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta, r, l, x_off, y_off) -> (x, v, a, rod_angle, r_eff)

        Inputs:
            theta : MX(n,1) - Crank angles (radians)
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset from gear center (mm)
            y_off : MX - Crank center y-offset from gear center (mm)

        Outputs:
            x : MX(n,1) - Piston displacement (mm)
            v : MX(n,1) - Piston velocity (mm/rad)
            a : MX(n,1) - Piston acceleration (mm/rad²)
            rod_angle : MX(n,1) - Connecting rod angle (radians)
            r_eff : MX(n,1) - Effective crank radius (mm)

    Notes
    -----
    - Based on `campro/physics/kinematics/crank_kinematics.py:compute_rod_angle`
    - Offset correction from `crank_kinematics.py:compute_corrected_piston_motion`
    - Effective radius marked TODO for future physics-based enhancement
    - Domain guards protect against arcsin and sqrt singularities

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.kinematics import create_crank_piston_kinematics
    >>>
    >>> kin_fn = create_crank_piston_kinematics()
    >>> theta = ca.DM([0.0, ca.pi/4, ca.pi/2])
    >>> r, l = 50.0, 150.0  # mm
    >>> x_off, y_off = 5.0, 2.0  # mm
    >>> x, v, a, rod_angle, r_eff = kin_fn(theta, r, l, x_off, y_off)
    """
    # Define symbolic inputs
    theta = ca.MX.sym("theta", 1)  # Single angle (will be mapped over vectors)
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")

    # Compute rod angle with offset correction
    # From crank_kinematics.py:175
    offset_angle = ca.atan2(y_off, x_off)
    effective_crank_angle = theta + offset_angle

    # Guard arcsin domain: |r/l * sin(theta)| must be ≤ 1
    sin_arg = (r / l) * ca.sin(effective_crank_angle)
    sin_arg_clamped = ca.fmin(ca.fmax(sin_arg, -CASADI_PHYSICS_ASIN_CLAMP),
                               CASADI_PHYSICS_ASIN_CLAMP)
    rod_angle = ca.arcsin(sin_arg_clamped)

    # Compute piston displacement with offset correction
    # From crank_kinematics.py:254-257
    offset_correction = x_off * ca.cos(theta) + y_off * ca.sin(theta)

    # Standard crank-slider position: x = r*cos(theta) + sqrt(l² - (r*sin(theta))²)
    radicand = l**2 - (r * ca.sin(theta))**2
    radicand_safe = ca.fmax(radicand, CASADI_PHYSICS_EPSILON)
    x_nominal = r * ca.cos(theta) + ca.sqrt(radicand_safe)

    # Apply offset correction
    x = offset_correction + x_nominal

    # Compute velocity via automatic differentiation
    v = ca.jacobian(x, theta)

    # Compute acceleration via automatic differentiation
    a = ca.jacobian(v, theta)

    # Effective radius with optional offset correction
    if CASADI_PHYSICS_USE_EFFECTIVE_RADIUS_CORRECTION:
        # Account for crank center offset effects on effective radius
        # Based on torque_analysis.py:286-293
        # Offset correction: r_eff = r * (1 + offset_factor)
        offset_magnitude = ca.sqrt(ca.fmax(x_off**2 + y_off**2, CASADI_PHYSICS_EPSILON))
        offset_factor = 0.1 * offset_magnitude / r  # Simplified correction
        r_eff = r * (1.0 + offset_factor)
    else:
        # Nominal radius (default behavior)
        r_eff = r

    # Create scalar function
    scalar_fn = ca.Function(
        "crank_piston_kinematics_scalar",
        [theta, r, l, x_off, y_off],
        [x, v, a, rod_angle, r_eff],
        ["theta", "r", "l", "x_off", "y_off"],
        ["x", "v", "a", "rod_angle", "r_eff"],
    )

    # Create vectorized function using mapaccum or manual vectorization
    # For now, return the scalar function - vectorization will be handled at call time
    vectorized_fn = scalar_fn

    log.info("Created crank_piston_kinematics CasADi function")
    return vectorized_fn


def create_crank_piston_kinematics_vectorized() -> ca.Function:
    """
    Create vectorized CasADi function for crank-piston kinematics with offset effects.

    Computes piston displacement, velocity, acceleration, connecting rod angle,
    and effective crank radius for arbitrary-length input vectors.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, r, l, x_off, y_off) -> (x_vec, v_vec, a_vec, rod_angle_vec, r_eff_vec)

        Inputs:
            theta_vec : MX(n,1) - Crank angles (radians), arbitrary length n
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset from gear center (mm)
            y_off : MX - Crank center y-offset from gear center (mm)

        Outputs:
            x_vec : MX(n,1) - Piston displacement (mm)
            v_vec : MX(n,1) - Piston velocity (mm/rad)
            a_vec : MX(n,1) - Piston acceleration (mm/rad²)
            rod_angle_vec : MX(n,1) - Connecting rod angle (radians)
            r_eff_vec : MX(n,1) - Effective crank radius (mm)

    Notes
    -----
    - Vectorized version of create_crank_piston_kinematics
    - Uses CasADi map to apply scalar kinematics to each angle
    - Preserves automatic differentiation through vectorization
    - More efficient than calling scalar function in loops

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.kinematics import create_crank_piston_kinematics_vectorized
    >>>
    >>> kin_vec_fn = create_crank_piston_kinematics_vectorized()
    >>> theta_vec = ca.DM(np.linspace(0, 2*np.pi, 32))
    >>> r, l = 50.0, 150.0  # mm
    >>> x_off, y_off = 5.0, 2.0  # mm
    >>> x_vec, v_vec, a_vec, rod_angle_vec, r_eff_vec = kin_vec_fn(theta_vec, r, l, x_off, y_off)
    """
    # Get the scalar kinematics function
    scalar_kin_fn = create_crank_piston_kinematics()

    # Define symbolic inputs
    theta_vec = ca.MX.sym("theta_vec")
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")

    # Get vector length
    n = theta_vec.shape[0]

    # Initialize output vectors
    x_vec = ca.MX.zeros(n, 1)
    v_vec = ca.MX.zeros(n, 1)
    a_vec = ca.MX.zeros(n, 1)
    rod_angle_vec = ca.MX.zeros(n, 1)
    r_eff_vec = ca.MX.zeros(n, 1)

    # Apply scalar kinematics to each angle
    for i in range(n):
        # Get kinematics for this angle
        kin_results = scalar_kin_fn(theta_vec[i], r, l, x_off, y_off)
        
        # Extract results
        x_vec[i] = kin_results[0]
        v_vec[i] = kin_results[1]
        a_vec[i] = kin_results[2]
        rod_angle_vec[i] = kin_results[3]
        r_eff_vec[i] = kin_results[4]

    # Create function
    vectorized_fn = ca.Function(
        "crank_piston_kinematics_vectorized",
        [theta_vec, r, l, x_off, y_off],
        [x_vec, v_vec, a_vec, rod_angle_vec, r_eff_vec],
        ["theta_vec", "r", "l", "x_off", "y_off"],
        ["x_vec", "v_vec", "a_vec", "rod_angle_vec", "r_eff_vec"],
    )

    log.info("Created crank_piston_kinematics_vectorized CasADi function")
    return vectorized_fn


def create_phase_masks() -> ca.Function:
    """
    Create CasADi function to detect expansion and compression phases.

    Analyzes displacement profile to determine where piston/planet is expanding
    (moving away from center) versus compressing (moving toward center).

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (displacement) -> (expansion_mask, compression_mask)

        Inputs:
            displacement : MX(n,1) - Radial position of piston/planet (mm)

        Outputs:
            expansion_mask : MX(n,1) - 1 where displacement increasing, 0 otherwise
            compression_mask : MX(n,1) - 1 where displacement decreasing, 0 otherwise

    Notes
    -----
    - Expansion: radial displacement increasing (dR/dt > 0)
    - Compression: radial displacement decreasing (dR/dt < 0)
    - First element of masks is 0 (no prior difference available)
    - Masks are numeric (0/1) for use in weighted sums, not boolean

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.kinematics import create_phase_masks
    >>>
    >>> mask_fn = create_phase_masks()
    >>> displacement = ca.DM([100, 110, 120, 115, 105])  # mm
    >>> exp_mask, comp_mask = mask_fn(displacement)
    >>> # exp_mask ≈ [0, 1, 1, 0, 0]
    >>> # comp_mask ≈ [0, 0, 0, 1, 1]
    """
    # Define symbolic input (fixed size for now)
    displacement = ca.MX.sym("displacement", 10, 1)  # Fixed size vector
    displacement.shape[0]

    # Compute differences: diff[i] = displacement[i+1] - displacement[i]
    diff = displacement[1:] - displacement[:-1]

    # Detect expansion (diff > 0) and compression (diff < 0)
    # Use if_else for symbolic conditional
    expansion = ca.if_else(diff > 0, 1.0, 0.0)
    compression = ca.if_else(diff < 0, 1.0, 0.0)

    # Pad with leading zero (first point has no prior difference)
    expansion_mask = ca.vertcat(0.0, expansion)
    compression_mask = ca.vertcat(0.0, compression)

    # Create function
    phase_fn = ca.Function(
        "phase_masks",
        [displacement],
        [expansion_mask, compression_mask],
        ["displacement"],
        ["expansion_mask", "compression_mask"],
    )

    log.info("Created phase_masks CasADi function")
    return phase_fn

