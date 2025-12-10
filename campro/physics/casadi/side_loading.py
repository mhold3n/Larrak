"""
CasADi-based side loading calculations for crank-piston systems.

This module provides side load computation from piston forces and crank kinematics,
ported to CasADi MX for use in symbolic optimization.
"""

import casadi as ca

from campro.constants import CASADI_PHYSICS_ASIN_CLAMP, CASADI_PHYSICS_EPSILON
from campro.logging import get_logger

log = get_logger(__name__)


def create_side_load_pointwise() -> ca.Function:
    """
    Create CasADi function for instantaneous side load at single crank angle.

    Computes lateral force using F_side = F * tan(rod_angle) where rod_angle
    is computed from crank kinematics with proper domain guards.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta, r, l, F, x_off, y_off) -> F_side

        Inputs:
            theta : MX - Crank angle (radians)
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            F : MX - Piston force (N)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)

        Outputs:
            F_side : MX - Lateral side load (N)

    Notes
    -----
    - Based on `campro/physics/mechanics/side_loading.py:292-328`
    - Formula: F_side = F * tan(rod_angle)
    - Rod angle: beta = arcsin(clamp((r/l)*sin(theta + offset_angle), ...))
    - Domain guards protect against arcsin singularities

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.side_loading import create_side_load_pointwise
    >>>
    >>> side_load_fn = create_side_load_pointwise()
    >>> F_side = side_load_fn(ca.pi/4, 50.0, 150.0, 1000.0, 5.0, 2.0)
    """
    # Define symbolic inputs
    theta = ca.MX.sym("theta")
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")  # noqa: E741
    F = ca.MX.sym("F")  # noqa: N806
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")

    # Compute rod angle with offset correction (from kinematics.py)
    offset_angle = ca.atan2(y_off, x_off)
    effective_crank_angle = theta + offset_angle

    # Guard arcsin domain: |r/l * sin(theta)| must be ≤ 1
    sin_arg = (r / l) * ca.sin(effective_crank_angle)
    sin_arg_clamped = ca.fmin(
        ca.fmax(sin_arg, -CASADI_PHYSICS_ASIN_CLAMP),
        CASADI_PHYSICS_ASIN_CLAMP,
    )
    rod_angle = ca.arcsin(sin_arg_clamped)

    # Compute side load: F_side = F * tan(rod_angle)
    # Guard tan domain by ensuring rod_angle is not near ±π/2
    rod_angle_clamped = ca.fmin(
        ca.fmax(rod_angle, -1.5),  # Clamp to avoid tan singularities
        1.5,
    )
    F_side = F * ca.tan(rod_angle_clamped)  # noqa: N806

    # Create function
    side_load_fn = ca.Function(
        "side_load_pointwise",
        [theta, r, l, F, x_off, y_off],
        [F_side],
        ["theta", "r", "l", "F", "x_off", "y_off"],
        ["F_side"],
    )

    log.info("Created side_load_pointwise CasADi function")
    return side_load_fn


def create_side_load_profile() -> ca.Function:
    """
    Create CasADi function for side load profile with statistics over full cycle.

    Computes side load at each crank angle and provides cycle statistics including
    maximum, average, and ripple coefficient.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, F_vec, r, l, x_off, y_off) -> (F_side_vec, F_side_max, F_side_avg, ripple)  # noqa: E501

        Inputs:
            theta_vec : MX(n,1) - Crank angles (radians)
            F_vec : MX(n,1) - Piston forces (N)
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)

        Outputs:
            F_side_vec : MX(n,1) - Side load at each angle (N)
            F_side_max : MX - Maximum side load (N)
            F_side_avg : MX - Average side load (N)
            ripple : MX - Side load variation coefficient (dimensionless)

    Notes
    -----
    - Uses pointwise side load function for each angle
    - Ripple coefficient: std(F_side) / |mean(F_side)|
    - Average computed as simple mean (not trapezoidal integration like torque)

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.side_loading import create_side_load_profile
    >>>
    >>> profile_fn = create_side_load_profile()
    >>> theta = ca.DM(np.linspace(0, 2*np.pi, 10))
    >>> F = ca.DM(1000.0 * np.ones(10))
    >>> F_side_vec, F_side_max, F_side_avg, ripple = profile_fn(theta, F, 50.0, 150.0, 5.0, 2.0)  # noqa: E501
    """
    # Define symbolic inputs (fixed size for CasADi compatibility)
    theta_vec = ca.MX.sym("theta_vec", 10, 1)  # Fixed size vector
    F_vec = ca.MX.sym("F_vec", 10, 1)  # noqa: N806
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")  # noqa: E741
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")

    n = theta_vec.shape[0]

    # Get pointwise side load function
    side_load_point_fn = create_side_load_pointwise()

    # For now, compute side load at each point individually
    F_side_vec = ca.MX.zeros(n, 1)  # noqa: N806

    for i in range(n):
        # Compute side load at this point
        F_side_i = side_load_point_fn(  # noqa: N806
            theta_vec[i], r, l, F_vec[i], x_off, y_off,
        )
        F_side_vec[i] = F_side_i

    # Statistics
    F_side_max = ca.mmax(F_side_vec)  # noqa: N806
    F_side_avg = ca.sum1(F_side_vec) / n  # noqa: N806

    # Ripple coefficient: std(F_side) / |mean(F_side)|
    F_side_mean = F_side_avg  # noqa: N806
    F_side_variance = ca.sum1((F_side_vec - F_side_mean) ** 2) / n  # noqa: N806
    F_side_std = ca.sqrt(ca.fmax(F_side_variance, CASADI_PHYSICS_EPSILON))  # noqa: N806
    ripple = F_side_std / ca.fmax(ca.fabs(F_side_mean), CASADI_PHYSICS_EPSILON)

    # Create function
    profile_fn = ca.Function(
        "side_load_profile",
        [theta_vec, F_vec, r, l, x_off, y_off],
        [F_side_vec, F_side_max, F_side_avg, ripple],
        ["theta_vec", "F_vec", "r", "l", "x_off", "y_off"],
        ["F_side_vec", "F_side_max", "F_side_avg", "ripple"],
    )

    log.info("Created side_load_profile CasADi function")
    return profile_fn


def create_side_load_penalty() -> ca.Function:
    """
    Create CasADi function for smooth side load penalty with phase weights.

    Computes smooth penalty for side loads exceeding threshold, with different
    weights for compression and combustion phases.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (F_side_vec, max_threshold, compression_mask, combustion_mask) -> penalty

        Inputs:
            F_side_vec : MX(n,1) - Side load profile (N)
            max_threshold : MX - Maximum allowed side load (N)
            compression_mask : MX(n,1) - 1 for compression phase, 0 otherwise
            combustion_mask : MX(n,1) - 1 for combustion phase, 0 otherwise

        Outputs:
            penalty : MX - Smooth penalty value (dimensionless)

    Notes
    -----
    - Based on `campro/physics/mechanics/side_loading.py:292-328`
    - Smooth ReLU: relu_smooth(x) = 0.5 * (x + sqrt(x^2 + epsilon))
    - Phase weights: compression=1.2, combustion=1.5
    - Penalty = sum(weight * relu_smooth(|F_side| - threshold)^2)

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.side_loading import create_side_load_penalty
    >>>
    >>> penalty_fn = create_side_load_penalty()
    >>> F_side = ca.DM([100, 200, 300, 150, 250])  # N
    >>> threshold = 200.0  # N
    >>> comp_mask = ca.DM([1, 1, 0, 0, 1])  # compression phases
    >>> comb_mask = ca.DM([0, 0, 1, 1, 0])  # combustion phases
    >>> penalty = penalty_fn(F_side, threshold, comp_mask, comb_mask)
    """
    # Define symbolic inputs (fixed size for CasADi compatibility)
    F_side_vec = ca.MX.sym("F_side_vec", 10, 1)  # Fixed size vector  # noqa: N806
    max_threshold = ca.MX.sym("max_threshold")
    compression_mask = ca.MX.sym("compression_mask", 10, 1)
    combustion_mask = ca.MX.sym("combustion_mask", 10, 1)

    # Phase weights (from side_loading.py constants)
    compression_weight = 1.2
    combustion_weight = 1.5
    normal_weight = 1.0

    # Compute excess side load: |F_side| - threshold
    excess = ca.fabs(F_side_vec) - max_threshold

    # Smooth ReLU: relu_smooth(x) = 0.5 * (x + sqrt(x^2 + epsilon))
    relu_smooth = 0.5 * (excess + ca.sqrt(excess**2 + CASADI_PHYSICS_EPSILON))

    # Apply phase weights
    phase_weight = (
        compression_weight * compression_mask +
        combustion_weight * combustion_mask +
        normal_weight * (1.0 - compression_mask - combustion_mask)
    )

    # Compute weighted penalty: sum(weight * relu_smooth^2)
    weighted_penalty = phase_weight * (relu_smooth**2)
    penalty = ca.sum1(weighted_penalty)

    # Create function
    penalty_fn = ca.Function(
        "side_load_penalty",
        [F_side_vec, max_threshold, compression_mask, combustion_mask],
        [penalty],
        ["F_side_vec", "max_threshold", "compression_mask", "combustion_mask"],
        ["penalty"],
    )

    log.info("Created side_load_penalty CasADi function")
    return penalty_fn
