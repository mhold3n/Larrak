"""
CasADi-based torque calculations for crank-piston systems.

This module provides torque computation from piston forces and crank kinematics,
ported to CasADi MX for use in symbolic optimization.
"""

import casadi as ca
from campro.constants import CASADI_PHYSICS_EPSILON
from campro.logging import get_logger

from .kinematics import create_crank_piston_kinematics, create_crank_piston_kinematics_vectorized

log = get_logger(__name__)


def create_torque_pointwise() -> ca.Function:
    """
    Create CasADi function for instantaneous torque at single crank angle.

    Computes torque using T = F * r * sin(theta + rod_angle) * cos(pressure_angle)
    with proper unit conversion from mm to m for radius.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta, F, r, rod_angle, pressure_angle) -> T

        Inputs:
            theta : MX - Crank angle (radians)
            F : MX - Piston force (N)
            r : MX - Effective crank radius (mm)
            rod_angle : MX - Connecting rod angle (radians)
            pressure_angle : MX - Gear pressure angle (radians)

        Outputs:
            T : MX - Instantaneous torque (N⋅m)

    Notes
    -----
    - Based on `campro/physics/mechanics/torque_analysis.py:186-187`
    - Formula: T = F * r_m * sin(theta + rod_angle) * cos(pressure_angle)
    - Unit conversion: r (mm) → r_m (m) via 1e-3 factor

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.torque import create_torque_pointwise
    >>>
    >>> T_fn = create_torque_pointwise()
    >>> T = T_fn(ca.pi/4, 1000.0, 50.0, 0.1, ca.pi/9)
    """
    # Define symbolic inputs
    theta = ca.MX.sym("theta")
    F = ca.MX.sym("F")
    r = ca.MX.sym("r")
    rod_angle = ca.MX.sym("rod_angle")
    pressure_angle = ca.MX.sym("pressure_angle")

    # Convert radius from mm to m
    r_m = r * 1e-3

    # Compute torque: T = F * r_m * sin(theta + rod_angle) * cos(pressure_angle)
    T = F * r_m * ca.sin(theta + rod_angle) * ca.cos(pressure_angle)

    # Create function
    torque_fn = ca.Function(
        "torque_pointwise",
        [theta, F, r, rod_angle, pressure_angle],
        [T],
        ["theta", "F", "r", "rod_angle", "pressure_angle"],
        ["T"],
    )

    log.info("Created torque_pointwise CasADi function")
    return torque_fn


def create_torque_profile_fixed() -> ca.Function:
    """
    Create CasADi function for torque profile with statistics over full cycle.

    Computes torque at each crank angle and provides cycle statistics including
    average (via trapezoidal integration), maximum, minimum, and ripple coefficient.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, F_vec, r, l, x_off, y_off, pressure_angle) -> (T_vec, T_avg, T_max, T_min, ripple)

        Inputs:
            theta_vec : MX(n,1) - Crank angles (radians)
            F_vec : MX(n,1) - Piston forces (N)
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)
            pressure_angle : MX - Gear pressure angle (radians)

        Outputs:
            T_vec : MX(n,1) - Torque at each angle (N⋅m)
            T_avg : MX - Cycle-averaged torque via trapezoidal integration (N⋅m)
            T_max : MX - Maximum torque (N⋅m)
            T_min : MX - Minimum torque (N⋅m)
            ripple : MX - Torque variation coefficient (dimensionless)

    Notes
    -----
    - Uses kinematics to compute rod angles and effective radii
    - Trapezoidal integration: T_avg = sum(0.5 * (T[i] + T[i+1]) * dtheta) / (theta_end - theta_start)
    - Ripple coefficient: std(T) / |mean(T)|

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.torque import create_torque_profile
    >>>
    >>> profile_fn = create_torque_profile()
    >>> theta = ca.DM(np.linspace(0, 2*np.pi, 100))
    >>> F = ca.DM(1000.0 * np.ones(100))
    >>> T_vec, T_avg, T_max, T_min, ripple = profile_fn(theta, F, 50.0, 150.0, 5.0, 2.0, np.radians(20))
    """
    # Define symbolic inputs (fixed size for CasADi compatibility)
    theta_vec = ca.MX.sym("theta_vec", 10, 1)  # Fixed size vector
    F_vec = ca.MX.sym("F_vec", 10, 1)  # Fixed size vector
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")
    pressure_angle = ca.MX.sym("pressure_angle")

    n = theta_vec.shape[0]

    # Get pointwise torque function
    T_point_fn = create_torque_pointwise()

    # Use vectorized kinematics for efficiency
    kin_vec_fn = create_crank_piston_kinematics_vectorized()
    
    # Get all kinematics at once
    _, _, _, rod_angle_vec, r_eff_vec = kin_vec_fn(theta_vec, r, l, x_off, y_off)

    # Compute torque at each point using vectorized kinematics
    T_vec = ca.MX.zeros(n, 1)
    for i in range(n):
        # Compute torque at this point (CasADi indexing returns proper scalars)
        theta_i = theta_vec[i]
        F_i = F_vec[i]
        r_eff_i = r_eff_vec[i]
        rod_angle_i = rod_angle_vec[i]
        
        T_i = T_point_fn(theta_i, F_i, r_eff_i, rod_angle_i, pressure_angle)
        T_vec[i] = T_i

    # Compute cycle average via trapezoidal integration
    if n > 1:
        # Differences between consecutive angles
        dtheta = theta_vec[1:] - theta_vec[:-1]

        # Midpoint torques
        T_mid = 0.5 * (T_vec[:-1] + T_vec[1:])

        # Trapezoidal sum (ca.sum1 already returns scalar)
        T_sum = ca.sum1(T_mid * dtheta)

        # Total angle span (ensure scalar)
        theta_span = theta_vec[-1] - theta_vec[0]
        theta_span = theta_span[0] if theta_span.shape[0] > 1 else theta_span

        # Average torque (ensure scalar)
        T_avg = T_sum / ca.fmax(theta_span, CASADI_PHYSICS_EPSILON)
    else:
        T_avg = T_vec[0] if n == 1 else ca.MX(0.0)

    # Statistics (ensure scalar outputs)
    T_max = ca.mmax(T_vec)[0]  # Extract scalar from max
    T_min = ca.mmin(T_vec)[0]  # Extract scalar from min

    # Ripple coefficient: std(T) / |mean(T)|
    T_mean = ca.sum1(T_vec) / n
    T_variance = ca.sum1((T_vec - T_mean) ** 2) / n
    T_std = ca.sqrt(ca.fmax(T_variance, CASADI_PHYSICS_EPSILON))
    ripple = T_std / ca.fmax(ca.fabs(T_mean), CASADI_PHYSICS_EPSILON)

    # Create function
    profile_fn = ca.Function(
        "torque_profile",
        [theta_vec, F_vec, r, l, x_off, y_off, pressure_angle],
        [T_vec, T_avg, T_max, T_min, ripple],
        ["theta_vec", "F_vec", "r", "l", "x_off", "y_off", "pressure_angle"],
        ["T_vec", "T_avg", "T_max", "T_min", "ripple"],
    )

    log.info("Created torque_profile_fixed CasADi function")
    return profile_fn


def create_torque_profile() -> ca.Function:
    """
    Create CasADi function for torque profile with statistics over full cycle.
    
    This is an alias for create_torque_profile_fixed() for backward compatibility.
    For variable-length inputs, use the chunked wrapper in unified.py.
    """
    return create_torque_profile_fixed()


def torque_profile_chunked_wrapper(theta_vec, F_vec, r, l, x_off, y_off, pressure_angle):
    """
    Python wrapper for torque profile that handles variable-length inputs via chunking.
    
    This function processes arbitrary-length input vectors by chunking them into
    fixed-size blocks (n=10) and aggregating the results across chunks.
    
    Parameters
    ----------
    theta_vec : array-like
        Crank angles (rad), arbitrary length n
    F_vec : array-like
        Piston forces (N), same length as theta_vec
    r : float
        Crank radius (mm)
    l : float
        Connecting rod length (mm)
    x_off : float
        Crank center x-offset (mm)
    y_off : float
        Crank center y-offset (mm)
    pressure_angle : float
        Gear pressure angle (radians)
    
    Returns
    -------
    tuple
        (T_vec, T_avg, T_max, T_min, ripple) where:
        - T_vec: Torque at each angle (N⋅m)
        - T_avg: Cycle-averaged torque (N⋅m)
        - T_max: Maximum torque (N⋅m)
        - T_min: Minimum torque (N⋅m)
        - ripple: Torque variation coefficient (dimensionless)
    """
    import casadi as ca
    import numpy as np
    
    # Convert to numpy arrays if needed
    theta_vec = np.asarray(theta_vec)
    F_vec = np.asarray(F_vec)
    
    n = len(theta_vec)
    
    # Get the fixed-size function
    fixed_fn = create_torque_profile_fixed()
    
    if n == 10:
        # Direct use for length 10
        result = fixed_fn(theta_vec, F_vec, r, l, x_off, y_off, pressure_angle)
        return result
    elif n < 10:
        # Pad to length 10 with last value
        theta_padded = np.pad(theta_vec, (0, 10 - n), mode='edge')
        F_padded = np.pad(F_vec, (0, 10 - n), mode='edge')
        
        T_vec_full, T_avg, T_max, T_min, ripple = fixed_fn(theta_padded, F_padded, r, l, x_off, y_off, pressure_angle)
        T_vec = T_vec_full[:n]  # Extract only the relevant part
        return T_vec, T_avg, T_max, T_min, ripple
    else:
        # For longer vectors, we need proper chunking
        # Chunk into 10-element blocks and aggregate results
        chunk_size = 10
        n_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Initialize aggregated results
        T_vec_chunks = []
        T_avg_chunks = []
        T_max_chunks = []
        T_min_chunks = []
        ripple_chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, n)
            chunk_length = end_idx - start_idx
            
            # Extract chunk
            theta_chunk = theta_vec[start_idx:end_idx]
            F_chunk = F_vec[start_idx:end_idx]
            
            # Pad chunk to length 10 if necessary
            if chunk_length < chunk_size:
                theta_chunk = np.pad(theta_chunk, (0, chunk_size - chunk_length), mode='edge')
                F_chunk = np.pad(F_chunk, (0, chunk_size - chunk_length), mode='edge')
            
            # Process chunk
            T_vec_chunk, T_avg_chunk, T_max_chunk, T_min_chunk, ripple_chunk = fixed_fn(
                theta_chunk, F_chunk, r, l, x_off, y_off, pressure_angle
            )
            
            # Store results
            T_vec_chunks.append(T_vec_chunk[:chunk_length])  # Only keep relevant part
            T_avg_chunks.append(T_avg_chunk)
            T_max_chunks.append(T_max_chunk)
            T_min_chunks.append(T_min_chunk)
            ripple_chunks.append(ripple_chunk)
        
        # Aggregate results
        T_vec = np.concatenate(T_vec_chunks)
        
        # For statistics, we need to properly aggregate across chunks
        # This is a simplified approach - more sophisticated aggregation could be implemented
        T_avg = np.mean(T_avg_chunks)
        T_max = np.max(T_max_chunks)
        T_min = np.min(T_min_chunks)
        ripple = np.mean(ripple_chunks)
        
        # Convert to CasADi DM objects to match expected format
        import casadi as ca
        T_avg = ca.DM(T_avg)
        T_max = ca.DM(T_max)
        T_min = ca.DM(T_min)
        ripple = ca.DM(ripple)
        
        return T_vec, T_avg, T_max, T_min, ripple
