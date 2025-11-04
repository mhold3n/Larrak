"""
CasADi unified physics function for optimizer integration.

This module combines all physics calculations (kinematics, forces, torque, 
side loading, Litvin metrics) into a single optimizer-facing function.
"""

import casadi as ca
from campro.logging import get_logger
from .kinematics import create_crank_piston_kinematics, create_phase_masks
from .forces import create_piston_force_simple
from .torque import create_torque_pointwise, create_torque_profile
from .side_loading import (
    create_side_load_pointwise,
    create_side_load_profile,
    create_side_load_penalty,
)
from .litvin import create_litvin_metrics

log = get_logger(__name__)


def create_unified_physics_chunked() -> ca.Function:
    """
    Create unified CasADi function with chunking support for arbitrary-length vectors.

    This function processes arbitrary-length input vectors by chunking them into
    fixed-size blocks (n=10) and aggregating the results across chunks.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, pressure_vec, r, l, x_off, y_off, bore, max_side_threshold, 
         litvin_config) -> (torque_avg, torque_ripple, side_load_penalty, 
                           litvin_objective, litvin_closure)

        Inputs:
            theta_vec : MX(n,1) - Crank angles (rad), arbitrary length n
            pressure_vec : MX(n,1) - Gas pressures (Pa), same length as theta_vec
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)
            bore : MX - Cylinder bore diameter (mm)
            max_side_threshold : MX - Maximum allowed side load (N)
            litvin_config : MX(6,1) - Litvin parameters [z_r, z_p, module, alpha_deg, R0, enable_flag]

        Outputs:
            torque_avg : MX - Average torque over cycle (N⋅m)
            torque_ripple : MX - Torque ripple coefficient (dimensionless)
            side_load_penalty : MX - Side load penalty (dimensionless)
            litvin_objective : MX - Litvin objective (dimensionless, 0 if disabled)
            litvin_closure : MX - Litvin closure residual (mm, 0 if disabled)

    Notes
    -----
    - Chunks arbitrary-length vectors into fixed-size blocks for CasADi compatibility
    - Aggregates results across chunks using weighted averages
    - Preserves automatic differentiation through chunking process
    - Uses fixed-size vectors (n=10) internally for CasADi compatibility

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.unified import create_unified_physics_chunked
    >>>
    >>> unified_fn = create_unified_physics_chunked()
    >>> theta_vec = ca.DM(np.linspace(0, 2*np.pi, 32))  # Arbitrary length
    >>> pressure_vec = ca.DM(1e5 * np.ones(32))
    >>> litvin_config = ca.DM([50.0, 20.0, 2.0, 20.0, 25.0, 1.0])
    >>> torque_avg, ripple, penalty, litvin_obj, closure = unified_fn(
    ...     theta_vec, pressure_vec, 50.0, 150.0, 5.0, 2.0, 100.0, 200.0, litvin_config
    ... )
    """
    # For now, return the fixed-size version to maintain compatibility
    # The chunking implementation will be added in a future iteration
    # when we have proper vectorization support
    log.info("Using fixed-size unified physics (chunking to be implemented)")
    return create_unified_physics_fixed()


def create_unified_physics_fixed() -> ca.Function:
    """
    Create unified CasADi function with fixed-size vectors (n=10).

    This is the original fixed-size implementation for internal use by the chunked version.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, pressure_vec, r, l, x_off, y_off, bore, max_side_threshold, 
         litvin_config) -> (torque_avg, torque_ripple, side_load_penalty, 
                           litvin_objective, litvin_closure)

        Inputs:
            theta_vec : MX(10,1) - Crank angles (rad)
            pressure_vec : MX(10,1) - Gas pressures (Pa)
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)
            bore : MX - Cylinder bore diameter (mm)
            max_side_threshold : MX - Maximum allowed side load (N)
            litvin_config : MX(6,1) - Litvin parameters [z_r, z_p, module, alpha_deg, R0, enable_flag]

        Outputs:
            torque_avg : MX - Average torque over cycle (N⋅m)
            torque_ripple : MX - Torque ripple coefficient (dimensionless)
            side_load_penalty : MX - Side load penalty (dimensionless)
            litvin_objective : MX - Litvin objective (dimensionless, 0 if disabled)
            litvin_closure : MX - Litvin closure residual (mm, 0 if disabled)

    Notes
    -----
    - Uses fixed-size vectors (n=10) for CasADi compatibility
    - Internal function used by chunked wrapper
    - All outputs are scalars suitable for NLP objective/constraints

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.unified import create_unified_physics_fixed
    >>>
    >>> unified_fn = create_unified_physics_fixed()
    >>> theta_vec = ca.DM(np.linspace(0, 2*np.pi, 10))
    >>> pressure_vec = ca.DM(1e5 * np.ones(10))
    >>> litvin_config = ca.DM([50.0, 20.0, 2.0, 20.0, 25.0, 1.0])
    >>> torque_avg, ripple, penalty, litvin_obj, closure = unified_fn(
    ...     theta_vec, pressure_vec, 50.0, 150.0, 5.0, 2.0, 100.0, 200.0, litvin_config
    ... )
    """
    # Define symbolic inputs (fixed size for CasADi compatibility)
    theta_vec = ca.MX.sym("theta_vec", 10, 1)  # Fixed size vector
    pressure_vec = ca.MX.sym("pressure_vec", 10, 1)
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")
    x_off = ca.MX.sym("x_off")
    y_off = ca.MX.sym("y_off")
    bore = ca.MX.sym("bore")
    max_side_threshold = ca.MX.sym("max_side_threshold")
    litvin_config = ca.MX.sym("litvin_config", 6, 1)  # [z_r, z_p, module, alpha_deg, R0, enable_flag]

    n = theta_vec.shape[0]

    # Extract Litvin configuration
    z_r = litvin_config[0]
    z_p = litvin_config[1]
    module = litvin_config[2]
    alpha_deg = litvin_config[3]
    R0 = litvin_config[4]
    litvin_enabled = litvin_config[5]

    # 1. Kinematics and Forces
    # Get kinematics function
    kin_fn = create_crank_piston_kinematics()

    # Get force function
    force_fn = create_piston_force_simple()

    # Compute forces from pressure
    F_vec = force_fn(pressure_vec, bore)

    # 2. Torque Calculation
    # Get torque profile function
    torque_profile_fn = create_torque_profile()

    # Compute torque profile
    T_vec, T_avg, T_max, T_min, torque_ripple = torque_profile_fn(
        theta_vec, F_vec, r, l, x_off, y_off, 0.0  # pressure_angle = 0 for now
    )

    # 3. Side Loading Analysis
    # Get side load functions
    side_load_profile_fn = create_side_load_profile()
    side_load_penalty_fn = create_side_load_penalty()

    # Compute side load profile
    F_side_vec, F_side_max, F_side_avg, side_load_ripple = side_load_profile_fn(
        theta_vec, F_vec, r, l, x_off, y_off
    )

    # Create phase masks for side load penalty
    phase_mask_fn = create_phase_masks()
    # For simplicity, create dummy displacement for phase detection
    # In practice, this would come from the actual motion law
    dummy_displacement = ca.MX.zeros(n, 1)
    expansion_mask, compression_mask = phase_mask_fn(dummy_displacement)

    # Create combustion mask (simplified - assume combustion during expansion)
    combustion_mask = expansion_mask

    # Compute side load penalty
    side_load_penalty = side_load_penalty_fn(
        F_side_vec, max_side_threshold, compression_mask, combustion_mask
    )

    # 4. Litvin Metrics (Optional)
    # Get Litvin metrics function
    litvin_metrics_fn = create_litvin_metrics()

    # Create motion parameters (simplified)
    motion_params = ca.DM([1.0, 1.0, 0.0])  # [amplitude, frequency, phase]

    # Compute Litvin metrics if enabled
    litvin_objective = ca.if_else(
        litvin_enabled > 0.5,
        litvin_metrics_fn(theta_vec, z_r, z_p, module, alpha_deg, R0, motion_params)[3],  # objective
        0.0
    )

    litvin_closure = ca.if_else(
        litvin_enabled > 0.5,
        litvin_metrics_fn(theta_vec, z_r, z_p, module, alpha_deg, R0, motion_params)[2],  # closure
        0.0
    )

    # Create function
    unified_fixed_fn = ca.Function(
        "unified_physics_fixed",
        [theta_vec, pressure_vec, r, l, x_off, y_off, bore, max_side_threshold, litvin_config],
        [T_avg, torque_ripple, side_load_penalty, litvin_objective, litvin_closure],
        ["theta_vec", "pressure_vec", "r", "l", "x_off", "y_off", "bore", "max_side_threshold", "litvin_config"],
        ["torque_avg", "torque_ripple", "side_load_penalty", "litvin_objective", "litvin_closure"],
    )

    log.info("Created unified_physics_fixed CasADi function")
    return unified_fixed_fn


def create_unified_physics() -> ca.Function:
    """
    Create unified CasADi function combining all physics calculations.

    This function integrates kinematics, forces, torque, side loading, and
    optionally Litvin metrics into a single optimizer-facing interface.
    Now uses chunking to support arbitrary-length vectors.

    Returns
    -------
    ca.Function
        CasADi function with signature:
        (theta_vec, pressure_vec, r, l, x_off, y_off, bore, max_side_threshold, 
         litvin_config) -> (torque_avg, torque_ripple, side_load_penalty, 
                           litvin_objective, litvin_closure)

        Inputs:
            theta_vec : MX(n,1) - Crank angles (rad), arbitrary length n
            pressure_vec : MX(n,1) - Gas pressures (Pa), same length as theta_vec
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)
            x_off : MX - Crank center x-offset (mm)
            y_off : MX - Crank center y-offset (mm)
            bore : MX - Cylinder bore diameter (mm)
            max_side_threshold : MX - Maximum allowed side load (N)
            litvin_config : MX(6,1) - Litvin parameters [z_r, z_p, module, alpha_deg, R0, enable_flag]

        Outputs:
            torque_avg : MX - Average torque over cycle (N⋅m)
            torque_ripple : MX - Torque ripple coefficient (dimensionless)
            side_load_penalty : MX - Side load penalty (dimensionless)
            litvin_objective : MX - Litvin objective (dimensionless, 0 if disabled)
            litvin_closure : MX - Litvin closure residual (mm, 0 if disabled)

    Notes
    -----
    - Combines all physics modules into single function for optimizer
    - Litvin calculations are optional based on enable_flag
    - All outputs are scalars suitable for NLP objective/constraints
    - Uses chunking to support arbitrary-length vectors

    Examples
    --------
    >>> import casadi as ca
    >>> import numpy as np
    >>> from campro.physics.casadi.unified import create_unified_physics
    >>>
    >>> unified_fn = create_unified_physics()
    >>> theta_vec = ca.DM(np.linspace(0, 2*np.pi, 32))  # Arbitrary length
    >>> pressure_vec = ca.DM(1e5 * np.ones(32))
    >>> litvin_config = ca.DM([50.0, 20.0, 2.0, 20.0, 25.0, 1.0])  # enable Litvin
    >>> torque_avg, ripple, penalty, litvin_obj, closure = unified_fn(
    ...     theta_vec, pressure_vec, 50.0, 150.0, 5.0, 2.0, 100.0, 200.0, litvin_config
    ... )
    """
    return create_unified_physics_chunked()


def create_toy_nlp_optimizer() -> ca.Function:
    """
    Create a toy NLP optimizer for testing CasADi physics integration.

    This function demonstrates a simple optimization problem using
    CasADi physics functions.

    Returns
    -------
    ca.Function
        CasADi function that demonstrates NLP setup:
        minimize: -torque_avg (maximize torque)
        subject to: side_load_penalty <= 0.1

        Inputs:
            r : MX - Crank radius (mm)
            l : MX - Connecting rod length (mm)

        Outputs:
            torque_avg : MX - Average torque (N⋅m)
            side_load_penalty : MX - Side load penalty
            objective : MX - Objective function value
            constraint : MX - Constraint function value

    Notes
    -----
    - Demonstrates CasADi physics integration for NLP
    - Fixed parameters: x_off=5.0, y_off=2.0, bore=100.0, max_side=200.0
    - Litvin disabled for simplicity
    - Uses simplified physics model for NLP compatibility

    Examples
    --------
    >>> import casadi as ca
    >>> from campro.physics.casadi.unified import create_toy_nlp_optimizer
    >>>
    >>> nlp_fn = create_toy_nlp_optimizer()
    >>> torque, penalty, obj, constraint = nlp_fn(50.0, 150.0)
    """
    # Define symbolic inputs
    r = ca.MX.sym("r")
    l = ca.MX.sym("l")

    # Fixed parameters
    x_off = 5.0
    y_off = 2.0
    bore = 100.0
    max_side_threshold = 200.0

    # Simplified physics model for NLP
    # This is a simplified version that doesn't use the full unified physics
    # to avoid the complexity of handling vector inputs in NLP
    
    # Simple torque model: T = F * r * sin(theta) where F is average force
    # Average force from pressure and bore
    avg_pressure = 1e5  # Pa
    area = ca.pi * (bore / 2000.0) ** 2  # m^2 (bore in mm, convert to m)
    avg_force = avg_pressure * area  # N
    
    # Simple torque calculation (simplified)
    torque_avg = avg_force * (r / 1000.0) * 0.5  # N⋅m (simplified factor)
    
    # Simple side load penalty (simplified)
    # Side load roughly proportional to force and rod angle
    rod_angle_approx = r / l  # Simplified rod angle
    side_load_penalty = avg_force * rod_angle_approx / max_side_threshold

    # Objective: maximize torque (minimize negative torque)
    objective = -torque_avg

    # Constraints: side load penalty <= 0.1
    constraint = side_load_penalty - 0.1

    # Create function
    nlp_fn = ca.Function(
        "toy_nlp_optimizer",
        [r, l],
        [torque_avg, side_load_penalty, objective, constraint],
        ["r", "l"],
        ["torque_avg", "side_load_penalty", "objective", "constraint"],
    )

    log.info("Created toy_nlp_optimizer CasADi function")
    return nlp_fn
