"""
Cam-ring processing functions for secondary optimization.

This module provides processing functions that can be used with the SecondaryOptimizer
to create circular follower (ring) designs based on linear follower motion laws
from primary optimization.
"""

from typing import Any, Dict, Optional

import numpy as np

from campro.logging import get_logger
from campro.physics.cam_ring_mapping import CamRingMapper, CamRingParameters

log = get_logger(__name__)


def process_linear_to_ring_follower(
    primary_data: Dict[str, np.ndarray],
    secondary_constraints: Dict[str, Any],
    secondary_relationships: Dict[str, Any],
    optimization_targets: Dict[str, Any],
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Process primary linear follower motion law to create ring follower design.

    This function takes the linear follower motion law from primary optimization
    and uses the cam-ring mapping framework to design the corresponding ring
    follower (circular follower) system.

    Parameters
    ----------
    primary_data : Dict[str, np.ndarray]
        Primary optimization results containing:
        - 'time': Time array
        - 'position': Linear follower position x(t)
        - 'velocity': Linear follower velocity
        - 'acceleration': Linear follower acceleration
        - 'control': Control input (jerk)
    secondary_constraints : Dict[str, Any]
        Constraints for ring design:
        - 'cam_parameters': CamRingParameters or dict
        - 'ring_design_type': Type of ring design
        - 'ring_design_params': Parameters for ring design
    secondary_relationships : Dict[str, Any]
        Relationships between primary and secondary optimization
    optimization_targets : Dict[str, Any]
        Specific targets for ring optimization
    **kwargs
        Additional parameters

    Returns
    -------
    Dict[str, np.ndarray]
        Ring follower design results including:
        - 'theta': Cam angles
        - 'x_theta': Linear follower displacement vs cam angle
        - 'psi': Ring angles
        - 'R_psi': Ring instantaneous radius
        - 'cam_curves': Cam geometry
        - 'validation': Design validation results
    """
    log.info("Processing linear follower motion law to ring follower design")

    # Extract primary motion law data
    time = primary_data.get("time", np.array([]))
    position = primary_data.get("position", np.array([]))
    velocity = primary_data.get("velocity", np.array([]))
    acceleration = primary_data.get("acceleration", np.array([]))
    cam_angle = primary_data.get("cam_angle", np.array([]))

    if len(time) == 0 or len(position) == 0:
        raise ValueError("Primary data must contain time and position arrays")

    # Use cam_angle if available, otherwise convert from time
    if len(cam_angle) > 0:
        theta = cam_angle  # Use provided cam angles (in degrees)
    else:
        # Convert time-based motion to cam angle-based motion
        cam_angular_velocity = secondary_constraints.get(
            "cam_angular_velocity", 1.0,
        )  # rad/s
        theta = cam_angular_velocity * time  # Cam angles

    # Linear follower displacement vs cam angle
    x_theta = position

    # Get cam-ring parameters
    cam_params = secondary_constraints.get("cam_parameters", {})
    if isinstance(cam_params, dict):
        cam_parameters = CamRingParameters(**cam_params)
    else:
        cam_parameters = cam_params

    # Get ring design parameters
    ring_design_type = secondary_constraints.get("ring_design_type", "constant")
    ring_design_params = secondary_constraints.get("ring_design_params", {})

    # Add optimization targets to ring design
    ring_design_params.update(optimization_targets)

    # Create cam-ring mapper
    mapper = CamRingMapper(cam_parameters)

    # Perform the mapping
    try:
        results = mapper.map_linear_to_ring_follower(
            theta,
            x_theta,
            ring_design={
                "design_type": ring_design_type,
                **ring_design_params,
            },
        )

        # Validate the design
        validation = mapper.validate_design(results)
        results["validation"] = validation

        # Add primary data for reference
        results["primary_time"] = time
        results["primary_position"] = position
        results["primary_velocity"] = velocity
        results["primary_acceleration"] = acceleration

        log.info("Successfully processed linear follower to ring follower design")
        return results

    except Exception as e:
        log.error(f"Failed to process linear to ring follower: {e}")
        raise


def process_ring_optimization(
    primary_data: Dict[str, np.ndarray],
    secondary_constraints: Dict[str, Any],
    secondary_relationships: Dict[str, Any],
    optimization_targets: Dict[str, Any],
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Optimize ring follower design based on primary motion law.

    This function performs optimization of the ring follower design parameters
    to achieve specific objectives while maintaining the linear follower motion law.

    Parameters
    ----------
    primary_data : Dict[str, np.ndarray]
        Primary optimization results
    secondary_constraints : Dict[str, Any]
        Constraints for ring optimization
    secondary_relationships : Dict[str, Any]
        Relationships between optimizations
    optimization_targets : Dict[str, Any]
        Optimization targets (e.g., minimize ring size, maximize efficiency)
    **kwargs
        Additional parameters

    Returns
    -------
    Dict[str, np.ndarray]
        Optimized ring follower design
    """
    log.info("Optimizing ring follower design")

    # Get optimization objective
    objective = optimization_targets.get("objective", "minimize_ring_size")

    # Get parameter bounds for optimization
    param_bounds = secondary_constraints.get("parameter_bounds", {})

    # For now, use the basic processing and add optimization logic
    # This is a placeholder for more sophisticated optimization
    results = process_linear_to_ring_follower(
        primary_data,
        secondary_constraints,
        secondary_relationships,
        optimization_targets,
        **kwargs,
    )

    # Add optimization metadata
    results["optimization_objective"] = objective
    results["parameter_bounds"] = param_bounds

    log.info(f"Completed ring optimization with objective: {objective}")
    return results


def process_multi_objective_ring_design(
    primary_data: Dict[str, np.ndarray],
    secondary_constraints: Dict[str, Any],
    secondary_relationships: Dict[str, Any],
    optimization_targets: Dict[str, Any],
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Multi-objective ring follower design optimization.

    Balances multiple objectives such as:
    - Minimize ring size
    - Maximize efficiency
    - Minimize stress
    - Maximize smoothness

    Parameters
    ----------
    primary_data : Dict[str, np.ndarray]
        Primary optimization results
    secondary_constraints : Dict[str, Any]
        Constraints for multi-objective optimization
    secondary_relationships : Dict[str, Any]
        Relationships between optimizations
    optimization_targets : Dict[str, Any]
        Multiple optimization targets with weights
    **kwargs
        Additional parameters

    Returns
    -------
    Dict[str, np.ndarray]
        Multi-objective optimized ring design
    """
    log.info("Performing multi-objective ring follower design")

    # Get objective weights
    weights = optimization_targets.get(
        "weights",
        {
            "ring_size": 0.3,
            "efficiency": 0.3,
            "smoothness": 0.2,
            "stress": 0.2,
        },
    )

    # Get multiple design alternatives
    design_alternatives = secondary_constraints.get(
        "design_alternatives",
        [
            {"design_type": "constant"},
            {"design_type": "linear"},
            {"design_type": "sinusoidal"},
        ],
    )

    best_result = None
    best_score = float("inf")

    # Evaluate each design alternative
    for i, design_params in enumerate(design_alternatives):
        try:
            # Create modified constraints for this alternative
            alt_constraints = secondary_constraints.copy()
            alt_constraints["ring_design_type"] = design_params["design_type"]
            alt_constraints["ring_design_params"] = design_params

            # Process this design
            result = process_linear_to_ring_follower(
                primary_data,
                alt_constraints,
                secondary_relationships,
                optimization_targets,
                **kwargs,
            )

            # Calculate multi-objective score
            score = _calculate_multi_objective_score(result, weights)
            result["multi_objective_score"] = score
            result["design_alternative"] = i

            # Keep track of best result
            if score < best_score:
                best_score = score
                best_result = result

        except Exception as e:
            log.warning(f"Failed to evaluate design alternative {i}: {e}")
            continue

    if best_result is None:
        raise RuntimeError("All design alternatives failed")

    log.info(f"Best multi-objective design has score: {best_score}")
    return best_result


def _calculate_multi_objective_score(
    result: Dict[str, np.ndarray], weights: Dict[str, float],
) -> float:
    """
    Calculate multi-objective score for ring design.

    Parameters
    ----------
    result : Dict[str, np.ndarray]
        Ring design result
    weights : Dict[str, float]
        Objective weights

    Returns
    -------
    float
        Combined multi-objective score (lower is better)
    """
    score = 0.0

    # Ring size objective (minimize)
    if "ring_size" in weights:
        R_psi = result["R_psi"]
        ring_size_score = np.mean(R_psi)  # Average ring radius
        score += weights["ring_size"] * ring_size_score

    # Efficiency objective (maximize smoothness of motion)
    if "efficiency" in weights:
        psi = result["psi"]
        # Measure smoothness as inverse of angular acceleration variation
        dpsi_dtheta = np.gradient(psi, result["theta"])
        d2psi_dtheta2 = np.gradient(dpsi_dtheta, result["theta"])
        efficiency_score = np.std(d2psi_dtheta2)  # Lower variation = higher efficiency
        score += weights["efficiency"] * efficiency_score

    # Smoothness objective (minimize curvature variation)
    if "smoothness" in weights:
        kappa_c = result["kappa_c"]
        smoothness_score = np.std(kappa_c)  # Lower variation = smoother
        score += weights["smoothness"] * smoothness_score

    # Stress objective (minimize high curvature regions)
    if "stress" in weights:
        kappa_c = result["kappa_c"]
        stress_score = np.max(np.abs(kappa_c))  # Peak stress
        score += weights["stress"] * stress_score

    return score


# Convenience functions for common processing scenarios


def create_constant_ring_design(
    primary_data: Dict[str, np.ndarray],
    ring_radius: float = 15.0,
    cam_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Create a constant radius ring design from linear follower motion.

    Parameters
    ----------
    primary_data : Dict[str, np.ndarray]
        Primary optimization results
    ring_radius : float
        Constant ring radius
    cam_parameters : Dict[str, Any], optional
        Cam-ring system parameters

    Returns
    -------
    Dict[str, np.ndarray]
        Constant ring design results
    """
    constraints = {
        "cam_parameters": cam_parameters or {},
        "ring_design_type": "constant",
        "ring_design_params": {"base_radius": ring_radius},
    }

    return process_linear_to_ring_follower(
        primary_data,
        constraints,
        {},
        {},
    )


def create_optimized_ring_design(
    primary_data: Dict[str, np.ndarray],
    optimization_objective: str = "minimize_ring_size",
    cam_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Create an optimized ring design from linear follower motion.

    Parameters
    ----------
    primary_data : Dict[str, np.ndarray]
        Primary optimization results
    optimization_objective : str
        Optimization objective
    cam_parameters : Dict[str, Any], optional
        Cam-ring system parameters

    Returns
    -------
    Dict[str, np.ndarray]
        Optimized ring design results
    """
    constraints = {
        "cam_parameters": cam_parameters or {},
        "ring_design_type": "constant",  # Will be optimized
    }

    targets = {
        "objective": optimization_objective,
    }

    return process_ring_optimization(
        primary_data,
        constraints,
        {},
        targets,
    )
