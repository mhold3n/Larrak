"""
CasADi physics module exports.

This module provides access to all CasADi-based physics functions for
symbolic optimization with automatic differentiation support.
"""

from .forces import create_piston_force_simple
from .kinematics import create_crank_piston_kinematics, create_crank_piston_kinematics_vectorized, create_phase_masks
from .litvin import (
    create_internal_flank_sampler,
    create_planet_transform,
    create_contact_phi_solver,
    create_litvin_metrics,
)
from .side_loading import (
    create_side_load_pointwise,
    create_side_load_profile,
    create_side_load_penalty,
)
from .torque import create_torque_pointwise, create_torque_profile, torque_profile_chunked_wrapper
from .unified import create_unified_physics, create_unified_physics_chunked, create_toy_nlp_optimizer

__all__ = [
    # Kinematics
    "create_crank_piston_kinematics",
    "create_crank_piston_kinematics_vectorized",
    "create_phase_masks",

    # Forces
    "create_piston_force_simple",

    # Torque
    "create_torque_pointwise",
    "create_torque_profile",
    "torque_profile_chunked_wrapper",

    # Side Loading
    "create_side_load_pointwise",
    "create_side_load_profile",
    "create_side_load_penalty",

    # Litvin Metrics
    "create_internal_flank_sampler",
    "create_planet_transform",
    "create_contact_phi_solver",
    "create_litvin_metrics",

    # Unified Physics
    "create_unified_physics",
    "create_unified_physics_chunked",
    "create_toy_nlp_optimizer",
]
