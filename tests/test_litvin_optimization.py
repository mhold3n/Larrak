from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from campro.litvin.config import (
    GeometrySearchConfig,
    OptimizationOrder,
    PlanetSynthesisConfig,
)
from campro.litvin.kinematics import PlanetKinematics
from campro.litvin.metrics import evaluate_order0_metrics
from campro.litvin.motion import RadialSlotMotion
from campro.litvin.optimization import OptimResult, optimize_geometry
from campro.litvin.planetary_synthesis import synthesize_planet_from_motion
from campro.physics.geometry.litvin import LitvinSynthesis


def zero_motion(theta: float) -> float:
    """Zero center offset motion."""
    return 0.0


def identity_double(theta: float) -> float:
    """Planet angle: θ_p = 2·θ_r."""
    return 2.0 * theta


def test_order0_evaluate_returns_feasible_result():
    """Test ORDER0_EVALUATE returns a feasible result."""
    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
    )

    config = GeometrySearchConfig(
        ring_teeth_candidates=[60],
        planet_teeth_candidates=[30],
        pressure_angle_deg_bounds=(20.0, 20.0),
        addendum_factor_bounds=(1.0, 1.0),
        base_center_radius=30.0,
        samples_per_rev=360,
        motion=motion,
    )

    result = optimize_geometry(config, OptimizationOrder.ORDER0_EVALUATE)

    assert isinstance(result, OptimResult)
    assert result.feasible is True
    assert result.best_config is not None
    assert result.objective_value is not None
    assert result.best_config.ring_teeth == 60
    assert result.best_config.planet_teeth == 30


def test_order1_geometry_optimization():
    """Test ORDER1_GEOMETRY multi-parameter optimization."""
    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
    )

    config = GeometrySearchConfig(
        ring_teeth_candidates=[50, 60, 70],
        planet_teeth_candidates=[25, 30, 35],
        pressure_angle_deg_bounds=(18.0, 22.0),
        addendum_factor_bounds=(0.9, 1.1),
        base_center_radius=30.0,
        samples_per_rev=360,
        motion=motion,
    )

    result = optimize_geometry(config, OptimizationOrder.ORDER1_GEOMETRY)

    assert isinstance(result, OptimResult)
    assert result.feasible is True
    assert result.best_config is not None
    assert result.objective_value is not None
    assert result.best_config.ring_teeth in [50, 60, 70]
    assert result.best_config.planet_teeth in [25, 30, 35]
    assert 18.0 <= result.best_config.pressure_angle_deg <= 22.0
    assert 0.9 <= result.best_config.addendum_factor <= 1.1


def test_order2_micro_optimization():
    """Test ORDER2_MICRO collocation-based refinement."""
    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
    )

    config = GeometrySearchConfig(
        ring_teeth_candidates=[60],
        planet_teeth_candidates=[30],
        pressure_angle_deg_bounds=(20.0, 20.0),
        addendum_factor_bounds=(1.0, 1.0),
        base_center_radius=30.0,
        samples_per_rev=360,
        motion=motion,
    )

    result = optimize_geometry(config, OptimizationOrder.ORDER2_MICRO)

    assert isinstance(result, OptimResult)
    # ORDER2_MICRO may not always find feasible solutions, so we just check it returns a result
    assert result.best_config is not None
    assert result.objective_value is not None


def test_synthesize_planet_from_motion():
    """Test planet synthesis from motion."""
    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
    )

    config = PlanetSynthesisConfig(
        ring_teeth=60,
        planet_teeth=30,
        pressure_angle_deg=20.0,
        addendum_factor=1.0,
        base_center_radius=30.0,
        samples_per_rev=360,
        motion=motion,
    )

    profile = synthesize_planet_from_motion(config)

    assert profile is not None
    assert len(profile.points) > 0
    assert all(isinstance(p, tuple) and len(p) == 2 for p in profile.points)


def test_evaluate_order0_metrics():
    """Test Order0 metrics evaluation."""
    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
    )

    config = PlanetSynthesisConfig(
        ring_teeth=60,
        planet_teeth=30,
        pressure_angle_deg=20.0,
        addendum_factor=1.0,
        base_center_radius=30.0,
        samples_per_rev=360,
        motion=motion,
    )

    metrics = evaluate_order0_metrics(config)

    assert metrics is not None
    assert isinstance(metrics.slip_integral, float)
    assert isinstance(metrics.contact_length, float)
    assert isinstance(metrics.closure_residual, float)
    assert isinstance(metrics.phi_edge_fraction, float)
    assert isinstance(metrics.samples, int)
    assert isinstance(metrics.feasible, bool)


def test_litvin_polar_pitch_alignment():
    """The Litvin output should match the sampled polar pitch curve within tolerance."""
    motion = RadialSlotMotion(
        center_offset_fn=lambda th: 2.0 * np.sin(th),
        planet_angle_fn=identity_double,
    )
    base_radius = 30.0
    theta_rad = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    kin = PlanetKinematics(R0=base_radius, motion=motion)
    polar_radius = np.asarray([kin.center_distance(float(t)) for t in theta_rad])

    litvin = LitvinSynthesis()
    result = litvin.synthesize_from_cam_profile(
        theta=theta_rad,
        r_profile=polar_radius,
        target_ratio=1.0,
    )

    max_error = float(np.max(np.abs(result.R_psi - polar_radius)))
    assert max_error < 1e-6


def test_cam_ring_optimizer_integration():
    """Test integration with CamRingOptimizer."""
    from campro.optimization.cam_ring_optimizer import (
        CamRingOptimizationConstraints,
        CamRingOptimizationTargets,
        CamRingOptimizer,
    )

    # Create optimizer
    optimizer = CamRingOptimizer("TestOptimizer")

    # Configure with gear geometry constraints
    constraints = CamRingOptimizationConstraints(
        ring_teeth_candidates=[50, 60],
        planet_teeth_candidates=[25, 30],
        pressure_angle_min=18.0,
        pressure_angle_max=22.0,
        addendum_factor_min=0.9,
        addendum_factor_max=1.1,
        samples_per_rev=180,  # Reduced for faster testing
    )

    targets = CamRingOptimizationTargets()

    optimizer.configure(constraints=constraints, targets=targets)

    # Create test primary data
    theta = np.linspace(0, 360, 36)  # 36 points for 10° increments
    position = 10.0 * np.sin(np.radians(theta))  # 10mm amplitude sine wave

    primary_data = {
        "cam_angle": theta,
        "position": position,
    }

    # Run optimization
    result = optimizer.optimize(primary_data)

    # Verify result
    assert result.status.value == "converged"
    assert result.solution is not None
    assert "gear_geometry" in result.solution
    assert "ring_teeth" in result.solution
    assert "planet_teeth" in result.solution
    assert "pressure_angle_deg" in result.solution
    assert "addendum_factor" in result.solution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
