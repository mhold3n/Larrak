from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch

import numpy as np
import pytest
from campro.optimization.cam_ring_processing import (
    _calculate_multi_objective_score,
    create_constant_ring_design,
    process_linear_to_ring_follower,
)

from campro.physics.cam_ring_mapping import CamRingMapper, CamRingParameters


def _theta_grid(count: int = 100) -> np.ndarray:
    """Generate cam angle grid in radians."""
    return np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)


def _simple_motion(theta: np.ndarray) -> np.ndarray:
    """Generate simple sinusoidal motion."""
    return 5.0 * np.sin(theta)


def _mapper() -> CamRingMapper:
    """Create default CamRingMapper instance."""
    return CamRingMapper()


def test_cam_ring_parameters_defaults() -> None:
    """Test CamRingParameters default values."""
    params = CamRingParameters()
    assert params.base_radius == 10.0
    assert params.connecting_rod_length == 25.0
    assert params.contact_type == "external"


def test_cam_ring_parameters_custom() -> None:
    """Test CamRingParameters with custom values."""
    params = CamRingParameters(
        base_radius=15.0,
        connecting_rod_length=30.0,
        contact_type="internal",
    )
    assert params.base_radius == 15.0
    assert params.connecting_rod_length == 30.0
    assert params.contact_type == "internal"
    param_dict = params.to_dict()
    assert isinstance(param_dict, dict)
    assert param_dict["base_radius"] == 15.0


def test_mapper_initialization() -> None:
    """Test CamRingMapper initialization."""
    mapper = CamRingMapper()
    assert isinstance(mapper.parameters, CamRingParameters)

    custom_params = CamRingParameters(base_radius=20.0)
    mapper_custom = CamRingMapper(custom_params)
    assert mapper_custom.parameters.base_radius == 20.0


def test_compute_cam_curves() -> None:
    """Test cam curve computation."""
    theta = _theta_grid()
    x_theta = _simple_motion(theta)
    mapper = _mapper()

    curves = mapper.compute_cam_curves(theta, x_theta)

    required_keys = ["pitch_radius", "profile_radius", "contact_radius", "theta"]
    for key in required_keys:
        assert key in curves

    assert curves["pitch_radius"].shape == theta.shape
    expected_pitch = mapper.parameters.base_radius + x_theta
    np.testing.assert_array_almost_equal(curves["pitch_radius"], expected_pitch)


def test_compute_cam_curvature() -> None:
    """Test cam curvature computation."""
    theta = _theta_grid()
    x_theta = _simple_motion(theta)
    mapper = _mapper()
    curves = mapper.compute_cam_curves(theta, x_theta)

    kappa_c = mapper.compute_cam_curvature(theta, curves["contact_radius"])
    assert np.all(np.isfinite(kappa_c))
    assert kappa_c.shape == theta.shape


def test_compute_osculating_radius() -> None:
    """Test osculating radius computation."""
    mapper = _mapper()
    kappa_c = np.array([0.1, 0.2, 0.5, 1.0])
    rho_c = mapper.compute_osculating_radius(kappa_c)

    expected = 1.0 / kappa_c
    np.testing.assert_array_almost_equal(rho_c, expected)

    # Test with zero curvature
    kappa_c_zero = np.array([0.0, 0.1, 0.0])
    rho_c_zero = mapper.compute_osculating_radius(kappa_c_zero)
    assert rho_c_zero[0] == np.inf
    assert rho_c_zero[2] == np.inf


@pytest.mark.parametrize("design_type,expected_func", [
    ("constant", lambda psi: np.full_like(psi, 20.0)),
    ("linear", lambda psi: 15.0 + 2.0 * psi),
    ("sinusoidal", lambda psi: 15.0 + 3.0 * np.sin(2.0 * psi)),
])
def test_design_ring_radius(design_type: str, expected_func) -> None:
    """Test ring radius design for different types."""
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, 50)
    
    if design_type == "constant":
        R_psi = mapper.design_ring_radius(psi, design_type, base_radius=20.0)
    elif design_type == "linear":
        R_psi = mapper.design_ring_radius(psi, design_type, base_radius=15.0, slope=2.0)
    elif design_type == "sinusoidal":
        R_psi = mapper.design_ring_radius(
            psi, design_type, base_radius=15.0, amplitude=3.0, frequency=2.0,
        )

    expected = expected_func(psi)
    np.testing.assert_array_almost_equal(R_psi, expected)


def test_design_ring_radius_invalid_type() -> None:
    """Test invalid ring design type raises error."""
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, 10)

    with pytest.raises(ValueError, match="Unknown design type"):
        mapper.design_ring_radius(psi, "invalid_type")


def test_solve_meshing_law() -> None:
    """Test meshing law solution."""
    theta = _theta_grid()
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, len(theta))
    rho_c = np.ones_like(theta) * 5.0
    R_psi = np.ones_like(psi) * 10.0

    result_psi = mapper.solve_meshing_law(theta, rho_c, psi, R_psi)

    assert result_psi.shape == psi.shape
    assert np.all(np.isfinite(result_psi))
    assert not np.all(result_psi == 0)


def test_solve_meshing_law_failure_fallback() -> None:
    """Test meshing law solution fallback on failure."""
    theta = _theta_grid()
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, len(theta))
    rho_c = np.ones_like(theta) * 5.0
    R_psi = np.ones_like(psi) * 10.0

    with patch("scipy.integrate.solve_ivp") as mock_solve_ivp:
        mock_solve_ivp.side_effect = Exception("ODE solver failed")
        result_psi = mapper.solve_meshing_law(theta, rho_c, psi, R_psi)

        assert result_psi.shape == psi.shape
        assert np.all(np.isfinite(result_psi))
    assert result_psi[0] == psi[0]


@pytest.mark.parametrize("driver,omega_key,omega_val", [
    ("cam", "omega", 2.0),
    ("ring", "Omega", 1.5),
])
def test_compute_time_kinematics(driver: str, omega_key: str, omega_val: float) -> None:
    """Test time kinematics computation for different drivers."""
    theta = _theta_grid()
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, len(theta))
    rho_c = np.ones_like(theta) * 5.0
    R_psi = np.ones_like(psi) * 10.0

    kinematics = mapper.compute_time_kinematics(
        theta, psi, rho_c, R_psi, driver=driver, **{omega_key: omega_val},
    )

    assert "time" in kinematics
    assert "theta" in kinematics
    assert "psi" in kinematics
    assert kinematics["driver"] == driver


def test_compute_time_kinematics_invalid_driver() -> None:
    """Test time kinematics with invalid driver parameters."""
    theta = _theta_grid()
    mapper = _mapper()
    psi = np.linspace(0, 2.0 * np.pi, len(theta))
    rho_c = np.ones_like(theta) * 5.0
    R_psi = np.ones_like(psi) * 10.0

    with pytest.raises(ValueError, match="Must specify either omega"):
        mapper.compute_time_kinematics(theta, psi, rho_c, R_psi, driver="cam")


def test_map_linear_to_ring_follower() -> None:
    """Test complete linear-to-ring follower mapping."""
    theta = _theta_grid()
    x_theta = _simple_motion(theta)
    mapper = _mapper()

    ring_design = {
        "design_type": "constant",
        "base_radius": 15.0,
    }

    results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)

    required_keys = [
        "theta", "x_theta", "cam_curves", "kappa_c", "rho_c", "psi", "R_psi",
    ]
    for key in required_keys:
        assert key in results

    # Check array consistency
    assert len(results["theta"]) == len(results["x_theta"])
    assert len(results["psi"]) == len(results["R_psi"])


def test_validate_design() -> None:
    """Test design validation."""
    theta = _theta_grid()
    x_theta = _simple_motion(theta)
    mapper = _mapper()

    ring_design = {"design_type": "constant", "base_radius": 15.0}
    results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)
    validation = mapper.validate_design(results)

    assert isinstance(validation, dict)
    for value in validation.values():
        assert isinstance(value, bool)


def test_process_linear_to_ring_follower() -> None:
    """Test linear-to-ring follower processing."""
    primary_data = {
        "time": np.linspace(0, 2.0 * np.pi, 100),
        "position": 5.0 * np.sin(np.linspace(0, 2.0 * np.pi, 100)),
        "velocity": 5.0 * np.cos(np.linspace(0, 2.0 * np.pi, 100)),
        "acceleration": -5.0 * np.sin(np.linspace(0, 2.0 * np.pi, 100)),
    }
    
    constraints = {
        "cam_parameters": {"base_radius": 12.0},
        "ring_design_type": "constant",
        "ring_design_params": {"base_radius": 20.0},
    }

    results = process_linear_to_ring_follower(primary_data, constraints, {}, {})

    expected_keys = ["theta", "x_theta", "psi", "R_psi", "cam_curves", "validation"]
    for key in expected_keys:
        assert key in results


def test_calculate_multi_objective_score() -> None:
    """Test multi-objective score calculation."""
    result = {
        "R_psi": np.array([10.0, 12.0, 15.0]),
        "psi": np.array([0.0, 1.0, 2.0]),
        "theta": np.array([0.0, 0.5, 1.0]),
        "kappa_c": np.array([0.1, 0.2, 0.15]),
    }

    weights = {
        "ring_size": 0.5,
        "efficiency": 0.3,
        "smoothness": 0.2,
    }

    score = _calculate_multi_objective_score(result, weights)
    assert isinstance(score, float)
    assert score >= 0.0


def test_create_constant_ring_design() -> None:
    """Test constant ring design creation."""
    primary_data = {
        "time": np.linspace(0, 2.0 * np.pi, 100),
        "position": 5.0 * np.sin(np.linspace(0, 2.0 * np.pi, 100)),
    }
    
    results = create_constant_ring_design(primary_data, ring_radius=25.0)
    R_psi = results["R_psi"]
    assert np.allclose(R_psi, 25.0, rtol=1e-10)


def test_end_to_end_mapping() -> None:
    """Test complete end-to-end mapping from linear follower to ring design."""
    theta = np.linspace(0, 2.0 * np.pi, 200)
    x_theta = 8.0 * (1 - np.cos(theta))

    params = CamRingParameters(
        base_radius=15.0,
        connecting_rod_length=25.0,
        contact_type="external",
    )
    mapper = CamRingMapper(params)

    ring_design = {
        "design_type": "sinusoidal",
        "base_radius": 20.0,
        "amplitude": 3.0,
        "frequency": 1.0,
    }

    results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)
    validation = mapper.validate_design(results)

    assert validation["positive_cam_radii"]
    assert validation["positive_ring_radii"]
    assert validation["reasonable_curvature"]

    R_psi = results["R_psi"]
    assert np.all(R_psi > 0)
    assert np.min(R_psi) >= 17.0
    assert np.max(R_psi) <= 23.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
