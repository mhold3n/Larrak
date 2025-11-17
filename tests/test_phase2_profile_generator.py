from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from numpy.typing import NDArray

from campro.physics.cam_ring_mapping import CamRingMapper, CamRingParameters


def _theta_grid(count: int = 181) -> NDArray[np.float64]:
    return np.linspace(0.0, 360.0, count)


def _cycloid_profile(theta_deg: NDArray[np.float64], stroke: float) -> NDArray[np.float64]:
    theta_rad = np.deg2rad(theta_deg)
    return (0.5 * stroke * (1.0 - np.cos(theta_rad))).astype(np.float64)


def test_cycloid_motion_generates_sinusoidal_ring_profile() -> None:
    theta = _theta_grid()
    stroke = 12.0
    x_theta = _cycloid_profile(theta, stroke)

    mapper = CamRingMapper(
        CamRingParameters(
            base_radius=25.0,
            connecting_rod_length=55.0,
        ),
    )
    ring_design: dict[str, float | str] = {
        "design_type": "sinusoidal",
        "base_radius": 40.0,
        "amplitude": 4.0,
        "frequency": 2.0,
    }

    result = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)

    R_psi = result["R_psi"]
    assert np.all(np.isfinite(R_psi))
    base_radius: float = float(ring_design["base_radius"])
    # Relax tolerance to account for numerical integration errors in meshing law solution
    assert np.isclose(R_psi.mean(), base_radius, atol=2e-2)
    span = R_psi.max() - R_psi.min()
    amplitude: float = float(ring_design["amplitude"])
    assert span == pytest.approx(2 * amplitude, rel=0.2)

    psi = result["psi"]
    assert np.isclose(psi.min(), 0.0, atol=1e-6)
    assert np.isclose(psi.max(), 2 * np.pi, atol=1e-6)
    assert np.all(np.diff(psi) >= 0.0)


def test_ring_profile_handles_increased_crank_center_offset() -> None:
    theta = _theta_grid()
    base_profile = _cycloid_profile(theta, stroke=10.0)
    scaled_profile = 1.5 * base_profile

    ring_design: dict[str, float | str] = {"design_type": "constant", "base_radius": 35.0}

    mapper_nominal = CamRingMapper(CamRingParameters(base_radius=30.0, connecting_rod_length=45.0))
    nominal = mapper_nominal.map_linear_to_ring_follower(theta, base_profile, ring_design)

    mapper_scaled = CamRingMapper(CamRingParameters(base_radius=30.0, connecting_rod_length=45.0))
    scaled = mapper_scaled.map_linear_to_ring_follower(theta, scaled_profile, ring_design)

    pitch_nominal = nominal["cam_curves"]["pitch_radius"]
    pitch_scaled = scaled["cam_curves"]["pitch_radius"]

    assert pitch_scaled.max() > pitch_nominal.max()
    assert np.min(pitch_scaled) >= mapper_scaled.parameters.base_radius - 1e-6
    assert scaled["x_theta"].max() == pytest.approx(scaled_profile.max(), rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
