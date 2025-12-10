from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from campro.physics import CamRingParameters  # noqa: F401 (API presence check)
from campro.physics.geometry.litvin import LitvinSynthesis, LitvinSynthesisResult


def _make_simple_cam_profile(n: int = 360) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r0 = 30.0
    amplitude = 2.0
    r_profile = r0 + amplitude * np.sin(theta)
    return theta, r_profile


def test_litvin_conjugacy_identity_ratio():
    theta, r_profile = _make_simple_cam_profile(720)
    synthesis = LitvinSynthesis()
    result: LitvinSynthesisResult = synthesis.synthesize_from_cam_profile(
        theta=theta,
        r_profile=r_profile,
        target_ratio=1.0,
    )

    assert result.psi.shape == theta.shape
    assert result.R_psi.shape == theta.shape

    # Conjugacy check: dpsi/dtheta ≈ rho_c / R(psi) with periodic boundaries handled
    dpsi_dtheta = np.gradient(result.psi, theta)
    mask = (theta > 1e-6) & (theta < (2.0 * np.pi - 1e-6))
    ratio = np.divide(result.rho_c[mask], result.R_psi[mask])
    assert np.allclose(dpsi_dtheta[mask], ratio, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 3.0])
def test_litvin_conjugacy_general_ratio(ratio: float):
    theta, r_profile = _make_simple_cam_profile(360)
    synthesis = LitvinSynthesis()
    result: LitvinSynthesisResult = synthesis.synthesize_from_cam_profile(
        theta=theta,
        r_profile=r_profile,
        target_ratio=ratio,
    )

    # Monotonic and periodic mapping over one cycle
    assert np.all(np.diff(result.psi) > 0.0)
    span = result.psi[-1] - result.psi[0]
    assert np.isclose(span, 2.0 * np.pi, rtol=1e-3, atol=1e-3)

    # Conjugacy law holds numerically
    dpsi_dtheta = np.gradient(result.psi, theta)
    mask = (theta > 1e-3) & (theta < (2.0 * np.pi - 1e-3))
    lhs = dpsi_dtheta[mask]
    rhs = np.divide(result.rho_c[mask], result.R_psi[mask])
    assert np.allclose(lhs, rhs, rtol=2e-2, atol=2e-2)


def test_api_types_and_metadata():
    theta, r_profile = _make_simple_cam_profile(360)
    synthesis = LitvinSynthesis()
    res = synthesis.synthesize_from_cam_profile(
        theta=theta, r_profile=r_profile, target_ratio=1.5,
    )
    assert isinstance(res.psi, np.ndarray)
    assert isinstance(res.R_psi, np.ndarray)
    assert isinstance(res.rho_c, np.ndarray)
    assert res.metadata["method"] == "litvin"
    assert "normalized_ratio" in res.metadata
