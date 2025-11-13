import numpy as np
import pytest

from campro.physics.geometry.litvin import (
    LitvinGearGeometry,
    LitvinSynthesis,
)


def _simple_motion_profile(n: int = 360):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    stroke = 20.0
    upfrac = 0.6
    up_end = int(n * upfrac)
    x = np.zeros_like(theta)
    # S-curve upstroke then mirror downstroke
    tau_up = np.linspace(0.0, 1.0, up_end, endpoint=False)
    x[:up_end] = stroke * (6 * tau_up**5 - 15 * tau_up**4 + 10 * tau_up**3)
    tau_dn = np.linspace(0.0, 1.0, n - up_end, endpoint=False)
    x[up_end:] = stroke * (1 - (6 * tau_dn**5 - 15 * tau_dn**4 + 10 * tau_dn**3))
    return theta, x


def test_gear_geometry_basic_metrics():
    theta, x_theta = _simple_motion_profile(720)
    base_radius = 30.0
    r_profile = base_radius + x_theta

    litvin = LitvinSynthesis()
    syn = litvin.synthesize_from_cam_profile(
        theta=theta, r_profile=r_profile, target_ratio=2.0,
    )

    geom = LitvinGearGeometry.from_synthesis(
        theta=theta,
        r_profile=r_profile,
        psi=syn.psi,
        R_psi=syn.R_psi,
        target_average_radius=np.mean(syn.R_psi),
        R_ring_profile=syn.R_psi,
    )

    # Base circles positive
    assert geom.base_circle_cam > 0.0
    assert geom.base_circle_ring > 0.0

    # Pressure angle bounded (typical involute bounds ~ [0°, 35°])
    assert np.nanmax(np.abs(geom.pressure_angle_deg)) < 60.0

    # Contact ratio positive
    assert geom.contact_ratio > 0.0

    # Path of contact spans a finite arc
    assert geom.path_of_contact_arc_length > 0.0


@pytest.mark.parametrize("ratio", [1.5, 2.0, 3.0])
def test_ratio_affects_base_circle_and_teeth(ratio: float):
    theta, x_theta = _simple_motion_profile(360)
    base_radius = 25.0
    r_profile = base_radius + x_theta

    litvin = LitvinSynthesis()
    syn = litvin.synthesize_from_cam_profile(
        theta=theta, r_profile=r_profile, target_ratio=ratio,
    )

    geom = LitvinGearGeometry.from_synthesis(
        theta=theta,
        r_profile=r_profile,
        psi=syn.psi,
        R_psi=syn.R_psi,
        target_average_radius=np.mean(syn.R_psi),
        R_ring_profile=syn.R_psi,
    )

    # Teeth counts must be integers and maintain ratio approximately
    assert geom.z_cam >= 8 and geom.z_ring >= 8
    approx_ratio = geom.z_ring / geom.z_cam
    assert np.isclose(approx_ratio, ratio, rtol=0.2)
