import numpy as np

from campro.physics.geometry.litvin import LitvinGearGeometry, LitvinSynthesis


def _simple_profile(n: int = 360):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r0 = 30.0
    x = 5.0 * (0.5 - 0.5 * np.cos(theta))
    r = r0 + x
    return theta, r


def test_flank_generation_and_pressure_angle_limits():
    theta, r = _simple_profile(720)
    syn = LitvinSynthesis().synthesize_from_cam_profile(theta=theta, r_profile=r, target_ratio=2.0)

    geom = LitvinGearGeometry.from_synthesis(
        theta=theta,
        r_profile=r,
        psi=syn.psi,
        R_psi=syn.R_psi,
        target_average_radius=float(np.mean(syn.R_psi)),
        module=2.0,
        max_pressure_angle_deg=35.0,
    )

    # Manufacturing tooth flanks present
    assert geom.flanks is not None
    assert "addendum" in geom.flanks
    assert "dedendum" in geom.flanks
    assert "fillet" in geom.flanks
    assert geom.flanks["addendum"].shape[1] == 2  # x,y pairs

    # Pressure angle enforcement within limit
    assert np.nanmax(np.abs(geom.pressure_angle_deg)) <= 35.0 + 1e-6

    # Interference/undercut detailed report present per flank
    assert isinstance(geom.undercut_flags, np.ndarray)
    assert geom.undercut_flags.shape[0] == geom.z_ring


