import numpy as np

from campro.physics.kinematics.phase2_relationships import (
    Phase2AnimationInputs,
    build_phase2_relationships,
)


def test_phase2_relationships_center_stepping_and_no_slip():
    # Synthetic inputs with matching grid lengths to avoid resampling complexity
    N = 360
    psi = np.linspace(0.0, 2.0 * np.pi, N)
    rb_cam = 20.0
    rb_ring = 60.0
    z_cam = 24
    contact_type = "internal"
    R_psi = 40.0 + 5.0 * np.sin(psi)  # arbitrary ring radius variation

    # Motion law: θ in degrees, x(θ) in mm (non-negative)
    theta_deg = np.linspace(0.0, 360.0, N)
    x_theta = 3.0 + 2.0 * np.sin(np.deg2rad(theta_deg))

    # Base planet center radius (C0)
    C0 = 50.0

    # Gear geometry bundle (bases and tooth count)
    gear_geometry = {
        "base_circle_cam": rb_cam,
        "base_circle_ring": rb_ring,
        "z_cam": z_cam,
    }

    inputs = Phase2AnimationInputs(
        theta_deg=theta_deg,
        x_theta_mm=x_theta,
        base_radius_mm=C0,
        psi_rad=psi,
        R_psi_mm=R_psi,
        gear_geometry=gear_geometry,
        contact_type=contact_type,
    )

    state = build_phase2_relationships(inputs)

    # Center stepping matches C0 + x(θ) (here grids already aligned)
    assert np.allclose(state.planet_center_radius, C0 + x_theta, atol=1e-9)

    # Angular synchronization
    assert np.allclose(state.planet_center_angle, psi, atol=1e-12)

    # No-slip check at base circles: rb_cam dφ/dt ≈ sign * rb_ring ω with ω=1 and dt = dψ/ω
    dphi = np.gradient(state.planet_spin_angle)
    dpsi = np.gradient(psi)
    dt = dpsi / 1.0
    dphi_dt = dphi / np.maximum(dt, 1e-9)
    sign = -1.0 if contact_type == "internal" else 1.0
    residual = rb_cam * dphi_dt - sign * rb_ring * 1.0
    rms = np.sqrt(np.mean(residual**2))
    assert rms < 1e-6
