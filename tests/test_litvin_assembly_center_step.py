import numpy as np

from campro.physics.kinematics.litvin_assembly import (
    AssemblyInputs,
    compute_assembly_state,
)


def test_center_stepping_and_no_slip():
    # Synthetic inputs
    N = 360
    psi = np.linspace(0.0, 2.0 * np.pi, N)
    rb_cam = 20.0
    rb_ring = 60.0
    z_cam = 24
    contact_type = "internal"
    R_psi = 40.0 + 5.0 * np.sin(psi)  # arbitrary ring radius variation

    # Motion law: θ in degrees, x(θ) in mm
    theta_deg = np.linspace(0.0, 360.0, N)
    x_theta = 3.0 + 2.0 * np.sin(np.deg2rad(theta_deg))  # ≥0

    C0 = 50.0

    inputs = AssemblyInputs(
        base_circle_cam=rb_cam,
        base_circle_ring=rb_ring,
        z_cam=z_cam,
        contact_type=contact_type,
        psi=psi,
        R_psi=R_psi,
        theta_cam_rad=np.deg2rad(theta_deg),
        center_base_radius=C0,
        motion_theta_deg=theta_deg,
        motion_offset_mm=x_theta,
    )

    state = compute_assembly_state(inputs, ring_omega=1.0)

    # Center stepping matches C0 + resampled x(θ) (here grids already aligned)
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
