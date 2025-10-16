import math


def zero_motion(theta: float) -> float:
    return 0.0


def identity_double(theta: float) -> float:
    # θ_p(θ_r) = 2·θ_r
    return 2.0 * theta


def make_motion_class():
    # Import late to allow module to be missing initially (TDD)
    from CamPro_LitvinPlanetary import RadialSlotMotion  # type: ignore

    return RadialSlotMotion


def make_config_classes():
    from CamPro_LitvinPlanetary import PlanetSynthesisConfig  # type: ignore

    return PlanetSynthesisConfig


def test_baseline_c_zero_envelope_periodicity():
    # Arrange
    RadialSlotMotion = make_motion_class()
    PlanetSynthesisConfig = make_config_classes()

    motion = RadialSlotMotion(
        center_offset_fn=zero_motion,
        planet_angle_fn=identity_double,
        d_center_offset_fn=None,
        d2_center_offset_fn=None,
    )

    cfg = PlanetSynthesisConfig(
        ring_teeth=60,
        planet_teeth=30,
        pressure_angle_deg=20.0,
        addendum_factor=1.0,
        base_center_radius=30.0,  # R0 at TDC
        samples_per_rev=360,
        motion=motion,
    )

    # Act
    from CamPro_LitvinPlanetary import synthesize_planet_from_motion  # type: ignore

    profile = synthesize_planet_from_motion(cfg)

    # Assert basic shape properties
    assert hasattr(profile, "points"), "profile should expose points"
    assert len(profile.points) > 100, "profile should be well sampled"

    # Periodicity closure check (start ~ end within tolerance)
    x0, y0 = profile.points[0]
    x1, y1 = profile.points[-1]
    tol = 1e-2
    assert math.hypot(x1 - x0, y1 - y0) < tol


def test_small_sinusoidal_motion_produces_continuous_profile():
    RadialSlotMotion = make_motion_class()
    PlanetSynthesisConfig = make_config_classes()

    def c(theta: float) -> float:
        return 0.1 * math.sin(theta)

    motion = RadialSlotMotion(
        center_offset_fn=c,
        planet_angle_fn=identity_double,
        d_center_offset_fn=None,
        d2_center_offset_fn=None,
    )

    cfg = PlanetSynthesisConfig(
        ring_teeth=60,
        planet_teeth=30,
        pressure_angle_deg=20.0,
        addendum_factor=1.0,
        base_center_radius=30.0,
        samples_per_rev=720,
        motion=motion,
    )

    from CamPro_LitvinPlanetary import synthesize_planet_from_motion  # type: ignore

    profile = synthesize_planet_from_motion(cfg)

    # Local continuity: successive segments small
    diffs = [
        math.hypot(profile.points[i + 1][0] - profile.points[i][0],
                   profile.points[i + 1][1] - profile.points[i][1])
        for i in range(len(profile.points) - 1)
    ]
    assert max(diffs) < 2.0, "profile has unreasonable jumps"


