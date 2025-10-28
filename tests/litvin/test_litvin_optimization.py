from CamPro_LitvinPlanetary import (
    GeometrySearchConfig,
    OptimizationOrder,
    RadialSlotMotion,
    optimize_geometry,
)


def test_order0_returns_feasible_result():
    motion = RadialSlotMotion(
        center_offset_fn=lambda th: 0.0, planet_angle_fn=lambda th: 2.0 * th,
    )
    cfg = GeometrySearchConfig(
        ring_teeth_candidates=[60, 62],
        planet_teeth_candidates=[30, 31],
        pressure_angle_deg_bounds=(18.0, 22.0),
        addendum_factor_bounds=(0.9, 1.1),
        base_center_radius=30.0,
        samples_per_rev=120,
        motion=motion,
    )

    res = optimize_geometry(cfg, order=OptimizationOrder.ORDER0_EVALUATE)
    assert res.feasible is True
    assert res.best_config is not None
    assert res.objective_value is not None
