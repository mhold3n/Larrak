
from campro.optimization.unified_framework import UnifiedOptimizationFramework


def test_primary_uses_360_points_and_assumptions():
    fw = UnifiedOptimizationFramework()
    data = fw.optimize_cascaded({
        "stroke": 20.0,
        "cycle_time": 1.0,
        "upstroke_duration_percent": 60.0,
        "zero_accel_duration_percent": 0.0,
        "motion_type": "minimum_jerk",
    })

    # Primary outputs present
    assert data.primary_theta is not None
    assert data.primary_position is not None

    # 360-point sampling expected in radians internally; stored degrees externally
    # We only verify count consistency with the chosen sampling
    assert len(data.primary_theta) == 360

    # Assumptions present in convergence info or accessible via framework summary
    summary = fw.get_optimization_summary()
    assert "primary" in summary["convergence_info"]


