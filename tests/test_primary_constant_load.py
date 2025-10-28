import numpy as np

from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)


def test_primary_constant_load_profile():
    settings = UnifiedOptimizationSettings()
    settings.constant_load_value = 2.5
    settings.constant_temperature_K = 800.0

    fw = UnifiedOptimizationFramework(settings=settings)
    data = fw.optimize_cascaded(
        {
            "stroke": 20.0,
            "cycle_time": 1.0,
            "upstroke_duration_percent": 60.0,
            "zero_accel_duration_percent": 0.0,
            "motion_type": "minimum_jerk",
            "constant_temperature_K": 850.0,  # override via input
        },
    )

    assert data.primary_theta is not None
    assert data.primary_load_profile is not None
    assert len(data.primary_load_profile) == len(data.primary_theta)
    assert np.allclose(data.primary_load_profile, settings.constant_load_value)
    # Temperature persisted and overridable via input
    assert data.primary_constant_temperature_K == 850.0
