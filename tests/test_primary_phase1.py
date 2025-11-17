from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import patch

import numpy as np

from campro.optimization.base import OptimizationResult, OptimizationStatus
from campro.optimization.unified_framework import UnifiedOptimizationFramework


def test_primary_uses_360_points_and_assumptions():
    fw = UnifiedOptimizationFramework()

    n_points = int(fw.settings.universal_n_points)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    position = np.sin(theta)

    def fake_optimize_primary(self, initial_guess=None):
        return OptimizationResult(
            solution={"cam_angle": theta, "position": position},
            status=OptimizationStatus.CONVERGED,
        )

    def fake_update_from_primary(self, result):
        self.data.primary_theta = result.solution["cam_angle"]
        self.data.primary_position = result.solution["position"]
        self.data.convergence_info["primary"] = {
            "assumptions": {"angular_sampling_points": len(theta)},
        }

    def fake_optimize_secondary(self):
        return OptimizationResult(status=OptimizationStatus.CONVERGED)

    def fake_update_from_secondary(self, result):
        self.data.secondary_theta = theta

    def fake_optimize_tertiary(self):
        return OptimizationResult(status=OptimizationStatus.CONVERGED)

    def fake_update_from_tertiary(self, result):
        self.data.tertiary_theta = theta

    with (
        patch.object(
            UnifiedOptimizationFramework,
            "_optimize_primary",
            fake_optimize_primary,
        ),
        patch.object(
            UnifiedOptimizationFramework,
            "_update_data_from_primary",
            fake_update_from_primary,
        ),
        patch.object(
            UnifiedOptimizationFramework,
            "_optimize_secondary",
            fake_optimize_secondary,
        ),
        patch.object(
            UnifiedOptimizationFramework,
            "_update_data_from_secondary",
            fake_update_from_secondary,
        ),
        patch.object(
            UnifiedOptimizationFramework,
            "_optimize_tertiary",
            fake_optimize_tertiary,
        ),
        patch.object(
            UnifiedOptimizationFramework,
            "_update_data_from_tertiary",
            fake_update_from_tertiary,
        ),
    ):
        data = fw.optimize_cascaded(
            {
                "stroke": 20.0,
                "cycle_time": 1.0,
                "upstroke_duration_percent": 60.0,
                "zero_accel_duration_percent": 0.0,
                "motion_type": "minimum_jerk",
            },
        )

    # Primary outputs present
    assert data.primary_theta is not None
    assert data.primary_position is not None

    # 360-point sampling expected in radians internally; stored degrees externally
    # We only verify count consistency with the chosen sampling
    assert len(data.primary_theta) == 360

    # Assumptions present in convergence info or accessible via framework summary
    summary = fw.get_optimization_summary()
    assert "primary" in summary["convergence_info"]


def _main() -> None:
    """Allow running this test module directly with Python."""
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    _main()
