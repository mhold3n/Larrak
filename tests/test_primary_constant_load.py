from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from campro.optimization.base import OptimizationResult, OptimizationStatus
from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)


@contextmanager
def _patched_framework(flow_name: str, theta: np.ndarray):
    executed_flows: list[str] = []

    def fake_optimize_primary(self, initial_guess=None):
        executed_flows.append(flow_name)
        return OptimizationResult(
            solution={
                "cam_angle": theta,
                "position": np.sin(theta),
            },
            status=OptimizationStatus.CONVERGED,
            objective_value=0.0,
            solve_time=0.01,
            iterations=3,
            metadata={"flow": flow_name},
        )

    def fake_update_from_primary(self, result):
        self.data.primary_theta = result.solution["cam_angle"]
        self.data.primary_load_profile = np.full_like(
            self.data.primary_theta,
            fill_value=self.settings.constant_load_value,
            dtype=float,
        )
        self.data.primary_constant_temperature_K = self.settings.constant_temperature_K

    def fake_optimize_secondary(self):
        return OptimizationResult(status=OptimizationStatus.CONVERGED)

    def fake_update_from_secondary(self, result):
        self.data.secondary_theta = self.data.primary_theta

    def fake_optimize_tertiary(self):
        return OptimizationResult(status=OptimizationStatus.CONVERGED)

    def fake_update_from_tertiary(self, result):
        self.data.tertiary_theta = self.data.primary_theta

    patches = [
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
    ]

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        yield executed_flows


def _run_primary_flow(flow_name: str, flow_overrides: dict[str, bool]) -> None:
    """Execute a single primary-flow scenario using lightweight stubs."""
    settings = UnifiedOptimizationSettings()
    settings.constant_load_value = 2.5
    settings.constant_temperature_K = 800.0
    settings.max_iterations = 100
    settings.tolerance = 1e-6
    settings.enable_ipopt_analysis = False
    settings.verbose = False
    settings.universal_n_points = 90

    for attr, value in flow_overrides.items():
        setattr(settings, attr, value)

    framework = UnifiedOptimizationFramework(settings=settings)
    print(f"[{flow_name}] Settings overrides: {flow_overrides}")

    theta = np.linspace(0.0, 2.0 * np.pi, settings.universal_n_points, endpoint=False)

    with _patched_framework(flow_name, theta) as executed_flows:
        start_time = time.perf_counter()
        data = framework.optimize_cascaded(
            {
                "stroke": 20.0,
                "cycle_time": 1.0,
                "upstroke_duration_percent": 60.0,
                "zero_accel_duration_percent": 0.0,
                "motion_type": "minimum_jerk",
                "constant_temperature_K": 850.0,
            },
        )
        elapsed = time.perf_counter() - start_time
        from campro.utils import format_duration
        print(f"[{flow_name}] Cascaded optimization completed in {format_duration(elapsed)}")

    assert executed_flows == [flow_name], f"{flow_name} flow was not executed"
    assert data.primary_theta is not None
    assert data.primary_load_profile is not None
    assert len(data.primary_load_profile) == len(data.primary_theta)
    assert np.allclose(data.primary_load_profile, settings.constant_load_value)
    assert data.primary_constant_temperature_K == settings.constant_temperature_K


def test_primary_constant_load_profile():
    """Validate both thermal-efficiency and invariance flows sequentially."""
    flow_configs = [
        ("thermal_efficiency", {"use_thermal_efficiency": True}),
        ("pressure_invariance", {"use_pressure_invariance": True}),
    ]

    for flow_name, overrides in flow_configs:
        _run_primary_flow(flow_name, overrides)


def _main() -> None:
    """Allow running this module directly with python."""
    print("Running primary constant load flow validation via __main__")
    test_primary_constant_load_profile()


if __name__ == "__main__":
    _main()
