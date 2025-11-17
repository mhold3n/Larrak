from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import Mock, patch

from campro.litvin.config import GeometrySearchConfig
from campro.litvin.metrics import evaluate_order0_metrics_given_phi
from campro.litvin.motion import RadialSlotMotion
from campro.litvin.optimization import OptimResult, _order2_ipopt_optimization


def _test_config() -> GeometrySearchConfig:
    """Create test configuration."""
    return GeometrySearchConfig(
        ring_teeth_candidates=[60],
        planet_teeth_candidates=[30],
        pressure_angle_deg_bounds=[20.0, 25.0],
        addendum_factor_bounds=[0.9, 1.1],
        base_center_radius=45.0,
        samples_per_rev=8,  # Small for fast test
        motion=RadialSlotMotion(
            center_offset_fn=lambda th: 0.0,
            planet_angle_fn=lambda th: 2.0 * th,
        ),
    )


def _mock_metrics(feasible: bool = True, slip_integral: float = 0.5) -> Mock:
    """Create mock metrics result."""
    return Mock(
        feasible=feasible,
        slip_integral=slip_integral,
        contact_length=10.0,
    )


def test_litvin_hybrid_physics_objective() -> None:
    """Test hybrid CasADi + Python physics objective."""
    config = _test_config()

    with patch(
        "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
    ) as mock_metrics:
        mock_metrics.return_value = _mock_metrics()
        result = _order2_ipopt_optimization(config)

        assert isinstance(result, OptimResult)
        assert result.best_config is not None
        assert result.feasible is True


def test_litvin_physics_validation() -> None:
    """Test physics validation of optimized solution."""
    config = _test_config()

    with patch(
        "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
    ) as mock_metrics:
        mock_metrics.return_value = _mock_metrics(feasible=False, slip_integral=2.0)
        result = _order2_ipopt_optimization(config)

        assert isinstance(result, OptimResult)
        assert result.best_config is not None
        assert result.feasible is False


def test_litvin_physics_metrics_in_analysis() -> None:
    """Test physics metrics appear in analysis report."""
    config = _test_config()

    with patch(
        "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
    ) as mock_metrics:
        mock_metrics.return_value = _mock_metrics(slip_integral=0.3)
        result = _order2_ipopt_optimization(config)

        assert isinstance(result, OptimResult)
        assert result.best_config is not None
        assert result.feasible is True


def test_litvin_physics_objective_function_interface() -> None:
    """Test that physics objective function has correct interface."""
    from campro.litvin.config import PlanetSynthesisConfig

    config = PlanetSynthesisConfig(
        ring_teeth=60,
        planet_teeth=30,
        pressure_angle_deg=22.5,
        addendum_factor=1.0,
        base_center_radius=45.0,
        samples_per_rev=8,
        motion=RadialSlotMotion(
            center_offset_fn=lambda th: 0.0,
            planet_angle_fn=lambda th: 2.0 * th,
        ),
    )

    test_phi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    result = evaluate_order0_metrics_given_phi(config, test_phi)

    assert hasattr(result, "feasible")
    assert hasattr(result, "slip_integral")
    assert hasattr(result, "contact_length")
    assert isinstance(result.feasible, bool)
    assert isinstance(result.slip_integral, (int, float))
    assert isinstance(result.contact_length, (int, float))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
