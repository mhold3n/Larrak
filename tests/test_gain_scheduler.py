from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from campro.physics.simple_cycle_adapter import SimpleCycleAdapter, WiebeParams


def _adapter() -> SimpleCycleAdapter:
    """Create default adapter instance."""
    return SimpleCycleAdapter(
        wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
        alpha_fuel_to_base=30.0,
        beta_base=100.0,
    )


def test_empty_table_fallback() -> None:
    """Test fallback to default alpha/beta when table is empty."""
    adapter = _adapter()
    alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
    assert abs(alpha - 30.0) < 0.01
    assert abs(beta - 100.0) < 0.01


def test_single_point_constant_return() -> None:
    """Test that single point returns constant value."""
    adapter = _adapter()
    adapter._update_gain_table(1.0, 5e-4, 40.0, 120.0)
    alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
    assert abs(alpha - 40.0) < 0.01
    assert abs(beta - 120.0) < 0.01


def test_bilinear_interpolation() -> None:
    """Test bilinear interpolation on 2x2 grid."""
    adapter = _adapter()
    # Create 2x2 grid
    adapter._update_gain_table(0.8, 4e-4, 30.0, 100.0)
    adapter._update_gain_table(1.2, 4e-4, 40.0, 110.0)
    adapter._update_gain_table(0.8, 6e-4, 35.0, 105.0)
    adapter._update_gain_table(1.2, 6e-4, 45.0, 115.0)

    # Test at grid center (should be average)
    alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
    expected_alpha = 37.5
    expected_beta = 107.5
    assert abs(alpha - expected_alpha) < 0.1
    assert abs(beta - expected_beta) < 0.1


def test_point_outside_grid_clamping() -> None:
    """Test that points outside grid are clamped to bounds."""
    adapter = _adapter()
    adapter._update_gain_table(0.8, 4e-4, 30.0, 100.0)
    adapter._update_gain_table(1.2, 4e-4, 40.0, 110.0)
    adapter._update_gain_table(0.8, 6e-4, 35.0, 105.0)
    adapter._update_gain_table(1.2, 6e-4, 45.0, 115.0)

    # Point below grid bounds
    alpha, beta = adapter._get_scheduled_base_pressure(0.5, 1e-4)
    assert abs(alpha - 30.0) < 0.1
    assert abs(beta - 100.0) < 0.1

    # Point above grid bounds
    alpha, beta = adapter._get_scheduled_base_pressure(2.0, 1e-3)
    assert abs(alpha - 45.0) < 0.1
    assert abs(beta - 115.0) < 0.1


def test_reset_clears_table() -> None:
    """Test that reset clears the gain table."""
    adapter = _adapter()
    adapter._update_gain_table(1.0, 5e-4, 40.0, 120.0)
    assert len(adapter._gain_table) == 1

    adapter.reset_gain_table()
    assert len(adapter._gain_table) == 0

    # Should fall back to default after reset
    alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
    assert abs(alpha - 30.0) < 0.01
    assert abs(beta - 100.0) < 0.01


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
