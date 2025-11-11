"""Unit tests for gain-scheduled base pressure mapping."""

from __future__ import annotations

import numpy as np
import pytest

from campro.physics.simple_cycle_adapter import SimpleCycleAdapter, WiebeParams


class TestGainScheduler:
    """Test gain-scheduled base pressure mapping."""

    def test_empty_table_fallback(self):
        """Test fallback to default alpha/beta when table is empty."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        assert abs(alpha - 30.0) < 0.01
        assert abs(beta - 100.0) < 0.01

    def test_single_point_constant_return(self):
        """Test that single point returns constant value."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        adapter._update_gain_table(1.0, 5e-4, 40.0, 120.0)
        alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        assert abs(alpha - 40.0) < 0.01
        assert abs(beta - 120.0) < 0.01

    def test_2x2_grid_bilinear_interpolation(self):
        """Test bilinear interpolation on 2x2 grid."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        # Create 2x2 grid
        adapter._update_gain_table(0.8, 4e-4, 30.0, 100.0)  # Corner 1
        adapter._update_gain_table(1.2, 4e-4, 40.0, 110.0)  # Corner 2
        adapter._update_gain_table(0.8, 6e-4, 35.0, 105.0)  # Corner 3
        adapter._update_gain_table(1.2, 6e-4, 45.0, 115.0)  # Corner 4
        
        # Test at grid center (should be average)
        alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        # Expected: bilinear interpolation of corners
        # phi: 0.8 -> 1.0 (midpoint between 0.8 and 1.2)
        # fuel: 4e-4 -> 5e-4 (midpoint between 4e-4 and 6e-4)
        # Top edge: (30+40)/2 = 35, (100+110)/2 = 105
        # Bottom edge: (35+45)/2 = 40, (105+115)/2 = 110
        # Center: (35+40)/2 = 37.5, (105+110)/2 = 107.5
        expected_alpha = 37.5
        expected_beta = 107.5
        assert abs(alpha - expected_alpha) < 0.1
        assert abs(beta - expected_beta) < 0.1

    def test_point_outside_grid_clamping(self):
        """Test that points outside grid are clamped to bounds."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        # Create a proper 2x2 grid for interpolation
        adapter._update_gain_table(0.8, 4e-4, 30.0, 100.0)
        adapter._update_gain_table(1.2, 4e-4, 40.0, 110.0)
        adapter._update_gain_table(0.8, 6e-4, 35.0, 105.0)
        adapter._update_gain_table(1.2, 6e-4, 45.0, 115.0)
        
        # Point below grid bounds - should clamp to (0.8, 4e-4)
        alpha, beta = adapter._get_scheduled_base_pressure(0.5, 1e-4)
        # Clamped to phi=0.8, fuel=4e-4, which should return corner value
        assert abs(alpha - 30.0) < 0.1
        assert abs(beta - 100.0) < 0.1
        
        # Point above grid bounds - should clamp to (1.2, 6e-4)
        alpha, beta = adapter._get_scheduled_base_pressure(2.0, 1e-3)
        # Clamped to phi=1.2, fuel=6e-4, which should return corner value
        assert abs(alpha - 45.0) < 0.1
        assert abs(beta - 115.0) < 0.1

    def test_reset_clears_table(self):
        """Test that reset clears the gain table."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        adapter._update_gain_table(1.0, 5e-4, 40.0, 120.0)
        assert len(adapter._gain_table) == 1
        
        adapter.reset_gain_table()
        assert len(adapter._gain_table) == 0
        
        # Should fall back to default after reset
        alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        assert abs(alpha - 30.0) < 0.01
        assert abs(beta - 100.0) < 0.01

    def test_missing_corner_fallback(self):
        """Test fallback when corner is missing from grid."""
        adapter = SimpleCycleAdapter(
            wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
            alpha_fuel_to_base=30.0,
            beta_base=100.0,
        )
        # Create incomplete grid (missing one corner)
        adapter._update_gain_table(0.8, 4e-4, 30.0, 100.0)
        adapter._update_gain_table(1.2, 4e-4, 40.0, 110.0)
        adapter._update_gain_table(0.8, 6e-4, 35.0, 105.0)
        # Missing (1.2, 6e-4)
        
        # Interpolation at center should fallback
        alpha, beta = adapter._get_scheduled_base_pressure(1.0, 5e-4)
        assert abs(alpha - 30.0) < 0.01  # Should fallback to default
        assert abs(beta - 100.0) < 0.01

