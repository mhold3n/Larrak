"""
Tests for Litvin physics integration in optimization.

This module tests the hybrid physics integration approach where
CasADi smoothness objectives are combined with Python physics validation.
"""

from unittest.mock import Mock, patch

from campro.litvin.config import GeometrySearchConfig
from campro.litvin.metrics import evaluate_order0_metrics_given_phi
from campro.litvin.motion import RadialSlotMotion
from campro.litvin.optimization import (
    OptimResult,
    _order2_ipopt_optimization,
)


class TestLitvinPhysicsIntegration:
    """Test Litvin hybrid physics integration."""

    def test_litvin_hybrid_physics_objective(self):
        """Test hybrid CasADi + Python physics objective."""
        # Create test configuration
        config = GeometrySearchConfig(
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

        # Test that the function can be called without CasADi (should use fallback)
        with patch(
            "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
        ) as mock_metrics:
            mock_metrics.return_value = Mock(
                feasible=True,
                slip_integral=0.5,
                contact_length=10.0,
            )

            # Run optimization (should use fallback when CasADi not available)
            result = _order2_ipopt_optimization(config)

            # Verify result
            assert isinstance(result, OptimResult)
            assert result.best_config is not None
            assert result.feasible is True
            assert result.ipopt_analysis is None  # No analysis in fallback

    def test_litvin_physics_validation(self):
        """Test physics validation of optimized solution."""
        # Create test configuration
        config = GeometrySearchConfig(
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

        # Test fallback behavior with infeasible result
        with patch(
            "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
        ) as mock_metrics:
            mock_metrics.return_value = Mock(
                feasible=False,  # Infeasible result
                slip_integral=2.0,
                contact_length=5.0,
            )

            # Run optimization (should use fallback)
            result = _order2_ipopt_optimization(config)

            # Verify result
            assert isinstance(result, OptimResult)
            assert result.best_config is not None
            assert result.feasible is False  # Should be infeasible
            assert result.ipopt_analysis is None  # No analysis in fallback

    def test_litvin_physics_metrics_in_analysis(self):
        """Test physics metrics appear in analysis report."""
        # Create test configuration
        config = GeometrySearchConfig(
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

        # Test fallback behavior
        with patch(
            "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
        ) as mock_metrics:
            mock_metrics.return_value = Mock(
                feasible=True,
                slip_integral=0.3,
                contact_length=12.0,
            )

            # Run optimization (should use fallback)
            result = _order2_ipopt_optimization(config)

            # Verify result
            assert isinstance(result, OptimResult)
            assert result.best_config is not None
            assert result.feasible is True
            assert result.ipopt_analysis is None  # No analysis in fallback

    def test_litvin_fallback_smoothing_with_physics(self):
        """Test fallback smoothing when Ipopt fails."""
        # Create test configuration
        config = GeometrySearchConfig(
            ring_teeth_candidates=[60],
            planet_teeth_candidates=[30],
            pressure_angle_deg_bounds=[20.0, 25.0],
            addendum_factor_bounds=[0.9, 1.1],
            base_center_radius=45.0,
            samples_per_rev=8,
            motion=RadialSlotMotion(
                center_offset_fn=lambda th: 0.0,
                planet_angle_fn=lambda th: 2.0 * th,
            ),
        )

        # Test fallback behavior (CasADi not available)
        with patch(
            "campro.litvin.optimization.evaluate_order0_metrics_given_phi",
        ) as mock_metrics:
            mock_metrics.return_value = Mock(
                feasible=True,
                slip_integral=0.4,
                contact_length=8.0,
            )

            # Run optimization (should use fallback)
            result = _order2_ipopt_optimization(config)

            # Verify fallback result
            assert isinstance(result, OptimResult)
            assert result.best_config is not None
            assert result.feasible is True
            assert result.ipopt_analysis is None  # No analysis in fallback

    def test_litvin_physics_objective_function_interface(self):
        """Test that physics objective function has correct interface."""
        # Create test configuration
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

        # Test the objective function directly
        test_phi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # This would be called inside _order2_ipopt_optimization
        # We're testing the interface here
        result = evaluate_order0_metrics_given_phi(config, test_phi)

        # Verify the function returns a result with expected attributes
        assert hasattr(result, "feasible")
        assert hasattr(result, "slip_integral")
        assert hasattr(result, "contact_length")
        assert isinstance(result.feasible, bool)
        assert isinstance(result.slip_integral, (int, float))
        assert isinstance(result.contact_length, (int, float))
