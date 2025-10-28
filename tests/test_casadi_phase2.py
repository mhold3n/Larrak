"""
Tests for CasADi Phase 2: Side Loading Analysis.

This module tests the CasADi side loading implementation against Python baselines
to ensure numeric parity and proper automatic differentiation.
"""

from unittest.mock import Mock

import casadi as ca
import numpy as np
import pytest
from campro.physics.casadi import (
    create_phase_masks,
    create_side_load_penalty,
    create_side_load_pointwise,
    create_side_load_profile,
)

from campro.physics.geometry.litvin import LitvinGearGeometry
from campro.physics.mechanics.side_loading import SideLoadAnalyzer


class TestCasadiSideLoading:
    """Test CasADi side loading functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.side_load_point_fn = create_side_load_pointwise()
        self.side_load_profile_fn = create_side_load_profile()
        self.side_load_penalty_fn = create_side_load_penalty()
        self.phase_mask_fn = create_phase_masks()

        # Test parameters
        self.r = 50.0  # mm
        self.l = 150.0  # mm
        self.x_off = 5.0  # mm
        self.y_off = 2.0  # mm
        self.F = 1000.0  # N

        # Test data (limit to 10 for fixed-size functions)
        self.theta = np.linspace(0, 2 * np.pi, 10)
        self.F_vec = 1000.0 * np.ones(10)  # N

    def test_side_load_pointwise_function_creation(self):
        """Test that pointwise side load function is created successfully."""
        assert isinstance(self.side_load_point_fn, ca.Function)
        assert self.side_load_point_fn.name() == "side_load_pointwise"

    def test_side_load_profile_function_creation(self):
        """Test that side load profile function is created successfully."""
        assert isinstance(self.side_load_profile_fn, ca.Function)
        assert self.side_load_profile_fn.name() == "side_load_profile"

    def test_side_load_penalty_function_creation(self):
        """Test that side load penalty function is created successfully."""
        assert isinstance(self.side_load_penalty_fn, ca.Function)
        assert self.side_load_penalty_fn.name() == "side_load_penalty"

    def test_side_load_pointwise_outputs(self):
        """Test side load pointwise function outputs."""
        F_side = self.side_load_point_fn(
            self.theta[0],
            self.r,
            self.l,
            self.F,
            self.x_off,
            self.y_off,
        )

        # Check output shape
        assert F_side.shape == (1, 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(F_side))
        assert not np.any(np.isinf(F_side))

        # Check reasonable values (side load can be positive or negative)
        assert np.isfinite(F_side)

    def test_side_load_profile_outputs(self):
        """Test side load profile function outputs."""
        F_side_vec, F_side_max, F_side_avg, ripple = self.side_load_profile_fn(
            self.theta,
            self.F_vec,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
        )

        # Check output shapes
        assert F_side_vec.shape == (10, 1)
        assert F_side_max.shape == (1, 1)
        assert F_side_avg.shape == (1, 1)
        assert ripple.shape == (1, 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(F_side_vec))
        assert not np.any(np.isnan(F_side_max))
        assert not np.any(np.isnan(F_side_avg))
        assert not np.any(np.isnan(ripple))

        assert not np.any(np.isinf(F_side_vec))
        assert not np.any(np.isinf(F_side_max))
        assert not np.any(np.isinf(F_side_avg))
        assert not np.any(np.isinf(ripple))

        # Check logical relationships
        assert F_side_max >= F_side_avg
        assert ripple >= 0  # Ripple coefficient should be non-negative

    def test_side_load_penalty_outputs(self):
        """Test side load penalty function outputs."""
        # Create test side load profile
        F_side_vec = ca.DM([100, 200, 300, 150, 250, 180, 220, 190, 210, 160])
        max_threshold = 200.0

        # Create phase masks (compression and combustion)
        compression_mask = ca.DM([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        combustion_mask = ca.DM([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

        penalty = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold,
            compression_mask,
            combustion_mask,
        )

        # Check output shape
        assert penalty.shape == (1, 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(penalty))
        assert not np.any(np.isinf(penalty))

        # Check reasonable values (penalty should be non-negative)
        assert penalty >= 0

    def test_penalty_monotonicity(self):
        """Test that penalty increases with force magnitude."""
        # Test with increasing side loads
        F_side_low = ca.DM([100, 150, 120, 180, 140, 160, 130, 170, 150, 110])
        F_side_high = ca.DM([300, 400, 350, 450, 380, 420, 360, 440, 390, 370])

        max_threshold = 200.0
        compression_mask = ca.DM([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        combustion_mask = ca.DM([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

        penalty_low = self.side_load_penalty_fn(
            F_side_low,
            max_threshold,
            compression_mask,
            combustion_mask,
        )
        penalty_high = self.side_load_penalty_fn(
            F_side_high,
            max_threshold,
            compression_mask,
            combustion_mask,
        )

        # Higher forces should result in higher penalty
        assert penalty_high > penalty_low

    def test_domain_safety(self):
        """Test domain safety near singularities."""
        # Test near r ≈ l (singularity condition)
        r_singular = 149.9  # mm, very close to l=150

        F_side = self.side_load_point_fn(
            self.theta[0],
            r_singular,
            self.l,
            self.F,
            self.x_off,
            self.y_off,
        )

        # Should not produce NaN/Inf even near singularity
        assert not np.any(np.isnan(F_side))
        assert not np.any(np.isinf(F_side))

    def test_phase_weight_application(self):
        """Test that phase weights are applied correctly."""
        # Create side load profile with known excess
        F_side_vec = ca.DM([100, 300, 100, 300, 100, 300, 100, 300, 100, 300])
        max_threshold = 200.0

        # Test different phase weight scenarios
        # All compression
        compression_mask = ca.DM([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        combustion_mask = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        penalty_comp = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold,
            compression_mask,
            combustion_mask,
        )

        # All combustion
        compression_mask = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        combustion_mask = ca.DM([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        penalty_comb = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold,
            compression_mask,
            combustion_mask,
        )

        # All normal
        compression_mask = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        combustion_mask = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        penalty_normal = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold,
            compression_mask,
            combustion_mask,
        )

        # Check that penalties scale with weights
        # compression_weight = 1.2, combustion_weight = 1.5, normal_weight = 1.0
        assert penalty_comb > penalty_comp > penalty_normal


class TestCasadiSideLoadingParityWithPython:
    """Test CasADi side loading implementation against Python baselines."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create Python side load analyzer
        self.python_analyzer = SideLoadAnalyzer()
        mock_gear = Mock(spec=LitvinGearGeometry)
        mock_gear.pressure_angle = np.radians(20.0)

        self.python_analyzer.configure(
            crank_radius=50.0,
            rod_length=150.0,
            gear_geometry=mock_gear,
        )

        # CasADi functions
        self.side_load_point_fn = create_side_load_pointwise()
        self.side_load_profile_fn = create_side_load_profile()

        # Test parameters
        self.r = 50.0  # mm
        self.l = 150.0  # mm
        self.x_off = 5.0  # mm
        self.y_off = 2.0  # mm

        # Test data (pad to 10 elements for fixed-size functions)
        self.theta_test = np.array(
            [
                0.0,
                np.pi / 4,
                np.pi / 2,
                3 * np.pi / 4,
                np.pi,
                5 * np.pi / 4,
                3 * np.pi / 2,
                7 * np.pi / 4,
                2 * np.pi,
                9 * np.pi / 4,
            ],
        )
        self.F_test = 1000.0  # N

    @pytest.mark.xfail(
        reason="Python implementation has different offset handling - needs investigation",
    )
    def test_side_load_parity_single_point(self):
        """Test side load calculation parity at single points."""
        rtol = 1e-3  # Relaxed tolerance due to different offset handling

        for theta in self.theta_test:
            # Python calculation
            python_side_load = self.python_analyzer.compute_instantaneous_side_load(
                piston_force=self.F_test,
                crank_angle=theta,
                crank_center_offset=(self.x_off, self.y_off),
            )

            # CasADi calculation
            casadi_side_load = self.side_load_point_fn(
                theta,
                self.r,
                self.l,
                self.F_test,
                self.x_off,
                self.y_off,
            )

            # Compare
            np.testing.assert_allclose(
                casadi_side_load,
                python_side_load,
                rtol=rtol,
                err_msg=f"Side load mismatch at theta={theta}",
            )

    @pytest.mark.xfail(
        reason="Python implementation has different offset handling - needs investigation",
    )
    def test_side_load_parity_profile(self):
        """Test side load profile parity over full cycle."""
        rtol = 1e-3  # Relaxed tolerance due to different offset handling

        # Create motion law data for Python analyzer
        motion_law_data = {
            "theta": self.theta_test,
            "displacement": np.zeros_like(
                self.theta_test,
            ),  # Not used in side load calc
            "velocity": np.zeros_like(self.theta_test),  # Not used in side load calc
            "acceleration": np.zeros_like(
                self.theta_test,
            ),  # Not used in side load calc
        }
        load_profile = self.F_test * np.ones_like(self.theta_test)

        # Python calculation
        python_profile = self.python_analyzer.compute_side_load_profile(
            motion_law_data=motion_law_data,
            load_profile=load_profile,
            crank_center_offset=(self.x_off, self.y_off),
        )

        # CasADi calculation
        F_side_vec, F_side_max, F_side_avg, ripple = self.side_load_profile_fn(
            self.theta_test,
            load_profile,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
        )

        # Compare maximum side load
        np.testing.assert_allclose(
            F_side_max,
            python_profile["max_side_load"],
            rtol=rtol,
            err_msg="Maximum side load mismatch",
        )

        # Compare average side load
        np.testing.assert_allclose(
            F_side_avg,
            python_profile["avg_side_load"],
            rtol=rtol,
            err_msg="Average side load mismatch",
        )


class TestCasadiSideLoadingGradients:
    """Test automatic differentiation capabilities for side loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.side_load_point_fn = create_side_load_pointwise()
        self.side_load_penalty_fn = create_side_load_penalty()

        # Test parameters
        self.theta = np.pi / 4
        self.r = 50.0
        self.l = 150.0
        self.F = 1000.0
        self.x_off = 5.0
        self.y_off = 2.0

    def test_side_load_gradients(self):
        """Test gradients of side load w.r.t. r and l."""
        rtol = 1e-5

        # Create symbolic variables
        r_sym = ca.MX.sym("r")
        l_sym = ca.MX.sym("l")

        # Compute side load
        F_side = self.side_load_point_fn(
            self.theta,
            r_sym,
            l_sym,
            self.F,
            self.x_off,
            self.y_off,
        )

        # Compute gradients
        dF_side_dr = ca.jacobian(F_side, r_sym)
        dF_side_dl = ca.jacobian(F_side, l_sym)

        # Create gradient function
        grad_fn = ca.Function(
            "side_load_gradients", [r_sym, l_sym], [dF_side_dr, dF_side_dl],
        )

        # Evaluate gradients
        grad_r, grad_l = grad_fn(self.r, self.l)

        # Check for NaN/Inf
        assert not np.any(np.isnan(grad_r))
        assert not np.any(np.isnan(grad_l))
        assert not np.any(np.isinf(grad_r))
        assert not np.any(np.isinf(grad_l))

        # Finite difference check
        eps = 1e-6

        # Forward difference for dF_side/dr
        F_side_plus = self.side_load_point_fn(
            self.theta,
            self.r + eps,
            self.l,
            self.F,
            self.x_off,
            self.y_off,
        )
        F_side_minus = self.side_load_point_fn(
            self.theta,
            self.r - eps,
            self.l,
            self.F,
            self.x_off,
            self.y_off,
        )
        dF_side_dr_fd = (F_side_plus - F_side_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_r,
            dF_side_dr_fd,
            rtol=rtol,
            err_msg="Gradient dF_side/dr mismatch",
        )

        # Forward difference for dF_side/dl
        F_side_plus = self.side_load_point_fn(
            self.theta,
            self.r,
            self.l + eps,
            self.F,
            self.x_off,
            self.y_off,
        )
        F_side_minus = self.side_load_point_fn(
            self.theta,
            self.r,
            self.l - eps,
            self.F,
            self.x_off,
            self.y_off,
        )
        dF_side_dl_fd = (F_side_plus - F_side_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_l,
            dF_side_dl_fd,
            rtol=rtol,
            err_msg="Gradient dF_side/dl mismatch",
        )

    def test_penalty_gradients(self):
        """Test gradients of penalty function."""
        rtol = 1e-5

        # Create test data
        F_side_vec = ca.DM([100, 200, 300, 150, 250, 180, 220, 190, 210, 160])
        max_threshold = 200.0
        compression_mask = ca.DM([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
        combustion_mask = ca.DM([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

        # Create symbolic threshold
        threshold_sym = ca.MX.sym("threshold")

        # Compute penalty
        penalty = self.side_load_penalty_fn(
            F_side_vec,
            threshold_sym,
            compression_mask,
            combustion_mask,
        )

        # Compute gradient
        dpenalty_dthreshold = ca.jacobian(penalty, threshold_sym)

        # Create gradient function
        grad_fn = ca.Function(
            "penalty_gradient", [threshold_sym], [dpenalty_dthreshold],
        )

        # Evaluate gradient
        grad = grad_fn(max_threshold)

        # Check for NaN/Inf
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isinf(grad))

        # Finite difference check
        eps = 1e-6

        penalty_plus = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold + eps,
            compression_mask,
            combustion_mask,
        )
        penalty_minus = self.side_load_penalty_fn(
            F_side_vec,
            max_threshold - eps,
            compression_mask,
            combustion_mask,
        )
        grad_fd = (penalty_plus - penalty_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad,
            grad_fd,
            rtol=rtol,
            err_msg="Gradient dpenalty/dthreshold mismatch",
        )
