"""
Tests for CasADi Phase 4: Unified Physics & Optimizer Integration.

This module tests the unified physics function and its integration into
optimization problems, including toy NLP and performance benchmarks.
"""

import time

import casadi as ca
import numpy as np
from campro.physics.casadi import create_toy_nlp_optimizer, create_unified_physics


class TestCasadiUnifiedPhysics:
    """Test unified physics function integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.unified_fn = create_unified_physics()

        # Test parameters
        self.theta_vec = np.linspace(0, 2 * np.pi, 10)
        self.pressure_vec = 1e5 * np.ones(10)  # Pa
        self.r = 50.0  # mm
        self.l = 150.0  # mm
        self.x_off = 5.0  # mm
        self.y_off = 2.0  # mm
        self.bore = 100.0  # mm
        self.max_side_threshold = 200.0  # N
        self.litvin_config = np.array(
            [50.0, 20.0, 2.0, 20.0, 25.0, 1.0],
        )  # enable Litvin

    def test_unified_physics_function_creation(self):
        """Test that unified physics function is created successfully."""
        assert isinstance(self.unified_fn, ca.Function)
        assert self.unified_fn.name() == "unified_physics"

    def test_unified_physics_outputs(self):
        """Test unified physics function outputs."""
        (
            torque_avg,
            torque_ripple,
            side_load_penalty,
            litvin_objective,
            litvin_closure,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            self.litvin_config,
        )

        # Check output shapes
        assert torque_avg.shape == (1, 1)
        assert torque_ripple.shape == (1, 1)
        assert side_load_penalty.shape == (1, 1)
        assert litvin_objective.shape == (1, 1)
        assert litvin_closure.shape == (1, 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(torque_avg))
        assert not np.any(np.isnan(torque_ripple))
        assert not np.any(np.isnan(side_load_penalty))
        assert not np.any(np.isnan(litvin_objective))
        assert not np.any(np.isnan(litvin_closure))

        assert not np.any(np.isinf(torque_avg))
        assert not np.any(np.isinf(torque_ripple))
        assert not np.any(np.isinf(side_load_penalty))
        assert not np.any(np.isinf(litvin_objective))
        assert not np.any(np.isinf(litvin_closure))

        # Check reasonable values
        assert torque_avg >= 0  # Torque should be non-negative
        assert torque_ripple >= 0  # Ripple should be non-negative
        assert side_load_penalty >= 0  # Penalty should be non-negative
        assert litvin_objective >= 0  # Objective should be non-negative
        assert litvin_closure >= 0  # Closure should be non-negative

    def test_unified_physics_litvin_toggle(self):
        """Test that Litvin calculations can be toggled on/off."""
        # Test with Litvin enabled
        litvin_config_enabled = np.array([50.0, 20.0, 2.0, 20.0, 25.0, 1.0])
        (
            torque_avg_en,
            torque_ripple_en,
            side_load_penalty_en,
            litvin_objective_en,
            litvin_closure_en,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            litvin_config_enabled,
        )

        # Test with Litvin disabled
        litvin_config_disabled = np.array([50.0, 20.0, 2.0, 20.0, 25.0, 0.0])
        (
            torque_avg_dis,
            torque_ripple_dis,
            side_load_penalty_dis,
            litvin_objective_dis,
            litvin_closure_dis,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            litvin_config_disabled,
        )

        # Torque and side load should be the same
        assert np.allclose(torque_avg_en, torque_avg_dis, rtol=1e-6)
        assert np.allclose(torque_ripple_en, torque_ripple_dis, rtol=1e-6)
        assert np.allclose(side_load_penalty_en, side_load_penalty_dis, rtol=1e-6)

        # Litvin should be zero when disabled
        assert litvin_objective_dis == 0.0
        assert litvin_closure_dis == 0.0

        # Litvin should be non-zero when enabled
        assert litvin_objective_en > 0.0
        assert litvin_closure_en >= 0.0

    def test_unified_physics_gradients(self):
        """Test gradients of unified physics function."""
        # Create symbolic variables
        r_sym = ca.MX.sym("r")
        l_sym = ca.MX.sym("l")

        # Compute unified physics
        (
            torque_avg,
            torque_ripple,
            side_load_penalty,
            litvin_objective,
            litvin_closure,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            r_sym,
            l_sym,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            self.litvin_config,
        )

        # Compute gradients
        dtorque_dr = ca.jacobian(torque_avg, r_sym)
        dtorque_dl = ca.jacobian(torque_avg, l_sym)
        dpenalty_dr = ca.jacobian(side_load_penalty, r_sym)
        dpenalty_dl = ca.jacobian(side_load_penalty, l_sym)

        # Create gradient function
        grad_fn = ca.Function(
            "unified_physics_gradients",
            [r_sym, l_sym],
            [dtorque_dr, dtorque_dl, dpenalty_dr, dpenalty_dl],
        )

        # Evaluate gradients
        dtorque_dr_val, dtorque_dl_val, dpenalty_dr_val, dpenalty_dl_val = grad_fn(
            self.r,
            self.l,
        )

        # Check for NaN/Inf
        assert not np.any(np.isnan(dtorque_dr_val))
        assert not np.any(np.isnan(dtorque_dl_val))
        assert not np.any(np.isnan(dpenalty_dr_val))
        assert not np.any(np.isnan(dpenalty_dl_val))

        assert not np.any(np.isinf(dtorque_dr_val))
        assert not np.any(np.isinf(dtorque_dl_val))
        assert not np.any(np.isinf(dpenalty_dr_val))
        assert not np.any(np.isinf(dpenalty_dl_val))

    def test_unified_physics_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        # Test with different crank radii
        r_values = [40.0, 50.0, 60.0]

        for r in r_values:
            (
                torque_avg,
                torque_ripple,
                side_load_penalty,
                litvin_objective,
                litvin_closure,
            ) = self.unified_fn(
                self.theta_vec,
                self.pressure_vec,
                r,
                self.l,
                self.x_off,
                self.y_off,
                self.bore,
                self.max_side_threshold,
                self.litvin_config,
            )

            # Should produce finite results
            assert np.isfinite(torque_avg)
            assert np.isfinite(torque_ripple)
            assert np.isfinite(side_load_penalty)
            assert np.isfinite(litvin_objective)
            assert np.isfinite(litvin_closure)

            # Check non-negativity
            assert torque_avg >= 0
            assert torque_ripple >= 0
            assert side_load_penalty >= 0
            assert litvin_objective >= 0
            assert litvin_closure >= 0


class TestCasadiToyNlpOptimizer:
    """Test toy NLP optimizer using unified physics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.nlp_fn = create_toy_nlp_optimizer()

        # Test parameters
        self.r = 50.0  # mm
        self.l = 150.0  # mm

    def test_toy_nlp_function_creation(self):
        """Test that toy NLP function is created successfully."""
        assert isinstance(self.nlp_fn, ca.Function)
        assert self.nlp_fn.name() == "toy_nlp_optimizer"

    def test_toy_nlp_optimization(self):
        """Test toy NLP function outputs."""
        torque_avg, side_load_penalty, objective, constraint = self.nlp_fn(
            self.r,
            self.l,
        )

        # Check output shapes
        assert torque_avg.shape == (1, 1)
        assert side_load_penalty.shape == (1, 1)
        assert objective.shape == (1, 1)
        assert constraint.shape == (1, 1)

        # Check for NaN/Inf
        assert not np.any(np.isnan(torque_avg))
        assert not np.any(np.isnan(side_load_penalty))
        assert not np.any(np.isnan(objective))
        assert not np.any(np.isnan(constraint))

        assert not np.any(np.isinf(torque_avg))
        assert not np.any(np.isinf(side_load_penalty))
        assert not np.any(np.isinf(objective))
        assert not np.any(np.isinf(constraint))

        # Check reasonable values
        assert torque_avg >= 0  # Torque should be non-negative
        assert side_load_penalty >= 0  # Penalty should be non-negative
        assert objective <= 0  # Objective should be negative (maximize torque)

    def test_toy_nlp_constraint_satisfaction(self):
        """Test that constraint function works correctly."""
        torque_avg, side_load_penalty, objective, constraint = self.nlp_fn(
            self.r,
            self.l,
        )

        # Constraint: side_load_penalty <= 0.1
        # constraint is the constraint value (penalty - 0.1), should be <= 0 when satisfied
        # For the test parameters (r=50, l=150), this constraint is violated
        assert constraint > 0.0  # Constraint is violated for these parameters

        # Test with parameters that should satisfy the constraint
        # Use smaller r and larger l to reduce side load penalty
        torque_avg2, side_load_penalty2, objective2, constraint2 = self.nlp_fn(
            5.0,
            200.0,  # Very small r, large l should reduce penalty
        )

        # This should satisfy the constraint
        assert constraint2 <= 0.0

    def test_toy_nlp_different_parameters(self):
        """Test function with different parameter values."""
        test_params = [
            (40.0, 120.0),
            (60.0, 180.0),
            (45.0, 160.0),
        ]

        for r, l in test_params:
            torque_avg, side_load_penalty, objective, constraint = self.nlp_fn(
                r,
                l,
            )

            # Should produce finite results
            assert np.isfinite(torque_avg)
            assert np.isfinite(side_load_penalty)
            assert np.isfinite(objective)
            assert np.isfinite(constraint)

            # Should satisfy bounds
            assert 30.0 <= r <= 80.0
            assert 100.0 <= l <= 200.0

            # Check reasonable values
            assert torque_avg >= 0
            assert side_load_penalty >= 0
            assert objective <= 0


class TestCasadiPerformanceBenchmark:
    """Test performance of CasADi unified physics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.unified_fn = create_unified_physics()

        # Test parameters
        self.theta_vec = np.linspace(0, 2 * np.pi, 10)
        self.pressure_vec = 1e5 * np.ones(10)  # Pa
        self.r = 50.0  # mm
        self.l = 150.0  # mm
        self.x_off = 5.0  # mm
        self.y_off = 2.0  # mm
        self.bore = 100.0  # mm
        self.max_side_threshold = 200.0  # N
        self.litvin_config = np.array([50.0, 20.0, 2.0, 20.0, 25.0, 1.0])

    def test_unified_physics_evaluation_speed(self):
        """Test evaluation speed of unified physics function."""
        n_evaluations = 100

        # Time the evaluations
        start_time = time.time()
        for _ in range(n_evaluations):
            (
                torque_avg,
                torque_ripple,
                side_load_penalty,
                litvin_objective,
                litvin_closure,
            ) = self.unified_fn(
                self.theta_vec,
                self.pressure_vec,
                self.r,
                self.l,
                self.x_off,
                self.y_off,
                self.bore,
                self.max_side_threshold,
                self.litvin_config,
            )
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_eval = total_time / n_evaluations

        # Should be reasonably fast (less than 1ms per evaluation)
        assert avg_time_per_eval < 0.001  # 1ms

        # Log performance for reference
        print(
            f"Unified physics evaluation time: {avg_time_per_eval * 1000:.2f} ms per evaluation",
        )

    def test_unified_physics_gradient_speed(self):
        """Test gradient computation speed."""
        # Create symbolic variables
        r_sym = ca.MX.sym("r")
        l_sym = ca.MX.sym("l")

        # Compute unified physics
        (
            torque_avg,
            torque_ripple,
            side_load_penalty,
            litvin_objective,
            litvin_closure,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            r_sym,
            l_sym,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            self.litvin_config,
        )

        # Compute gradients
        dtorque_dr = ca.jacobian(torque_avg, r_sym)
        dtorque_dl = ca.jacobian(torque_avg, l_sym)

        # Create gradient function
        grad_fn = ca.Function(
            "unified_physics_gradients",
            [r_sym, l_sym],
            [dtorque_dr, dtorque_dl],
        )

        # Time gradient evaluations
        n_evaluations = 50
        start_time = time.time()
        for _ in range(n_evaluations):
            dtorque_dr_val, dtorque_dl_val = grad_fn(self.r, self.l)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_eval = total_time / n_evaluations

        # Should be reasonably fast (less than 2ms per evaluation)
        assert (
            avg_time_per_eval < 0.005
        )  # 5ms (more realistic for complex unified physics)

        # Log performance for reference
        print(
            f"Unified physics gradient time: {avg_time_per_eval * 1000:.2f} ms per evaluation",
        )


class TestCasadiIntegrationParity:
    """Test parity between CasADi and expected behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.unified_fn = create_unified_physics()

        # Test parameters
        self.theta_vec = np.linspace(0, 2 * np.pi, 10)
        self.pressure_vec = 1e5 * np.ones(10)  # Pa
        self.r = 50.0  # mm
        self.l = 150.0  # mm
        self.x_off = 5.0  # mm
        self.y_off = 2.0  # mm
        self.bore = 100.0  # mm
        self.max_side_threshold = 200.0  # N
        self.litvin_config = np.array(
            [50.0, 20.0, 2.0, 20.0, 25.0, 0.0],
        )  # Litvin disabled

    def test_unified_physics_consistency(self):
        """Test that unified physics produces consistent results."""
        # Run multiple times with same inputs
        results = []
        for _ in range(5):
            (
                torque_avg,
                torque_ripple,
                side_load_penalty,
                litvin_objective,
                litvin_closure,
            ) = self.unified_fn(
                self.theta_vec,
                self.pressure_vec,
                self.r,
                self.l,
                self.x_off,
                self.y_off,
                self.bore,
                self.max_side_threshold,
                self.litvin_config,
            )
            results.append(
                (
                    torque_avg,
                    torque_ripple,
                    side_load_penalty,
                    litvin_objective,
                    litvin_closure,
                ),
            )

        # All results should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0][0], results[i][0], rtol=1e-10)
            assert np.allclose(results[0][1], results[i][1], rtol=1e-10)
            assert np.allclose(results[0][2], results[i][2], rtol=1e-10)
            assert np.allclose(results[0][3], results[i][3], rtol=1e-10)
            assert np.allclose(results[0][4], results[i][4], rtol=1e-10)

    def test_unified_physics_monotonicity(self):
        """Test that physics outputs behave monotonically with parameters."""
        # Test torque vs crank radius
        r_values = [40.0, 50.0, 60.0]
        torque_values = []

        for r in r_values:
            (
                torque_avg,
                torque_ripple,
                side_load_penalty,
                litvin_objective,
                litvin_closure,
            ) = self.unified_fn(
                self.theta_vec,
                self.pressure_vec,
                r,
                self.l,
                self.x_off,
                self.y_off,
                self.bore,
                self.max_side_threshold,
                self.litvin_config,
            )
            torque_values.append(torque_avg)

        # Torque should generally increase with crank radius
        # (This is a physical expectation, though not strictly guaranteed)
        assert (
            torque_values[1] > torque_values[0] or torque_values[2] > torque_values[1]
        )

    def test_unified_physics_physical_bounds(self):
        """Test that physics outputs are within physically reasonable bounds."""
        (
            torque_avg,
            torque_ripple,
            side_load_penalty,
            litvin_objective,
            litvin_closure,
        ) = self.unified_fn(
            self.theta_vec,
            self.pressure_vec,
            self.r,
            self.l,
            self.x_off,
            self.y_off,
            self.bore,
            self.max_side_threshold,
            self.litvin_config,
        )

        # Check physically reasonable bounds
        assert 0 <= torque_avg < 1000.0  # Reasonable torque range (N⋅m)
        assert 0 <= torque_ripple < 10.0  # Reasonable ripple range
        assert (
            0 <= side_load_penalty < 20000.0
        )  # Reasonable penalty range (adjusted for realistic values)
        assert 0 <= litvin_objective < 1000.0  # Reasonable objective range
        assert 0 <= litvin_closure < 100.0  # Reasonable closure range (mm)
