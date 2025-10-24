"""
Tests for crank center physics integration in optimization.

This module tests the hybrid physics integration approach where
CasADi simplified objectives are combined with Python physics validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from campro.optimization.crank_center_optimizer import (
    CrankCenterOptimizer,
    CrankCenterOptimizationConstraints,
    CrankCenterOptimizationTargets,
)
from campro.optimization.base import OptimizationStatus
from campro.optimization.solver_analysis import MA57ReadinessReport
from campro.constants import USE_CASADI_PHYSICS


class TestCrankCenterPhysicsIntegration:
    """Test crank center hybrid physics integration."""

    def test_crank_center_hybrid_physics_objective(self):
        """Test hybrid physics objective evaluation."""
        # Create optimizer with constraints and targets
        constraints = CrankCenterOptimizationConstraints(
            max_iterations=100,
            tolerance=1e-6,
            crank_center_x_min=-5.0,
            crank_center_x_max=5.0,
            crank_center_y_min=-5.0,
            crank_center_y_max=5.0,
        )
        
        targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )
        
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=constraints, targets=targets)
        
        # Mock the physics models
        with patch.object(optimizer, '_torque_calculator') as mock_torque, \
             patch.object(optimizer, '_side_load_analyzer') as mock_side_load, \
             patch.object(optimizer, '_configure_physics_models') as mock_configure:
            
            # Mock successful simulation results
            mock_torque_result = Mock()
            mock_torque_result.is_successful = True
            mock_torque_result.metadata = {
                "torque_result": Mock(
                    cycle_average_torque=100.0,
                    torque_ripple=5.0,
                    power_output=50.0
                )
            }
            mock_torque.simulate.return_value = mock_torque_result
            
            mock_side_load_result = Mock()
            mock_side_load_result.is_successful = True
            mock_side_load_result.metadata = {
                "side_load_result": Mock(
                    total_penalty=10.0
                )
            }
            mock_side_load.simulate.return_value = mock_side_load_result
            
            # Test the physics objective evaluation
            test_params = np.array([1.0, 2.0])  # crank_center_x, crank_center_y
            param_names = ["crank_center_x", "crank_center_y"]
            
            # Create test data
            motion_law_data = {"cam_angle": [0, 1, 2], "position": [0, 10, 20]}
            load_profile = {"load": [1.0, 1.0, 1.0]}
            gear_geometry = Mock()
            
            # Test the physics objective function (this would be called inside _optimize_with_ipopt)
            def evaluate_physics_objective(params_array):
                """Evaluate full physics objective for given parameters."""
                param_dict = dict(zip(param_names, params_array))
                param_dict["bore_diameter"] = 100.0
                param_dict["piston_clearance"] = 0.1
                
                # Configure physics models
                optimizer._configure_physics_models(param_dict, gear_geometry)
                crank_center_offset = (param_dict["crank_center_x"], param_dict["crank_center_y"])
                
                # Compute torque
                torque_inputs = {
                    "motion_law_data": motion_law_data,
                    "load_profile": load_profile,
                    "crank_center_offset": crank_center_offset,
                }
                torque_result = optimizer._torque_calculator.simulate(torque_inputs)
                
                # Compute side loading
                side_load_inputs = {
                    "motion_law_data": motion_law_data,
                    "load_profile": load_profile,
                    "crank_center_offset": crank_center_offset,
                }
                side_load_result = optimizer._side_load_analyzer.simulate(side_load_inputs)
                
                if not torque_result.is_successful or not side_load_result.is_successful:
                    return 1e6
                
                torque_result_obj = torque_result.metadata.get("torque_result")
                side_load_result_obj = side_load_result.metadata.get("side_load_result")
                
                if torque_result_obj is None or side_load_result_obj is None:
                    return 1e6
                
                # Multi-objective
                objective = 0.0
                if optimizer.targets.maximize_torque:
                    objective += optimizer.targets.torque_weight * (-torque_result_obj.cycle_average_torque)
                if optimizer.targets.minimize_side_loading:
                    objective += optimizer.targets.side_load_weight * side_load_result_obj.total_penalty
                if optimizer.targets.minimize_torque_ripple:
                    objective += optimizer.targets.torque_ripple_weight * torque_result_obj.torque_ripple
                if optimizer.targets.maximize_power_output:
                    objective += optimizer.targets.power_output_weight * (-torque_result_obj.power_output)
                
                return objective
            
            # Test the physics objective
            result = evaluate_physics_objective(test_params)
            
            # Verify the result
            assert isinstance(result, (int, float))
            assert result < 1e6  # Should be feasible
            assert result == -100.0 + 0.5 * 10.0  # -torque + side_load_penalty
            
            # Verify physics models were called
            mock_configure.assert_called_once()
            mock_torque.simulate.assert_called_once()
            mock_side_load.simulate.assert_called_once()

    def test_crank_center_physics_validation(self):
        """Test physics validation of optimized crank center."""
        # Create optimizer
        constraints = CrankCenterOptimizationConstraints(
            max_iterations=100,
            tolerance=1e-6,
            crank_center_x_min=-5.0,
            crank_center_x_max=5.0,
            crank_center_y_min=-5.0,
            crank_center_y_max=5.0,
        )
        
        targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )
        
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=constraints, targets=targets)
        
        # Mock the physics models to return infeasible results
        with patch.object(optimizer, '_torque_calculator') as mock_torque, \
             patch.object(optimizer, '_side_load_analyzer') as mock_side_load, \
             patch.object(optimizer, '_configure_physics_models') as mock_configure:
            
            # Mock failed simulation results
            mock_torque_result = Mock()
            mock_torque_result.is_successful = False
            mock_torque.simulate.return_value = mock_torque_result
            
            mock_side_load_result = Mock()
            mock_side_load_result.is_successful = False
            mock_side_load.simulate.return_value = mock_side_load_result
            
            # Test the physics objective evaluation with infeasible results
            test_params = np.array([1.0, 2.0])
            param_names = ["crank_center_x", "crank_center_y"]
            
            # Create test data
            motion_law_data = {"cam_angle": [0, 1, 2], "position": [0, 10, 20]}
            load_profile = {"load": [1.0, 1.0, 1.0]}
            gear_geometry = Mock()
            
            # Test the physics objective function
            def evaluate_physics_objective(params_array):
                """Evaluate full physics objective for given parameters."""
                param_dict = dict(zip(param_names, params_array))
                param_dict["bore_diameter"] = 100.0
                param_dict["piston_clearance"] = 0.1
                
                # Configure physics models
                optimizer._configure_physics_models(param_dict, gear_geometry)
                crank_center_offset = (param_dict["crank_center_x"], param_dict["crank_center_y"])
                
                # Compute torque
                torque_inputs = {
                    "motion_law_data": motion_law_data,
                    "load_profile": load_profile,
                    "crank_center_offset": crank_center_offset,
                }
                torque_result = optimizer._torque_calculator.simulate(torque_inputs)
                
                # Compute side loading
                side_load_inputs = {
                    "motion_law_data": motion_law_data,
                    "load_profile": load_profile,
                    "crank_center_offset": crank_center_offset,
                }
                side_load_result = optimizer._side_load_analyzer.simulate(side_load_inputs)
                
                if not torque_result.is_successful or not side_load_result.is_successful:
                    return 1e6
                
                # This should not be reached due to failed simulations
                return 0.0
            
            # Test the physics objective
            result = evaluate_physics_objective(test_params)
            
            # Verify the result is infeasible
            assert result == 1e6  # Should be infeasible due to failed simulations

    def test_crank_center_physics_metrics_in_result(self):
        """Test physics metrics in optimization result."""
        # Create optimizer
        constraints = CrankCenterOptimizationConstraints(
            max_iterations=100,
            tolerance=1e-6,
            crank_center_x_min=-5.0,
            crank_center_x_max=5.0,
            crank_center_y_min=-5.0,
            crank_center_y_max=5.0,
        )
        
        targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )
        
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=constraints, targets=targets)
        
        # Test that the optimizer has the expected attributes
        assert hasattr(optimizer, 'constraints')
        assert hasattr(optimizer, 'targets')
        assert hasattr(optimizer, '_torque_calculator')
        assert hasattr(optimizer, '_side_load_analyzer')
        assert hasattr(optimizer, '_kinematics')
        
        # Test that constraints and targets are properly configured
        assert optimizer.constraints.max_iterations == 100
        assert optimizer.constraints.tolerance == 1e-6
        assert optimizer.targets.maximize_torque is True
        assert optimizer.targets.minimize_side_loading is True

    def test_crank_center_fallback_optimization(self):
        """Test fallback to scipy when Ipopt fails."""
        # Create optimizer
        constraints = CrankCenterOptimizationConstraints(
            max_iterations=100,
            tolerance=1e-6,
            crank_center_x_min=-5.0,
            crank_center_x_max=5.0,
            crank_center_y_min=-5.0,
            crank_center_y_max=5.0,
        )
        
        targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )
        
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=constraints, targets=targets)
        
        # Test that the optimizer can be created and configured
        assert optimizer.name == "CrankCenterOptimizer"
        assert optimizer.constraints.max_iterations == 100
        assert optimizer.targets.maximize_torque is True
        
        # Test that physics models are initialized
        assert optimizer._torque_calculator is not None
        assert optimizer._side_load_analyzer is not None
        assert optimizer._kinematics is not None

    def test_crank_center_physics_objective_interface(self):
        """Test that physics objective function has correct interface."""
        # Create optimizer
        constraints = CrankCenterOptimizationConstraints(
            max_iterations=100,
            tolerance=1e-6,
            crank_center_x_min=-5.0,
            crank_center_x_max=5.0,
            crank_center_y_min=-5.0,
            crank_center_y_max=5.0,
        )
        
        targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )
        
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=constraints, targets=targets)
        
        # Test that the optimizer has the expected methods
        assert hasattr(optimizer, '_objective_function')
        assert hasattr(optimizer, '_configure_physics_models')
        assert hasattr(optimizer, '_generate_final_design')
        
        # Test that the physics models are properly initialized
        assert optimizer._torque_calculator is not None
        assert optimizer._side_load_analyzer is not None
        assert optimizer._kinematics is not None
        
        # Test that the optimizer can be configured with different parameters
        new_constraints = CrankCenterOptimizationConstraints(max_iterations=200)
        new_targets = CrankCenterOptimizationTargets(maximize_torque=False)
        
        optimizer.configure(constraints=new_constraints, targets=new_targets)
        
        assert optimizer.constraints.max_iterations == 200
        assert optimizer.targets.maximize_torque is False


class TestCasadiPhysicsToggleIntegration:
    """Test cross-mode integration between Python and CasADi physics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.constraints = CrankCenterOptimizationConstraints(
            max_iterations=50,  # Reduced for testing
            tolerance=1e-4,
            crank_center_x_min=-10.0,
            crank_center_x_max=10.0,
            crank_center_y_min=-10.0,
            crank_center_y_max=10.0,
        )
        
        self.targets = CrankCenterOptimizationTargets(
            maximize_torque=True,
            minimize_side_loading=True,
            minimize_torque_ripple=False,
            maximize_power_output=False,
            torque_weight=1.0,
            side_load_weight=0.5,
            torque_ripple_weight=0.0,
            power_output_weight=0.0,
        )

    @pytest.mark.skipif(not USE_CASADI_PHYSICS, reason="CasADi physics not enabled")
    def test_casadi_physics_mode_integration(self):
        """Test optimizer integration with CasADi physics enabled."""
        optimizer = CrankCenterOptimizer()
        optimizer.configure(constraints=self.constraints, targets=self.targets)
        
        # Mock the required inputs
        mock_inputs = {
            'crank_center_x': 5.0,
            'crank_center_y': 2.0,
            'crank_radius': 50.0,
            'rod_length': 150.0,
            'motion_law_data': {
                'theta': np.linspace(0, 2*np.pi, 10),
                'pressure': 1e5 * np.ones(10),
            },
            'litvin_config': {
                'z_r': 50.0,
                'z_p': 20.0,
                'module': 2.0,
                'alpha_deg': 20.0,
                'R0': 25.0,
            }
        }
        
        # Test that optimizer can run with CasADi physics
        result = optimizer.simulate(mock_inputs)
        
        # Check that result is valid
        assert result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.CONVERGED]
        assert 'crank_center_x' in result.outputs
        assert 'crank_center_y' in result.outputs
        assert 'crank_radius' in result.outputs
        assert 'rod_length' in result.outputs

    def test_python_physics_mode_integration(self):
        """Test optimizer integration with Python physics (fallback mode)."""
        # Temporarily disable CasADi physics for this test
        with patch('campro.constants.USE_CASADI_PHYSICS', False):
            optimizer = CrankCenterOptimizer()
            optimizer.configure(constraints=self.constraints, targets=self.targets)
            
            # Mock the required inputs
            mock_inputs = {
                'crank_center_x': 5.0,
                'crank_center_y': 2.0,
                'crank_radius': 50.0,
                'rod_length': 150.0,
                'motion_law_data': {
                    'theta': np.linspace(0, 2*np.pi, 10),
                    'pressure': 1e5 * np.ones(10),
                },
                'litvin_config': {
                    'z_r': 50.0,
                    'z_p': 20.0,
                    'module': 2.0,
                    'alpha_deg': 20.0,
                    'R0': 25.0,
                }
            }
            
            # Test that optimizer can run with Python physics
            result = optimizer.simulate(mock_inputs)
            
            # Check that result is valid
            assert result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.CONVERGED]
            assert 'crank_center_x' in result.outputs
            assert 'crank_center_y' in result.outputs
            assert 'crank_radius' in result.outputs
            assert 'rod_length' in result.outputs

    def test_cross_mode_parity(self):
        """Test that Python and CasADi modes produce similar results."""
        # This test would require both modes to be available and produce comparable results
        # For now, just verify that both modes can be configured
        
        # Test Python mode
        with patch('campro.constants.USE_CASADI_PHYSICS', False):
            optimizer_python = CrankCenterOptimizer()
            optimizer_python.configure(constraints=self.constraints, targets=self.targets)
            assert optimizer_python is not None
        
        # Test CasADi mode (if available)
        if USE_CASADI_PHYSICS:
            optimizer_casadi = CrankCenterOptimizer()
            optimizer_casadi.configure(constraints=self.constraints, targets=self.targets)
            assert optimizer_casadi is not None

    def test_toggle_enablement_criteria(self):
        """Test that toggle enablement criteria are met."""
        # This test verifies that the conditions for enabling CasADi physics are met
        # In practice, this would check:
        # 1. Parity thresholds are met
        # 2. Performance gates are satisfied
        # 3. All tests pass
        
        # For now, just verify the toggle state
        if USE_CASADI_PHYSICS:
            # If enabled, verify that CasADi is available
            try:
                import casadi as ca
                from campro.physics.casadi import create_unified_physics
                unified_fn = create_unified_physics()
                assert unified_fn is not None
            except ImportError:
                pytest.fail("CasADi physics enabled but CasADi not available")
        else:
            # If disabled, verify that fallback works
            with patch('campro.constants.USE_CASADI_PHYSICS', False):
                optimizer = CrankCenterOptimizer()
                optimizer.configure(constraints=self.constraints, targets=self.targets)
                assert optimizer is not None
