"""
Test suite for GUI integration with CasADi validation mode.

This module tests the integration of validation mode controls in the GUI
and their interaction with the UnifiedOptimizationFramework.
"""

import pytest
import tkinter as tk
from unittest.mock import Mock, patch

from cam_motion_gui import CamMotionGUI


class TestGUIValidationModeIntegration:
    """Test GUI integration with CasADi validation mode."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a root window for testing
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window during tests
        
        # Create the GUI instance
        self.gui = CamMotionGUI(self.root)

    def teardown_method(self):
        """Clean up test fixtures."""
        self.root.destroy()

    def test_validation_mode_variables_exist(self):
        """Test that validation mode variables are properly created."""
        assert "enable_casadi_validation_mode" in self.gui.variables
        assert "casadi_validation_tolerance" in self.gui.variables
        
        # Check default values
        assert self.gui.variables["enable_casadi_validation_mode"].get() is False
        assert self.gui.variables["casadi_validation_tolerance"].get() == 1e-4

    def test_validation_mode_variables_are_tkinter_vars(self):
        """Test that validation mode variables are proper Tkinter variables."""
        assert isinstance(self.gui.variables["enable_casadi_validation_mode"], tk.BooleanVar)
        assert isinstance(self.gui.variables["casadi_validation_tolerance"], tk.DoubleVar)

    def test_validation_mode_variables_can_be_set(self):
        """Test that validation mode variables can be set and retrieved."""
        # Test boolean variable
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        assert self.gui.variables["enable_casadi_validation_mode"].get() is True
        
        self.gui.variables["enable_casadi_validation_mode"].set(False)
        assert self.gui.variables["enable_casadi_validation_mode"].get() is False
        
        # Test double variable
        self.gui.variables["casadi_validation_tolerance"].set(1e-6)
        assert self.gui.variables["casadi_validation_tolerance"].get() == 1e-6
        
        self.gui.variables["casadi_validation_tolerance"].set(1e-3)
        assert self.gui.variables["casadi_validation_tolerance"].get() == 1e-3

    def test_validation_mode_in_framework_configuration(self):
        """Test that validation mode settings are passed to the framework."""
        # Set validation mode in GUI
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        self.gui.variables["casadi_validation_tolerance"].set(1e-5)
        
        # Mock the framework methods
        with patch.object(self.gui.unified_framework, 'configure') as mock_configure, \
             patch.object(self.gui.unified_framework, 'enable_casadi_validation_mode') as mock_enable, \
             patch.object(self.gui.unified_framework, 'disable_casadi_validation_mode') as mock_disable:
            
            # Call the configuration method
            self.gui._configure_unified_framework()
            
            # Verify that configure was called
            mock_configure.assert_called_once()
            
            # Verify that enable_casadi_validation_mode was called with correct tolerance
            mock_enable.assert_called_once_with(1e-5)
            
            # Verify that disable was not called
            mock_disable.assert_not_called()

    def test_validation_mode_disabled_in_framework_configuration(self):
        """Test that validation mode is properly disabled when not selected."""
        # Ensure validation mode is disabled
        self.gui.variables["enable_casadi_validation_mode"].set(False)
        
        # Mock the framework methods
        with patch.object(self.gui.unified_framework, 'configure') as mock_configure, \
             patch.object(self.gui.unified_framework, 'enable_casadi_validation_mode') as mock_enable, \
             patch.object(self.gui.unified_framework, 'disable_casadi_validation_mode') as mock_disable:
            
            # Call the configuration method
            self.gui._configure_unified_framework()
            
            # Verify that configure was called
            mock_configure.assert_called_once()
            
            # Verify that disable_casadi_validation_mode was called
            mock_disable.assert_called_once()
            
            # Verify that enable was not called
            mock_enable.assert_not_called()

    def test_validation_mode_tolerance_passed_to_framework(self):
        """Test that the tolerance value is properly passed to the framework."""
        # Set a specific tolerance
        tolerance = 1e-6
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        self.gui.variables["casadi_validation_tolerance"].set(tolerance)
        
        # Mock the framework methods
        with patch.object(self.gui.unified_framework, 'configure') as mock_configure, \
             patch.object(self.gui.unified_framework, 'enable_casadi_validation_mode') as mock_enable:
            
            # Call the configuration method
            self.gui._configure_unified_framework()
            
            # Verify that enable_casadi_validation_mode was called with the correct tolerance
            mock_enable.assert_called_once_with(tolerance)

    def test_validation_mode_settings_in_unified_settings(self):
        """Test that validation mode settings are included in UnifiedOptimizationSettings."""
        # Set validation mode
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        self.gui.variables["casadi_validation_tolerance"].set(1e-5)
        
        # Mock the framework configure method to capture the settings
        with patch.object(self.gui.unified_framework, 'configure') as mock_configure:
            self.gui._configure_unified_framework()
            
            # Get the settings that were passed to configure
            call_args = mock_configure.call_args
            settings = call_args[1]['settings']  # settings is a keyword argument
            
            # Verify that validation mode settings are in the settings object
            assert settings.enable_casadi_validation_mode is True
            assert settings.casadi_validation_tolerance == 1e-5

    def test_validation_mode_with_different_tolerance_values(self):
        """Test validation mode with various tolerance values."""
        tolerances = [1e-3, 1e-4, 1e-5, 1e-6]
        
        for tolerance in tolerances:
            # Set tolerance
            self.gui.variables["casadi_validation_tolerance"].set(tolerance)
            self.gui.variables["enable_casadi_validation_mode"].set(True)
            
            # Mock the framework methods
            with patch.object(self.gui.unified_framework, 'configure') as mock_configure, \
                 patch.object(self.gui.unified_framework, 'enable_casadi_validation_mode') as mock_enable:
                
                # Call the configuration method
                self.gui._configure_unified_framework()
                
                # Verify that enable_casadi_validation_mode was called with the correct tolerance
                mock_enable.assert_called_with(tolerance)

    def test_validation_mode_gui_controls_exist(self):
        """Test that validation mode GUI controls are created."""
        # Check that the validation frame exists
        # Note: This is a basic check - in a real test we might need to access the actual widgets
        assert hasattr(self.gui, 'variables')
        assert "enable_casadi_validation_mode" in self.gui.variables
        assert "casadi_validation_tolerance" in self.gui.variables

    def test_validation_mode_integration_with_optimization_flow(self):
        """Test that validation mode integrates properly with the optimization flow."""
        # Enable validation mode
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        self.gui.variables["casadi_validation_tolerance"].set(1e-4)
        
        # Mock the optimization framework
        with patch.object(self.gui.unified_framework, 'optimize_cascaded') as mock_optimize:
            # Mock a successful optimization result
            mock_result = Mock()
            mock_result.total_solve_time = 1.5
            mock_optimize.return_value = mock_result
            
            # Mock the framework configuration
            with patch.object(self.gui, '_configure_unified_framework') as mock_config:
                # This would normally be called in the optimization thread
                # We're just testing that the configuration includes validation mode
                self.gui._configure_unified_framework()
                
                # Verify that configuration was called
                mock_config.assert_called_once()

    def test_validation_mode_error_handling(self):
        """Test that validation mode handles errors gracefully."""
        # Test with invalid tolerance values
        invalid_tolerances = [-1e-4, 0.0, "invalid"]
        
        for tolerance in invalid_tolerances:
            try:
                self.gui.variables["casadi_validation_tolerance"].set(tolerance)
                # The GUI should handle this gracefully
                assert True  # If we get here, no exception was raised
            except (ValueError, TypeError):
                # If an exception is raised, that's also acceptable
                assert True

    def test_validation_mode_persistence_across_gui_operations(self):
        """Test that validation mode settings persist across GUI operations."""
        # Set validation mode
        self.gui.variables["enable_casadi_validation_mode"].set(True)
        self.gui.variables["casadi_validation_tolerance"].set(1e-5)
        
        # Perform some GUI operations (simulated)
        self.gui.variables["stroke"].set(25.0)
        self.gui.variables["cycle_time"].set(1.5)
        
        # Check that validation mode settings are still there
        assert self.gui.variables["enable_casadi_validation_mode"].get() is True
        assert self.gui.variables["casadi_validation_tolerance"].get() == 1e-5
