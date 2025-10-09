"""
Unit tests for Phase 1 physics foundation modules.

Tests the torque analysis, side-loading analysis, and crank kinematics modules
that form the foundation for crank center optimization.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from campro.physics.base import PhysicsStatus
from campro.physics.geometry.litvin import LitvinGearGeometry
from campro.physics.kinematics.crank_kinematics import (
    CrankKinematics,
)
from campro.physics.mechanics.side_loading import SideLoadAnalyzer
from campro.physics.mechanics.torque_analysis import (
    PistonTorqueCalculator,
)


class TestPistonTorqueCalculator:
    """Test cases for PistonTorqueCalculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PistonTorqueCalculator()

        # Mock gear geometry
        self.mock_gear_geometry = Mock(spec=LitvinGearGeometry)
        self.mock_gear_geometry.pressure_angle = np.radians(20.0)

        # Configure calculator
        self.calculator.configure(
            crank_radius=50.0,
            rod_length=150.0,
            gear_geometry=self.mock_gear_geometry,
        )

        # Create test motion law data
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.motion_law_data = {
            "theta": self.theta,
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
        }

        # Create test load profile
        self.load_profile = 1000.0 * np.ones_like(self.theta)

        # Test crank center offset
        self.crank_center_offset = (5.0, 2.0)

    def test_configure_valid_parameters(self):
        """Test configuration with valid parameters."""
        calculator = PistonTorqueCalculator()
        mock_gear = Mock(spec=LitvinGearGeometry)

        calculator.configure(
            crank_radius=50.0,
            rod_length=150.0,
            gear_geometry=mock_gear,
        )

        assert calculator.is_configured()

    def test_configure_invalid_parameters(self):
        """Test configuration with invalid parameters."""
        calculator = PistonTorqueCalculator()
        mock_gear = Mock(spec=LitvinGearGeometry)

        # Test negative crank radius
        with pytest.raises(ValueError, match="Crank radius must be positive"):
            calculator.configure(
                crank_radius=-50.0,
                rod_length=150.0,
                gear_geometry=mock_gear,
            )

        # Test negative rod length
        with pytest.raises(ValueError, match="Rod length must be positive"):
            calculator.configure(
                crank_radius=50.0,
                rod_length=-150.0,
                gear_geometry=mock_gear,
            )

        # Test invalid gear geometry
        with pytest.raises(ValueError, match="gear_geometry must be a LitvinGearGeometry instance"):
            calculator.configure(
                crank_radius=50.0,
                rod_length=150.0,
                gear_geometry="invalid",
            )

    def test_compute_instantaneous_torque(self):
        """Test instantaneous torque computation."""
        torque = self.calculator.compute_instantaneous_torque(
            piston_force=1000.0,
            crank_angle=np.pi/4,
            crank_center_offset=self.crank_center_offset,
            pressure_angle=np.radians(20.0),
        )

        assert isinstance(torque, float)
        assert not np.isnan(torque)
        assert not np.isinf(torque)

    def test_compute_cycle_average_torque(self):
        """Test cycle-averaged torque computation."""
        avg_torque = self.calculator.compute_cycle_average_torque(
            self.motion_law_data,
            self.load_profile,
            self.crank_center_offset,
        )

        assert isinstance(avg_torque, float)
        assert not np.isnan(avg_torque)
        assert not np.isinf(avg_torque)

    def test_simulate_success(self):
        """Test successful simulation."""
        inputs = {
            "motion_law_data": self.motion_law_data,
            "load_profile": self.load_profile,
            "crank_center_offset": self.crank_center_offset,
        }

        result = self.calculator.simulate(inputs)

        assert result.status == PhysicsStatus.COMPLETED
        assert result.is_successful
        assert "instantaneous_torque" in result.data
        assert "cycle_average_torque" in result.data
        assert "crank_angles" in result.data

    def test_simulate_invalid_inputs(self):
        """Test simulation with invalid inputs."""
        # Test missing motion law data
        inputs = {
            "load_profile": self.load_profile,
            "crank_center_offset": self.crank_center_offset,
        }

        result = self.calculator.simulate(inputs)
        assert result.status == PhysicsStatus.FAILED
        assert result.error_message is not None

        # Test mismatched array lengths
        inputs = {
            "motion_law_data": self.motion_law_data,
            "load_profile": self.load_profile[:50],  # Different length
            "crank_center_offset": self.crank_center_offset,
        }

        result = self.calculator.simulate(inputs)
        assert result.status == PhysicsStatus.FAILED
        assert result.error_message is not None


class TestSideLoadAnalyzer:
    """Test cases for SideLoadAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SideLoadAnalyzer()

        # Configure analyzer
        self.piston_geometry = {
            "bore_diameter": 100.0,
            "piston_clearance": 0.1,
            "rod_length": 150.0,
            "crank_radius": 50.0,
        }
        self.analyzer.configure(piston_geometry=self.piston_geometry)

        # Create test motion law data
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.motion_law_data = {
            "theta": self.theta,
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
        }

        # Create test load profile
        self.load_profile = 1000.0 * np.ones_like(self.theta)

        # Test crank center offset
        self.crank_center_offset = (5.0, 2.0)

        # Create phase arrays
        self.compression_phases = np.zeros_like(self.theta, dtype=bool)
        self.compression_phases[20:40] = True  # Some compression phases

        self.combustion_phases = np.zeros_like(self.theta, dtype=bool)
        self.combustion_phases[60:80] = True  # Some combustion phases

    def test_configure_valid_parameters(self):
        """Test configuration with valid parameters."""
        analyzer = SideLoadAnalyzer()
        piston_geometry = {
            "bore_diameter": 100.0,
            "piston_clearance": 0.1,
            "rod_length": 150.0,
            "crank_radius": 50.0,
        }

        analyzer.configure(piston_geometry=piston_geometry)
        assert analyzer.is_configured()

    def test_configure_invalid_parameters(self):
        """Test configuration with invalid parameters."""
        analyzer = SideLoadAnalyzer()

        # Test missing required key
        with pytest.raises(ValueError, match="piston_geometry missing required key"):
            analyzer.configure(piston_geometry={"bore_diameter": 100.0})

        # Test negative value
        piston_geometry = {
            "bore_diameter": -100.0,
            "piston_clearance": 0.1,
            "rod_length": 150.0,
            "crank_radius": 50.0,
        }

        with pytest.raises(ValueError, match="piston_geometry\\[bore_diameter\\] must be positive"):
            analyzer.configure(piston_geometry=piston_geometry)

    def test_compute_side_load_profile(self):
        """Test side-loading profile computation."""
        profile = self.analyzer.compute_side_load_profile(
            self.motion_law_data,
            self.crank_center_offset,
        )

        assert isinstance(profile, np.ndarray)
        assert len(profile) == len(self.theta)
        assert not np.any(np.isnan(profile))
        assert not np.any(np.isinf(profile))

    def test_compute_side_load_penalty(self):
        """Test side-loading penalty computation."""
        profile = self.analyzer.compute_side_load_profile(
            self.motion_law_data,
            self.crank_center_offset,
        )

        penalty = self.analyzer.compute_side_load_penalty(
            profile,
            self.compression_phases,
            self.combustion_phases,
        )

        assert isinstance(penalty, float)
        assert penalty >= 0.0
        assert not np.isnan(penalty)
        assert not np.isinf(penalty)

    def test_simulate_success(self):
        """Test successful simulation."""
        inputs = {
            "motion_law_data": self.motion_law_data,
            "load_profile": self.load_profile,
            "crank_center_offset": self.crank_center_offset,
            "compression_phases": self.compression_phases,
            "combustion_phases": self.combustion_phases,
        }

        result = self.analyzer.simulate(inputs)

        assert result.status == PhysicsStatus.COMPLETED
        assert result.is_successful
        assert "side_load_profile" in result.data
        assert "max_side_load" in result.data
        assert "total_penalty" in result.data


class TestCrankKinematics:
    """Test cases for CrankKinematics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kinematics = CrankKinematics()

        # Configure kinematics
        self.kinematics.configure(
            crank_radius=50.0,
            rod_length=150.0,
        )

        # Create test motion law data
        self.theta = np.linspace(0, 2*np.pi, 100)
        self.motion_law_data = {
            "theta": self.theta,
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
        }

        # Test crank center offset
        self.crank_center_offset = (5.0, 2.0)

    def test_configure_valid_parameters(self):
        """Test configuration with valid parameters."""
        kinematics = CrankKinematics()

        kinematics.configure(
            crank_radius=50.0,
            rod_length=150.0,
        )

        assert kinematics.is_configured()

    def test_configure_invalid_parameters(self):
        """Test configuration with invalid parameters."""
        kinematics = CrankKinematics()

        # Test negative crank radius
        with pytest.raises(ValueError, match="Crank radius must be positive"):
            kinematics.configure(
                crank_radius=-50.0,
                rod_length=150.0,
            )

        # Test negative rod length
        with pytest.raises(ValueError, match="Rod length must be positive"):
            kinematics.configure(
                crank_radius=50.0,
                rod_length=-150.0,
            )

    def test_compute_rod_angle(self):
        """Test rod angle computation."""
        rod_angle = self.kinematics.compute_rod_angle(
            crank_angle=np.pi/4,
            crank_center_offset=self.crank_center_offset,
        )

        assert isinstance(rod_angle, float)
        assert not np.isnan(rod_angle)
        assert not np.isinf(rod_angle)
        assert abs(rod_angle) <= np.pi/2  # Rod angle should be reasonable

    def test_compute_rod_angular_velocity(self):
        """Test rod angular velocity computation."""
        rod_angular_velocity = self.kinematics.compute_rod_angular_velocity(
            crank_angle=np.pi/4,
            crank_angular_velocity=100.0,
            crank_center_offset=self.crank_center_offset,
        )

        assert isinstance(rod_angular_velocity, float)
        assert not np.isnan(rod_angular_velocity)
        assert not np.isinf(rod_angular_velocity)

    def test_compute_corrected_piston_motion(self):
        """Test corrected piston motion computation."""
        corrected_motion = self.kinematics.compute_corrected_piston_motion(
            self.motion_law_data,
            self.crank_center_offset,
        )

        assert isinstance(corrected_motion, dict)
        assert "displacement" in corrected_motion
        assert "velocity" in corrected_motion
        assert "acceleration" in corrected_motion

        for key, values in corrected_motion.items():
            assert isinstance(values, np.ndarray)
            assert len(values) == len(self.theta)
            assert not np.any(np.isnan(values))

    def test_simulate_success(self):
        """Test successful simulation."""
        inputs = {
            "motion_law_data": self.motion_law_data,
            "crank_center_offset": self.crank_center_offset,
            "angular_velocity": 100.0,
        }

        result = self.kinematics.simulate(inputs)

        assert result.status == PhysicsStatus.COMPLETED
        assert result.is_successful
        assert "rod_angles" in result.data
        assert "piston_displacements" in result.data
        assert "max_rod_angle" in result.data


class TestIntegration:
    """Integration tests for Phase 1 modules."""

    def test_torque_and_side_loading_integration(self):
        """Test integration between torque and side-loading analysis."""
        # Create shared test data
        theta = np.linspace(0, 2*np.pi, 100)
        motion_law_data = {
            "theta": theta,
            "displacement": 10.0 * np.sin(theta),
            "velocity": 10.0 * np.cos(theta),
            "acceleration": -10.0 * np.sin(theta),
        }
        load_profile = 1000.0 * np.ones_like(theta)
        crank_center_offset = (5.0, 2.0)

        # Configure torque calculator
        torque_calc = PistonTorqueCalculator()
        mock_gear = Mock(spec=LitvinGearGeometry)
        mock_gear.pressure_angle = np.radians(20.0)
        torque_calc.configure(
            crank_radius=50.0,
            rod_length=150.0,
            gear_geometry=mock_gear,
        )

        # Configure side-load analyzer
        side_load_analyzer = SideLoadAnalyzer()
        piston_geometry = {
            "bore_diameter": 100.0,
            "piston_clearance": 0.1,
            "rod_length": 150.0,
            "crank_radius": 50.0,
        }
        side_load_analyzer.configure(piston_geometry=piston_geometry)

        # Run both analyses
        torque_inputs = {
            "motion_law_data": motion_law_data,
            "load_profile": load_profile,
            "crank_center_offset": crank_center_offset,
        }
        torque_result = torque_calc.simulate(torque_inputs)

        side_load_inputs = {
            "motion_law_data": motion_law_data,
            "load_profile": load_profile,
            "crank_center_offset": crank_center_offset,
        }
        side_load_result = side_load_analyzer.simulate(side_load_inputs)

        # Verify both analyses succeeded
        assert torque_result.status == PhysicsStatus.COMPLETED
        assert side_load_result.status == PhysicsStatus.COMPLETED

        # Verify data consistency
        assert len(torque_result.data["crank_angles"]) == len(side_load_result.data["crank_angles"])
        assert np.allclose(torque_result.data["crank_angles"], side_load_result.data["crank_angles"])

    def test_kinematics_integration(self):
        """Test integration with kinematics analysis."""
        # Create test data
        theta = np.linspace(0, 2*np.pi, 100)
        motion_law_data = {
            "theta": theta,
            "displacement": 10.0 * np.sin(theta),
            "velocity": 10.0 * np.cos(theta),
            "acceleration": -10.0 * np.sin(theta),
        }
        crank_center_offset = (5.0, 2.0)

        # Configure kinematics
        kinematics = CrankKinematics()
        kinematics.configure(
            crank_radius=50.0,
            rod_length=150.0,
        )

        # Run kinematics analysis
        inputs = {
            "motion_law_data": motion_law_data,
            "crank_center_offset": crank_center_offset,
            "angular_velocity": 100.0,
        }
        result = kinematics.simulate(inputs)

        # Verify successful analysis
        assert result.status == PhysicsStatus.COMPLETED
        assert result.is_successful

        # Verify kinematic relationships
        rod_angles = result.data["rod_angles"]
        rod_velocities = result.data["rod_angular_velocities"]

        # Rod angles should be reasonable
        assert np.all(np.abs(rod_angles) <= np.pi/2)

        # Rod velocities should be finite
        assert not np.any(np.isnan(rod_velocities))
        assert not np.any(np.isinf(rod_velocities))
