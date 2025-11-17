
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from campro.physics.base import PhysicsStatus
from campro.physics.geometry.litvin import LitvinGearGeometry
from campro.physics.kinematics.crank_kinematics import CrankKinematics
from campro.physics.mechanics.side_loading import SideLoadAnalyzer
from campro.physics.mechanics.torque_analysis import PistonTorqueCalculator

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _motion_law_data() -> dict[str, np.ndarray]:
    theta = np.linspace(0, 2.0 * np.pi, 100)
    return {
        "theta": theta,
        "displacement": 10.0 * np.sin(theta),
        "velocity": 10.0 * np.cos(theta),
        "acceleration": -10.0 * np.sin(theta),
    }


def _mock_gear_geometry() -> Mock:
    mock_gear = Mock(spec=LitvinGearGeometry)
    mock_gear.pressure_angle = np.radians(20.0)
    return mock_gear


def _piston_geometry() -> dict[str, float]:
    return {
        "bore_diameter": 100.0,
        "piston_clearance": 0.1,
        "rod_length": 150.0,
        "crank_radius": 50.0,
    }


def test_piston_torque_calculator_configure() -> None:
    calculator = PistonTorqueCalculator()
    mock_gear = _mock_gear_geometry()
    calculator.configure(crank_radius=50.0, rod_length=150.0, gear_geometry=mock_gear)
    assert calculator.is_configured()

    with pytest.raises(ValueError, match="Crank radius must be positive"):
        calculator.configure(crank_radius=-50.0, rod_length=150.0, gear_geometry=mock_gear)

    with pytest.raises(ValueError, match="Rod length must be positive"):
        calculator.configure(crank_radius=50.0, rod_length=-150.0, gear_geometry=mock_gear)


def test_piston_torque_calculator_compute() -> None:
    calculator = PistonTorqueCalculator()
    calculator.configure(crank_radius=50.0, rod_length=150.0, gear_geometry=_mock_gear_geometry())

    motion_law_data = _motion_law_data()
    load_profile = 1000.0 * np.ones_like(motion_law_data["theta"])
    crank_center_offset = (5.0, 2.0)

    torque = calculator.compute_instantaneous_torque(
        motion_law_data, load_profile, crank_center_offset,
    )
    assert isinstance(torque, np.ndarray)
    assert len(torque) == len(motion_law_data["theta"])
    assert np.all(np.isfinite(torque))

    avg_torque = calculator.compute_cycle_average_torque(torque)
    assert isinstance(avg_torque, float)
    assert np.isfinite(avg_torque)


def test_piston_torque_calculator_simulate() -> None:
    calculator = PistonTorqueCalculator()
    calculator.configure(crank_radius=50.0, rod_length=150.0, gear_geometry=_mock_gear_geometry())

    motion_law_data = _motion_law_data()
    inputs = {
        "motion_law_data": motion_law_data,
        "load_profile": 1000.0 * np.ones_like(motion_law_data["theta"]),
        "crank_center_offset": (5.0, 2.0),
    }

    result = calculator.simulate(inputs)
    assert result.status == PhysicsStatus.COMPLETED
    assert result.is_successful
    assert "torque_profile" in result.data
    assert "average_torque" in result.data


def test_side_load_analyzer_configure() -> None:
    analyzer = SideLoadAnalyzer()
    analyzer.configure(piston_geometry=_piston_geometry())
    assert analyzer.is_configured()

    with pytest.raises(ValueError, match="piston_geometry missing required key"):
        analyzer.configure(piston_geometry={"bore_diameter": 100.0})

    invalid_geometry = _piston_geometry()
    invalid_geometry["bore_diameter"] = -100.0
    with pytest.raises(ValueError, match="must be positive"):
        analyzer.configure(piston_geometry=invalid_geometry)


def test_side_load_analyzer_compute() -> None:
    analyzer = SideLoadAnalyzer()
    analyzer.configure(piston_geometry=_piston_geometry())

    motion_law_data = _motion_law_data()
    crank_center_offset = (5.0, 2.0)

    profile = analyzer.compute_side_load_profile(motion_law_data, crank_center_offset)
    assert isinstance(profile, np.ndarray)
    assert len(profile) == len(motion_law_data["theta"])
    assert np.all(np.isfinite(profile))

    compression_phases = np.zeros_like(motion_law_data["theta"], dtype=bool)
    combustion_phases = np.zeros_like(motion_law_data["theta"], dtype=bool)
    penalty = analyzer.compute_side_load_penalty(profile, compression_phases, combustion_phases)
    assert isinstance(penalty, float)
    assert penalty >= 0.0
    assert np.isfinite(penalty)


def test_side_load_analyzer_simulate() -> None:
    analyzer = SideLoadAnalyzer()
    analyzer.configure(piston_geometry=_piston_geometry())

    motion_law_data = _motion_law_data()
    inputs = {
        "motion_law_data": motion_law_data,
        "crank_center_offset": (5.0, 2.0),
        "load_profile": 1000.0 * np.ones_like(motion_law_data["theta"]),
    }

    result = analyzer.simulate(inputs)
    assert result.status == PhysicsStatus.COMPLETED
    assert result.is_successful
    assert "side_load_profile" in result.data
    assert "penalty" in result.data


def test_crank_kinematics_configure() -> None:
    kinematics = CrankKinematics()
    kinematics.configure(piston_geometry=_piston_geometry())
    assert kinematics.is_configured()

    invalid_geometry = _piston_geometry()
    invalid_geometry["rod_length"] = -150.0
    with pytest.raises(ValueError, match="must be positive"):
        kinematics.configure(piston_geometry=invalid_geometry)


def test_crank_kinematics_simulate() -> None:
    kinematics = CrankKinematics()
    kinematics.configure(piston_geometry=_piston_geometry())

    motion_law_data = _motion_law_data()
    inputs = {
        "motion_law_data": motion_law_data,
        "load_profile": 1000.0 * np.ones_like(motion_law_data["theta"]),
    }

    result = kinematics.simulate(inputs)
    assert result.status == PhysicsStatus.COMPLETED
    assert result.is_successful
    assert "follower_position" in result.data
    assert "follower_velocity" in result.data


def test_torque_and_side_loading_integration() -> None:
    torque_calculator = PistonTorqueCalculator()
    torque_calculator.configure(
        crank_radius=50.0,
        rod_length=150.0,
        gear_geometry=_mock_gear_geometry(),
    )

    side_analyzer = SideLoadAnalyzer()
    side_analyzer.configure(piston_geometry=_piston_geometry())

    motion_law_data = _motion_law_data()
    load_profile = 1000.0 * np.ones_like(motion_law_data["theta"])
    crank_center_offset = (5.0, 2.0)

    torque = torque_calculator.compute_instantaneous_torque(
        motion_law_data, load_profile, crank_center_offset,
    )
    profile = side_analyzer.compute_side_load_profile(
        motion_law_data, crank_center_offset,
    )

    assert torque.shape == profile.shape
    assert np.all(np.isfinite(torque))
    assert np.all(np.isfinite(profile))

    compression_phases = np.zeros_like(motion_law_data["theta"], dtype=bool)
    combustion_phases = np.zeros_like(motion_law_data["theta"], dtype=bool)
    penalty = side_analyzer.compute_side_load_penalty(
        profile, compression_phases, combustion_phases,
    )
    assert penalty >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
