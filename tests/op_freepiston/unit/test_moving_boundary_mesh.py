"""
Unit tests for enhanced moving boundary mesh and gas-structure coupling.

Tests the enhanced MovingBoundaryMesh class with ALE capabilities and
gas-structure coupled time stepping functionality.
"""

import numpy as np
import pytest

from campro.freepiston.net1d.mesh import (
    MovingBoundaryMesh,
    calculate_face_velocity,
    calculate_mesh_velocity,
    check_mesh_quality,
    conservative_remapping_ale,
    create_ale_mesh,
    linear_mesh_motion,
    piston_boundary_ale_motion,
    sinusoidal_mesh_motion,
    smooth_mesh,
    validate_conservation,
)
from campro.freepiston.net1d.stepper import (
    TimeStepParameters,
    calculate_ale_fluxes,
    calculate_piston_forces,
    calculate_source_terms,
    gas_structure_coupled_step,
)


class TestMovingBoundaryMesh:
    """Test enhanced MovingBoundaryMesh class."""

    def test_initialization(self):
        """Test mesh initialization."""
        mesh = MovingBoundaryMesh(n_cells=10, x_left=0.0, x_right=1.0)

        assert mesh.n_cells == 10
        assert mesh.x_left == 0.0
        assert mesh.x_right == 1.0
        assert mesh.motion_type == "linear"
        assert mesh.piston_positions["left"] == 0.0
        assert mesh.piston_positions["right"] == 1.0
        assert mesh.piston_velocities["left"] == 0.0
        assert mesh.piston_velocities["right"] == 0.0
        assert len(mesh.x_faces) == 11
        assert len(mesh.v_faces) == 11
        assert len(mesh.mesh_velocity) == 10
        assert len(mesh.volume_change_rate) == 10

    def test_linear_mesh_motion(self):
        """Test linear mesh motion."""
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        # Update piston boundaries
        mesh.update_piston_boundaries(0.1, 0.9, 0.5, -0.5)

        assert mesh.x_left == 0.1
        assert mesh.x_right == 0.9
        assert mesh.piston_positions["left"] == 0.1
        assert mesh.piston_positions["right"] == 0.9
        assert mesh.piston_velocities["left"] == 0.5
        assert mesh.piston_velocities["right"] == -0.5

        # Check face positions are linear
        expected_faces = np.linspace(0.1, 0.9, 6)
        np.testing.assert_array_almost_equal(mesh.x_faces, expected_faces)

        # Check face velocities are linear
        expected_velocities = np.linspace(0.5, -0.5, 6)
        np.testing.assert_array_almost_equal(mesh.v_faces, expected_velocities)

    def test_sinusoidal_mesh_motion(self):
        """Test sinusoidal mesh motion."""
        mesh = MovingBoundaryMesh(
            n_cells=5,
            x_left=0.0,
            x_right=1.0,
            motion_type="sinusoidal",
            motion_params={"amplitude": 0.1, "frequency": 1.0, "phase": 0.0},
        )

        # Update piston boundaries
        mesh.update_piston_boundaries(0.1, 0.9, 0.5, -0.5)

        assert mesh.x_left == 0.1
        assert mesh.x_right == 0.9

        # Check that face positions are not linear (have sinusoidal perturbation)
        linear_faces = np.linspace(0.1, 0.9, 6)
        assert not np.allclose(mesh.x_faces, linear_faces, atol=1e-6)

    def test_cell_volumes(self):
        """Test cell volume calculation."""
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        volumes = mesh.cell_volumes()
        expected_volume = 0.2  # 1.0 / 5

        assert len(volumes) == 5
        np.testing.assert_array_almost_equal(volumes, [expected_volume] * 5)

    def test_volume_change_rate(self):
        """Test volume change rate calculation."""
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        # Set face velocities
        mesh.v_faces = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        mesh._update_volume_change_rate()

        volume_rates = mesh.calculate_volume_change_rate()
        expected_rates = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        np.testing.assert_array_almost_equal(volume_rates, expected_rates)

    def test_mesh_velocity_update(self):
        """Test mesh velocity update."""
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        # Set face velocities
        mesh.v_faces = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        mesh._update_mesh_velocity()

        expected_velocities = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
        np.testing.assert_array_almost_equal(mesh.mesh_velocity, expected_velocities)

    def test_invalid_boundaries(self):
        """Test error handling for invalid boundaries."""
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        with pytest.raises(ValueError, match="Right boundary must be greater than left boundary"):
            mesh.update_piston_boundaries(0.5, 0.3, 0.0, 0.0)


class TestALEMeshFunctions:
    """Test ALE mesh utility functions."""

    def test_create_ale_mesh(self):
        """Test ALE mesh creation."""
        mesh = create_ale_mesh(0.0, 1.0, 10, "linear")

        assert mesh.n_cells == 10
        assert mesh.motion_type == "linear"
        assert len(mesh.x) == 10
        assert len(mesh.dx) == 10
        assert len(mesh.x_faces) == 11

    def test_linear_mesh_motion_function(self):
        """Test linear mesh motion function."""
        mesh = create_ale_mesh(0.0, 1.0, 5, "linear")
        new_mesh = linear_mesh_motion(mesh, 0.1, 0.9)

        assert new_mesh.n_cells == 5
        expected_faces = np.linspace(0.1, 0.9, 6)
        np.testing.assert_array_almost_equal(new_mesh.x_faces, expected_faces)

    def test_sinusoidal_mesh_motion_function(self):
        """Test sinusoidal mesh motion function."""
        mesh = create_ale_mesh(0.0, 1.0, 5, "sinusoidal")
        new_mesh = sinusoidal_mesh_motion(mesh, 0.1, 0.9, amplitude=0.1, frequency=1.0)

        assert new_mesh.n_cells == 5
        # Check that faces are not linear
        linear_faces = np.linspace(0.1, 0.9, 6)
        assert not np.allclose(new_mesh.x_faces, linear_faces, atol=1e-6)

    def test_calculate_mesh_velocity(self):
        """Test mesh velocity calculation."""
        mesh_old = create_ale_mesh(0.0, 1.0, 5, "linear")
        mesh_new = create_ale_mesh(0.1, 0.9, 5, "linear")

        v_mesh = calculate_mesh_velocity(mesh_old, mesh_new, 0.1)

        assert len(v_mesh) == 5
        # Should be non-zero due to mesh motion
        assert not np.allclose(v_mesh, 0.0)

    def test_calculate_face_velocity(self):
        """Test face velocity calculation."""
        mesh_old = create_ale_mesh(0.0, 1.0, 5, "linear")
        mesh_new = create_ale_mesh(0.1, 0.9, 5, "linear")

        v_faces = calculate_face_velocity(mesh_old, mesh_new, 0.1)

        assert len(v_faces) == 6
        # Should be non-zero due to mesh motion
        assert not np.allclose(v_faces, 0.0)

    def test_mesh_quality_check(self):
        """Test mesh quality metrics."""
        mesh = create_ale_mesh(0.0, 1.0, 10, "linear")

        quality = check_mesh_quality(mesh)

        assert "aspect_ratio" in quality
        assert "skewness" in quality
        assert "size_variation" in quality
        assert "face_consistency" in quality

        # For uniform mesh, aspect ratio should be 1.0
        assert abs(quality["aspect_ratio"] - 1.0) < 1e-12

    def test_mesh_smoothing(self):
        """Test mesh smoothing."""
        mesh = create_ale_mesh(0.0, 1.0, 10, "linear")

        # Add some perturbation
        mesh.x[5] += 0.1

        smoothed_mesh = smooth_mesh(mesh, smoothing_factor=0.1)

        assert smoothed_mesh.n_cells == mesh.n_cells
        assert smoothed_mesh.motion_type == mesh.motion_type


class TestGasStructureCoupling:
    """Test gas-structure coupling functions."""

    def test_gas_structure_coupled_step(self):
        """Test gas-structure coupled time step."""
        # Create test mesh
        mesh = MovingBoundaryMesh(n_cells=5, x_left=0.0, x_right=1.0)

        # Create test state
        U = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],  # density
                      [0.0, 0.0, 0.0, 0.0, 0.0],  # momentum
                      [2.5, 2.5, 2.5, 2.5, 2.5]])  # energy

        # Create piston forces
        piston_forces = {
            "x_L": 0.1,
            "x_R": 0.9,
            "v_L": 0.5,
            "v_R": -0.5,
        }

        # Create time step parameters
        params = TimeStepParameters(
            rtol=1e-6,
            atol=1e-8,
            dt_max=1e-3,
            dt_min=1e-12,
        )

        # Add parameters for gas dynamics
        params_dict = {
            "gamma": 1.4,
            "bore": 0.1,
        }

        # Perform coupled step
        result = gas_structure_coupled_step(U, mesh, piston_forces, 1e-4, params_dict)

        assert result.success
        assert result.dt_used == 1e-4
        assert result.iterations == 1
        assert "successful" in result.message

    def test_calculate_ale_fluxes(self):
        """Test ALE flux calculation."""
        # Create test mesh
        mesh = MovingBoundaryMesh(n_cells=3, x_left=0.0, x_right=1.0)

        # Create test state
        U = np.array([[1.0, 1.0, 1.0],  # density
                      [0.0, 0.0, 0.0],  # momentum
                      [2.5, 2.5, 2.5]])  # energy

        # Create time step parameters
        params = TimeStepParameters()
        params_dict = {"gamma": 1.4}

        # Calculate ALE fluxes
        flux = calculate_ale_fluxes(U, mesh, params_dict)

        assert flux.shape == U.shape
        # Flux should be finite
        assert np.all(np.isfinite(flux))

    def test_calculate_source_terms(self):
        """Test source term calculation."""
        # Create test mesh
        mesh = MovingBoundaryMesh(n_cells=3, x_left=0.0, x_right=1.0)
        mesh.volume_change_rate = np.array([0.1, 0.1, 0.1])

        # Create test state
        U = np.array([[1.0, 1.0, 1.0],  # density
                      [0.0, 0.0, 0.0],  # momentum
                      [2.5, 2.5, 2.5]])  # energy

        # Create time step parameters
        params = TimeStepParameters()

        # Calculate source terms
        source = calculate_source_terms(U, mesh, 1e-4, params)

        assert source.shape == U.shape
        # Source terms should be finite
        assert np.all(np.isfinite(source))

    def test_calculate_piston_forces(self):
        """Test piston force calculation."""
        # Create test mesh
        mesh = MovingBoundaryMesh(n_cells=3, x_left=0.0, x_right=1.0)
        mesh.piston_positions = {"left": 0.1, "right": 0.9}
        mesh.piston_velocities = {"left": 0.5, "right": -0.5}

        # Create test state
        U = np.array([[1.0, 1.0, 1.0],  # density
                      [0.0, 0.0, 0.0],  # momentum
                      [2.5, 2.5, 2.5]])  # energy

        # Create time step parameters
        params = TimeStepParameters()
        params_dict = {"gamma": 1.4, "bore": 0.1}

        # Calculate piston forces
        forces = calculate_piston_forces(U, mesh, params_dict)

        assert "x_L" in forces
        assert "x_R" in forces
        assert "v_L" in forces
        assert "v_R" in forces
        assert "F_gas_L" in forces
        assert "F_gas_R" in forces

        assert forces["x_L"] == 0.1
        assert forces["x_R"] == 0.9
        assert forces["v_L"] == 0.5
        assert forces["v_R"] == -0.5

        # Gas forces should be finite
        assert np.isfinite(forces["F_gas_L"])
        assert np.isfinite(forces["F_gas_R"])


class TestConservativeRemapping:
    """Test conservative remapping functions."""

    def test_conservative_remapping_ale(self):
        """Test conservative ALE remapping."""
        # Create old and new meshes
        mesh_old = create_ale_mesh(0.0, 1.0, 5, "linear")
        mesh_new = create_ale_mesh(0.1, 0.9, 5, "linear")

        # Create test solution
        U_old = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Perform conservative remapping
        U_new = conservative_remapping_ale(U_old, mesh_old, mesh_new)

        assert len(U_new) == 5
        assert np.all(np.isfinite(U_new))

    def test_piston_boundary_ale_motion(self):
        """Test piston boundary ALE motion."""
        # Create mesh
        mesh_old = create_ale_mesh(0.0, 1.0, 5, "linear")

        # Create test solution
        U_old = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Perform piston boundary motion
        mesh_new, U_new = piston_boundary_ale_motion(
            mesh_old, (0.0, 1.0), (0.1, 0.9), U_old,
        )

        assert mesh_new.n_cells == 5
        assert len(U_new) == 5
        assert np.all(np.isfinite(U_new))

    def test_validate_conservation(self):
        """Test conservation validation."""
        # Create meshes with same domain size
        mesh_old = create_ale_mesh(0.0, 1.0, 5, "linear")
        mesh_new = create_ale_mesh(0.0, 1.0, 5, "linear")

        # Create test solutions
        U_old = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        U_new = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Validate conservation
        conservation = validate_conservation(U_old, mesh_old, U_new, mesh_new)

        assert "total_old" in conservation
        assert "total_new" in conservation
        assert "conservation_error" in conservation
        assert "relative_error" in conservation
        assert "conservation_ok" in conservation

        # For identical solutions and meshes, conservation should be perfect
        assert conservation["conservation_ok"]


if __name__ == "__main__":
    pytest.main([__file__])
