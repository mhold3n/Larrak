from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class FVMesh:
    """1D finite-volume mesh for cylinder/ports."""

    x: List[float]  # cell centers
    dx: List[float]  # cell widths
    n_cells: int

    def __post_init__(self) -> None:
        self.n_cells = len(self.x)


@dataclass
class ALEMesh:
    """Arbitrary Lagrangian-Eulerian mesh for moving boundaries."""

    x: np.ndarray  # cell centers
    dx: np.ndarray  # cell widths
    x_faces: np.ndarray  # face positions
    n_cells: int

    # Mesh motion parameters
    motion_type: str = "linear"  # "linear", "sinusoidal", "adaptive"
    motion_params: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.motion_params is None:
            self.motion_params = {}
        self.n_cells = len(self.x)


@dataclass
class MeshMotion:
    """Mesh motion configuration for ALE methods."""

    # Boundary motion
    x_left: float  # Left boundary position
    x_right: float  # Right boundary position
    v_left: float  # Left boundary velocity
    v_right: float  # Right boundary velocity

    # Motion type
    motion_type: str = "linear"  # "linear", "sinusoidal", "adaptive"

    # Motion parameters
    motion_params: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.motion_params is None:
            self.motion_params = {}


@dataclass
class MovingBoundaryMesh:
    """Enhanced 1D moving-boundary mesh for opposed-piston cylinder with ALE capabilities.

    Stores face positions and face velocities; provides cell volumes and
    per-cell volume change rates suitable for ALE formulations.
    """

    n_cells: int
    x_left: float
    x_right: float

    # Derived arrays (faces and their velocities)
    x_faces: np.ndarray | None = None
    v_faces: np.ndarray | None = None

    # Enhanced ALE capabilities
    piston_positions: Dict[str, float] = None  # {"left": x_L, "right": x_R}
    piston_velocities: Dict[str, float] = None  # {"left": v_L, "right": v_R}
    mesh_velocity: np.ndarray = None  # Grid velocity at each cell
    volume_change_rate: np.ndarray = None  # dV/dt for each cell

    # Mesh motion parameters
    motion_type: str = "linear"  # "linear", "sinusoidal", "adaptive"
    motion_params: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.piston_positions is None:
            self.piston_positions = {"left": self.x_left, "right": self.x_right}
        if self.piston_velocities is None:
            self.piston_velocities = {"left": 0.0, "right": 0.0}
        if self.motion_params is None:
            self.motion_params = {}

        length = max(self.x_right - self.x_left, 1e-12)
        # Uniform initial face positions
        self.x_faces = np.linspace(self.x_left, self.x_right, self.n_cells + 1)
        # Initialize face velocities to zero
        self.v_faces = np.zeros(self.n_cells + 1)
        # Initialize mesh velocity and volume change rate
        self.mesh_velocity = np.zeros(self.n_cells)
        self.volume_change_rate = np.zeros(self.n_cells)
        # Basic monotonicity guard
        if not np.all(np.diff(self.x_faces) > 0.0):
            raise ValueError("Face coordinates must be strictly increasing")

    def update_piston_boundaries(
        self, x_L: float, x_R: float, v_L: float, v_R: float,
    ) -> None:
        """Update mesh based on piston positions and velocities with ALE motion."""
        if x_R <= x_L:
            raise ValueError("Right boundary must be greater than left boundary")

        # Update piston state
        self.piston_positions["left"] = x_L
        self.piston_positions["right"] = x_R
        self.piston_velocities["left"] = v_L
        self.piston_velocities["right"] = v_R

        # Update boundary positions
        self.x_left = x_L
        self.x_right = x_R

        # Apply mesh motion based on type
        if self.motion_type == "linear":
            self._linear_mesh_motion(x_L, x_R, v_L, v_R)
        elif self.motion_type == "sinusoidal":
            self._sinusoidal_mesh_motion(x_L, x_R, v_L, v_R)
        elif self.motion_type == "adaptive":
            self._adaptive_mesh_motion(x_L, x_R, v_L, v_R)
        else:
            # Default to linear motion
            self._linear_mesh_motion(x_L, x_R, v_L, v_R)

        # Update mesh velocity and volume change rate
        self._update_mesh_velocity()
        self._update_volume_change_rate()

    def _linear_mesh_motion(
        self, x_L: float, x_R: float, v_L: float, v_R: float,
    ) -> None:
        """Apply linear mesh motion."""
        # Update face positions uniformly between new boundaries
        self.x_faces = np.linspace(x_L, x_R, self.n_cells + 1)
        # Linear profile of face velocities from left to right boundary
        self.v_faces = np.linspace(v_L, v_R, self.n_cells + 1)

    def _sinusoidal_mesh_motion(
        self, x_L: float, x_R: float, v_L: float, v_R: float,
    ) -> None:
        """Apply sinusoidal mesh motion."""
        # Get sinusoidal parameters
        amplitude = self.motion_params.get("amplitude", 0.1)
        frequency = self.motion_params.get("frequency", 1.0)
        phase = self.motion_params.get("phase", 0.0)

        # Create base linear mesh
        self._linear_mesh_motion(x_L, x_R, v_L, v_R)

        # Apply sinusoidal perturbation to face positions
        length = x_R - x_L
        for i in range(self.n_cells + 1):
            # Normalized position in domain
            xi = (self.x_faces[i] - x_L) / length if length > 0 else 0.0
            # Sinusoidal perturbation
            perturbation = amplitude * math.sin(2 * math.pi * frequency * xi + phase)
            self.x_faces[i] += perturbation

        # Update face velocities with sinusoidal profile
        for i in range(self.n_cells + 1):
            xi = (self.x_faces[i] - x_L) / length if length > 0 else 0.0
            velocity_perturbation = (
                amplitude
                * 2
                * math.pi
                * frequency
                * math.cos(2 * math.pi * frequency * xi + phase)
            )
            self.v_faces[i] = v_L + (v_R - v_L) * xi + velocity_perturbation

    def _adaptive_mesh_motion(
        self, x_L: float, x_R: float, v_L: float, v_R: float,
    ) -> None:
        """Apply adaptive mesh motion based on solution gradients."""
        # For now, default to linear motion
        # In practice, this would use solution gradients to refine mesh
        self._linear_mesh_motion(x_L, x_R, v_L, v_R)

    def _update_mesh_velocity(self) -> None:
        """Update mesh velocity at cell centers."""
        if self.x_faces is not None and len(self.x_faces) > 1:
            # Mesh velocity at cell centers (average of adjacent face velocities)
            for i in range(self.n_cells):
                self.mesh_velocity[i] = 0.5 * (self.v_faces[i] + self.v_faces[i + 1])

    def _update_volume_change_rate(self) -> None:
        """Update volume change rate for each cell."""
        if self.v_faces is not None and len(self.v_faces) > 1:
            # dV/dt = v_{i+1/2} - v_{i-1/2}
            self.volume_change_rate = np.diff(self.v_faces)

    def calculate_volume_change_rate(self) -> np.ndarray:
        """Calculate dV/dt for each cell due to piston motion."""
        return self.volume_change_rate.copy()

    def update(
        self, *, x_left: float, x_right: float, v_left: float, v_right: float,
    ) -> None:
        """Update boundary positions and face velocities assuming linear mapping.

        - Updates `x_faces` by linear stretch/compress between new boundaries
        - Sets `v_faces` as a linear interpolation from v_left to v_right
        """
        self.update_piston_boundaries(x_left, x_right, v_left, v_right)

    def cell_volumes(self) -> np.ndarray:
        """Return per-cell volumes (unit area -> equal to cell lengths)."""
        assert self.x_faces is not None
        return np.diff(self.x_faces)

    def cell_volume_rate(self) -> np.ndarray:
        """Return per-cell volume time-derivatives dV/dt from face velocities.

        For unit area, dV_i/dt = v_{i+1/2} - v_{i-1/2} using linear face velocities.
        The sum over cells equals (v_right - v_left).
        """
        return self.calculate_volume_change_rate()


def uniform_mesh(length: float, n_cells: int) -> FVMesh:
    """Create uniform 1D mesh."""
    dx = length / n_cells
    x = [dx * (i + 0.5) for i in range(n_cells)]
    dx_list = [dx] * n_cells
    return FVMesh(x=x, dx=dx_list, n_cells=n_cells)


def moving_boundary_mesh(x_L: float, x_R: float, n_cells: int) -> FVMesh:
    """Create 1D mesh for moving boundary (cylinder length = x_R - x_L)."""
    length = max(x_R - x_L, 1e-6)  # avoid zero length
    return uniform_mesh(length, n_cells)


def create_ale_mesh(
    x_left: float,
    x_right: float,
    n_cells: int,
    motion_type: str = "linear",
    motion_params: Optional[Dict[str, Any]] = None,
) -> ALEMesh:
    """
    Create ALE mesh for moving boundaries.

    Args:
        x_left: Left boundary position
        x_right: Right boundary position
        n_cells: Number of cells
        motion_type: Type of mesh motion
        motion_params: Motion parameters

    Returns:
        ALEMesh object
    """
    if motion_params is None:
        motion_params = {}

    # Create uniform mesh initially
    length = max(x_right - x_left, 1e-6)
    dx = length / n_cells

    # Cell centers
    x = np.linspace(x_left + dx / 2, x_right - dx / 2, n_cells)

    # Face positions
    x_faces = np.linspace(x_left, x_right, n_cells + 1)

    # Cell widths
    dx_array = np.full(n_cells, dx)

    return ALEMesh(
        x=x,
        dx=dx_array,
        x_faces=x_faces,
        n_cells=n_cells,
        motion_type=motion_type,
        motion_params=motion_params,
    )


def linear_mesh_motion(
    mesh: ALEMesh,
    x_left_new: float,
    x_right_new: float,
) -> ALEMesh:
    """
    Apply linear mesh motion to ALE mesh.

    Args:
        mesh: Current ALE mesh
        x_left_new: New left boundary position
        x_right_new: New right boundary position

    Returns:
        Updated ALE mesh
    """
    # Calculate new length
    length_new = max(x_right_new - x_left_new, 1e-6)

    # Update face positions
    x_faces_new = np.linspace(x_left_new, x_right_new, mesh.n_cells + 1)

    # Update cell centers
    dx_new = length_new / mesh.n_cells
    x_new = np.linspace(x_left_new + dx_new / 2, x_right_new - dx_new / 2, mesh.n_cells)

    # Update cell widths
    dx_array_new = np.full(mesh.n_cells, dx_new)

    return ALEMesh(
        x=x_new,
        dx=dx_array_new,
        x_faces=x_faces_new,
        n_cells=mesh.n_cells,
        motion_type=mesh.motion_type,
        motion_params=mesh.motion_params,
    )


def sinusoidal_mesh_motion(
    mesh: ALEMesh,
    x_left_new: float,
    x_right_new: float,
    amplitude: float = 0.1,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> ALEMesh:
    """
    Apply sinusoidal mesh motion to ALE mesh.

    Args:
        mesh: Current ALE mesh
        x_left_new: New left boundary position
        x_right_new: New right boundary position
        amplitude: Amplitude of sinusoidal motion
        frequency: Frequency of sinusoidal motion
        phase: Phase of sinusoidal motion

    Returns:
        Updated ALE mesh
    """
    # Calculate new length
    length_new = max(x_right_new - x_left_new, 1e-6)

    # Create base linear mesh
    base_mesh = linear_mesh_motion(mesh, x_left_new, x_right_new)

    # Apply sinusoidal perturbation
    x_perturbed = base_mesh.x.copy()

    for i in range(mesh.n_cells):
        # Normalized position in domain
        xi = (base_mesh.x[i] - x_left_new) / length_new

        # Sinusoidal perturbation
        perturbation = amplitude * math.sin(2 * math.pi * frequency * xi + phase)
        x_perturbed[i] += perturbation

    # Update face positions based on perturbed cell centers
    x_faces_new = np.zeros(mesh.n_cells + 1)
    x_faces_new[0] = x_left_new
    x_faces_new[-1] = x_right_new

    for i in range(1, mesh.n_cells):
        x_faces_new[i] = 0.5 * (x_perturbed[i - 1] + x_perturbed[i])

    # Update cell widths
    dx_new = np.diff(x_faces_new)

    return ALEMesh(
        x=x_perturbed,
        dx=dx_new,
        x_faces=x_faces_new,
        n_cells=mesh.n_cells,
        motion_type=mesh.motion_type,
        motion_params=mesh.motion_params,
    )


def adaptive_mesh_motion(
    mesh: ALEMesh,
    x_left_new: float,
    x_right_new: float,
    refinement_criteria: Optional[Dict[str, Any]] = None,
) -> ALEMesh:
    """
    Apply adaptive mesh motion based on solution gradients.

    Args:
        mesh: Current ALE mesh
        x_left_new: New left boundary position
        x_right_new: New right boundary position
        refinement_criteria: Criteria for mesh refinement

    Returns:
        Updated ALE mesh
    """
    if refinement_criteria is None:
        refinement_criteria = {}

    # Default to linear motion if no criteria provided
    return linear_mesh_motion(mesh, x_left_new, x_right_new)


def calculate_mesh_velocity(
    mesh_old: ALEMesh,
    mesh_new: ALEMesh,
    dt: float,
) -> np.ndarray:
    """
    Calculate mesh velocity for ALE formulation.

    Args:
        mesh_old: Old mesh state
        mesh_new: New mesh state
        dt: Time step

    Returns:
        Mesh velocity at cell centers
    """
    if dt <= 0:
        return np.zeros(mesh_old.n_cells)

    # Mesh velocity at cell centers
    v_mesh = (mesh_new.x - mesh_old.x) / dt

    return v_mesh


def calculate_face_velocity(
    mesh_old: ALEMesh,
    mesh_new: ALEMesh,
    dt: float,
) -> np.ndarray:
    """
    Calculate mesh velocity at faces for ALE formulation.

    Args:
        mesh_old: Old mesh state
        mesh_new: New mesh state
        dt: Time step

    Returns:
        Mesh velocity at faces
    """
    if dt <= 0:
        return np.zeros(mesh_old.n_cells + 1)

    # Mesh velocity at faces
    v_faces = (mesh_new.x_faces - mesh_old.x_faces) / dt

    return v_faces


def check_mesh_quality(mesh: ALEMesh) -> Dict[str, float]:
    """
    Check mesh quality metrics.

    Args:
        mesh: ALE mesh

    Returns:
        Dictionary of quality metrics
    """
    # Aspect ratio (max/min cell width)
    aspect_ratio = np.max(mesh.dx) / np.min(mesh.dx)

    # Skewness (deviation from uniform)
    dx_mean = np.mean(mesh.dx)
    skewness = np.std(mesh.dx) / dx_mean

    # Cell size variation
    size_variation = (np.max(mesh.dx) - np.min(mesh.dx)) / dx_mean

    # Face spacing consistency
    face_spacing = np.diff(mesh.x_faces)
    face_consistency = np.std(face_spacing) / np.mean(face_spacing)

    return {
        "aspect_ratio": aspect_ratio,
        "skewness": skewness,
        "size_variation": size_variation,
        "face_consistency": face_consistency,
    }


def smooth_mesh(
    mesh: ALEMesh,
    smoothing_factor: float = 0.1,
    max_iterations: int = 10,
) -> ALEMesh:
    """
    Apply mesh smoothing to improve quality.

    Args:
        mesh: ALE mesh to smooth
        smoothing_factor: Smoothing strength (0-1)
        max_iterations: Maximum smoothing iterations

    Returns:
        Smoothed ALE mesh
    """
    x_smooth = mesh.x.copy()
    x_faces_smooth = mesh.x_faces.copy()

    for iteration in range(max_iterations):
        # Smooth cell centers
        for i in range(1, mesh.n_cells - 1):
            x_smooth[i] = (1 - smoothing_factor) * x_smooth[
                i
            ] + smoothing_factor * 0.5 * (x_smooth[i - 1] + x_smooth[i + 1])

        # Update face positions
        x_faces_smooth[0] = mesh.x_faces[0]
        x_faces_smooth[-1] = mesh.x_faces[-1]

        for i in range(1, mesh.n_cells):
            x_faces_smooth[i] = 0.5 * (x_smooth[i - 1] + x_smooth[i])

        # Update cell widths
        dx_smooth = np.diff(x_faces_smooth)

        # Check convergence
        if np.max(np.abs(x_smooth - mesh.x)) < 1e-12:
            break

    return ALEMesh(
        x=x_smooth,
        dx=dx_smooth,
        x_faces=x_faces_smooth,
        n_cells=mesh.n_cells,
        motion_type=mesh.motion_type,
        motion_params=mesh.motion_params,
    )


def conservative_remapping_ale(
    U_old: np.ndarray,
    mesh_old: ALEMesh,
    mesh_new: ALEMesh,
    interpolation_method: str = "linear",
    conservation_tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Perform conservative remapping for ALE mesh motion.

    This function conservatively remaps the solution from the old mesh to the new mesh
    while preserving the total conserved quantity (mass, momentum, energy).

    Args:
        U_old: Conservative variables on old mesh
        mesh_old: Old ALE mesh
        mesh_new: New ALE mesh
        interpolation_method: Interpolation method ("linear", "cubic", "weno", "monotonic")
        conservation_tolerance: Tolerance for conservation enforcement

    Returns:
        Conservative variables on new mesh
    """
    from campro.freepiston.net1d.interpolation import (
        InterpolationParameters,
        MeshState,
        conservative_interpolation,
    )

    # Convert ALEMesh to MeshState for interpolation
    mesh_state_old = MeshState(
        x=mesh_old.x,
        dx=mesh_old.dx,
        n_cells=mesh_old.n_cells,
    )

    mesh_state_new = MeshState(
        x=mesh_new.x,
        dx=mesh_new.dx,
        n_cells=mesh_new.n_cells,
    )

    # Set up interpolation parameters
    params = InterpolationParameters(
        method=interpolation_method,
        enforce_conservation=True,
        conservation_tolerance=conservation_tolerance,
        monotonicity_preserving=True,
    )

    # Perform conservative interpolation
    U_new = conservative_interpolation(U_old, mesh_state_old, mesh_state_new, params)

    return U_new


def piston_boundary_ale_motion(
    mesh_old: ALEMesh,
    piston_position_old: Tuple[float, float],
    piston_position_new: Tuple[float, float],
    U_old: np.ndarray,
    interpolation_method: str = "linear",
) -> Tuple[ALEMesh, np.ndarray]:
    """
    Handle piston boundary motion with conservative ALE remapping.

    Args:
        mesh_old: Old ALE mesh
        piston_position_old: Old piston positions (x_L, x_R)
        piston_position_new: New piston positions (x_L, x_R)
        U_old: Conservative variables on old mesh
        interpolation_method: Interpolation method

    Returns:
        Tuple of (new_mesh, new_solution)
    """
    x_L_old, x_R_old = piston_position_old
    x_L_new, x_R_new = piston_position_new

    # Create new mesh based on piston motion
    if mesh_old.motion_type == "linear":
        mesh_new = linear_mesh_motion(mesh_old, x_L_new, x_R_new)
    elif mesh_old.motion_type == "sinusoidal":
        amplitude = mesh_old.motion_params.get("amplitude", 0.1)
        frequency = mesh_old.motion_params.get("frequency", 1.0)
        phase = mesh_old.motion_params.get("phase", 0.0)
        mesh_new = sinusoidal_mesh_motion(
            mesh_old, x_L_new, x_R_new, amplitude, frequency, phase,
        )
    elif mesh_old.motion_type == "adaptive":
        refinement_criteria = mesh_old.motion_params.get("refinement_criteria", {})
        mesh_new = adaptive_mesh_motion(mesh_old, x_L_new, x_R_new, refinement_criteria)
    else:
        # Default to linear motion
        mesh_new = linear_mesh_motion(mesh_old, x_L_new, x_R_new)

    # Perform conservative remapping
    U_new = conservative_remapping_ale(U_old, mesh_old, mesh_new, interpolation_method)

    return mesh_new, U_new


def validate_conservation(
    U_old: np.ndarray,
    mesh_old: ALEMesh,
    U_new: np.ndarray,
    mesh_new: ALEMesh,
    tolerance: float = 1e-12,
) -> Dict[str, float]:
    """
    Validate conservation during ALE remapping.

    Args:
        U_old: Conservative variables on old mesh
        mesh_old: Old ALE mesh
        U_new: Conservative variables on new mesh
        mesh_new: New ALE mesh
        tolerance: Conservation tolerance

    Returns:
        Dictionary with conservation metrics
    """
    # Calculate total conserved quantity on old mesh
    total_old = np.sum(U_old * mesh_old.dx)

    # Calculate total conserved quantity on new mesh
    total_new = np.sum(U_new * mesh_new.dx)

    # Conservation error
    conservation_error = abs(total_new - total_old)
    relative_error = (
        conservation_error / abs(total_old) if abs(total_old) > 1e-15 else 0.0
    )

    # Conservation status
    conservation_ok = conservation_error < tolerance

    return {
        "total_old": total_old,
        "total_new": total_new,
        "conservation_error": conservation_error,
        "relative_error": relative_error,
        "conservation_ok": conservation_ok,
    }


def get_mesh_motion_function(motion_type: str) -> callable:
    """
    Get mesh motion function by type.

    Args:
        motion_type: Type of mesh motion

    Returns:
        Mesh motion function
    """
    functions = {
        "linear": linear_mesh_motion,
        "sinusoidal": sinusoidal_mesh_motion,
        "adaptive": adaptive_mesh_motion,
    }

    if motion_type not in functions:
        raise ValueError(f"Unknown mesh motion type: {motion_type}")

    return functions[motion_type]
