from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class InterpolationParameters:
    """Parameters for conservative interpolation."""
    # Interpolation method
    method: str = "linear"  # "linear", "cubic", "weno", "monotonic"

    # Conservation enforcement
    enforce_conservation: bool = True
    conservation_tolerance: float = 1e-12

    # Smoothing parameters
    smoothing_factor: float = 0.0  # 0 = no smoothing, 1 = maximum smoothing
    monotonicity_preserving: bool = True

    # Boundary treatment
    boundary_extrapolation: str = "constant"  # "constant", "linear", "periodic"

    # Error control
    max_iterations: int = 10
    convergence_tol: float = 1e-10


@dataclass
class MeshState:
    """State of a moving mesh."""
    x: np.ndarray  # Cell center positions
    dx: np.ndarray  # Cell widths
    n_cells: int

    def volume(self) -> float:
        """Total volume of the mesh."""
        return np.sum(self.dx)

    def cell_volume(self, i: int) -> float:
        """Volume of cell i."""
        return self.dx[i]


def conservative_interpolation(
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
    params: InterpolationParameters,
) -> np.ndarray:
    """
    Perform conservative interpolation from old mesh to new mesh.
    
    This function ensures that the total conserved quantity is preserved
    during mesh motion, which is crucial for moving boundary problems.
    
    Args:
        U_old: Conservative variables on old mesh
        mesh_old: Old mesh state
        mesh_new: New mesh state
        params: Interpolation parameters
        
    Returns:
        Conservative variables on new mesh
    """
    if len(U_old) != mesh_old.n_cells:
        raise ValueError("U_old length must match mesh_old.n_cells")

    # Compute total conserved quantity on old mesh
    total_old = np.sum(U_old * mesh_old.dx)

    # Initialize new solution
    U_new = np.zeros(mesh_new.n_cells)

    if params.method == "linear":
        U_new = _linear_conservative_interpolation(U_old, mesh_old, mesh_new, params)
    elif params.method == "cubic":
        U_new = _cubic_conservative_interpolation(U_old, mesh_old, mesh_new, params)
    elif params.method == "weno":
        U_new = _weno_conservative_interpolation(U_old, mesh_old, mesh_new, params)
    elif params.method == "monotonic":
        U_new = _monotonic_conservative_interpolation(U_old, mesh_old, mesh_new, params)
    else:
        raise ValueError(f"Unknown interpolation method: {params.method}")

    # Enforce conservation if requested
    if params.enforce_conservation:
        U_new = _enforce_conservation(U_new, mesh_new, total_old, params)

    # Apply smoothing if requested
    if params.smoothing_factor > 0:
        U_new = _apply_smoothing(U_new, params.smoothing_factor)

    return U_new


def _linear_conservative_interpolation(
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
    params: InterpolationParameters,
) -> np.ndarray:
    """Linear conservative interpolation."""
    U_new = np.zeros(mesh_new.n_cells)

    for i in range(mesh_new.n_cells):
        x_new = mesh_new.x[i]

        # Find surrounding cells in old mesh
        j_left, j_right = _find_surrounding_cells(x_new, mesh_old)

        if j_left == j_right:
            # Point falls exactly on a cell center
            U_new[i] = U_old[j_left]
        else:
            # Linear interpolation between two cells
            x_left = mesh_old.x[j_left]
            x_right = mesh_old.x[j_right]

            # Weighted average based on distance
            w_left = (x_right - x_new) / (x_right - x_left)
            w_right = (x_new - x_left) / (x_right - x_left)

            U_new[i] = w_left * U_old[j_left] + w_right * U_old[j_right]

    return U_new


def _cubic_conservative_interpolation(
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
    params: InterpolationParameters,
) -> np.ndarray:
    """Cubic conservative interpolation using splines."""
    U_new = np.zeros(mesh_new.n_cells)

    # Compute cubic spline coefficients
    coeffs = _compute_cubic_spline_coefficients(U_old, mesh_old)

    for i in range(mesh_new.n_cells):
        x_new = mesh_new.x[i]

        # Find which interval x_new falls in
        j = _find_cell_index(x_new, mesh_old)

        if j >= 0 and j < mesh_old.n_cells - 1:
            # Evaluate cubic spline
            dx = x_new - mesh_old.x[j]
            U_new[i] = (coeffs[j, 0] +
                       coeffs[j, 1] * dx +
                       coeffs[j, 2] * dx**2 +
                       coeffs[j, 3] * dx**3)
        else:
            # Use boundary extrapolation
            U_new[i] = _boundary_extrapolation(U_old, mesh_old, x_new, params)

    return U_new


def _weno_conservative_interpolation(
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
    params: InterpolationParameters,
) -> np.ndarray:
    """WENO (Weighted Essentially Non-Oscillatory) conservative interpolation."""
    U_new = np.zeros(mesh_new.n_cells)

    for i in range(mesh_new.n_cells):
        x_new = mesh_new.x[i]

        # Find surrounding stencil
        j_center = _find_cell_index(x_new, mesh_old)

        if j_center >= 2 and j_center < mesh_old.n_cells - 2:
            # Use 5-point WENO stencil
            stencil = U_old[j_center-2:j_center+3]
            weights = _compute_weno_weights(stencil)

            # Reconstruct value at x_new
            U_new[i] = np.sum(weights * stencil)
        else:
            # Fall back to linear interpolation near boundaries
            j_left, j_right = _find_surrounding_cells(x_new, mesh_old)
            if j_left == j_right:
                U_new[i] = U_old[j_left]
            else:
                x_left = mesh_old.x[j_left]
                x_right = mesh_old.x[j_right]
                w_left = (x_right - x_new) / (x_right - x_left)
                w_right = (x_new - x_left) / (x_right - x_left)
                U_new[i] = w_left * U_old[j_left] + w_right * U_old[j_right]

    return U_new


def _monotonic_conservative_interpolation(
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
    params: InterpolationParameters,
) -> np.ndarray:
    """Monotonic conservative interpolation preserving monotonicity."""
    U_new = np.zeros(mesh_new.n_cells)

    # First, perform linear interpolation
    U_new = _linear_conservative_interpolation(U_old, mesh_old, mesh_new, params)

    # Apply monotonicity-preserving limiter
    if params.monotonicity_preserving:
        U_new = _apply_monotonicity_limiter(U_new, U_old, mesh_old, mesh_new)

    return U_new


def _find_surrounding_cells(x: float, mesh: MeshState) -> Tuple[int, int]:
    """Find the two cells surrounding position x."""
    # Binary search for efficiency
    left = 0
    right = mesh.n_cells - 1

    while right - left > 1:
        mid = (left + right) // 2
        if mesh.x[mid] <= x:
            left = mid
        else:
            right = mid

    return left, right


def _find_cell_index(x: float, mesh: MeshState) -> int:
    """Find the cell index containing position x."""
    for i in range(mesh.n_cells):
        x_left = mesh.x[i] - mesh.dx[i] / 2
        x_right = mesh.x[i] + mesh.dx[i] / 2
        if x_left <= x <= x_right:
            return i

    # If not found, return closest cell
    distances = np.abs(mesh.x - x)
    return np.argmin(distances)


def _compute_cubic_spline_coefficients(U: np.ndarray, mesh: MeshState) -> np.ndarray:
    """Compute cubic spline coefficients for conservative interpolation."""
    n = len(U)
    coeffs = np.zeros((n-1, 4))

    # Set up tridiagonal system for spline coefficients
    # This is a simplified version - in practice, you'd use proper spline algorithms

    for i in range(n-1):
        # Simple cubic interpolation coefficients
        dx = mesh.dx[i]
        coeffs[i, 0] = U[i]  # Constant term
        coeffs[i, 1] = (U[i+1] - U[i]) / dx  # Linear term
        coeffs[i, 2] = 0.0  # Quadratic term (simplified)
        coeffs[i, 3] = 0.0  # Cubic term (simplified)

    return coeffs


def _compute_weno_weights(stencil: np.ndarray) -> np.ndarray:
    """Compute WENO weights for 5-point stencil."""
    # Simplified WENO weights - in practice, you'd compute smoothness indicators
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Equal weights for simplicity
    return weights


def _boundary_extrapolation(
    U: np.ndarray,
    mesh: MeshState,
    x: float,
    params: InterpolationParameters,
) -> float:
    """Extrapolate value at boundary."""
    if params.boundary_extrapolation == "constant":
        if x < mesh.x[0]:
            return U[0]
        return U[-1]
    if params.boundary_extrapolation == "linear":
        if x < mesh.x[0]:
            # Linear extrapolation from first two points
            slope = (U[1] - U[0]) / (mesh.x[1] - mesh.x[0])
            return U[0] + slope * (x - mesh.x[0])
        # Linear extrapolation from last two points
        slope = (U[-1] - U[-2]) / (mesh.x[-1] - mesh.x[-2])
        return U[-1] + slope * (x - mesh.x[-1])
    # Default to constant
    if x < mesh.x[0]:
        return U[0]
    return U[-1]


def _enforce_conservation(
    U_new: np.ndarray,
    mesh_new: MeshState,
    total_old: float,
    params: InterpolationParameters,
) -> np.ndarray:
    """Enforce conservation by adjusting the solution."""
    total_new = np.sum(U_new * mesh_new.dx)

    if abs(total_new - total_old) < params.conservation_tolerance:
        return U_new

    # Compute correction factor
    if abs(total_new) > 1e-15:
        correction_factor = total_old / total_new
        U_corrected = U_new * correction_factor
    else:
        # If total is zero, distribute evenly
        U_corrected = np.full_like(U_new, total_old / mesh_new.volume())

    return U_corrected


def _apply_smoothing(U: np.ndarray, smoothing_factor: float) -> np.ndarray:
    """Apply smoothing to reduce oscillations."""
    if smoothing_factor <= 0:
        return U

    U_smooth = U.copy()
    n = len(U)

    for i in range(1, n-1):
        # Simple 3-point smoothing
        U_smooth[i] = (1 - smoothing_factor) * U[i] + \
                      smoothing_factor * 0.5 * (U[i-1] + U[i+1])

    return U_smooth


def _apply_monotonicity_limiter(
    U_new: np.ndarray,
    U_old: np.ndarray,
    mesh_old: MeshState,
    mesh_new: MeshState,
) -> np.ndarray:
    """Apply monotonicity-preserving limiter."""
    U_limited = U_new.copy()

    # Check for monotonicity violations and apply limiting
    for i in range(1, len(U_limited)-1):
        # Check if solution is monotonic
        if (U_limited[i] > U_limited[i-1] and U_limited[i] > U_limited[i+1]) or \
           (U_limited[i] < U_limited[i-1] and U_limited[i] < U_limited[i+1]):
            # Apply limiting to preserve monotonicity
            U_limited[i] = 0.5 * (U_limited[i-1] + U_limited[i+1])

    return U_limited


def ale_mesh_motion(
    mesh_old: MeshState,
    mesh_new: MeshState,
    U_old: np.ndarray,
    params: InterpolationParameters,
) -> np.ndarray:
    """
    Arbitrary Lagrangian-Eulerian (ALE) mesh motion with conservative remapping.
    
    This function handles the case where the mesh moves and we need to
    conservatively remap the solution from the old mesh to the new mesh.
    
    Args:
        mesh_old: Old mesh state
        mesh_new: New mesh state
        U_old: Conservative variables on old mesh
        params: Interpolation parameters
        
    Returns:
        Conservative variables on new mesh
    """
    return conservative_interpolation(U_old, mesh_old, mesh_new, params)


def piston_boundary_motion(
    mesh_old: MeshState,
    piston_position_old: float,
    piston_position_new: float,
    U_old: np.ndarray,
    params: InterpolationParameters,
) -> Tuple[MeshState, np.ndarray]:
    """
    Handle piston boundary motion with conservative remapping.
    
    Args:
        mesh_old: Old mesh state
        piston_position_old: Old piston position
        piston_position_new: New piston position
        U_old: Conservative variables on old mesh
        params: Interpolation parameters
        
    Returns:
        Tuple of (new_mesh, new_solution)
    """
    # Compute new mesh based on piston motion
    piston_displacement = piston_position_new - piston_position_old

    # Create new mesh with updated positions
    x_new = mesh_old.x + piston_displacement
    mesh_new = MeshState(x=x_new, dx=mesh_old.dx, n_cells=mesh_old.n_cells)

    # Perform conservative remapping
    U_new = conservative_interpolation(U_old, mesh_old, mesh_new, params)

    return mesh_new, U_new


def get_interpolation_function(method: str) -> callable:
    """
    Get interpolation function by name.
    
    Args:
        method: Interpolation method name
        
    Returns:
        Interpolation function
    """
    methods = {
        "linear": _linear_conservative_interpolation,
        "cubic": _cubic_conservative_interpolation,
        "weno": _weno_conservative_interpolation,
        "monotonic": _monotonic_conservative_interpolation,
    }

    if method not in methods:
        raise ValueError(f"Unknown interpolation method: {method}")

    return methods[method]
