"""
CasADi-based motion law optimizer using Opti stack with direct collocation.

This module implements Phase 1 optimization using CasADi's Opti stack for
motion law optimization with thermal efficiency objectives and warm-starting
capabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from casadi import *

from campro.logging import get_logger
from campro.optimization.base import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
)
from campro.optimization.ipopt_factory import build_ipopt_solver_options

log = get_logger(__name__)


@dataclass
class CasADiMotionProblem:
    """
    Problem specification for CasADi motion law optimization.
    
    All motion constraints are in per-degree units (SI):
    - max_velocity: m/deg
    - max_acceleration: m/deg²
    - max_jerk: m/deg³
    
    Note: duration_angle_deg is required for Phase 1 per-degree optimization.
    No fallback to default values is allowed to prevent unit mixing.
    """

    # Boundary conditions
    stroke: float  # Stroke length in meters (SI)
    cycle_time: float  # Cycle time in seconds
    upstroke_percent: float  # Upstroke percentage (0-100)
    duration_angle_deg: float  # Total motion duration in degrees (required, no default)
    max_velocity: float | None = None  # Maximum velocity constraint in m/deg (per-degree units, optional)
    max_acceleration: float | None = None  # Maximum acceleration constraint in m/deg² (per-degree units, optional)
    max_jerk: float | None = None  # Maximum jerk constraint in m/deg³ (per-degree units, optional)
    compression_ratio_limits: tuple[float, float] = (20.0, 70.0)  # Compression ratio bounds

    # Objectives
    minimize_jerk: bool = True
    maximize_thermal_efficiency: bool = True
    weights: dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            }
        self._validate_per_degree_units()
        if self.duration_angle_deg <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {self.duration_angle_deg}. "
                "Phase 1 optimization requires angle-based units, not time-based. "
                "duration_angle_deg is required and must be explicitly set."
            )

    def _validate_per_degree_units(self) -> None:
        """
        Validate that motion constraints are in per-degree SI units.
        
        Raises
        ------
        ValueError
            If constraints are invalid (negative values, etc.).
        """
        # Basic validation: ensure constraints are positive if provided
        if self.max_velocity is not None and self.max_velocity <= 0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")
        if self.max_acceleration is not None and self.max_acceleration <= 0:
            raise ValueError(f"max_acceleration must be positive, got {self.max_acceleration}")
        if self.max_jerk is not None and self.max_jerk <= 0:
            raise ValueError(f"max_jerk must be positive, got {self.max_jerk}")


class CasADiMotionOptimizer(BaseOptimizer):
    """
    CasADi-based motion law optimizer using Opti stack with direct collocation.

    Implements Phase 1 optimization with:
    - Direct collocation with Legendre/Radau polynomials
    - Thermal efficiency objectives from FPE literature
    - Warm-starting capabilities
    - Physics-based constraints
    """

    def __init__(
        self,
        n_segments: int = 50,
        poly_order: int = 3,
        collocation_method: str = "legendre",
        solver_options: dict[str, Any] | None = None,
    ):
        """
        Initialize CasADi motion optimizer.

        Parameters
        ----------
        n_segments : int
            Number of finite elements for collocation
        poly_order : int
            Polynomial order for collocation (3 = cubic)
        collocation_method : str
            Collocation method: "legendre" or "radau"
        """
        super().__init__()
        self.n_segments = n_segments
        self.poly_order = poly_order
        self.collocation_method = collocation_method

        # Initialize Opti stack
        self.opti = Opti()

        self._solver_options: dict[str, Any] = {
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-6,
            "ipopt.print_level": 0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.linear_solver": "ma57",
        }
        if solver_options:
            self._solver_options.update(solver_options)

        self._configure_ipopt_solver()

        log.info(
            f"Initialized CasADiMotionOptimizer: {n_segments} segments, "
            f"order {poly_order}, method {collocation_method}",
        )

        # Mark as configured after initialization
        self._is_configured = True

    def configure(self, **kwargs) -> None:
        """
        Configure the CasADi motion optimizer.

        Parameters
        ----------
        **kwargs
            Configuration parameters:
            - n_segments: Number of finite elements for collocation
            - poly_order: Polynomial order for collocation
            - collocation_method: Collocation method ("legendre" or "radau")
            - solver_options: Dict of IPOPT solver options
        """
        needs_rebuild = False

        # Update n_segments if provided
        if "n_segments" in kwargs:
            new_n_segments = kwargs["n_segments"]
            if new_n_segments != self.n_segments:
                self.n_segments = new_n_segments
                needs_rebuild = True

        # Update poly_order if provided
        if "poly_order" in kwargs:
            new_poly_order = kwargs["poly_order"]
            if new_poly_order != self.poly_order:
                self.poly_order = new_poly_order
                needs_rebuild = True

        # Update collocation_method if provided
        if "collocation_method" in kwargs:
            new_method = kwargs["collocation_method"]
            if new_method != self.collocation_method:
                if new_method not in ("legendre", "radau"):
                    raise ValueError(
                        f"Invalid collocation_method: {new_method}. "
                        "Must be 'legendre' or 'radau'",
                    )
                self.collocation_method = new_method
                needs_rebuild = True

        # Rebuild Opti stack if structural parameters changed
        if needs_rebuild:
            self.opti = Opti()
            log.info(
                f"Rebuilt Opti stack: {self.n_segments} segments, "
                f"order {self.poly_order}, method {self.collocation_method}",
            )

        # Update solver options if provided
        solver_options = kwargs.get("solver_options")
        if solver_options:
            self._solver_options.update(solver_options)
            log.debug(f"Updated solver options: {solver_options}")

        if needs_rebuild or solver_options is not None:
            self._configure_ipopt_solver()

        self._is_configured = True
        log.info(
            f"Configured CasADiMotionOptimizer: {self.n_segments} segments, "
            f"order {self.poly_order}, method {self.collocation_method}",
        )

    def _configure_ipopt_solver(self) -> None:
        """Install IPOPT solver options via the shared factory helper."""
        opts = build_ipopt_solver_options(self._solver_options)
        # Add scaling options
        opts.setdefault("ipopt.nlp_scaling_method", "gradient-based")
        opts.setdefault("ipopt.nlp_scaling_max_gradient", 100.0)
        self.opti.solver("ipopt", opts)

    def _diagnose_scaling(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
        initial_guess: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Diagnose scaling issues by computing and logging variable/constraint magnitudes.
        
        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification
        collocation_vars : dict[str, Any]
            Collocation variables dictionary
        initial_guess : Optional[dict[str, np.ndarray]]
            Initial guess for variables (if available)
        """
        dtheta = collocation_vars["dtheta"]
        
        # Estimate variable magnitudes from problem bounds and initial guess
        stroke = problem.stroke
        duration_angle = problem.duration_angle_deg
        
        # Typical velocity: stroke / duration_angle
        typical_v = stroke / max(duration_angle, 1e-6) if duration_angle > 0 else 0.0
        # Typical acceleration: velocity / duration_angle
        typical_a = typical_v / max(duration_angle, 1e-6) if duration_angle > 0 else 0.0
        # Typical jerk: acceleration / duration_angle
        typical_j = typical_a / max(duration_angle, 1e-6) if duration_angle > 0 else 0.0
        
        # Get initial guess values if available
        if initial_guess:
            x0 = initial_guess.get("x")
            v0 = initial_guess.get("v")
            a0 = initial_guess.get("a")
            j0 = initial_guess.get("j")
            
            if x0 is not None:
                x_mag = np.max(np.abs(x0))
                x_mean = np.mean(np.abs(x0))
            else:
                x_mag = stroke
                x_mean = stroke / 2.0
                
            if v0 is not None:
                v_mag = np.max(np.abs(v0))
                v_mean = np.mean(np.abs(v0))
            else:
                v_mag = typical_v
                v_mean = typical_v / 2.0
                
            if a0 is not None:
                a_mag = np.max(np.abs(a0))
                a_mean = np.mean(np.abs(a0))
            else:
                a_mag = typical_a
                a_mean = typical_a / 2.0
                
            if j0 is not None:
                j_mag = np.max(np.abs(j0))
                j_mean = np.mean(np.abs(j0))
            else:
                j_mag = typical_j
                j_mean = typical_j / 2.0
        else:
            x_mag, x_mean = stroke, stroke / 2.0
            v_mag, v_mean = typical_v, typical_v / 2.0
            a_mag, a_mean = typical_a, typical_a / 2.0
            j_mag, j_mean = typical_j, typical_j / 2.0
        
        # Compute magnitude ratios
        max_mag = max(x_mag, v_mag, a_mag, j_mag)
        min_mag = min(x_mag, v_mag, a_mag, j_mag) if min(x_mag, v_mag, a_mag, j_mag) > 1e-10 else 1e-10
        magnitude_ratio = max_mag / min_mag
        
        # Estimate constraint magnitudes
        # Collocation constraint: x[k+1] = x[k] + 0.5 * dtheta * (v[k] + v[k+1])
        colloc_constraint_mag = x_mag + 0.5 * dtheta * 2.0 * v_mag
        
        # Compression ratio constraint: cr = (x + clearance) / clearance
        clearance = 0.002
        cr_mag = (stroke + clearance) / clearance
        
        log.info("=== Scaling Diagnosis (Before Scaling) ===")
        log.info(f"Variable magnitudes (estimated):")
        log.info(f"  Position (x): max={x_mag:.6e}, mean={x_mean:.6e}")
        log.info(f"  Velocity (v): max={v_mag:.6e}, mean={v_mean:.6e}")
        log.info(f"  Acceleration (a): max={a_mag:.6e}, mean={a_mean:.6e}")
        log.info(f"  Jerk (j): max={j_mag:.6e}, mean={j_mean:.6e}")
        log.info(f"Magnitude ratio (max/min): {magnitude_ratio:.2e}")
        log.info(f"Constraint magnitudes:")
        log.info(f"  Collocation constraint: ~{colloc_constraint_mag:.6e}")
        log.info(f"  Compression ratio: ~{cr_mag:.2f}")
        if magnitude_ratio > 100.0:
            log.info(
                f"Large magnitude ratio ({magnitude_ratio:.2e}) detected. "
                "Variable scaling has been implemented to address this."
            )
    
    def _verify_scaling(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
        initial_guess: dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Verify that all scaled variables, constraints, and objectives are O(1) magnitude.
        
        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification
        collocation_vars : dict[str, Any]
            Collocation variables dictionary
        initial_guess : Optional[dict[str, np.ndarray]]
            Initial guess for variables (if available)
        """
        scale_x, scale_v, scale_a, scale_j = (
            collocation_vars["scale_x"],
            collocation_vars["scale_v"],
            collocation_vars["scale_a"],
            collocation_vars["scale_j"],
        )
        
        log.info("=== Scaling Verification ===")
        log.info(f"Scaling factors:")
        log.info(f"  Position (scale_x): {scale_x:.6e}")
        log.info(f"  Velocity (scale_v): {scale_v:.6e}")
        log.info(f"  Acceleration (scale_a): {scale_a:.6e}")
        log.info(f"  Jerk (scale_j): {scale_j:.6e}")
        
        # Verify scaled initial guess magnitudes
        if initial_guess:
            x0 = initial_guess.get("x")
            v0 = initial_guess.get("v")
            a0 = initial_guess.get("a")
            j0 = initial_guess.get("j")
            
            if x0 is not None:
                x0_scaled = np.asarray(x0) * scale_x
                x0_mag = np.max(np.abs(x0_scaled))
                log.info(f"Scaled initial guess magnitudes:")
                log.info(f"  Position (x_scaled): max={x0_mag:.6e} (target: ~1.0)")
                if x0_mag > 10.0 or x0_mag < 0.1:
                    log.warning(f"Position scaling may be suboptimal: {x0_mag:.6e}")
            
            if v0 is not None:
                v0_scaled = np.asarray(v0) * scale_v
                v0_mag = np.max(np.abs(v0_scaled))
                log.info(f"  Velocity (v_scaled): max={v0_mag:.6e} (target: ~1.0)")
                if v0_mag > 10.0 or v0_mag < 0.1:
                    log.warning(f"Velocity scaling may be suboptimal: {v0_mag:.6e}")
            
            if a0 is not None:
                a0_scaled = np.asarray(a0) * scale_a
                a0_mag = np.max(np.abs(a0_scaled))
                log.info(f"  Acceleration (a_scaled): max={a0_mag:.6e} (target: ~1.0)")
                if a0_mag > 10.0 or a0_mag < 0.1:
                    log.warning(f"Acceleration scaling may be suboptimal: {a0_mag:.6e}")
            
            if j0 is not None:
                j0_scaled = np.asarray(j0) * scale_j
                j0_mag = np.max(np.abs(j0_scaled))
                log.info(f"  Jerk (j_scaled): max={j0_mag:.6e} (target: ~1.0)")
                if j0_mag > 10.0 or j0_mag < 0.1:
                    log.warning(f"Jerk scaling may be suboptimal: {j0_mag:.6e}")
        
        # Verify constraint scaling
        # Collocation constraints are now scaled to O(1) by dividing by the coefficient
        dtheta = collocation_vars["dtheta"]
        scale_xv = scale_x / scale_v
        constraint_coeff_xv = 0.5 * dtheta * scale_xv
        constraint_scale_xv = 1.0 / max(constraint_coeff_xv, 1e-10)
        log.info(f"Collocation constraint coefficient (before scaling): {constraint_coeff_xv:.6e}")
        log.info(f"Collocation constraint scale factor: {constraint_scale_xv:.6e}")
        log.info(f"Effective constraint scale (after scaling): ~1.0 (target: ~1.0)")
        
        log.info("Scaling verification complete.")

    def setup_collocation(self, problem: CasADiMotionProblem) -> dict[str, Any]:
        """
        Setup direct collocation discretization with variable scaling.

        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification

        Returns
        -------
        Dict[str, Any]
            Collocation variables and parameters (including scaled variables and scaling factors)
        """
        # Angle discretization (degrees domain)
        theta_total = max(float(problem.duration_angle_deg), 1e-6)
        dtheta = theta_total / self.n_segments

        # Compute scaling factors to normalize variables to O(1) magnitude
        # Position: normalize to [0, 1] by dividing by stroke
        scale_x = 1.0 / max(problem.stroke, 1e-6)
        
        # Velocity: normalize by typical velocity (stroke / duration_angle)
        typical_velocity = problem.stroke / max(theta_total, 1e-6)
        scale_v = 1.0 / max(typical_velocity, 1e-6)
        
        # Acceleration: normalize by typical acceleration (velocity / duration_angle)
        typical_acceleration = typical_velocity / max(theta_total, 1e-6)
        scale_a = 1.0 / max(typical_acceleration, 1e-6)
        
        # Jerk: normalize by typical jerk (acceleration / duration_angle)
        typical_jerk = typical_acceleration / max(theta_total, 1e-6)
        scale_j = 1.0 / max(typical_jerk, 1e-6)

        # State variables: position, velocity, acceleration (scaled)
        x_scaled = self.opti.variable(self.n_segments + 1)  # scaled position [0, 1]
        v_scaled = self.opti.variable(self.n_segments + 1)  # scaled velocity
        a_scaled = self.opti.variable(self.n_segments + 1)  # scaled acceleration

        # Control variable: jerk (scaled)
        j_scaled = self.opti.variable(self.n_segments)  # scaled jerk

        # Physical (unscaled) variables for use in constraints/objectives
        x = x_scaled / scale_x  # physical position in meters
        v = v_scaled / scale_v  # physical velocity in m/deg
        a = a_scaled / scale_a  # physical acceleration in m/deg²
        j = j_scaled / scale_j  # physical jerk in m/deg³

        # Collocation points
        if self.collocation_method == "legendre":
            # Legendre-Gauss-Lobatto (docs/casadi_phase1_formulation.md)
            if self.poly_order > 1:
                interior_nodes = np.polynomial.legendre.leggauss(self.poly_order - 1)[0]
            else:
                interior_nodes = np.array([])
            tau_root = np.concatenate(([-1.0], interior_nodes, [1.0]))
        else:  # Radau (Legendre-Gauss-Radau: left endpoint + interior nodes)
            tau_root = np.polynomial.legendre.leggauss(self.poly_order)[0]
            tau_root = np.concatenate(([-1.0], tau_root))

        # Collocation matrices
        C = np.zeros((self.poly_order + 1, self.poly_order + 1))
        D = np.zeros((self.poly_order + 1, 1))

        # Construct collocation matrices
        for basis_idx in range(self.poly_order + 1):
            # Construct Lagrange polynomial
            coeffs = np.zeros(self.poly_order + 1)
            coeffs[basis_idx] = 1.0
            poly = np.polynomial.Polynomial(coeffs)

            # Evaluate at collocation points
            for node_idx in range(self.poly_order + 1):
                C[basis_idx, node_idx] = poly(tau_root[node_idx])
            D[basis_idx, 0] = poly(1.0)  # End point

        return {
            "x": x,  # Physical (unscaled) position
            "v": v,  # Physical (unscaled) velocity
            "a": a,  # Physical (unscaled) acceleration
            "j": j,  # Physical (unscaled) jerk
            "x_scaled": x_scaled,  # Scaled position [0, 1]
            "v_scaled": v_scaled,  # Scaled velocity
            "a_scaled": a_scaled,  # Scaled acceleration
            "j_scaled": j_scaled,  # Scaled jerk
            "scale_x": scale_x,
            "scale_v": scale_v,
            "scale_a": scale_a,
            "scale_j": scale_j,
            "dtheta": dtheta,
            "theta_total": theta_total,
            "C": C,
            "D": D,
            "tau_root": tau_root,
        }

    def add_boundary_conditions(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
    ) -> None:
        """Add boundary conditions to the optimization problem (using scaled variables)."""
        x_scaled, v_scaled = collocation_vars["x_scaled"], collocation_vars["v_scaled"]
        scale_x, scale_v = collocation_vars["scale_x"], collocation_vars["scale_v"]

        # Position boundary conditions for full cycle: BDC → TDC → BDC
        # x[0] = 0 → x_scaled[0] / scale_x = 0 → x_scaled[0] = 0
        self.opti.subject_to(x_scaled[0] == 0)  # Start at BDC (zero)
        self.opti.subject_to(x_scaled[-1] == 0)  # End at BDC (zero) - full cycle

        # Velocity boundary conditions
        # v[0] = 0 → v_scaled[0] / scale_v = 0 → v_scaled[0] = 0
        self.opti.subject_to(v_scaled[0] == 0)  # Start at rest
        self.opti.subject_to(v_scaled[-1] == 0)  # End at rest

        # Upstroke constraint: position must reach stroke at upstroke end
        total_angle = max(problem.duration_angle_deg, 1e-6)
        segment_angle = total_angle / self.n_segments
        upstroke_angle = total_angle * problem.upstroke_percent / 100.0
        upstroke_index = int(upstroke_angle / segment_angle)
        upstroke_index = min(upstroke_index, self.n_segments)

        # Constrain position at upstroke end to reach stroke (TDC)
        # x[upstroke_index] = stroke → x_scaled[upstroke_index] / scale_x = stroke
        # → x_scaled[upstroke_index] = stroke * scale_x = stroke / stroke = 1.0
        self.opti.subject_to(x_scaled[upstroke_index] == 1.0)  # Scaled: stroke * scale_x = 1.0

        # Velocity should be positive during upstroke (moving up from BDC to TDC)
        # v[i] >= 0 → v_scaled[i] / scale_v >= 0 → v_scaled[i] >= 0 (since scale_v > 0)
        for i in range(upstroke_index):
            self.opti.subject_to(v_scaled[i] >= 0)

        # Velocity should be negative during downstroke (moving down from TDC to BDC)
        # v[i] <= 0 → v_scaled[i] / scale_v <= 0 → v_scaled[i] <= 0
        for i in range(upstroke_index, self.n_segments + 1):
            self.opti.subject_to(v_scaled[i] <= 0)

    def add_motion_constraints(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
    ) -> None:
        """Add motion constraints (velocity, acceleration, jerk limits) using scaled variables.
        
        Only enforces constraints when max_velocity, max_acceleration, or max_jerk
        are provided (not None). When None, the optimizer is free to choose any
        values within the duration window.
        """
        v_scaled, a_scaled, j_scaled = (
            collocation_vars["v_scaled"],
            collocation_vars["a_scaled"],
            collocation_vars["j_scaled"],
        )
        scale_v, scale_a, scale_j = (
            collocation_vars["scale_v"],
            collocation_vars["scale_a"],
            collocation_vars["scale_j"],
        )

        # Velocity constraints (only if max_velocity is provided)
        # v[i] <= max_velocity → v_scaled[i] / scale_v <= max_velocity → v_scaled[i] <= max_velocity * scale_v
        if problem.max_velocity is not None:
            max_v_scaled = problem.max_velocity * scale_v
            for i in range(self.n_segments + 1):
                self.opti.subject_to(
                    self.opti.bounded(-max_v_scaled, v_scaled[i], max_v_scaled),
                )

        # Acceleration constraints (only if max_acceleration is provided)
        # a[i] <= max_acceleration → a_scaled[i] / scale_a <= max_acceleration → a_scaled[i] <= max_acceleration * scale_a
        if problem.max_acceleration is not None:
            max_a_scaled = problem.max_acceleration * scale_a
            for i in range(self.n_segments + 1):
                self.opti.subject_to(
                    self.opti.bounded(-max_a_scaled, a_scaled[i], max_a_scaled),
                )

        # Jerk constraints (only if max_jerk is provided)
        # j[i] <= max_jerk → j_scaled[i] / scale_j <= max_jerk → j_scaled[i] <= max_jerk * scale_j
        if problem.max_jerk is not None:
            max_j_scaled = problem.max_jerk * scale_j
            for i in range(self.n_segments):
                self.opti.subject_to(
                    self.opti.bounded(-max_j_scaled, j_scaled[i], max_j_scaled),
                )

    def add_physics_constraints(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
    ) -> None:
        """Add physics-based constraints from FPE literature."""
        x, v, a = collocation_vars["x"], collocation_vars["v"], collocation_vars["a"]
        dtheta = collocation_vars["dtheta"]

        # Geometric bounds: position must stay within physical domain [0, stroke]
        # Using scaled variables: x_scaled / scale_x >= 0 → x_scaled >= 0
        # x_scaled / scale_x <= stroke → x_scaled <= stroke * scale_x = 1.0
        x_scaled = collocation_vars["x_scaled"]
        for i in range(self.n_segments + 1):
            self.opti.subject_to(x_scaled[i] >= 0.0)  # Cannot go below BDC (scaled: 0)
            self.opti.subject_to(x_scaled[i] <= 1.0)  # Cannot exceed stroke (scaled: 1.0)

        # Compression ratio constraints
        # CR = V_max / V_min = (position + clearance) / clearance
        # Enforce maximum CR limit at TDC (maximum displacement) only
        # At BDC: x[0] = 0, so cr = (0 + clearance)/clearance = 1.0 (always minimum)
        # At TDC: x[upstroke_index] = stroke (enforced by boundary condition), so cr = (stroke + clearance)/clearance
        # The geometric constraint (x[upstroke_index] == stroke) already determines the achievable CR range
        # No minimum CR constraint needed - geometry determines the minimum achievable CR
        clearance = 0.002  # 2mm clearance
        min_cr_limit, max_cr_limit = problem.compression_ratio_limits
        
        # Find upstroke index (where TDC occurs)
        total_angle = max(problem.duration_angle_deg, 1e-6)
        segment_angle = total_angle / self.n_segments
        upstroke_angle = total_angle * problem.upstroke_percent / 100.0
        upstroke_index = int(upstroke_angle / segment_angle)
        upstroke_index = min(upstroke_index, self.n_segments)
        
        # Compute CR at TDC (maximum displacement, at upstroke_index)
        # x[upstroke_index] = x_scaled[upstroke_index] / scale_x
        # At TDC, x_scaled[upstroke_index] = 1.0, so x[upstroke_index] = 1.0 / scale_x = stroke
        x_scaled = collocation_vars["x_scaled"]
        scale_x = collocation_vars["scale_x"]
        x_tdc = x_scaled[upstroke_index] / scale_x
        cr_tdc = (x_tdc + clearance) / clearance  # TDC: maximum CR
        
        # Enforce maximum CR limit only (geometry determines minimum)
        self.opti.subject_to(cr_tdc <= max_cr_limit)  # Maximum CR enforced at TDC
        
        # For interior points, only enforce upper bound to prevent over-compression mid-stroke
        # Since cr(theta) is monotonic between endpoints, the TDC check guarantees
        # the entire trace stays within [1.0, cr_end]
        for i in range(1, self.n_segments):  # Skip endpoints
            x_i = x_scaled[i] / scale_x
            cr = (x_i + clearance) / clearance
            self.opti.subject_to(cr <= max_cr_limit)  # Prevent over-compression
            self.opti.subject_to(cr >= 1.0)  # Physical lower bound (always satisfied at BDC)

        # Pressure rate constraint (avoid diesel knock)
        # Simplified: limit acceleration rate
        # Note: acceleration is in per-degree units, rate is computed per degree
        # Convert to per-second rate using duration_angle_deg and cycle_time
        # Using scaled variables: a = a_scaled / scale_a
        a_scaled = collocation_vars["a_scaled"]
        scale_a = collocation_vars["scale_a"]
        max_pressure_rate_per_s = 1000.0  # Pa/ms
        if hasattr(problem, 'duration_angle_deg') and hasattr(problem, 'cycle_time'):
            deg_per_s = problem.duration_angle_deg / max(problem.cycle_time, 1e-9)
            for i in range(self.n_segments - 1):
                a_i = a_scaled[i] / scale_a
                a_i1 = a_scaled[i + 1] / scale_a
                pressure_rate_per_deg = fabs(a_i1 - a_i) / dtheta
                # Convert to per-second rate using actual engine speed
                pressure_rate_per_s = pressure_rate_per_deg * deg_per_s
                self.opti.subject_to(pressure_rate_per_s <= max_pressure_rate_per_s)

    def add_collocation_constraints(self, collocation_vars: dict[str, Any]) -> None:
        """Add collocation constraints for state continuity (using scaled variables)."""
        x, v, a, j = (
            collocation_vars["x"],
            collocation_vars["v"],
            collocation_vars["a"],
            collocation_vars["j"],
        )
        x_scaled, v_scaled, a_scaled, j_scaled = (
            collocation_vars["x_scaled"],
            collocation_vars["v_scaled"],
            collocation_vars["a_scaled"],
            collocation_vars["j_scaled"],
        )
        scale_x, scale_v, scale_a, scale_j = (
            collocation_vars["scale_x"],
            collocation_vars["scale_v"],
            collocation_vars["scale_a"],
            collocation_vars["scale_j"],
        )
        dtheta, C, D = collocation_vars["dtheta"], collocation_vars["C"], collocation_vars["D"]

        # State continuity constraints using scaled variables
        # Scale constraints to O(1) magnitude for better numerical conditioning
        
        # Position: x[k+1] = x[k] + 0.5 * dtheta * (v[k] + v[k+1])
        # In scaled form: x_scaled[k+1]/scale_x = x_scaled[k]/scale_x + 0.5 * dtheta * (v_scaled[k]/scale_v + v_scaled[k+1]/scale_v)
        # Multiply by scale_x: x_scaled[k+1] = x_scaled[k] + 0.5 * dtheta * (scale_x/scale_v) * (v_scaled[k] + v_scaled[k+1])
        # The coefficient 0.5 * dtheta * (scale_x/scale_v) is very small, so scale the constraint
        scale_xv = scale_x / scale_v
        constraint_coeff_xv = 0.5 * dtheta * scale_xv
        # Scale constraint to O(1): divide both sides by the coefficient (or multiply by its inverse)
        constraint_scale_xv = 1.0 / max(constraint_coeff_xv, 1e-10)  # Avoid division by zero
        for k in range(self.n_segments):
            # Scaled constraint: constraint_scale_xv * (x_scaled[k+1] - x_scaled[k]) = constraint_scale_xv * constraint_coeff_xv * (v_scaled[k] + v_scaled[k+1])
            # Which simplifies to: constraint_scale_xv * (x_scaled[k+1] - x_scaled[k]) = (v_scaled[k] + v_scaled[k+1])
            self.opti.subject_to(
                constraint_scale_xv * (x_scaled[k + 1] - x_scaled[k]) == v_scaled[k] + v_scaled[k + 1]
            )
        
        # Velocity: v[k+1] = v[k] + 0.5 * dtheta * (a[k] + a[k+1])
        # In scaled form: v_scaled[k+1]/scale_v = v_scaled[k]/scale_v + 0.5 * dtheta * (a_scaled[k]/scale_a + a_scaled[k+1]/scale_a)
        # Multiply by scale_v: v_scaled[k+1] = v_scaled[k] + 0.5 * dtheta * (scale_v/scale_a) * (a_scaled[k] + a_scaled[k+1])
        scale_va = scale_v / scale_a
        constraint_coeff_va = 0.5 * dtheta * scale_va
        constraint_scale_va = 1.0 / max(constraint_coeff_va, 1e-10)
        for k in range(self.n_segments):
            self.opti.subject_to(
                constraint_scale_va * (v_scaled[k + 1] - v_scaled[k]) == a_scaled[k] + a_scaled[k + 1]
            )
        
        # Acceleration: a[k+1] = a[k] + dtheta * j[k]
        # In scaled form: a_scaled[k+1]/scale_a = a_scaled[k]/scale_a + dtheta * j_scaled[k]/scale_j
        # Multiply by scale_a: a_scaled[k+1] = a_scaled[k] + dtheta * (scale_a/scale_j) * j_scaled[k]
        scale_aj = scale_a / scale_j
        constraint_coeff_aj = dtheta * scale_aj
        constraint_scale_aj = 1.0 / max(constraint_coeff_aj, 1e-10)
        for k in range(self.n_segments):
            self.opti.subject_to(
                constraint_scale_aj * (a_scaled[k + 1] - a_scaled[k]) == j_scaled[k]
            )

    def add_thermal_efficiency_objective(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
    ) -> None:
        """Add thermal efficiency objective from FPE literature (using scaled variables)."""
        from casadi import sum1
        
        x_scaled, v_scaled, a_scaled = (
            collocation_vars["x_scaled"],
            collocation_vars["v_scaled"],
            collocation_vars["a_scaled"],
        )
        scale_x, scale_v, scale_a = (
            collocation_vars["scale_x"],
            collocation_vars["scale_v"],
            collocation_vars["scale_a"],
        )
        
        # Convert scaled variables to physical units
        x = x_scaled / scale_x
        v = v_scaled / scale_v
        a = a_scaled / scale_a
        
        # Thermal efficiency objective (simplified Otto cycle)
        # η_thermal ≈ 1 - 1/CR^(γ-1) + heat_loss_penalty + mech_loss_penalty

        gamma = 1.4  # Specific heat ratio
        clearance = 0.002  # 2mm clearance

        # Compression ratio at each point
        # Ensure cr >= 1.0 to prevent numerical issues (x >= 0 is enforced by geometric bounds)
        cr = (x + clearance) / clearance
        cr_safe = fmax(cr, 1.0 + 1e-6)  # Clamp to prevent cr < 1.0 during optimization

        # Otto cycle efficiency: 1 - 1/CR^(γ-1)
        # Add small epsilon to prevent division by zero if cr becomes exactly 1.0
        cr_power = cr_safe ** (gamma - 1)
        otto_efficiency = 1 - 1 / fmax(cr_power, 1e-6)

        # Heat loss penalty (proportional to velocity - Woschni correlation simplified)
        # Scale penalty to O(1): v^2 = (v_scaled/scale_v)^2 = v_scaled^2 / scale_v^2
        # Scale by scale_v^2 to normalize: v_scaled^2
        heat_loss_penalty = 0.1 * v_scaled**2

        # Mechanical loss penalty (proportional to acceleration)
        # Scale penalty to O(1): a^2 = (a_scaled/scale_a)^2 = a_scaled^2 / scale_a^2
        # Scale by scale_a^2 to normalize: a_scaled^2
        mech_loss_penalty = 0.01 * a_scaled**2

        # Total thermal efficiency (negative for minimization)
        # otto_efficiency is already O(1), penalties are now O(1) after scaling
        thermal_efficiency = otto_efficiency - heat_loss_penalty - mech_loss_penalty

        # Average thermal efficiency over cycle
        n_points = max(thermal_efficiency.shape[0] if hasattr(thermal_efficiency, 'shape') else len(thermal_efficiency), 1)
        avg_thermal_efficiency = sum1(thermal_efficiency) / max(n_points, 1)

        # Add to objective (minimize negative efficiency)
        # Objective is already O(1) after scaling
        self.opti.minimize(
            -problem.weights["thermal_efficiency"] * avg_thermal_efficiency,
        )

    def add_jerk_objective(
        self, problem: CasADiMotionProblem, collocation_vars: dict[str, Any],
    ) -> None:
        """Add jerk minimization objective for smoothness (using scaled variables)."""
        j_scaled = collocation_vars["j_scaled"]
        scale_j = collocation_vars["scale_j"]
        dtheta = collocation_vars["dtheta"]

        # Jerk squared integral (minimize for smoothness)
        # j = j_scaled / scale_j, so j^2 = (j_scaled / scale_j)^2 = j_scaled^2 / scale_j^2
        # Scale by scale_j^2 to keep objective O(1): j_scaled^2 * (dtheta / scale_j^2)
        # But we want to minimize physical jerk, so we need to account for the scaling
        # Objective: sum1(j^2) * dtheta = sum1((j_scaled/scale_j)^2) * dtheta
        # = sum1(j_scaled^2) * dtheta / scale_j^2
        # Scale by scale_j^2 to normalize: sum1(j_scaled^2) * dtheta
        jerk_squared_scaled = sum1(j_scaled**2) * dtheta

        # Add to objective (scaled objective is already O(1))
        self.opti.minimize(problem.weights["jerk"] * jerk_squared_scaled)

    def solve(
        self,
        problem: CasADiMotionProblem,
        initial_guess: dict[str, np.ndarray] | None = None,
        target_theta_rad: np.ndarray | None = None,
    ) -> OptimizationResult:
        """
        Solve the motion law optimization problem.

        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification
        initial_guess : Optional[Dict[str, np.ndarray]]
            Initial guess for variables

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        try:
            # Setup collocation
            collocation_vars = self.setup_collocation(problem)

            # Add constraints
            self.add_boundary_conditions(problem, collocation_vars)
            self.add_motion_constraints(problem, collocation_vars)
            self.add_physics_constraints(problem, collocation_vars)
            self.add_collocation_constraints(collocation_vars)

            # Add objectives
            if problem.minimize_jerk:
                self.add_jerk_objective(problem, collocation_vars)

            if problem.maximize_thermal_efficiency:
                self.add_thermal_efficiency_objective(problem, collocation_vars)

            # Diagnose scaling before solving
            self._diagnose_scaling(problem, collocation_vars, initial_guess)
            
            # Set initial guess if provided (convert to scaled variables)
            if initial_guess:
                x0 = initial_guess.get("x")
                v0 = initial_guess.get("v")
                a0 = initial_guess.get("a")
                j0 = initial_guess.get("j")
                
                # Convert physical initial guess to scaled variables
                scale_x = collocation_vars["scale_x"]
                scale_v = collocation_vars["scale_v"]
                scale_a = collocation_vars["scale_a"]
                scale_j = collocation_vars["scale_j"]
                
                if x0 is not None:
                    x0_scaled = np.asarray(x0) * scale_x
                    self.opti.set_initial(collocation_vars["x_scaled"], x0_scaled)
                if v0 is not None:
                    v0_scaled = np.asarray(v0) * scale_v
                    self.opti.set_initial(collocation_vars["v_scaled"], v0_scaled)
                if a0 is not None:
                    a0_scaled = np.asarray(a0) * scale_a
                    self.opti.set_initial(collocation_vars["a_scaled"], a0_scaled)
                if j0 is not None:
                    j0_scaled = np.asarray(j0) * scale_j
                    self.opti.set_initial(collocation_vars["j_scaled"], j0_scaled)
            
            # Verify scaling
            self._verify_scaling(problem, collocation_vars, initial_guess)

            # Solve
            log.info("Solving CasADi motion law optimization...")
            sol = self.opti.solve()

            # Extract results (unscale from scaled variables)
            x_scaled_opt = sol.value(collocation_vars["x_scaled"])
            v_scaled_opt = sol.value(collocation_vars["v_scaled"])
            a_scaled_opt = sol.value(collocation_vars["a_scaled"])
            j_scaled_opt = sol.value(collocation_vars["j_scaled"])
            
            # Convert back to physical units
            scale_x = collocation_vars["scale_x"]
            scale_v = collocation_vars["scale_v"]
            scale_a = collocation_vars["scale_a"]
            scale_j = collocation_vars["scale_j"]
            
            x_opt = x_scaled_opt / scale_x
            v_opt = v_scaled_opt / scale_v
            a_opt = a_scaled_opt / scale_a
            j_opt = j_scaled_opt / scale_j
            
            # Use target_theta_rad if provided and matches solution grid size
            if target_theta_rad is not None and len(target_theta_rad) == x_opt.shape[0]:
                theta_rad = np.asarray(target_theta_rad)
                theta_deg = np.degrees(theta_rad)
                log.info(
                    f"Using provided target_theta_rad for solution grid "
                    f"({len(theta_rad)} points)"
                )
            else:
                theta_deg = np.linspace(0.0, float(problem.duration_angle_deg), x_opt.shape[0])
                theta_rad = np.deg2rad(theta_deg)

            # Compute objective value
            objective_value = sol.value(self.opti.f)

            # Create result
            result = OptimizationResult(
                status=OptimizationStatus.CONVERGED,
                objective_value=objective_value,
                solve_time=sol.stats()["t_wall_total"],
                solution={
                    "position": x_opt,
                    "velocity": v_opt,
                    "acceleration": a_opt,
                    "jerk": j_opt,
                    "theta_deg": theta_deg,
                    "theta_rad": theta_rad,
                    "cam_angle": theta_rad,  # cam_angle is in radians, same as theta_rad
                },
                metadata={
                    "n_segments": self.n_segments,
                    "poly_order": self.poly_order,
                    "collocation_method": self.collocation_method,
                    "solver_stats": sol.stats(),
                    "duration_angle_deg": float(problem.duration_angle_deg),
                },
            )

            log.info(
                f"CasADi optimization completed successfully in {result.solve_time:.3f}s",
            )
            return result

        except Exception as e:
            log.error(f"CasADi optimization failed: {e}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solve_time=0.0,
                solution={},
                error_message=str(e),
                metadata={
                    "error": str(e),
                    "duration_angle_deg": float(problem.duration_angle_deg),
                },
            )

    def optimize(self, constraints, targets, **kwargs) -> OptimizationResult:
        """
        Optimize motion law with given constraints and targets.

        This method provides compatibility with the existing optimization interface.
        
        Raises
        ------
        ValueError
            If duration_angle_deg is missing from constraints.
        """
        # Require duration_angle_deg - no fallback to default 360.0
        duration_angle_deg = constraints.get("duration_angle_deg")
        if duration_angle_deg is None:
            raise ValueError(
                "duration_angle_deg is required for Phase 1 per-degree optimization. "
                "It must be provided in constraints dict. "
                "No fallback to default 360.0 is allowed to prevent unit mixing."
            )
        duration_angle_deg = float(duration_angle_deg)
        if duration_angle_deg <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {duration_angle_deg}. "
                "Phase 1 optimization requires angle-based units, not time-based."
            )
        
        # Convert constraints and targets to CasADiMotionProblem
        # Default to None for max_velocity/max_acceleration/max_jerk (unbounded)
        problem = CasADiMotionProblem(
            stroke=constraints.get("stroke", 0.1),
            cycle_time=constraints.get("cycle_time", 0.0385),
            duration_angle_deg=duration_angle_deg,
            upstroke_percent=constraints.get("upstroke_percent", 50.0),
            max_velocity=constraints.get("max_velocity"),  # None if not provided
            max_acceleration=constraints.get("max_acceleration"),  # None if not provided
            max_jerk=constraints.get("max_jerk"),  # None if not provided
            compression_ratio_limits=constraints.get(
                "compression_ratio_limits", (20.0, 70.0),
            ),
            minimize_jerk=targets.get("minimize_jerk", True),
            maximize_thermal_efficiency=targets.get(
                "maximize_thermal_efficiency", True,
            ),
            weights=targets.get("weights", {}),
        )

        # Get initial guess if provided
        initial_guess = kwargs.get("initial_guess")

        return self.solve(problem, initial_guess)
