from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple, Optional, Any

import numpy as np

from campro.logging import get_logger
from campro.constants import HSLLIB_PATH
from campro.freepiston.opt.ipopt_solver import IPOPTSolver, IPOPTOptions

from .involute_internal import InternalGearParams, sample_internal_flank
from .kinematics import PlanetKinematics
from .metrics import evaluate_order0_metrics, evaluate_order0_metrics_given_phi
from .motion import RadialSlotMotion
from .opt.collocation import make_uniform_grid
from .planetary_synthesis import _newton_solve_phi
from .config import GeometrySearchConfig, OptimizationOrder, PlanetSynthesisConfig

log = get_logger(__name__)




@dataclass(frozen=True)
class OptimResult:
    best_config: PlanetSynthesisConfig | None
    objective_value: float | None
    feasible: bool
    ipopt_analysis: Optional[Any] = None  # Will be MA57ReadinessReport when available


def _order0_objective(cfg: PlanetSynthesisConfig) -> float:
    m = evaluate_order0_metrics(cfg)
    # Objective combines slip and penalties on closure and edge-contact
    penalty = 0.0
    if not m.feasible:
        penalty += 1e3
    penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    return m.slip_integral - 0.1 * m.contact_length + penalty


def optimize_geometry(config: GeometrySearchConfig, order: int = OptimizationOrder.ORDER0_EVALUATE) -> OptimResult:
    if order == OptimizationOrder.ORDER0_EVALUATE:
        # Evaluate first candidate deterministically
        if not config.ring_teeth_candidates or not config.planet_teeth_candidates:
            return OptimResult(best_config=None, objective_value=None, feasible=False)
        cfg = PlanetSynthesisConfig(
            ring_teeth=config.ring_teeth_candidates[0],
            planet_teeth=config.planet_teeth_candidates[0],
            pressure_angle_deg=sum(config.pressure_angle_deg_bounds) / 2.0,
            addendum_factor=sum(config.addendum_factor_bounds) / 2.0,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            motion=config.motion,
        )
        obj = _order0_objective(cfg)
        return OptimResult(best_config=cfg, objective_value=obj, feasible=True)

    if order == OptimizationOrder.ORDER1_GEOMETRY:
        # Coarse grid + local refinement (Powell-like coordinate search)
        best_cfg: PlanetSynthesisConfig | None = None
        best_obj: float | None = None
        pa_lo, pa_hi = config.pressure_angle_deg_bounds
        af_lo, af_hi = config.addendum_factor_bounds

        def obj_for(pa: float, af: float, zr: int, zp: int) -> float:
            cand = PlanetSynthesisConfig(
                ring_teeth=zr,
                planet_teeth=zp,
                pressure_angle_deg=pa,
                addendum_factor=af,
                base_center_radius=config.base_center_radius,
                samples_per_rev=config.samples_per_rev,
                motion=config.motion,
            )
            return _order0_objective(cand)

        for zr in config.ring_teeth_candidates:
            for zp in config.planet_teeth_candidates:
                pa = 0.5 * (pa_lo + pa_hi)
                af = 0.5 * (af_lo + af_hi)
                step_pa = max(0.25, (pa_hi - pa_lo) / 8.0)
                step_af = max(0.02, (af_hi - af_lo) / 8.0)
                best_local = obj_for(pa, af, zr, zp)
                improved = True
                iters = 0
                while improved and iters < 20:
                    improved = False
                    iters += 1
                    # coordinate search in pa
                    for delta in (-step_pa, step_pa):
                        pa_try = min(pa_hi, max(pa_lo, pa + delta))
                        val = obj_for(pa_try, af, zr, zp)
                        if val < best_local:
                            best_local = val
                            pa = pa_try
                            improved = True
                    # coordinate search in af
                    for delta in (-step_af, step_af):
                        af_try = min(af_hi, max(af_lo, af + delta))
                        val = obj_for(pa, af_try, zr, zp)
                        if val < best_local:
                            best_local = val
                            af = af_try
                            improved = True
                    # decrease steps
                    step_pa *= 0.5
                    step_af *= 0.5

                if best_obj is None or best_local < best_obj:
                    best_obj = best_local
                    best_cfg = PlanetSynthesisConfig(
                        ring_teeth=zr,
                        planet_teeth=zp,
                        pressure_angle_deg=pa,
                        addendum_factor=af,
                        base_center_radius=config.base_center_radius,
                        samples_per_rev=config.samples_per_rev,
                        motion=config.motion,
                    )

        return OptimResult(best_config=best_cfg, objective_value=best_obj, feasible=best_cfg is not None)

    if order == OptimizationOrder.ORDER2_MICRO:
        # Ipopt-based NLP optimization of the contact parameter sequence phi(θ)
        return _order2_ipopt_optimization(config)

    # Higher orders will be implemented subsequently
    return OptimResult(best_config=None, objective_value=None, feasible=False)


def _order2_ipopt_optimization(config: GeometrySearchConfig) -> OptimResult:
    """
    Ipopt-based NLP optimization of the contact parameter sequence phi(θ).
    
    This replaces the simple smoothing approach with a proper constrained optimization
    that handles smoothness, contact constraints, and periodicity.
    """
    try:
        import casadi as ca
    except ImportError:
        log.error("CasADi not available for ORDER2_MICRO Ipopt optimization")
        return OptimResult(best_config=None, objective_value=None, feasible=False)
    
    # Set up problem dimensions
    n = max(64, config.samples_per_rev)
    grid = make_uniform_grid(n)
    
    # Construct flank/kinematics once
    module = config.base_center_radius * 2.0 / max(config.ring_teeth_candidates[0] - config.planet_teeth_candidates[0], 1)
    zr = config.ring_teeth_candidates[0]
    zp = config.planet_teeth_candidates[0]
    pa = sum(config.pressure_angle_deg_bounds) / 2.0
    af = sum(config.addendum_factor_bounds) / 2.0
    cand = PlanetSynthesisConfig(
        ring_teeth=zr,
        planet_teeth=zp,
        pressure_angle_deg=pa,
        addendum_factor=af,
        base_center_radius=config.base_center_radius,
        samples_per_rev=config.samples_per_rev,
        motion=config.motion,
    )
    params = InternalGearParams(teeth=zr, module=module, pressure_angle_deg=pa, addendum_factor=af)
    flank = sample_internal_flank(params, n=256)
    kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)

    # Initialize phi by Newton per node (same as before)
    phi_vals: list[float] = []
    seed = flank.phi[len(flank.phi) // 2]
    for theta in grid.theta:
        phi = _newton_solve_phi(flank, kin, theta, seed) or seed
        phi_vals.append(phi)
        seed = phi
    
    # Convert to numpy array for CasADi
    phi_init = np.array(phi_vals)
    
    # Create CasADi variables
    phi = ca.SX.sym('phi', n)
    
    # Objective function: slip integral - 0.1 * contact_length + feasibility penalty
    # We'll approximate this using the existing metrics function
    def objective_function_with_physics(phi_vec):
        """Evaluate objective with physics metrics."""
        m = evaluate_order0_metrics_given_phi(cand, phi_vec.tolist())
        penalty = 0.0 if m.feasible else 1e3
        return m.slip_integral - 0.1 * m.contact_length + penalty
    
    # Store reference for final validation
    objective_with_physics = objective_function_with_physics
    
    # Create CasADi function for objective (placeholder - not used in hybrid approach)
    # obj_func = ca.Function('obj', [phi], [ca.SX.sym('obj_val')])
    
    # For CasADi: use smoothness penalty as proxy, validate with physics
    # This is a hybrid approach - CasADi smoothness + Python physics validation
    smoothness_penalty = 0.0
    for i in range(n):
        im = (i - 1) % n
        ip = (i + 1) % n
        smoothness_penalty += (phi[i] - 0.5 * (phi[im] + phi[ip]))**2
    
    # Periodicity constraint: phi[n-1] should be close to phi[0]
    periodicity_constraint = phi[n-1] - phi[0]
    
    # Bounds on phi values (based on flank geometry)
    phi_min = float(np.min(flank.phi))
    phi_max = float(np.max(flank.phi))
    
    # Create NLP problem
    nlp = {
        'x': phi,
        'f': smoothness_penalty,
        'g': periodicity_constraint
    }
    
    # Create CasADi NLP solver
    solver_options = IPOPTOptions(
        max_iter=1000,
        tol=1e-6,
        linear_solver="ma27",
        enable_analysis=True,
        print_level=3
    )
    
    # Create CasADi solver using centralized factory
    from campro.optimization.ipopt_factory import create_ipopt_solver
    casadi_solver = create_ipopt_solver('solver', nlp, {
        'ipopt.max_iter': solver_options.max_iter,
        'ipopt.tol': solver_options.tol,
        'ipopt.linear_solver': solver_options.linear_solver,
        'ipopt.print_level': solver_options.print_level,
    }, force_linear_solver=True)
    
    # Set up Ipopt solver wrapper
    solver = IPOPTSolver(solver_options)
    
    # Solve the NLP
    try:
        result = solver.solve(
            nlp=casadi_solver,
            x0=phi_init,
            lbx=np.full(n, phi_min),
            ubx=np.full(n, phi_max),
            lbg=np.array([0.0]),  # periodicity constraint: phi[n-1] - phi[0] = 0
            ubg=np.array([0.0])
        )
        
        if result.success:
            # Extract optimized phi values
            phi_opt = result.x_opt
            
            # Validate with full physics
            physics_obj_val = objective_with_physics(phi_opt)
            log.info(f"Physics objective validation: {physics_obj_val:.6f}")
            
            # Check if physics-based objective is acceptable
            if physics_obj_val > 1e3:  # Infeasible
                log.warning("Optimized solution failed physics feasibility check")
            
            # Evaluate final metrics
            m = evaluate_order0_metrics_given_phi(cand, phi_opt.tolist())
            obj = m.slip_integral - 0.1 * m.contact_length + (0.0 if m.feasible else 1e3)
            
            # Perform analysis
            from campro.optimization.solver_analysis import analyze_ipopt_run
            analysis = analyze_ipopt_run({
                'success': result.success,
                'iterations': result.iterations,
                'primal_inf': result.primal_inf,
                'dual_inf': result.dual_inf,
                'return_status': result.status
            }, None)  # No log file for now
            
            # Add physics validation to analysis
            analysis.stats['physics_objective'] = physics_obj_val
            analysis.stats['physics_feasible'] = physics_obj_val < 1e3
            
            return OptimResult(
                best_config=cand,
                objective_value=obj,
                feasible=m.feasible,
                ipopt_analysis=analysis
            )
        else:
            log.warning(f"ORDER2_MICRO Ipopt optimization failed: {result.message}")
            # Fall back to simple smoothing
            return _order2_fallback_smoothing(config, phi_init, cand)
            
    except Exception as e:
        log.error(f"ORDER2_MICRO Ipopt optimization error: {e}")
        # Fall back to simple smoothing
        return _order2_fallback_smoothing(config, phi_init, cand)


def _order2_fallback_smoothing(config: GeometrySearchConfig, phi_init: np.ndarray, cand: PlanetSynthesisConfig) -> OptimResult:
    """Fallback to simple smoothing if Ipopt fails."""
    phi_vals = phi_init.tolist()
    
    # Simple smoothing (quadratic penalty) with few iterations
    lam = 1e-2
    for _ in range(5):
        # local averaging as a proxy for solving (I + λL)φ = rhs
        new_phi = phi_vals.copy()
        for i in range(len(phi_vals)):
            im = (i - 1) % len(phi_vals)
            ip = (i + 1) % len(phi_vals)
            new_phi[i] = (phi_vals[i] + lam * (phi_vals[im] + phi_vals[ip])) / (1.0 + 2.0 * lam)
        phi_vals = new_phi

    m = evaluate_order0_metrics_given_phi(cand, phi_vals)
    obj = m.slip_integral - 0.1 * m.contact_length + (0.0 if m.feasible else 1e3)
    return OptimResult(best_config=cand, objective_value=obj, feasible=m.feasible)

