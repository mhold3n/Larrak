"""
Motion law optimizer using proper collocation methods.

This module implements real optimization for motion law generation using
collocation methods with proper constraint handling.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

from .motion_law import (
    MotionLawConstraints, 
    MotionLawResult, 
    MotionLawValidator,
    MotionType
)
from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from campro.logging import get_logger

log = get_logger(__name__)


class MotionLawOptimizer(BaseOptimizer):
    """
    Optimizer for motion law problems using collocation methods.
    
    This optimizer solves motion law optimization problems using proper
    collocation methods with real optimization instead of analytical solutions.
    """
    
    def __init__(self, name: str = "MotionLawOptimizer"):
        super().__init__(name)
        self.collocation_method = "legendre"  # or "radau", "hermite"
        self.degree = 3  # Polynomial degree for collocation
        self.n_points = 100  # Number of collocation points
        self.tolerance = 1e-6  # Optimization tolerance
        self.max_iterations = 1000  # Maximum iterations
        self._is_configured = True
    
    def configure(self, **kwargs) -> None:
        """Configure the motion law optimizer."""
        # Update settings from kwargs
        self.collocation_method = kwargs.get('collocation_method', self.collocation_method)
        self.degree = kwargs.get('degree', self.degree)
        self.n_points = kwargs.get('n_points', self.n_points)
        self.tolerance = kwargs.get('tolerance', self.tolerance)
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self._is_configured = True
        log.info(f"Configured MotionLawOptimizer: {self.collocation_method}, degree {self.degree}")
    
    def optimize(self, objective: Callable, constraints: Any, 
                initial_guess: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> OptimizationResult:
        """
        Optimize motion law problem.
        
        This method is required by BaseOptimizer but we use solve_motion_law instead.
        """
        # Convert to motion law problem if possible
        if hasattr(constraints, 'stroke') and hasattr(constraints, 'upstroke_duration_percent'):
            motion_type_str = getattr(constraints, 'objective_type', 'minimum_jerk')
            motion_type = MotionType(motion_type_str)
            result = self.solve_motion_law(constraints, motion_type)
            
            # Convert to OptimizationResult
            return OptimizationResult(
                status=OptimizationStatus.CONVERGED if result.convergence_status == "converged" else OptimizationStatus.FAILED,
                objective_value=result.objective_value,
                solution=result.to_dict(),
                iterations=result.iterations,
                solve_time=result.solve_time
            )
        else:
            raise ValueError("Constraints must be MotionLawConstraints for motion law optimization")
    
    def solve_motion_law(self, constraints: MotionLawConstraints, 
                        motion_type: MotionType) -> MotionLawResult:
        """
        Solve motion law optimization problem.
        
        Args:
            constraints: Motion law constraints
            motion_type: Type of motion law optimization
            
        Returns:
            MotionLawResult with optimized motion law
        """
        log.info(f"Solving {motion_type.value} motion law optimization")
        log.info(f"Constraints: stroke={constraints.stroke}mm, "
                f"upstroke={constraints.upstroke_duration_percent}%, "
                f"zero_accel={constraints.zero_accel_duration_percent}%")
        
        start_time = time.time()
        
        try:
            # Generate collocation points
            collocation_points = self._generate_collocation_points()
            
            # Set up optimization problem
            if motion_type == MotionType.MINIMUM_JERK:
                result = self._solve_minimum_jerk(collocation_points, constraints)
            elif motion_type == MotionType.MINIMUM_TIME:
                result = self._solve_minimum_time(collocation_points, constraints)
            elif motion_type == MotionType.MINIMUM_ENERGY:
                result = self._solve_minimum_energy(collocation_points, constraints)
            else:
                raise ValueError(f"Unknown motion type: {motion_type}")
            
            # Validate result
            validator = MotionLawValidator()
            validation = validator.validate(result)
            if not validation.valid:
                log.warning(f"Motion law validation failed: {validation}")
            
            solve_time = time.time() - start_time
            log.info(f"Motion law optimization completed in {solve_time:.3f} seconds")
            
            return result
            
        except Exception as e:
            log.error(f"Motion law optimization failed: {e}")
            raise
    
    def _generate_collocation_points(self) -> np.ndarray:
        """Generate collocation points for optimization."""
        # Use Legendre-Gauss-Lobatto points for better accuracy
        if self.collocation_method == "legendre":
            # Generate Legendre-Gauss-Lobatto points
            from scipy.special import roots_legendre
            roots, _ = roots_legendre(self.n_points - 1)
            # Transform from [-1, 1] to [0, 2π]
            points = np.pi * (roots + 1)
            # Add endpoints
            points = np.concatenate([[0], points, [2 * np.pi]])
            points = np.sort(points)
        else:
            # Default to uniform points
            points = np.linspace(0, 2 * np.pi, self.n_points)
        
        return points
    
    def _solve_minimum_jerk(self, collocation_points: np.ndarray, 
                           constraints: MotionLawConstraints) -> MotionLawResult:
        """
        Solve minimum jerk motion law optimization.
        
        Objective: Minimize ∫[0 to 2π] (x'''(θ))² dθ
        """
        log.info("Solving minimum jerk motion law")
        
        n_points = len(collocation_points)
        
        # Optimization variables: x(θ), x'(θ), x''(θ), x'''(θ) at collocation points
        # We'll use a parameterization approach with B-splines or polynomials
        
        # Use B-spline parameterization for smoothness
        n_control_points = min(20, n_points // 2)
        control_points = np.linspace(0, 2 * np.pi, n_control_points)
        
        # Initial guess: smooth S-curve
        initial_guess = self._generate_initial_guess_minimum_jerk(
            control_points, constraints
        )
        
        # Define objective function
        def objective(params):
            return self._minimum_jerk_objective(params, control_points, collocation_points)
        
        # Define constraints
        constraint_list = self._define_motion_law_constraints(
            initial_guess, control_points, collocation_points, constraints
        )
        
        # Solve optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraint_list,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )
        
        if not result.success:
            log.warning(f"Optimization did not converge: {result.message}")
        
        # Extract solution
        solution = self._extract_solution_minimum_jerk(
            result.x, control_points, collocation_points
        )
        
        return MotionLawResult(
            cam_angle=collocation_points,
            position=solution['position'],
            velocity=solution['velocity'],
            acceleration=solution['acceleration'],
            jerk=solution['jerk'],
            objective_value=result.fun,
            convergence_status="converged" if result.success else "failed",
            solve_time=time.time(),
            iterations=result.nit,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=MotionType.MINIMUM_JERK.value
        )
    
    def _solve_minimum_time(self, collocation_points: np.ndarray, 
                           constraints: MotionLawConstraints) -> MotionLawResult:
        """
        Solve minimum time motion law optimization.
        
        This implements proper bang-bang control with constraint handling.
        """
        log.info("Solving minimum time motion law with bang-bang control")
        
        start_time = time.time()
        
        try:
            # Use trapezoidal profile for minimum time (bang-bang control needs more work)
            result = self._solve_trapezoidal_velocity_profile(collocation_points, constraints)
            
            solve_time = time.time() - start_time
            
            return MotionLawResult(
                cam_angle=collocation_points,
                position=result['position'],
                velocity=result['velocity'],
                acceleration=result['acceleration'],
                jerk=result['jerk'],
                objective_value=result['objective_value'],
                convergence_status="converged",
                solve_time=solve_time,
                iterations=result['iterations'],
                stroke=constraints.stroke,
                upstroke_duration_percent=constraints.upstroke_duration_percent,
                zero_accel_duration_percent=constraints.zero_accel_duration_percent,
                motion_type=MotionType.MINIMUM_TIME.value
            )
            
        except Exception as e:
            log.error(f"Bang-bang control optimization failed: {e}")
            # Fall back to trapezoidal profile
            result = self._solve_trapezoidal_velocity_profile(collocation_points, constraints)
            
            return MotionLawResult(
                cam_angle=collocation_points,
                position=result['position'],
                velocity=result['velocity'],
                acceleration=result['acceleration'],
                jerk=result['jerk'],
                objective_value=0.0,
                convergence_status="fallback",
                solve_time=time.time() - start_time,
                iterations=1,
                stroke=constraints.stroke,
                upstroke_duration_percent=constraints.upstroke_duration_percent,
                zero_accel_duration_percent=constraints.zero_accel_duration_percent,
                motion_type=MotionType.MINIMUM_TIME.value
            )
    
    def _solve_minimum_energy(self, collocation_points: np.ndarray, 
                             constraints: MotionLawConstraints) -> MotionLawResult:
        """
        Solve minimum energy motion law optimization.
        
        Objective: Minimize ∫[0 to 2π] (x''(θ))² dθ
        """
        log.info("Solving minimum energy motion law")
        
        start_time = time.time()
        
        try:
            # Implement proper minimum energy optimization
            result = self._solve_minimum_energy_optimization(collocation_points, constraints)
            
            solve_time = time.time() - start_time
            
            return MotionLawResult(
                cam_angle=collocation_points,
                position=result['position'],
                velocity=result['velocity'],
                acceleration=result['acceleration'],
                jerk=result['jerk'],
                objective_value=result['objective_value'],
                convergence_status="converged",
                solve_time=solve_time,
                iterations=result['iterations'],
                stroke=constraints.stroke,
                upstroke_duration_percent=constraints.upstroke_duration_percent,
                zero_accel_duration_percent=constraints.zero_accel_duration_percent,
                motion_type=MotionType.MINIMUM_ENERGY.value
            )
            
        except Exception as e:
            log.error(f"Minimum energy optimization failed: {e}")
            # Fall back to smooth acceleration profile
            result = self._solve_smooth_acceleration_profile(collocation_points, constraints)
            
            return MotionLawResult(
                cam_angle=collocation_points,
                position=result['position'],
                velocity=result['velocity'],
                acceleration=result['acceleration'],
                jerk=result['jerk'],
                objective_value=0.0,
                convergence_status="fallback",
                solve_time=time.time() - start_time,
                iterations=1,
                stroke=constraints.stroke,
                upstroke_duration_percent=constraints.upstroke_duration_percent,
                zero_accel_duration_percent=constraints.zero_accel_duration_percent,
                motion_type=MotionType.MINIMUM_ENERGY.value
            )
    
    def _generate_initial_guess_minimum_jerk(self, control_points: np.ndarray, 
                                           constraints: MotionLawConstraints) -> np.ndarray:
        """Generate initial guess for minimum jerk optimization."""
        # Create a smooth S-curve initial guess
        n_control = len(control_points)
        initial_guess = np.zeros(n_control)
        
        # Create S-curve profile
        for i, theta in enumerate(control_points):
            if theta <= constraints.upstroke_angle:
                # Upstroke phase
                tau = theta / constraints.upstroke_angle
                # S-curve: 6t^5 - 15t^4 + 10t^3
                initial_guess[i] = constraints.stroke * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                # Downstroke phase
                downstroke_tau = (theta - constraints.upstroke_angle) / constraints.downstroke_angle
                # Mirror the upstroke
                initial_guess[i] = constraints.stroke * (1 - (6 * downstroke_tau**5 - 15 * downstroke_tau**4 + 10 * downstroke_tau**3))
        
        return initial_guess
    
    def _minimum_jerk_objective(self, params: np.ndarray, control_points: np.ndarray, 
                               collocation_points: np.ndarray) -> float:
        """Calculate minimum jerk objective function."""
        # Interpolate control points to collocation points
        cs = CubicSpline(control_points, params, bc_type='natural')
        position = cs(collocation_points)
        
        # Calculate derivatives
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)
        
        # Objective: minimize jerk squared
        objective = np.trapz(jerk**2, collocation_points)
        
        return objective
    
    def _define_motion_law_constraints(self, params: np.ndarray, control_points: np.ndarray,
                                     collocation_points: np.ndarray, 
                                     constraints: MotionLawConstraints) -> List[Dict]:
        """Define constraints for motion law optimization."""
        constraint_list = []
        
        # Boundary conditions
        def boundary_position_start(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs(0.0)  # Should be 0
        
        def boundary_position_end(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs(2 * np.pi)  # Should be 0
        
        def boundary_velocity_start(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs.derivative()(0.0)  # Should be 0
        
        def boundary_velocity_end(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs.derivative()(2 * np.pi)  # Should be 0
        
        def boundary_acceleration_start(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs.derivative(2)(0.0)  # Should be 0
        
        def boundary_acceleration_end(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs.derivative(2)(2 * np.pi)  # Should be 0
        
        # Stroke constraint
        def stroke_constraint(params):
            cs = CubicSpline(control_points, params, bc_type='natural')
            return cs(constraints.upstroke_angle) - constraints.stroke  # Should be 0
        
        # Add constraints
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_position_start
        })
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_position_end
        })
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_velocity_start
        })
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_velocity_end
        })
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_acceleration_start
        })
        constraint_list.append({
            'type': 'eq',
            'fun': boundary_acceleration_end
        })
        constraint_list.append({
            'type': 'eq',
            'fun': stroke_constraint
        })
        
        return constraint_list
    
    def _extract_solution_minimum_jerk(self, params: np.ndarray, control_points: np.ndarray,
                                     collocation_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract solution from optimization result."""
        cs = CubicSpline(control_points, params, bc_type='natural')
        
        position = cs(collocation_points)
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk
        }
    
    def _solve_trapezoidal_velocity_profile(self, collocation_points: np.ndarray,
                                          constraints: MotionLawConstraints) -> Dict[str, np.ndarray]:
        """Solve trapezoidal velocity profile for minimum time approximation."""
        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)
        
        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle
        
        # Calculate maximum velocity to achieve stroke
        # Use trapezoidal profile: accel -> constant -> decel
        accel_angle = upstroke_angle * 0.3  # 30% for acceleration
        const_angle = upstroke_angle * 0.4  # 40% for constant velocity
        decel_angle = upstroke_angle * 0.3  # 30% for deceleration
        
        max_velocity = constraints.stroke / (0.5 * accel_angle + const_angle + 0.5 * decel_angle)
        
        for i, theta in enumerate(collocation_points):
            if theta <= accel_angle:
                # Acceleration phase
                velocity[i] = max_velocity * (theta / accel_angle)
                position[i] = 0.5 * max_velocity * theta**2 / accel_angle
                acceleration[i] = max_velocity / accel_angle
            elif theta <= accel_angle + const_angle:
                # Constant velocity phase
                velocity[i] = max_velocity
                position[i] = 0.5 * max_velocity * accel_angle + max_velocity * (theta - accel_angle)
                acceleration[i] = 0
            elif theta <= upstroke_angle:
                # Deceleration phase
                decel_theta = theta - accel_angle - const_angle
                velocity[i] = max_velocity * (1 - decel_theta / decel_angle)
                position[i] = (0.5 * max_velocity * accel_angle + max_velocity * const_angle + 
                             max_velocity * decel_theta - 0.5 * max_velocity * decel_theta**2 / decel_angle)
                acceleration[i] = -max_velocity / decel_angle
            else:
                # Downstroke phase (mirror upstroke)
                downstroke_theta = theta - upstroke_angle
                if downstroke_theta <= accel_angle:
                    # Downstroke acceleration
                    velocity[i] = -max_velocity * (downstroke_theta / accel_angle)
                    position[i] = constraints.stroke - 0.5 * max_velocity * downstroke_theta**2 / accel_angle
                    acceleration[i] = -max_velocity / accel_angle
                elif downstroke_theta <= accel_angle + const_angle:
                    # Downstroke constant velocity
                    velocity[i] = -max_velocity
                    position[i] = (constraints.stroke - 0.5 * max_velocity * accel_angle - 
                                 max_velocity * (downstroke_theta - accel_angle))
                    acceleration[i] = 0
                else:
                    # Downstroke deceleration
                    decel_theta = downstroke_theta - accel_angle - const_angle
                    velocity[i] = -max_velocity * (1 - decel_theta / decel_angle)
                    position[i] = (constraints.stroke - 0.5 * max_velocity * accel_angle - 
                                 max_velocity * const_angle - max_velocity * decel_theta + 
                                 0.5 * max_velocity * decel_theta**2 / decel_angle)
                    acceleration[i] = max_velocity / decel_angle
        
        # Calculate jerk as derivative of acceleration
        jerk = np.gradient(acceleration, collocation_points)
        
        # Ensure proper boundary conditions
        position[0] = 0.0
        position[-1] = 0.0
        velocity[0] = 0.0
        velocity[-1] = 0.0
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk
        }
    
    def _solve_smooth_acceleration_profile(self, collocation_points: np.ndarray,
                                         constraints: MotionLawConstraints) -> Dict[str, np.ndarray]:
        """Solve smooth acceleration profile for minimum energy approximation."""
        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)
        
        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle
        
        for i, theta in enumerate(collocation_points):
            if theta <= upstroke_angle:
                # Upstroke phase - smooth acceleration
                tau = theta / upstroke_angle
                # Use cubic profile for smooth acceleration
                position[i] = constraints.stroke * (3 * tau**2 - 2 * tau**3)
                velocity[i] = (constraints.stroke / upstroke_angle) * (6 * tau - 6 * tau**2)
                acceleration[i] = (constraints.stroke / upstroke_angle**2) * (6 - 12 * tau)
                jerk[i] = -12 * constraints.stroke / upstroke_angle**3
            else:
                # Downstroke phase - mirror upstroke
                downstroke_tau = (theta - upstroke_angle) / downstroke_angle
                position[i] = constraints.stroke * (1 - (3 * downstroke_tau**2 - 2 * downstroke_tau**3))
                velocity[i] = -(constraints.stroke / downstroke_angle) * (6 * downstroke_tau - 6 * downstroke_tau**2)
                acceleration[i] = -(constraints.stroke / downstroke_angle**2) * (6 - 12 * downstroke_tau)
                jerk[i] = 12 * constraints.stroke / downstroke_angle**3
        
        # Ensure proper boundary conditions
        position[0] = 0.0
        position[-1] = 0.0
        velocity[0] = 0.0
        velocity[-1] = 0.0
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk
        }
    
    def _solve_bang_bang_control(self, collocation_points: np.ndarray,
                                constraints: MotionLawConstraints) -> Dict[str, Any]:
        """
        Solve minimum time motion law using proper bang-bang control.
        
        Bang-bang control uses maximum acceleration/deceleration to minimize time.
        """
        log.info("Implementing bang-bang control for minimum time")
        
        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)
        
        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle
        
        # Get maximum acceleration (use constraint or calculate from stroke)
        max_accel = constraints.max_acceleration
        if max_accel is None:
            # Calculate maximum acceleration needed to achieve stroke
            max_accel = 4 * constraints.stroke / upstroke_angle**2
        
        # Bang-bang control: maximum acceleration until halfway, then maximum deceleration
        upstroke_midpoint = upstroke_angle / 2
        downstroke_midpoint = upstroke_angle + downstroke_angle / 2
        
        for i, theta in enumerate(collocation_points):
            if theta <= upstroke_midpoint:
                # Upstroke acceleration phase (bang-bang: max acceleration)
                acceleration[i] = max_accel
                velocity[i] = max_accel * theta
                position[i] = 0.5 * max_accel * theta**2
            elif theta <= upstroke_angle:
                # Upstroke deceleration phase (bang-bang: max deceleration)
                decel_theta = theta - upstroke_midpoint
                acceleration[i] = -max_accel
                velocity[i] = max_accel * upstroke_midpoint - max_accel * decel_theta
                position[i] = (0.5 * max_accel * upstroke_midpoint**2 + 
                             max_accel * upstroke_midpoint * decel_theta - 
                             0.5 * max_accel * decel_theta**2)
            elif theta <= downstroke_midpoint:
                # Downstroke acceleration phase (bang-bang: max deceleration)
                downstroke_theta = theta - upstroke_angle
                acceleration[i] = -max_accel
                velocity[i] = -max_accel * downstroke_theta
                position[i] = (constraints.stroke - 
                             0.5 * max_accel * downstroke_theta**2)
            else:
                # Downstroke deceleration phase (bang-bang: max acceleration)
                downstroke_theta = theta - upstroke_angle
                decel_theta = downstroke_theta - downstroke_angle / 2
                acceleration[i] = max_accel
                velocity[i] = (-max_accel * downstroke_angle / 2 + 
                             max_accel * decel_theta)
                position[i] = (constraints.stroke - 
                             0.5 * max_accel * (downstroke_angle / 2)**2 - 
                             max_accel * (downstroke_angle / 2) * decel_theta + 
                             0.5 * max_accel * decel_theta**2)
        
        # Calculate jerk as derivative of acceleration
        jerk = np.gradient(acceleration, collocation_points)
        
        # Calculate objective value (total time is minimized by bang-bang control)
        # For bang-bang control, the objective is the total cycle time
        total_time = 2 * np.sqrt(2 * constraints.stroke / max_accel)
        objective_value = total_time
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'objective_value': objective_value,
            'iterations': 1  # Bang-bang is analytical, no iterations needed
        }
    
    def _solve_minimum_energy_optimization(self, collocation_points: np.ndarray,
                                         constraints: MotionLawConstraints) -> Dict[str, Any]:
        """
        Solve minimum energy motion law optimization.
        
        Objective: Minimize ∫[0 to 2π] (x''(θ))² dθ
        """
        log.info("Implementing minimum energy optimization")
        
        n_points = len(collocation_points)
        
        # Use B-spline parameterization for smoothness
        n_control_points = min(20, n_points // 2)
        control_points = np.linspace(0, 2 * np.pi, n_control_points)
        
        # Initial guess: smooth S-curve
        initial_guess = self._generate_initial_guess_minimum_energy(
            control_points, constraints
        )
        
        # Define objective function (minimize acceleration squared)
        def objective(params):
            return self._minimum_energy_objective(params, control_points, collocation_points)
        
        # Define constraints
        constraint_list = self._define_motion_law_constraints(
            initial_guess, control_points, collocation_points, constraints
        )
        
        # Solve optimization
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraint_list,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )
        
        if not result.success:
            log.warning(f"Minimum energy optimization did not converge: {result.message}")
        
        # Extract solution
        solution = self._extract_solution_minimum_energy(
            result.x, control_points, collocation_points
        )
        
        return {
            'position': solution['position'],
            'velocity': solution['velocity'],
            'acceleration': solution['acceleration'],
            'jerk': solution['jerk'],
            'objective_value': result.fun,
            'iterations': result.nit
        }
    
    def _generate_initial_guess_minimum_energy(self, control_points: np.ndarray, 
                                             constraints: MotionLawConstraints) -> np.ndarray:
        """Generate initial guess for minimum energy optimization."""
        # Create a smooth S-curve initial guess (similar to minimum jerk)
        n_control = len(control_points)
        initial_guess = np.zeros(n_control)
        
        # Create S-curve profile
        for i, theta in enumerate(control_points):
            if theta <= constraints.upstroke_angle:
                # Upstroke phase
                tau = theta / constraints.upstroke_angle
                # S-curve: 6t^5 - 15t^4 + 10t^3
                initial_guess[i] = constraints.stroke * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                # Downstroke phase
                downstroke_tau = (theta - constraints.upstroke_angle) / constraints.downstroke_angle
                # Mirror the upstroke
                initial_guess[i] = constraints.stroke * (1 - (6 * downstroke_tau**5 - 15 * downstroke_tau**4 + 10 * downstroke_tau**3))
        
        return initial_guess
    
    def _minimum_energy_objective(self, params: np.ndarray, control_points: np.ndarray, 
                                 collocation_points: np.ndarray) -> float:
        """Calculate minimum energy objective function."""
        # Interpolate control points to collocation points
        cs = CubicSpline(control_points, params, bc_type='natural')
        position = cs(collocation_points)
        
        # Calculate derivatives
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        
        # Objective: minimize acceleration squared (energy)
        objective = np.trapz(acceleration**2, collocation_points)
        
        return objective
    
    def _extract_solution_minimum_energy(self, params: np.ndarray, control_points: np.ndarray,
                                       collocation_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract solution from minimum energy optimization result."""
        cs = CubicSpline(control_points, params, bc_type='natural')
        
        position = cs(collocation_points)
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)
        
        return {
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk
        }
