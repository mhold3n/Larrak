"""
Motion-specific constraint definitions.

This module defines constraints for general motion law problems,
including position, velocity, acceleration, and jerk bounds,
as well as boundary conditions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from .base import BaseConstraints, ConstraintType, ConstraintViolation
from campro.logging import get_logger

log = get_logger(__name__)


class MotionConstraintType(Enum):
    """Types of motion constraints."""
    
    POSITION_BOUNDS = "position_bounds"
    VELOCITY_BOUNDS = "velocity_bounds"
    ACCELERATION_BOUNDS = "acceleration_bounds"
    JERK_BOUNDS = "jerk_bounds"
    CONTROL_BOUNDS = "control_bounds"
    INITIAL_POSITION = "initial_position"
    INITIAL_VELOCITY = "initial_velocity"
    INITIAL_ACCELERATION = "initial_acceleration"
    FINAL_POSITION = "final_position"
    FINAL_VELOCITY = "final_velocity"
    FINAL_ACCELERATION = "final_acceleration"


@dataclass
class MotionConstraints(BaseConstraints):
    """
    Constraints for optimal motion law problems.
    
    This class defines all constraints that can be applied to motion law
    optimization problems, including state bounds and boundary conditions.
    """
    
    # State constraints
    position_bounds: Optional[Tuple[float, float]] = None
    velocity_bounds: Optional[Tuple[float, float]] = None
    acceleration_bounds: Optional[Tuple[float, float]] = None
    jerk_bounds: Optional[Tuple[float, float]] = None
    
    # Control constraints
    control_bounds: Optional[Tuple[float, float]] = None
    
    # Boundary conditions
    initial_position: Optional[float] = None
    initial_velocity: Optional[float] = None
    initial_acceleration: Optional[float] = None
    final_position: Optional[float] = None
    final_velocity: Optional[float] = None
    final_acceleration: Optional[float] = None
    
    def __post_init__(self):
        """Initialize the constraint system."""
        super().__init__()
        self._register_constraints()
    
    def _register_constraints(self):
        """Register all constraints in the constraint system."""
        constraints = {
            'position_bounds': self.position_bounds,
            'velocity_bounds': self.velocity_bounds,
            'acceleration_bounds': self.acceleration_bounds,
            'jerk_bounds': self.jerk_bounds,
            'control_bounds': self.control_bounds,
            'initial_position': self.initial_position,
            'initial_velocity': self.initial_velocity,
            'initial_acceleration': self.initial_acceleration,
            'final_position': self.final_position,
            'final_velocity': self.final_velocity,
            'final_acceleration': self.final_acceleration,
        }
        
        for name, value in constraints.items():
            if value is not None:
                self.add_constraint(name, value)
    
    def validate(self) -> bool:
        """
        Validate all motion constraints.
        
        Returns:
            bool: True if all constraints are valid, False otherwise
        """
        self.clear_violations()
        is_valid = True
        
        # Validate bounds
        bounds_to_check = [
            ('position_bounds', self.position_bounds),
            ('velocity_bounds', self.velocity_bounds),
            ('acceleration_bounds', self.acceleration_bounds),
            ('jerk_bounds', self.jerk_bounds),
            ('control_bounds', self.control_bounds),
        ]
        
        for name, bounds in bounds_to_check:
            if bounds is not None:
                min_val, max_val = bounds
                if min_val is not None and max_val is not None and min_val > max_val:
                    violation = ConstraintViolation(
                        constraint_type=ConstraintType.CUSTOM,
                        constraint_name=f"{name}_validation",
                        violation_value=min_val,
                        limit_value=max_val,
                        message=f"{name}: minimum value {min_val} exceeds maximum value {max_val}"
                    )
                    self._add_violation(violation)
                    is_valid = False
        
        # Validate boundary conditions
        if self.initial_position is not None and self.final_position is not None:
            if self.position_bounds is not None:
                min_pos, max_pos = self.position_bounds
                if min_pos is not None and self.initial_position < min_pos:
                    violation = ConstraintViolation(
                        constraint_type=ConstraintType.INITIAL_STATE,
                        constraint_name="initial_position_bounds",
                        violation_value=self.initial_position,
                        limit_value=min_pos,
                        message=f"Initial position {self.initial_position} below minimum {min_pos}"
                    )
                    self._add_violation(violation)
                    is_valid = False
                
                if max_pos is not None and self.initial_position > max_pos:
                    violation = ConstraintViolation(
                        constraint_type=ConstraintType.INITIAL_STATE,
                        constraint_name="initial_position_bounds",
                        violation_value=self.initial_position,
                        limit_value=max_pos,
                        message=f"Initial position {self.initial_position} above maximum {max_pos}"
                    )
                    self._add_violation(violation)
                    is_valid = False
        
        log.info(f"Motion constraints validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid
    
    def check_violations(self, solution: Dict[str, np.ndarray]) -> List[ConstraintViolation]:
        """
        Check for constraint violations in a motion law solution.
        
        Args:
            solution: Dictionary containing solution arrays
                - 'time': Time array
                - 'position': Position array
                - 'velocity': Velocity array
                - 'acceleration': Acceleration array
                - 'control': Control (jerk) array
                
        Returns:
            List of constraint violations found
        """
        self.clear_violations()
        
        # Check state bounds
        if 'position' in solution:
            violations = self._check_bounds(
                solution['position'], self.position_bounds,
                ConstraintType.POSITION, 'position'
            )
            self._violations.extend(violations)
        
        if 'velocity' in solution:
            violations = self._check_bounds(
                solution['velocity'], self.velocity_bounds,
                ConstraintType.VELOCITY, 'velocity'
            )
            self._violations.extend(violations)
        
        if 'acceleration' in solution:
            violations = self._check_bounds(
                solution['acceleration'], self.acceleration_bounds,
                ConstraintType.ACCELERATION, 'acceleration'
            )
            self._violations.extend(violations)
        
        if 'control' in solution:
            violations = self._check_bounds(
                solution['control'], self.jerk_bounds,
                ConstraintType.JERK, 'jerk'
            )
            self._violations.extend(violations)
            
            # Also check control bounds if different from jerk bounds
            if self.control_bounds != self.jerk_bounds:
                violations = self._check_bounds(
                    solution['control'], self.control_bounds,
                    ConstraintType.CONTROL, 'control'
                )
                self._violations.extend(violations)
        
        # Check boundary conditions
        if 'position' in solution and len(solution['position']) > 0:
            # Initial position
            if self.initial_position is not None:
                violation = self._check_boundary_condition(
                    solution['position'][0], self.initial_position,
                    ConstraintType.INITIAL_STATE, 'initial_position'
                )
                if violation:
                    self._violations.append(violation)
            
            # Final position
            if self.final_position is not None:
                violation = self._check_boundary_condition(
                    solution['position'][-1], self.final_position,
                    ConstraintType.FINAL_STATE, 'final_position'
                )
                if violation:
                    self._violations.append(violation)
        
        if 'velocity' in solution and len(solution['velocity']) > 0:
            # Initial velocity
            if self.initial_velocity is not None:
                violation = self._check_boundary_condition(
                    solution['velocity'][0], self.initial_velocity,
                    ConstraintType.INITIAL_STATE, 'initial_velocity'
                )
                if violation:
                    self._violations.append(violation)
            
            # Final velocity
            if self.final_velocity is not None:
                violation = self._check_boundary_condition(
                    solution['velocity'][-1], self.final_velocity,
                    ConstraintType.FINAL_STATE, 'final_velocity'
                )
                if violation:
                    self._violations.append(violation)
        
        if 'acceleration' in solution and len(solution['acceleration']) > 0:
            # Initial acceleration
            if self.initial_acceleration is not None:
                violation = self._check_boundary_condition(
                    solution['acceleration'][0], self.initial_acceleration,
                    ConstraintType.INITIAL_STATE, 'initial_acceleration'
                )
                if violation:
                    self._violations.append(violation)
            
            # Final acceleration
            if self.final_acceleration is not None:
                violation = self._check_boundary_condition(
                    solution['acceleration'][-1], self.final_acceleration,
                    ConstraintType.FINAL_STATE, 'final_acceleration'
                )
                if violation:
                    self._violations.append(violation)
        
        log.info(f"Found {len(self._violations)} constraint violations")
        return self._violations.copy()
    
    def to_dict(self) -> Dict[str, any]:
        """Convert constraints to dictionary format."""
        return {
            'position_bounds': self.position_bounds,
            'velocity_bounds': self.velocity_bounds,
            'acceleration_bounds': self.acceleration_bounds,
            'jerk_bounds': self.jerk_bounds,
            'control_bounds': self.control_bounds,
            'initial_position': self.initial_position,
            'initial_velocity': self.initial_velocity,
            'initial_acceleration': self.initial_acceleration,
            'final_position': self.final_position,
            'final_velocity': self.final_velocity,
            'final_acceleration': self.final_acceleration,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'MotionConstraints':
        """Create constraints from dictionary format."""
        return cls(**data)
