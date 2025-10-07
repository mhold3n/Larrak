# Optimal Motion Law Design Document

## Overview

This document describes the design and implementation of the optimal motion law solver using CasADi and Ipopt with direct collocation methods.

## Architecture

### Core Components

1. **OptimalMotionSolver**: Main solver class that implements various optimal motion law problems
2. **MotionConstraints**: Data structure for defining motion constraints and boundary conditions
3. **CollocationSettings**: Configuration for collocation method parameters
4. **Convenience Functions**: High-level functions for common motion law problems

### Motion Law Types

The solver supports several types of optimal motion laws:

1. **Minimum Time**: Minimize the time to complete a motion while satisfying constraints
2. **Minimum Energy**: Minimize energy consumption for a fixed time horizon
3. **Minimum Jerk**: Minimize jerk (third derivative of position) for smooth motion
4. **Custom Objectives**: User-defined objective functions

### Collocation Methods

The implementation supports three collocation methods:

1. **Legendre**: Legendre-Gauss-Radau collocation
2. **Radau**: Radau collocation
3. **Lobatto**: Lobatto collocation

## Mathematical Formulation

### State Variables

- `x`: Position
- `v`: Velocity (dx/dt)
- `a`: Acceleration (dv/dt)
- `u`: Control input (jerk, da/dt)

### Dynamics

The system dynamics are:
```
dx/dt = v
dv/dt = a
da/dt = u
```

### Constraints

1. **Path Constraints**: Limits on velocity, acceleration, and jerk
2. **Boundary Conditions**: Initial and final position, velocity, and acceleration
3. **Control Constraints**: Limits on control input

### Objective Functions

1. **Minimum Time**: `min T` where T is the final time
2. **Minimum Energy**: `min ∫ u² dt`
3. **Minimum Jerk**: `min ∫ u² dt` where u is jerk
4. **Custom**: User-defined function of states and control

## Implementation Details

### CasADi Integration

The solver uses CasADi's optimal control problem (OCP) interface:

```python
ocp = ca.Ocp()
ocp.set_state(x)
ocp.set_state(v)
ocp.set_state(a)
ocp.set_control(u)
```

### Collocation Setup

Collocation is configured using CasADi's collocation methods:

```python
ocp.method(ca.Collocation(ocp, degree, method))
```

### Ipopt Solver

The OCP is solved using Ipopt with configurable options:

```python
ocp.solver('ipopt', {
    'ipopt.max_iter': max_iterations,
    'ipopt.tol': tolerance,
    'ipopt.print_level': verbose_level
})
```

## Usage Examples

### Minimum Time Motion

```python
from CamPro_OptimalMotion import solve_minimum_time_motion

solution = solve_minimum_time_motion(
    distance=10.0,
    max_velocity=5.0,
    max_acceleration=2.0,
    max_jerk=1.0
)
```

### Minimum Energy Motion

```python
from CamPro_OptimalMotion import solve_minimum_energy_motion

solution = solve_minimum_energy_motion(
    distance=10.0,
    time_horizon=5.0,
    max_velocity=5.0,
    max_acceleration=2.0
)
```

### Custom Objective

```python
from CamPro_OptimalMotion import OptimalMotionSolver, MotionConstraints

def custom_objective(t, x, v, a, u):
    return ca.integral(u**2 + 0.1*v**2)

solver = OptimalMotionSolver()
constraints = MotionConstraints(
    initial_position=0.0,
    final_position=10.0
)

solution = solver.solve_custom_objective(
    custom_objective, constraints, distance=10.0, time_horizon=5.0
)
```

## Performance Considerations

### Collocation Degree

- Higher degrees provide better accuracy but increase computational cost
- Default degree of 3 provides good balance between accuracy and speed
- For smooth motions, degree 5-7 may be beneficial

### Solver Settings

- Tolerance: Lower values provide better accuracy but slower convergence
- Max iterations: Higher values allow more complex problems but slower solving
- Verbose output: Useful for debugging but adds overhead

### Problem Scaling

- Ensure problem variables are well-scaled (order of magnitude 1)
- Use appropriate units to avoid numerical issues
- Consider constraint bounds carefully

## Validation and Testing

### Unit Tests

- Test individual components (constraints, settings, solver initialization)
- Test boundary conditions and constraint satisfaction
- Test error handling and edge cases

### Integration Tests

- Test complete motion law problems
- Verify solution properties (boundary conditions, constraints)
- Test different collocation methods

### Property-Based Tests

- Use Hypothesis for parameter space exploration
- Test solution properties across different parameter combinations
- Verify mathematical invariants

## Future Enhancements

### Multi-Dimensional Motion

- Extend to 2D and 3D motion problems
- Support for path constraints and obstacles
- Multi-segment trajectories

### Advanced Constraints

- State-dependent constraints
- Nonlinear constraints
- Inequality constraints

### Optimization Improvements

- Warm-start capabilities
- Parallel solving for multiple problems
- Adaptive collocation refinement

### Visualization

- Interactive plotting and animation
- Constraint visualization
- Solution comparison tools

## Dependencies

- **CasADi**: Symbolic computation and optimal control
- **Ipopt**: Interior-point optimizer
- **NumPy**: Numerical computations
- **Matplotlib**: Plotting and visualization
- **SciPy**: Additional numerical tools

## References

1. Betts, J. T. (2010). Practical Methods for Optimal Control and Estimation Using Nonlinear Programming
2. Rao, A. V. (2009). A Survey of Numerical Methods for Optimal Control
3. CasADi Documentation: https://web.casadi.org/
4. Ipopt Documentation: https://coin-or.github.io/Ipopt/





