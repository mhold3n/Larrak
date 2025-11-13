# CasADi API Examples

This document provides practical code snippets and examples for using the CasADi-based optimization framework, following CasADi documentation patterns.

## Table of Contents

1. [Basic Phase 1 Optimization](#basic-phase-1-optimization)
2. [Deterministic Seeding Examples](#deterministic-seeding-examples)
3. [Problem Specification](#problem-specification)
4. [Thermal Efficiency Integration](#thermal-efficiency-integration)
5. [Advanced Configuration](#advanced-configuration)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)

## Basic Phase 1 Optimization

### Example 1: Simple Motion Law Optimization

```python
from campro.optimization.casadi_motion_optimizer import CasADiMotionOptimizer, CasADiMotionProblem

# Create optimization problem
# All motion constraints must be in per-degree units (m/deg, m/deg², m/deg³)
# duration_angle_deg is required
problem = CasADiMotionProblem(
    stroke=0.100,  # 100mm stroke (in meters)
    duration_angle_deg=360.0,  # Required: motion law duration in degrees
    cycle_time=0.0385,  # 26 Hz (derived from engine_speed_rpm and duration_angle_deg)
    upstroke_percent=50.0,
    max_velocity=0.00019,  # m/deg (~0.19 mm/deg)
    max_acceleration=0.0019,  # m/deg² (~1.9 mm/deg²)
    max_jerk=0.019,  # m/deg³ (~19 mm/deg³)
    compression_ratio_limits=(20.0, 70.0),
    minimize_jerk=True,
    maximize_thermal_efficiency=True,
    weights={
        'jerk': 1.0,
        'thermal_efficiency': 0.1,
        'smoothness': 0.01
    }
)

# Initialize optimizer
optimizer = CasADiMotionOptimizer(
    n_segments=50,
    poly_order=3,
    collocation_method="legendre"
)

# Solve optimization
result = optimizer.solve(problem)

# Check results
if result.successful:
    print(f"Optimization successful!")
    print(f"Solve time: {result.solve_time:.3f}s")
    print(f"Objective value: {result.objective_value:.6f}")
    
    # Extract motion profile
    position = result.variables['position']
    velocity = result.variables['velocity']
    acceleration = result.variables['acceleration']
    jerk = result.variables['jerk']
else:
    print(f"Optimization failed: {result.metadata.get('error', 'Unknown error')}")
```

### Example 2: Using Predefined Problem Types

```python
from campro.optimization.casadi_problem_spec import (
    create_default_problem,
    create_high_efficiency_problem,
    create_smooth_motion_problem
)

# Create different problem types
# duration_angle_deg is required for all problems
default_problem = create_default_problem(
    stroke=0.1,
    duration_angle_deg=360.0,
    cycle_time=0.0385,
    upstroke_percent=50.0
)

high_efficiency_problem = create_high_efficiency_problem(
    stroke=0.1,
    duration_angle_deg=360.0,
    cycle_time=0.0385
)

smooth_motion_problem = create_smooth_motion_problem(
    stroke=0.1,
    duration_angle_deg=360.0,
    cycle_time=0.0385
)

# Solve each problem
optimizer = CasADiMotionOptimizer()

for name, problem in [
    ("Default", default_problem),
    ("High Efficiency", high_efficiency_problem),
    ("Smooth Motion", smooth_motion_problem)
]:
    print(f"\nSolving {name} problem...")
    result = optimizer.solve(problem)
    
    if result.successful:
        print(f"  Success: {result.successful}")
        print(f"  Solve time: {result.solve_time:.3f}s")
        print(f"  Max velocity: {max(abs(v) for v in result.variables['velocity']):.3f} m/s")
    else:
        print(f"  Failed: {result.metadata.get('error', 'Unknown error')}")
```

## Deterministic Seeding Examples

### Example 3: Basic Seed Generation and Polish

```python
from campro.optimization.casadi_motion_optimizer import CasADiMotionOptimizer
from campro.optimization.casadi_problem_spec import create_default_problem
from campro.optimization.initial_guess import InitialGuessBuilder

# Build problem/optimizer
# duration_angle_deg is required
problem = create_default_problem(stroke=0.1, duration_angle_deg=360.0, cycle_time=0.0385)
optimizer = CasADiMotionOptimizer(n_segments=50, poly_order=3)

# Deterministic seed
builder = InitialGuessBuilder(n_segments=optimizer.n_segments)
seed = builder.build_seed(problem)

# Optional polish (clips velocity/acc/jerk to constraints)
polished_seed = builder.polish_seed(problem, seed)

# Solve with polished seed
result = optimizer.solve(problem, polished_seed)

if result.successful:
    print(f"Solve time: {result.solve_time:.3f}s")
    print(f"Objective: {result.objective_value:.6f}")
else:
    print("Optimization failed")
```

### Example 4: Comparing Raw vs Polished Seeds

```python
import time
import numpy as np

from campro.optimization.casadi_motion_optimizer import CasADiMotionOptimizer
from campro.optimization.casadi_problem_spec import create_default_problem
from campro.optimization.initial_guess import InitialGuessBuilder

optimizer = CasADiMotionOptimizer()
builder = InitialGuessBuilder(n_segments=optimizer.n_segments)

stroke_values = np.linspace(0.08, 0.12, 5)
cycle_time_values = np.linspace(0.035, 0.042, 5)

raw_times = []
polished_times = []

for stroke in stroke_values:
    for cycle_time in cycle_time_values:
        # duration_angle_deg is required
        problem = create_default_problem(stroke=stroke, duration_angle_deg=360.0, cycle_time=cycle_time)

        seed = builder.build_seed(problem)

        # Raw seed solve
        start = time.time()
        raw_result = optimizer.solve(problem, seed)
        raw_times.append(time.time() - start)

        # Polished seed solve
        polished_seed = builder.polish_seed(problem, seed)
        start = time.time()
        polished_result = optimizer.solve(problem, polished_seed)
        polished_times.append(time.time() - start)

print(f"Average raw seed time: {np.mean(raw_times):.3f}s")
print(f"Average polished seed time: {np.mean(polished_times):.3f}s")
print(f"Speedup factor: {np.mean(raw_times) / np.mean(polished_times):.2f}x")
```

## Problem Specification

### Example 5: Custom Problem Configuration

```python
from campro.optimization.casadi_problem_spec import CasADiMotionProblem, OptimizationObjective

# Create custom problem with specific objectives
# All constraints must be in per-degree units
problem = CasADiMotionProblem(
    stroke=0.1,
    duration_angle_deg=360.0,  # Required
    cycle_time=0.0385,
    upstroke_percent=50.0,
    max_velocity=0.00015,  # m/deg (~0.15 mm/deg)
    max_acceleration=0.0015,  # m/deg² (~1.5 mm/deg²)
    max_jerk=0.015,  # m/deg³ (~15 mm/deg³)
    compression_ratio_limits=(25.0, 70.0),
    objectives=[
        OptimizationObjective.MAXIMIZE_THERMAL_EFFICIENCY,
        OptimizationObjective.MINIMIZE_JERK
    ],
    weights={
        'thermal_efficiency': 1.0,
        'jerk': 0.1,
        'smoothness': 0.01
    },
    n_segments=100,  # Higher resolution
    poly_order=4,    # Higher order polynomials
    collocation_method="radau",
    solver_options={
        'ipopt.linear_solver': 'ma27',
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-8,
        'ipopt.print_level': 1
    }
)

print("Custom problem configuration:")
print(f"Stroke: {problem.stroke} m")
print(f"Duration angle: {problem.duration_angle_deg} deg (required)")
print(f"Cycle time: {problem.cycle_time} s (derived)")
print(f"Frequency: {problem.get_frequency():.1f} Hz")
print(f"Objectives: {[obj.value for obj in problem.objectives]}")
print(f"Weights: {problem.weights}")
print(f"Segments: {problem.n_segments}")
print(f"Polynomial order: {problem.poly_order}")
print("Note: All motion constraints are in per-degree units (m/deg, m/deg², m/deg³)")
```

### Example 6: Problem Validation and Modification

```python
# Validate problem
if problem.is_feasible():
    print("Problem is feasible")
else:
    print("Problem is not feasible")

# Update weights
problem.update_weights(
    thermal_efficiency=2.0,
    jerk=0.05
)

# Add/remove objectives
problem.add_objective(OptimizationObjective.SMOOTHNESS, weight=0.5)
problem.remove_objective(OptimizationObjective.MINIMIZE_JERK)

print(f"Updated objectives: {[obj.value for obj in problem.objectives]}")
print(f"Updated weights: {problem.weights}")

# Get problem summary
print(problem.get_problem_summary())
```

## Thermal Efficiency Integration

### Example 7: Thermal Efficiency Evaluation

```python
from campro.physics.thermal_efficiency_simple import SimplifiedThermalModel

# Initialize thermal model
thermal_model = SimplifiedThermalModel()

# Evaluate thermal efficiency for a motion profile
if result.successful:
    efficiency_metrics = thermal_model.evaluate_efficiency(
        result.variables['position'],
        result.variables['velocity'],
        result.variables['acceleration']
    )
    
    print("Thermal Efficiency Analysis:")
    print(f"Compression ratio: {efficiency_metrics['compression_ratio']:.2f}")
    print(f"Otto cycle efficiency: {efficiency_metrics['otto_efficiency']:.3f}")
    print(f"Heat loss penalty: {efficiency_metrics['heat_loss_penalty']:.3f}")
    print(f"Mechanical loss: {efficiency_metrics['mechanical_loss']:.3f}")
    print(f"Total efficiency: {efficiency_metrics['total_efficiency']:.3f}")
    print(f"Efficiency target: {efficiency_metrics['efficiency_target']:.3f}")
    print(f"Target achieved: {efficiency_metrics['efficiency_achieved']}")
```

### Example 8: Thermal Efficiency Optimization

```python
# Create high-efficiency problem
efficiency_problem = create_high_efficiency_problem(
    stroke=0.1,
    duration_angle_deg=360.0,
    cycle_time=0.0385
)

# Solve with thermal efficiency focus
result = optimizer.solve(efficiency_problem)

if result.successful:
    # Evaluate thermal efficiency
    efficiency_metrics = thermal_model.evaluate_efficiency(
        result.variables['position'],
        result.variables['velocity'],
        result.variables['acceleration']
    )
    
    print(f"Thermal efficiency: {efficiency_metrics['total_efficiency']:.3f}")
    print(f"Target achieved: {efficiency_metrics['efficiency_achieved']}")
    
    # Compare with default problem
    default_result = optimizer.solve(create_default_problem())
    if default_result.successful:
        default_efficiency = thermal_model.evaluate_efficiency(
            default_result.variables['position'],
            default_result.variables['velocity'],
            default_result.variables['acceleration']
        )
        
        print(f"Default efficiency: {default_efficiency['total_efficiency']:.3f}")
        print(f"Efficiency improvement: {efficiency_metrics['total_efficiency'] - default_efficiency['total_efficiency']:.3f}")
```

## Advanced Configuration

### Example 9: Unified Flow Manager

```python
from campro.optimization.casadi_unified_flow import CasADiUnifiedFlow, CasADiOptimizationSettings

# Create unified flow with custom settings
settings = CasADiOptimizationSettings(
    enable_warmstart=True,  # toggles deterministic seed polishing
    coarse_resolution_segments=(40, 80, 160),  # coarse -> fine bootstrap levels
    target_angle_resolution_deg=0.05,        # target angular step for finest collocation grid
    poly_order=4,
    efficiency_target=0.60,
    solver_options={
        'ipopt.linear_solver': 'ma27',
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-8,
        'ipopt.print_level': 0
    }
)

# Initialize unified flow
unified_flow = CasADiUnifiedFlow(settings)

# Optimize with unified flow
# All constraints must be in per-degree units
constraints = {
    'stroke': 0.1,
    'duration_angle_deg': 360.0,  # Required
    'cycle_time': 0.0385,
    'upstroke_percent': 50.0,
    'max_velocity': 0.00019,  # m/deg (~0.19 mm/deg)
    'max_acceleration': 0.0019,  # m/deg² (~1.9 mm/deg²)
    'max_jerk': 0.019,  # m/deg³ (~19 mm/deg³)
    'compression_ratio_limits': (20.0, 70.0)
}

targets = {
    'minimize_jerk': True,
    'maximize_thermal_efficiency': True,
    'weights': {
        'jerk': 1.0,
        'thermal_efficiency': 0.1,
        'smoothness': 0.01
    }
}

result = unified_flow.optimize_phase1(constraints, targets)

if result.successful:
    print(f"Unified optimization successful!")
    print(f"Solve time: {result.solve_time:.3f}s")
    print(f"Thermal efficiency: {result.metadata.get('total_efficiency', 0.0):.3f}")
else:
    print(f"Unified optimization failed: {result.metadata.get('error', 'Unknown error')}")
```

### Example 10: Benchmarking

```python
# Benchmark across multiple problem specifications
problem_specs = [
    {
        'constraints': {'stroke': 0.08, 'duration_angle_deg': 360.0, 'cycle_time': 0.035, 'upstroke_percent': 45.0},
        'targets': {'minimize_jerk': True, 'maximize_thermal_efficiency': True}
    },
    {
        'constraints': {'stroke': 0.10, 'duration_angle_deg': 360.0, 'cycle_time': 0.0385, 'upstroke_percent': 50.0},
        'targets': {'minimize_jerk': True, 'maximize_thermal_efficiency': True}
    },
    {
        'constraints': {'stroke': 0.12, 'duration_angle_deg': 360.0, 'cycle_time': 0.042, 'upstroke_percent': 55.0},
        'targets': {'minimize_jerk': True, 'maximize_thermal_efficiency': True}
    }
]

# Run benchmark
benchmark_results = unified_flow.benchmark_optimization(problem_specs)

print("Benchmark Results:")
print(f"Total problems: {benchmark_results['total_problems']}")
print(f"Successful: {benchmark_results['successful_problems']}")
print(f"Success rate: {benchmark_results['success_rate']:.1%}")
print(f"Average solve time: {benchmark_results['avg_solve_time']:.3f}s")
print(f"Average efficiency: {benchmark_results['avg_efficiency']:.3f}")

# Inspect deterministic seed configuration
seed_stats = unified_flow.get_warmstart_stats()
print(f"\nInitial guess strategy: {seed_stats['strategy']}")
print(f"Segments: {seed_stats['n_segments']}")
```

## Error Handling

### Example 11: Robust Error Handling

```python
def safe_optimize(problem, optimizer, max_retries=3):
    """Safely optimize with retry logic."""
    for attempt in range(max_retries):
        try:
            result = optimizer.solve(problem)
            if result.successful:
                return result
            else:
                print(f"Attempt {attempt + 1} failed: {result.metadata.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Attempt {attempt + 1} exception: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying optimization (attempt {attempt + 2}/{max_retries})")
            # Could modify problem parameters here for retry
    
    # All attempts failed
    return OptimizationResult(
        status=OptimizationStatus.FAILED,
        successful=False,
        objective_value=float('inf'),
        solve_time=0.0,
        variables={},
        metadata={'error': 'All optimization attempts failed'}
    )

# Use safe optimization
problem = create_default_problem()
result = safe_optimize(problem, optimizer)

if result.successful:
    print("Optimization successful after retries")
else:
    print("All optimization attempts failed")
```

### Example 12: Constraint Validation

```python
def validate_problem(problem):
    """Validate problem constraints and parameters."""
    errors = []
    
    # Check basic constraints
    if problem.stroke <= 0:
        errors.append("Stroke must be positive")
    
    if not hasattr(problem, 'duration_angle_deg') or problem.duration_angle_deg <= 0:
        errors.append("duration_angle_deg is required and must be positive")
    
    if problem.cycle_time <= 0:
        errors.append("Cycle time must be positive")
    
    if not 0 < problem.upstroke_percent < 100:
        errors.append("Upstroke percent must be between 0 and 100")
    
    # Check motion limits (must be in per-degree units)
    if problem.max_velocity is not None and problem.max_velocity <= 0:
        errors.append("Max velocity must be positive (in m/deg)")
    
    if problem.max_acceleration is not None and problem.max_acceleration <= 0:
        errors.append("Max acceleration must be positive (in m/deg²)")
    
    if problem.max_jerk is not None and problem.max_jerk <= 0:
        errors.append("Max jerk must be positive (in m/deg³)")
    
    # Check compression ratio limits
    if problem.compression_ratio_limits[0] >= problem.compression_ratio_limits[1]:
        errors.append("Compression ratio limits must be ordered")
    
    # Check feasibility
    if not problem.is_feasible():
        errors.append("Problem is not feasible")
    
    return errors

# Validate problem before optimization
errors = validate_problem(problem)
if errors:
    print("Problem validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Problem validation passed")
    result = optimizer.solve(problem)
```

## Performance Optimization

### Example 13: Solver Configuration

```python
# Configure solver for performance
optimizer = CasADiMotionOptimizer(
    n_segments=50,
    poly_order=3,
    collocation_method="legendre"
)

# Set solver options for performance
optimizer.opti.solver('ipopt', {
    'ipopt.linear_solver': 'ma27',
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6,
    'ipopt.print_level': 0,  # Suppress output for performance
    'ipopt.warm_start_init_point': 'yes',  # Let IPOPT reuse the provided seed
    'ipopt.mu_strategy': 'adaptive',  # Adaptive barrier parameter
    'ipopt.hessian_approximation': 'limited-memory'  # Use L-BFGS
})

# Solve with performance configuration
result = optimizer.solve(problem)
```

### Example 14: Memory Management

```python
# Keep seed builder aligned with solver grid to avoid stale arrays
builder = InitialGuessBuilder(n_segments=optimizer.n_segments)

# If you change the discretization at runtime:
new_segments = 80
optimizer = CasADiMotionOptimizer(n_segments=new_segments)
builder.update_segments(new_segments)

# Monitor memory usage
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.1f} MB")
```

### Example 15: Parallel Optimization

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def optimize_parallel(problem_specs, max_workers=None):
    """Optimize multiple problems in parallel."""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    def solve_problem(spec):
        problem = create_default_problem(**spec['constraints'])
        optimizer = CasADiMotionOptimizer()
        return optimizer.solve(problem)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(solve_problem, spec) for spec in problem_specs]
        results = [future.result() for future in futures]
    
    return results

# Example parallel optimization
problem_specs = [
    {'constraints': {'stroke': 0.08, 'duration_angle_deg': 360.0, 'cycle_time': 0.035}},
    {'constraints': {'stroke': 0.10, 'duration_angle_deg': 360.0, 'cycle_time': 0.0385}},
    {'constraints': {'stroke': 0.12, 'duration_angle_deg': 360.0, 'cycle_time': 0.042}}
]

results = optimize_parallel(problem_specs)
successful_count = sum(1 for r in results if r.successful)
print(f"Parallel optimization: {successful_count}/{len(results)} successful")
```

## Summary

These examples demonstrate the key features of the CasADi optimization framework:

1. **Basic optimization** with CasADi Opti stack
2. **Warm-starting** for improved performance
3. **Problem specification** with validation
4. **Thermal efficiency** integration
5. **Advanced configuration** options
6. **Error handling** and robustness
7. **Performance optimization** techniques

## Important: Per-Degree Units Contract

**All motion-law inputs must be in per-degree units:**
- `duration_angle_deg` is **required** (no fallback)
- Motion constraints: `max_velocity` (m/deg), `max_acceleration` (m/deg²), `max_jerk` (m/deg³)
- `cycle_time` is derived from `engine_speed_rpm` and `duration_angle_deg`, not a primary input
- No per-second units are accepted; no compatibility mode or auto-conversion

The framework provides a comprehensive solution for Phase 1 motion law optimization with thermal efficiency objectives, following CasADi best practices and documentation patterns.

