# CasADi API Examples

This document provides practical code snippets and examples for using the CasADi-based optimization framework, following CasADi documentation patterns.

## Table of Contents

1. [Basic Phase 1 Optimization](#basic-phase-1-optimization)
2. [Warm-Starting Examples](#warm-starting-examples)
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
problem = CasADiMotionProblem(
    stroke=0.100,  # 100mm stroke
    cycle_time=0.0385,  # 26 Hz
    upstroke_percent=50.0,
    max_velocity=5.0,
    max_acceleration=500.0,
    max_jerk=50000.0,
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
default_problem = create_default_problem(
    stroke=0.1,
    cycle_time=0.0385,
    upstroke_percent=50.0
)

high_efficiency_problem = create_high_efficiency_problem(
    stroke=0.1,
    cycle_time=0.0385
)

smooth_motion_problem = create_smooth_motion_problem(
    stroke=0.1,
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

## Warm-Starting Examples

### Example 3: Basic Warm-Starting

```python
from campro.optimization.warmstart_manager import WarmStartManager
from campro.optimization.casadi_motion_optimizer import CasADiMotionOptimizer

# Initialize warm-start manager
warmstart_mgr = WarmStartManager(
    max_history=50,
    tolerance=0.1,
    storage_path="warmstart_history.json"
)

# Initialize optimizer
optimizer = CasADiMotionOptimizer()

# First optimization (cold start)
problem1 = create_default_problem(stroke=0.1, cycle_time=0.0385)
result1 = optimizer.solve(problem1)

if result1.successful:
    # Store solution for warm-starting
    warmstart_mgr.store_solution(
        problem1.to_dict(),
        {
            'position': result1.variables['position'],
            'velocity': result1.variables['velocity'],
            'acceleration': result1.variables['acceleration'],
            'jerk': result1.variables['jerk']
        },
        {
            'solve_time': result1.solve_time,
            'objective_value': result1.objective_value,
            'n_segments': 50,
            'timestamp': time.time()
        }
    )
    print("Solution stored for warm-starting")

# Second optimization (warm start)
problem2 = create_default_problem(stroke=0.105, cycle_time=0.040)  # Similar parameters
initial_guess = warmstart_mgr.get_initial_guess(problem2.to_dict())

if initial_guess:
    print("Using warm-start initial guess")
    result2 = optimizer.solve(problem2, initial_guess)
    
    print(f"Cold start time: {result1.solve_time:.3f}s")
    print(f"Warm start time: {result2.solve_time:.3f}s")
    print(f"Speedup: {result1.solve_time / result2.solve_time:.2f}x")
else:
    print("No suitable warm-start found, using default initial guess")
    result2 = optimizer.solve(problem2)
```

### Example 4: Warm-Start Strategy Comparison

```python
import time
import numpy as np

# Test warm-starting across parameter ranges
stroke_values = np.linspace(0.08, 0.12, 5)
cycle_time_values = np.linspace(0.035, 0.042, 5)

cold_start_times = []
warm_start_times = []

for stroke in stroke_values:
    for cycle_time in cycle_time_values:
        problem = create_default_problem(stroke=stroke, cycle_time=cycle_time)
        
        # Cold start
        start_time = time.time()
        cold_result = optimizer.solve(problem)
        cold_time = time.time() - start_time
        cold_start_times.append(cold_time)
        
        if cold_result.successful:
            # Store for warm-starting
            warmstart_mgr.store_solution(
                problem.to_dict(),
                {
                    'position': cold_result.variables['position'],
                    'velocity': cold_result.variables['velocity'],
                    'acceleration': cold_result.variables['acceleration'],
                    'jerk': cold_result.variables['jerk']
                },
                {
                    'solve_time': cold_result.solve_time,
                    'objective_value': cold_result.objective_value,
                    'n_segments': 50,
                    'timestamp': time.time()
                }
            )
            
            # Warm start
            initial_guess = warmstart_mgr.get_initial_guess(problem.to_dict())
            start_time = time.time()
            warm_result = optimizer.solve(problem, initial_guess)
            warm_time = time.time() - start_time
            warm_start_times.append(warm_time)
        else:
            warm_start_times.append(cold_time)

# Analyze results
avg_cold_time = np.mean(cold_start_times)
avg_warm_time = np.mean(warm_start_times)
speedup = avg_cold_time / avg_warm_time

print(f"Average cold start time: {avg_cold_time:.3f}s")
print(f"Average warm start time: {avg_warm_time:.3f}s")
print(f"Average speedup: {speedup:.2f}x")
```

## Problem Specification

### Example 5: Custom Problem Configuration

```python
from campro.optimization.casadi_problem_spec import CasADiMotionProblem, OptimizationObjective

# Create custom problem with specific objectives
problem = CasADiMotionProblem(
    stroke=0.1,
    cycle_time=0.0385,
    upstroke_percent=50.0,
    max_velocity=4.0,
    max_acceleration=400.0,
    max_jerk=40000.0,
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
        'ipopt.linear_solver': 'ma57',
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-8,
        'ipopt.print_level': 1
    }
)

print("Custom problem configuration:")
print(f"Stroke: {problem.stroke} m")
print(f"Cycle time: {problem.cycle_time} s")
print(f"Frequency: {problem.get_frequency():.1f} Hz")
print(f"Objectives: {[obj.value for obj in problem.objectives]}")
print(f"Weights: {problem.weights}")
print(f"Segments: {problem.n_segments}")
print(f"Polynomial order: {problem.poly_order}")
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
    enable_warmstart=True,
    max_history=100,
    tolerance=0.05,
    storage_path="optimization_history.json",
    n_segments=100,
    poly_order=4,
    efficiency_target=0.60,
    solver_options={
        'ipopt.linear_solver': 'ma57',
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-8,
        'ipopt.print_level': 0
    }
)

# Initialize unified flow
unified_flow = CasADiUnifiedFlow(settings)

# Optimize with unified flow
constraints = {
    'stroke': 0.1,
    'cycle_time': 0.0385,
    'upstroke_percent': 50.0,
    'max_velocity': 5.0,
    'max_acceleration': 500.0,
    'max_jerk': 50000.0,
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
        'constraints': {'stroke': 0.08, 'cycle_time': 0.035, 'upstroke_percent': 45.0},
        'targets': {'minimize_jerk': True, 'maximize_thermal_efficiency': True}
    },
    {
        'constraints': {'stroke': 0.10, 'cycle_time': 0.0385, 'upstroke_percent': 50.0},
        'targets': {'minimize_jerk': True, 'maximize_thermal_efficiency': True}
    },
    {
        'constraints': {'stroke': 0.12, 'cycle_time': 0.042, 'upstroke_percent': 55.0},
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

# Get warm-start statistics
warmstart_stats = unified_flow.get_warmstart_stats()
print(f"\nWarm-start statistics:")
print(f"History size: {warmstart_stats['count']}")
print(f"Average solve time: {warmstart_stats['avg_solve_time']:.3f}s")
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
    
    if problem.cycle_time <= 0:
        errors.append("Cycle time must be positive")
    
    if not 0 < problem.upstroke_percent < 100:
        errors.append("Upstroke percent must be between 0 and 100")
    
    # Check motion limits
    if problem.max_velocity <= 0:
        errors.append("Max velocity must be positive")
    
    if problem.max_acceleration <= 0:
        errors.append("Max acceleration must be positive")
    
    if problem.max_jerk <= 0:
        errors.append("Max jerk must be positive")
    
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
    'ipopt.linear_solver': 'ma57',  # Use MA57 for better performance
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6,
    'ipopt.print_level': 0,  # Suppress output for performance
    'ipopt.warm_start_init_point': 'yes',  # Enable warm-starting
    'ipopt.mu_strategy': 'adaptive',  # Adaptive barrier parameter
    'ipopt.hessian_approximation': 'limited-memory'  # Use L-BFGS
})

# Solve with performance configuration
result = optimizer.solve(problem)
```

### Example 14: Memory Management

```python
# Clear warm-start history periodically
warmstart_mgr = WarmStartManager(max_history=50)

# Check history size
stats = warmstart_mgr.get_history_stats()
print(f"History size: {stats['count']}")

# Clear history if it gets too large
if stats['count'] > 100:
    warmstart_mgr.clear_history()
    print("Cleared warm-start history")

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
    {'constraints': {'stroke': 0.08, 'cycle_time': 0.035}},
    {'constraints': {'stroke': 0.10, 'cycle_time': 0.0385}},
    {'constraints': {'stroke': 0.12, 'cycle_time': 0.042}}
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

The framework provides a comprehensive solution for Phase 1 motion law optimization with thermal efficiency objectives, following CasADi best practices and documentation patterns.








