# Motion Law Optimization Library

A comprehensive, reusable library for OP engine motion law optimization. This library encapsulates the complex optimization pipeline into a clean, configurable API that can be used as part of larger optimization routines.

## Features

- **Clean Architecture**: Modular design with clear separation of concerns
- **Multiple Solvers**: IPOPT, robust IPOPT, and adaptive solver backends
- **Flexible Configuration**: Presets, scenarios, and custom configurations
- **Comprehensive Results**: Built-in validation, metrics, and post-processing
- **Extensible**: Easy to add custom objectives, constraints, and solvers
- **Well Tested**: Extensive test coverage and examples
- **Backward Compatible**: Maintains compatibility with existing code

## Quick Start

### Basic Usage

```python
from campro.optimization import MotionLawOptimizer, ConfigFactory

# Create default configuration
config = ConfigFactory.create_default_config()

# Create optimizer
optimizer = MotionLawOptimizer(config)

# Run optimization
result = optimizer.optimize()

# Check results
if result.success:
    print(f"Objective value: {result.objective_value:.6e}")
    print(f"Iterations: {result.iterations}")
    print(f"CPU time: {result.cpu_time:.2f}s")
else:
    print(f"Optimization failed: {result.errors}")
```

### Quick Optimization

```python
from campro.optimization import quick_optimize, get_preset_config

# Use preset configuration
config = get_preset_config("high_performance")

# Quick optimization with robust solver
result = quick_optimize(config, backend="robust")
```

### Custom Configuration

```python
from campro.optimization import create_engine_config, create_optimization_scenario

# Create engine-specific configuration
config = create_engine_config("opposed_piston")

# Apply optimization scenario
config = create_optimization_scenario("efficiency", **config.__dict__)

# Run optimization
result = quick_optimize(config)
```

## Architecture

### Core Components

1. **MotionLawOptimizer**: Main optimization class
2. **OptimizationConfig**: Configuration management
3. **ConfigFactory**: Factory for creating configurations
4. **Solver Backends**: IPOPT, robust, and adaptive solvers
5. **Result Processing**: Validation and metrics computation

### Class Hierarchy

```
MotionLawOptimizer
├── ProblemBuilder
├── SolverBackend (Protocol)
│   ├── IPOPTBackend
│   ├── RobustIPOPTBackend
│   └── AdaptiveBackend
└── ResultProcessor
    ├── SolutionValidator
    └── PhysicsValidator
```

## Configuration

### Preset Configurations

```python
from campro.optimization import get_preset_config

# Available presets
configs = {
    "default": get_preset_config("default"),
    "high_performance": get_preset_config("high_performance"),
    "quick_test": get_preset_config("quick_test"),
    "1d": get_preset_config("1d"),
    "robust": get_preset_config("robust"),
}
```

### Engine-Specific Configurations

```python
from campro.optimization import create_engine_config

# Available engine types
engines = {
    "opposed_piston": create_engine_config("opposed_piston"),
    "free_piston": create_engine_config("free_piston"),
    "conventional": create_engine_config("conventional"),
}
```

### Optimization Scenarios

```python
from campro.optimization import create_optimization_scenario

# Available scenarios
scenarios = {
    "efficiency": create_optimization_scenario("efficiency"),
    "power": create_optimization_scenario("power"),
    "emissions": create_optimization_scenario("emissions"),
    "durability": create_optimization_scenario("durability"),
}
```

### Custom Configuration

```python
from campro.optimization import ConfigFactory

# Create custom configuration
config = ConfigFactory.create_custom_config(
    geometry={
        "bore": 0.12,
        "stroke": 0.08,
        "compression_ratio": 15.0,
    },
    bounds={
        "v_max": 40.0,
        "a_max": 800.0,
    },
    num={"K": 30, "C": 4},
    objective={
        "method": "thermal_efficiency",
        "w": {
            "smooth": 0.02,
            "short_circuit": 1.5,
            "eta_th": 0.8,
        }
    }
)
```

## Solver Backends

### Standard IPOPT

```python
### Standard IPOPT

```python
from campro.optimization import create_standard_optimizer

optimizer = create_standard_optimizer(config)
result = optimizer.optimize()
```

### Robust IPOPT

```python
from campro.optimization import create_robust_optimizer

optimizer = create_robust_optimizer(config)
result = optimizer.optimize()
```

### Adaptive Solver

```python
from campro.optimization import create_adaptive_optimizer

optimizer = create_adaptive_optimizer(config, max_refinements=3)
result = optimizer.optimize()
```

## Problem Builder

The problem builder provides a fluent interface for configuring optimization problems:

```python
from campro.optimization import MotionLawOptimizer, ConfigFactory

config = ConfigFactory.create_default_config()
optimizer = MotionLawOptimizer(config)

# Use fluent interface
builder = optimizer.get_problem_builder()
builder.with_geometry({
    "bore": 0.15,
    "stroke": 0.1,
    "compression_ratio": 12.0,
}).with_bounds({
    "v_max": 30.0,
    "a_max": 800.0,
}).with_objective({
    "method": "indicated_work",
    "w": {
        "smooth": 0.01,
        "short_circuit": 1.0,
    }
}).with_1d_model(n_cells=40)

# Run optimization with modified configuration
result = optimizer.optimize()
```

## Results and Validation

### Basic Results

```python
result = optimizer.optimize()

print(f"Success: {result.success}")
print(f"Objective value: {result.objective_value:.6e}")
print(f"Iterations: {result.iterations}")
print(f"CPU time: {result.cpu_time:.2f}s")
print(f"KKT error: {result.kkt_error:.2e}")
print(f"Feasibility error: {result.feasibility_error:.2e}")
```

### Validation and Metrics

```python
# Run optimization with validation
result = optimizer.optimize_with_validation(validate=True)

# Access validation results
print(f"Validation metrics: {result.validation_metrics}")
print(f"Physics validation: {result.physics_validation}")
print(f"Performance metrics: {result.performance_metrics}")

# Check warnings and errors
if result.warnings:
    print(f"Warnings: {result.warnings}")
if result.errors:
    print(f"Errors: {result.errors}")
```

### Performance Metrics

The library automatically computes performance metrics:

```python
metrics = result.performance_metrics

print(f"Max pressure: {metrics['max_pressure']:.0f} Pa")
print(f"Max temperature: {metrics['max_temperature']:.0f} K")
print(f"Min piston gap: {metrics['min_piston_gap']:.6f} m")
```

## Advanced Usage

### Custom Solver Backend

```python
from campro.optimization import MotionLawOptimizer, SolverBackend

class CustomBackend(SolverBackend):
    def solve(self, problem):
        # Custom solver implementation
        return OptimizationResult(success=True, objective_value=1.5e6)

config = ConfigFactory.create_default_config()
optimizer = MotionLawOptimizer(config, CustomBackend())
result = optimizer.optimize()
```

### Configuration from File

```python
from campro.optimization import ConfigFactory

# Save configuration
config = ConfigFactory.create_default_config()
ConfigFactory.to_yaml(config, "my_config.yaml")

# Load configuration
loaded_config = ConfigFactory.from_yaml("my_config.yaml")
```

### Comparison Studies

```python
from campro.optimization import get_preset_config, quick_optimize

# Compare different configurations
configs = {
    "default": get_preset_config("default"),
    "high_performance": get_preset_config("high_performance"),
    "quick_test": get_preset_config("quick_test"),
}

results = {}
for name, config in configs.items():
    results[name] = quick_optimize(config, backend="standard")

# Print comparison
for name, result in results.items():
    print(f"{name:15} | Success: {result.success:5} | "
          f"Objective: {result.objective_value:12.6e} | "
          f"Time: {result.cpu_time:6.2f}s")
```

## Examples

The library includes comprehensive examples in `examples.py`:

```python
from campro.optimization.examples import run_all_examples

# Run all examples
results = run_all_examples()
```

Available examples:

1. Basic optimization
2. Custom configuration
3. 1D gas model optimization
4. Robust optimization
5. Adaptive optimization
6. Engine-specific optimization
7. Scenario-based optimization
8. Quick optimization
9. Problem builder usage
10. Validation and metrics
11. Comparison studies
12. Configuration save/load

## Testing

Run the test suite:

```bash
pytest tests/unit/test_optimization_lib.py -v
```

The test suite covers:

- Configuration creation and validation
- Solver backend functionality
- Problem building and optimization
- Result processing and validation
- Convenience functions
- Error handling and edge cases

## Backward Compatibility

The library maintains backward compatibility with existing code:

```python
# Legacy usage still works
from campro.optimization import solve_cycle, build_collocation_nlp

# New library usage
from campro.optimization import MotionLawOptimizer, ConfigFactory
```

## Performance Tips

1. **Use appropriate presets**: Choose the right preset for your use case
2. **Start with quick_test**: Use for initial testing and debugging
3. **Use robust solver**: For difficult convergence problems
4. **Enable validation**: For production runs and critical applications
5. **Save configurations**: Reuse successful configurations
6. **Monitor metrics**: Use performance metrics to assess solution quality

## Troubleshooting

### Common Issues

1. **Convergence failures**: Try robust solver or adaptive refinement
2. **High CPU time**: Use lower resolution (smaller K, C) for initial testing
3. **Invalid configurations**: Use presets or validate configuration parameters
4. **Memory issues**: Reduce problem size or use 0D model instead of 1D

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run optimization with debug logging
result = optimizer.optimize()
```

## Contributing

The library is designed to be extensible. Key extension points:

1. **Custom solver backends**: Implement the `SolverBackend` protocol
2. **Custom configurations**: Add new presets and scenarios
3. **Custom objectives**: Extend the objective function system
4. **Custom constraints**: Add new constraint types
5. **Custom validation**: Extend the validation system

## License

This library is part of the OP Engine Optimization project and follows the same licensing terms.
