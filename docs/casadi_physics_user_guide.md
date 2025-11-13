# CasADi Physics User Guide

## Overview

This guide provides comprehensive information about using CasADi physics in the Cam-Ring System Designer. CasADi physics offers significant performance improvements over traditional Python implementations while maintaining numerical accuracy through automatic differentiation.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Validation Mode](#validation-mode)
4. [Performance Expectations](#performance-expectations)
5. [Configuration Options](#configuration-options)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [API Reference](#api-reference)

## Introduction

### What is CasADi Physics?

CasADi physics is a symbolic computation framework that provides:

- **Automatic Differentiation**: Exact gradients without finite differences
- **Performance**: 3-4x speedup in evaluation and gradient computation
- **Numerical Stability**: Symbolic computation reduces numerical errors
- **Optimization Integration**: Direct integration with Ipopt solver

### Benefits

- **Faster Optimization**: Reduced solve times for complex problems
- **Better Convergence**: Exact gradients improve optimization convergence
- **Higher Accuracy**: Symbolic computation maintains precision
- **Scalability**: Efficient handling of large-scale problems

## Getting Started

### Prerequisites

- Python 3.11+
- CasADi 3.6.0+
- Ipopt solver
- All standard project dependencies

### Enabling CasADi Physics

#### Method 1: GUI Control (Recommended)

1. Launch the Cam-Ring System Designer GUI
2. Navigate to the "Crank Center Optimization" section
3. Check "Enable Validation Mode" to run both Python and CasADi physics
4. Set tolerance (default: 1e-4)
5. Run optimization to collect validation statistics

#### Method 2: Constants Configuration

Edit `campro/constants.py`:

```python
# Enable CasADi physics
USE_CASADI_PHYSICS = True

# Enable validation mode for confidence building
CASADI_PHYSICS_VALIDATION_MODE = True
CASADI_PHYSICS_VALIDATION_TOLERANCE = 1e-4
```

#### Method 3: Programmatic Control

```python
from campro.optimization.unified_framework import UnifiedOptimizationFramework

# Create framework
framework = UnifiedOptimizationFramework()

# Enable validation mode
framework.enable_casadi_validation_mode(tolerance=1e-4)

# Or disable validation mode
framework.disable_casadi_validation_mode()
```

## Validation Mode

### Overview

Validation mode runs both Python and CasADi physics in parallel, providing:

- **Zero Risk**: Python results used for optimization
- **Confidence Building**: Accumulate validation statistics
- **Issue Detection**: Identify edge cases where implementations diverge
- **Performance Monitoring**: Track speedup and convergence improvements

### How It Works

1. **Parallel Execution**: Both Python and CasADi physics are evaluated
2. **Comparison**: Results are compared within specified tolerance
3. **Logging**: Validation metrics are logged for analysis
4. **Statistics**: Comprehensive statistics are collected over time

### Validation Statistics

The system collects detailed statistics including:

- **Parity Metrics**: Differences between Python and CasADi results
- **Performance Metrics**: Evaluation and gradient computation times
- **Convergence Metrics**: Iteration counts and success rates
- **Problem Analysis**: Breakdown by problem type and parameters

### Statistics Collection

Statistics are automatically saved to `logs/validation_stats/`:

- Individual validation metrics (JSON)
- Aggregated statistics (JSON)
- Human-readable reports (Markdown)

### Decision Criteria

CasADi physics is recommended for production when:

- **95%+ Success Rate**: 95% of validations within tolerance
- **No Systematic Bias**: Differences are random, not systematic
- **Performance Gains**: 3x+ speedup in evaluation/gradient computation
- **Stable Convergence**: No regression in optimization convergence

## Performance Expectations

### Typical Performance Improvements

| Metric | Python Baseline | CasADi | Speedup |
|--------|----------------|--------|---------|
| Evaluation Time | 2-5 ms | 0.5-1.5 ms | 3-4x |
| Gradient Time | 4-8 ms | 1-2.5 ms | 3-4x |
| Memory Usage | Baseline | +10-20% | - |
| Convergence | Baseline | Same/Better | - |

### Performance Thresholds

The system enforces performance thresholds:

```python
# Maximum acceptable evaluation time
CASADI_PHYSICS_MAX_EVALUATION_TIME_MS = 2.0  # 2ms with 2x safety margin

# Maximum acceptable gradient time  
CASADI_PHYSICS_MAX_GRADIENT_TIME_MS = 3.0    # 3ms with 2x safety margin
```

### Hardware Considerations

Performance improvements vary by hardware:

- **Apple M2/M3**: 3-4x speedup typical
- **Intel x86**: 2-3x speedup typical
- **Memory**: 16GB+ recommended for large problems
- **Storage**: SSD recommended for log files

## Configuration Options

### Core Settings

```python
# campro/constants.py

# Main toggle
USE_CASADI_PHYSICS = True

# Validation mode
CASADI_PHYSICS_VALIDATION_MODE = True
CASADI_PHYSICS_VALIDATION_TOLERANCE = 1e-4

# Performance thresholds
CASADI_PHYSICS_MAX_EVALUATION_TIME_MS = 2.0
CASADI_PHYSICS_MAX_GRADIENT_TIME_MS = 3.0

# Numerical safety
CASADI_PHYSICS_EPSILON = 1e-12
CASADI_PHYSICS_ASIN_CLAMP = 0.999999
```

### Feature Flags

```python
# Enable effective radius correction
CASADI_PHYSICS_USE_EFFECTIVE_RADIUS_CORRECTION = True

# Enable chunking for variable-length inputs
CASADI_PHYSICS_ENABLE_CHUNKING = True
```

### Solver Settings

```python
# Ipopt configuration for CasADi
ipopt_options = {
    "linear_solver": "ma27",
    "max_iter": 1000,
    "tol": 1e-6,
    "print_level": 0
}
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: No module named 'casadi'`

**Solution**:
```bash
pip install casadi
# or
conda install -c conda-forge casadi
```

#### 2. Performance Below Thresholds

**Problem**: Evaluation/gradient times exceed thresholds

**Solutions**:
- Check hardware specifications
- Reduce problem size
- Adjust performance thresholds
- Enable JIT compilation

#### 3. Validation Failures

**Problem**: High percentage of validation failures

**Solutions**:
- Check tolerance settings
- Verify input data quality
- Review numerical precision
- Check for edge cases

#### 4. Convergence Issues

**Problem**: Optimization fails to converge

**Solutions**:
- Check initial guesses
- Verify constraint bounds
- Review problem formulation
- Try different solver settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('campro.physics.casadi').setLevel(logging.DEBUG)
```

### Performance Profiling

Profile CasADi functions:

```python
from campro.physics.casadi import create_unified_physics
import time

# Create function
fn = create_unified_physics()

# Profile evaluation
start = time.perf_counter()
result = fn(*args)
eval_time = (time.perf_counter() - start) * 1000  # ms

print(f"Evaluation time: {eval_time:.2f} ms")
```

### Validation Debugging

Check validation statistics:

```python
from campro.optimization.validation_statistics import ValidationStatisticsCollector

# Load existing statistics
collector = ValidationStatisticsCollector()
statistics = collector.compute_statistics()

# Check tolerance success rate
print(f"Success rate: {statistics.tolerance_success_rate:.1%}")
```

## Advanced Usage

### Custom Physics Functions

Create custom CasADi physics functions:

```python
import casadi as ca
from campro.physics.casadi import create_torque_pointwise

# Create custom function
def create_custom_physics():
    # Define symbolic variables
    theta = ca.MX.sym("theta")
    F = ca.MX.sym("F")
    r = ca.MX.sym("r")
    
    # Custom physics calculation
    torque = F * r * ca.sin(theta)
    
    # Create function
    return ca.Function("custom_physics", [theta, F, r], [torque])
```

### Batch Processing

Process multiple problems efficiently:

```python
from campro.optimization.unified_framework import UnifiedOptimizationFramework

# Create framework
framework = UnifiedOptimizationFramework()

# Enable validation mode
framework.enable_casadi_validation_mode()

# Process multiple problems
for problem_data in problem_list:
    result = framework.optimize_cascaded(problem_data)
    # Results automatically collected for statistics
```

### Custom Validation Metrics

Extend validation statistics:

```python
from campro.optimization.validation_statistics import ValidationMetrics

# Create custom metrics
metrics = ValidationMetrics(
    problem_id="custom_001",
    timestamp=datetime.now().isoformat(),
    # ... standard fields ...
    custom_field="custom_value"  # Add custom fields
)

# Add to collector
collector.add_validation_metrics(metrics)
```

## API Reference

### Core Functions

#### `create_unified_physics()`

Creates the main CasADi physics function.

**Returns**: `casadi.Function` - Unified physics function

**Usage**:
```python
from campro.physics.casadi import create_unified_physics

fn = create_unified_physics()
result = fn(theta_vec, pressure_vec, crank_radius, rod_length, 
           crank_center_x, crank_center_y, x_off, y_off, litvin_config)
```

#### `torque_profile_chunked_wrapper()`

Handles variable-length inputs for torque profiles.

**Parameters**:
- `theta_vec`: Crank angle vector
- `F_vec`: Force vector  
- `r`: Crank radius
- `l`: Rod length
- `x_off`: X offset
- `y_off`: Y offset
- `pressure_angle`: Pressure angle

**Returns**: Tuple of (T_vec, T_avg, T_max, T_min, ripple)

### Validation Framework

#### `ValidationStatisticsCollector`

Main class for collecting validation statistics.

**Methods**:
- `add_validation_metrics(metrics)`: Add validation metrics
- `compute_statistics()`: Compute aggregated statistics
- `generate_report(statistics)`: Generate human-readable report
- `save_statistics(statistics)`: Save statistics to file

#### `UnifiedOptimizationFramework`

Extended framework with validation mode support.

**Methods**:
- `enable_casadi_validation_mode(tolerance)`: Enable validation mode
- `disable_casadi_validation_mode()`: Disable validation mode

### Constants

#### Performance Thresholds

```python
CASADI_PHYSICS_MAX_EVALUATION_TIME_MS = 2.0
CASADI_PHYSICS_MAX_GRADIENT_TIME_MS = 3.0
```

#### Validation Settings

```python
CASADI_PHYSICS_VALIDATION_MODE = True
CASADI_PHYSICS_VALIDATION_TOLERANCE = 1e-4
```

#### Numerical Safety

```python
CASADI_PHYSICS_EPSILON = 1e-12
CASADI_PHYSICS_ASIN_CLAMP = 0.999999
```

## Best Practices

### 1. Gradual Adoption

- Start with validation mode enabled
- Collect statistics for 2-4 weeks
- Enable production mode only after validation success

### 2. Performance Monitoring

- Monitor evaluation and gradient times
- Track convergence rates
- Watch for performance regressions

### 3. Validation Maintenance

- Run validation mode periodically
- Review validation statistics
- Investigate any tolerance failures

### 4. Problem-Specific Tuning

- Adjust tolerances for specific problem types
- Use appropriate solver settings
- Consider problem size and complexity

### 5. Documentation

- Document any custom configurations
- Keep validation statistics for reference
- Update performance baselines as needed

## Support

### Getting Help

1. **Check Logs**: Review validation statistics and error logs
2. **Run Tests**: Execute test suite to verify functionality
3. **Review Documentation**: Check this guide and API reference
4. **Contact Team**: Reach out to development team for issues

### Reporting Issues

When reporting issues, include:

- Problem description and steps to reproduce
- Relevant log files and error messages
- System information (OS, Python version, hardware)
- Validation statistics if applicable

### Contributing

To contribute to CasADi physics:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new functionality
3. Update documentation as needed
4. Ensure validation mode compatibility

---

*This guide is maintained as part of the CasADi Physics Integration project. For updates and additional information, see the project documentation.*
