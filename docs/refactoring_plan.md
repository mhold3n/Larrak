# Library-Based Refactoring Plan

## Overview

This document outlines the comprehensive refactoring plan to transform the Larrak project from a monolithic structure to a modular, library-based architecture. This refactoring prepares the codebase for future combustion physics simulation while maintaining all current functionality.

## Goals

1. **Modularity**: Separate concerns into distinct libraries
2. **Extensibility**: Easy addition of new physics domains (combustion, thermodynamics)
3. **Maintainability**: Clean interfaces and consistent patterns
4. **Reusability**: Components can be used independently
5. **Testing**: Isolated testing of individual components

## New Library Structure

```
campro/
├── __init__.py
├── constraints/          # Constraint definitions and validation
│   ├── __init__.py
│   ├── base.py          # Base constraint classes
│   ├── motion.py        # Motion-specific constraints
│   ├── cam.py           # Cam-specific constraints
│   └── physics.py       # Future: combustion physics constraints
├── optimization/         # Optimization routines and solvers
│   ├── __init__.py
│   ├── base.py          # Base solver interface
│   ├── collocation.py   # Collocation methods
│   ├── motion.py        # Motion law optimization
│   └── physics.py       # Future: combustion optimization
├── physics/              # Physics simulation modules
│   ├── __init__.py
│   ├── base.py          # Base physics interface
│   ├── combustion.py    # Future: combustion simulation
│   └── thermodynamics.py # Future: thermodynamic models
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── plotting.py      # Plotting utilities
│   ├── validation.py    # Input validation
│   └── conversion.py    # Unit conversions
├── gui/                  # GUI components
│   ├── __init__.py
│   ├── main.py          # Main GUI class
│   ├── widgets.py       # Custom widgets
│   └── dialogs.py       # Dialog boxes
├── constants.py          # Project constants
└── logging.py           # Logging configuration
```

## Implementation Phases

### Phase 1: Core Libraries (Current)
- [x] Create constraint library structure
- [x] Implement base constraint classes
- [x] Implement motion constraints
- [x] Implement cam constraints
- [x] Create optimization library structure
- [x] Implement base optimizer classes
- [x] Implement collocation optimizer
- [x] Implement motion optimizer
- [x] Create physics library foundation
- [x] Create utils library with plotting functions

### Phase 2: Refactor Existing Code
- [ ] Update `CamPro_OptimalMotion.py` to use new libraries
- [ ] Refactor GUI to use new constraint and optimization libraries
- [ ] Update all imports and dependencies
- [ ] Ensure backward compatibility

### Phase 3: Testing and Validation
- [ ] Update test suite for new library structure
- [ ] Add integration tests between libraries
- [ ] Validate all existing functionality works
- [ ] Performance testing

### Phase 4: Documentation and Cleanup
- [ ] Update all documentation
- [ ] Create library-specific documentation
- [ ] Remove deprecated code
- [ ] Final code review

## Library Details

### Constraints Library (`campro.constraints`)

**Purpose**: Define and validate constraints for optimization problems.

**Key Classes**:
- `BaseConstraints`: Abstract base class for all constraint systems
- `MotionConstraints`: General motion law constraints
- `CamMotionConstraints`: Cam-specific constraints
- `ConstraintViolation`: Represents constraint violations

**Features**:
- Consistent validation interface
- Violation tracking and reporting
- Conversion between constraint types
- Dictionary serialization

### Optimization Library (`campro.optimization`)

**Purpose**: Provide optimization routines and solvers.

**Key Classes**:
- `BaseOptimizer`: Abstract base class for all optimizers
- `CollocationOptimizer`: Direct collocation methods
- `MotionOptimizer`: High-level motion law optimization
- `OptimizationResult`: Standardized result format

**Features**:
- Pluggable optimization methods
- Performance tracking
- Result standardization
- History management

### Physics Library (`campro.physics`)

**Purpose**: Physics simulation for future combustion analysis.

**Key Classes**:
- `BasePhysicsModel`: Abstract base for physics models
- `PhysicsResult`: Standardized physics simulation results

**Future Extensions**:
- `CombustionModel`: Combustion simulation
- `ThermodynamicsModel`: Thermodynamic calculations
- `ValveTimingModel`: Valve timing optimization

### Utils Library (`campro.utils`)

**Purpose**: Common utility functions and helpers.

**Key Modules**:
- `plotting.py`: Advanced plotting with smart scaling
- `validation.py`: Input validation helpers
- `conversion.py`: Unit conversions and transformations

## Migration Strategy

### Backward Compatibility

The refactoring maintains backward compatibility through:

1. **Wrapper Functions**: Keep existing function signatures
2. **Import Aliases**: Maintain existing import paths
3. **Gradual Migration**: Phase out old code gradually
4. **Deprecation Warnings**: Notify users of changes

### Example Migration

**Before**:
```python
from CamPro_OptimalMotion import solve_cam_motion_law

solution = solve_cam_motion_law(
    stroke=20.0,
    upstroke_duration_percent=60.0,
    motion_type="minimum_jerk"
)
```

**After** (New Library):
```python
from campro.optimization import MotionOptimizer
from campro.constraints import CamMotionConstraints

# Create constraints
constraints = CamMotionConstraints(
    stroke=20.0,
    upstroke_duration_percent=60.0
)

# Create optimizer
optimizer = MotionOptimizer()

# Solve
result = optimizer.solve_cam_motion_law(
    constraints, 
    motion_type="minimum_jerk"
)
```

**After** (Backward Compatible):
```python
from CamPro_OptimalMotion import solve_cam_motion_law  # Still works!

solution = solve_cam_motion_law(
    stroke=20.0,
    upstroke_duration_percent=60.0,
    motion_type="minimum_jerk"
)
```

## Benefits

### For Current Users
- **No Breaking Changes**: Existing code continues to work
- **Better Performance**: Optimized library structure
- **Enhanced Features**: New plotting and validation capabilities

### For Future Development
- **Combustion Simulation**: Ready for physics-based optimization
- **Modular Design**: Easy to add new constraint types
- **Extensible**: Simple to add new optimization methods
- **Testable**: Isolated components for better testing

### For Maintenance
- **Clear Separation**: Each library has a single responsibility
- **Consistent Interfaces**: Standardized patterns across libraries
- **Documentation**: Comprehensive documentation for each library
- **Type Safety**: Better type hints and validation

## Future Extensions

### Combustion Physics Integration

The new structure enables easy integration of combustion physics:

```python
from campro.physics import CombustionModel
from campro.constraints import CombustionConstraints
from campro.optimization import PhysicsOptimizer

# Create combustion model
combustion = CombustionModel()
combustion.configure(
    cylinder_volume=500e-6,  # 500cc
    compression_ratio=10.5,
    fuel_type="gasoline"
)

# Define combustion constraints
constraints = CombustionConstraints(
    max_pressure=100e5,  # 100 bar
    max_temperature=2500,  # K
    valve_timing_bounds=(0, 720)  # degrees
)

# Optimize with physics
optimizer = PhysicsOptimizer()
result = optimizer.optimize_combustion(
    combustion_model=combustion,
    constraints=constraints,
    objective="max_efficiency"
)
```

### Multi-Domain Optimization

The library structure supports optimization across multiple domains:

```python
# Combine motion and combustion optimization
motion_result = motion_optimizer.solve_cam_motion_law(motion_constraints)
combustion_result = physics_optimizer.optimize_combustion(combustion_constraints)

# Joint optimization
joint_result = optimizer.optimize_joint(
    motion_constraints=motion_constraints,
    combustion_constraints=combustion_constraints,
    coupling_constraints=coupling_constraints
)
```

## Conclusion

This refactoring plan provides a solid foundation for the Larrak project's evolution from a motion law solver to a comprehensive engine optimization platform. The modular design ensures maintainability while enabling rapid development of new features.

The library-based architecture will significantly improve code organization, testing capabilities, and extensibility, making it much easier to add combustion physics simulation and other advanced features in the future.


