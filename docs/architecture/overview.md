# System Architecture Overview

This document describes the modular, library-based architecture of the Larrak/CamPro optimization system.

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Modularity** | Separate concerns into distinct libraries |
| **Extensibility** | Easy addition of new physics domains |
| **Maintainability** | Clean interfaces and consistent patterns |
| **Reusability** | Components can be used independently |
| **Testability** | Isolated testing of individual components |

---

## Library Structure

```
campro/
├── constraints/          # Constraint definitions and validation
│   ├── base.py          # Base constraint classes
│   ├── motion.py        # Motion-specific constraints
│   ├── cam.py           # Cam-specific constraints
│   └── physics.py       # Combustion physics constraints
├── optimization/         # Optimization routines and solvers
│   ├── base.py          # Base solver interface
│   ├── collocation.py   # Collocation methods
│   ├── motion.py        # Motion law optimization
│   └── casadi_*.py      # CasADi integration
├── physics/              # Physics simulation modules
│   ├── base.py          # Base physics interface
│   ├── casadi/          # CasADi physics port
│   └── thermodynamics.py
├── diagnostics/          # Analysis and debugging
│   ├── scaling.py       # NLP scaling diagnostics
│   └── feasibility.py   # Constraint feasibility checks
├── environment/          # Runtime environment
│   ├── env_manager.py   # Environment detection
│   └── hsl_detector.py  # HSL solver detection
├── utils/                # Utility functions
│   ├── plotting.py      # Visualization
│   ├── validation.py    # Input validation
│   └── conversion.py    # Unit conversions
└── config/               # Configuration
    └── system_builder.py # Builder pattern system config
```

---

## Core Libraries

### Constraints Library (`campro.constraints`)

**Purpose**: Define and validate constraints for optimization problems.

**Components**:
- `BaseConstraints`: Abstract base class with common interface
- `MotionConstraints`: General motion law constraints with bounds
- `CamMotionConstraints`: Cam-specific constraints with intuitive parameters
- `ConstraintViolation`: Standardized violation tracking

**Features**:
- Consistent validation interface across all constraint types
- Violation tracking with detailed error messages
- Conversion between constraint types (cam → motion)
- Dictionary serialization for persistence

### Optimization Library (`campro.optimization`)

**Purpose**: Provide optimization routines and solvers.

**Components**:
- `BaseOptimizer`: Abstract base with common optimization interface
- `CollocationOptimizer`: Direct collocation with CasADi integration
- `MotionOptimizer`: High-level motion law optimization
- `CasADiMotionOptimizer`: CasADi Opti stack implementation
- `OptimizationResult`: Standardized result format

**Features**:
- Pluggable optimization methods
- Performance tracking and history management
- Support for multiple motion law types (min time, energy, jerk)
- Custom objective function support

### Physics Library (`campro.physics`)

**Purpose**: Foundation for physics simulation.

**Components**:
- `BasePhysicsModel`: Abstract base class for physics models
- `PhysicsResult`: Standardized simulation results
- `casadi/`: CasADi symbolic physics for auto-differentiation

**Features**:
- Extensible framework for physics simulation
- Standardized simulation interface
- Performance tracking for physics calculations

---

## Modular Component Architecture

### Base Interfaces

```python
class BaseComponent(ABC):
    """Standard interface for all physics components."""
    def compute(self, inputs: Dict[str, np.ndarray]) -> ComponentResult
    def validate_parameters(self) -> None
    def get_required_inputs(self) -> List[str]
    def get_outputs(self) -> List[str]
```

### Modular Physics Components

| Component | Purpose |
|-----------|---------|
| `CamCurveComponent` | Computes cam curves from linear follower motion |
| `CurvatureComponent` | Calculates curvature and osculating radius |
| `MeshingLawComponent` | Solves meshing law between cam and ring |
| `CoordinateTransformComponent` | Handles coordinate transformations |

### System Builder Pattern

```python
builder = SystemBuilder("MySystem")
builder.add_component('cam_curves', 'cam_curve', {'base_radius': 15.0})
builder.add_component('curvature', 'curvature', {})
builder.connect_components('cam_curves', 'curvature')
system = builder.create_system()
```

### Result Handling

```python
result = component.compute(inputs)
if result.is_successful:
    outputs = result.outputs
    metadata = result.metadata
else:
    error = result.error_message
```

---

## Architecture Benefits

### Modularity
- **Clear Separation**: Each library has a single responsibility
- **Independent Development**: Libraries can be developed/tested independently
- **Easy Maintenance**: Changes in one library don't affect others

### Extensibility
- **New Constraint Types**: Easy to add new constraint systems
- **New Optimization Methods**: Pluggable optimization framework
- **Physics Integration**: Ready for combustion simulation
- **Custom Objectives**: Support for user-defined optimization goals

### Consistency
- **Standardized Interfaces**: Common patterns across all libraries
- **Uniform Error Handling**: Consistent error reporting and validation
- **Performance Tracking**: Standardized metrics across all components

---

## Implementation Status

### Completed

| Component | Status |
|-----------|--------|
| Constraint library structure | ✅ |
| Base constraint classes | ✅ |
| Motion/cam constraints | ✅ |
| Optimization library structure | ✅ |
| Base optimizer classes | ✅ |
| Collocation optimizer | ✅ |
| CasADi motion optimizer | ✅ |
| Physics library foundation | ✅ |
| Utils library (plotting) | ✅ |
| System builder pattern | ✅ |
| Modular physics components | ✅ |

### In Progress

| Component | Status |
|-----------|--------|
| Full CasADi physics port | 🔄 |
| Combustion physics integration | 🔄 |
| GUI library modernization | 📋 |

---

## Related Documentation

- **CasADi API**: See `architecture/casadi-api.md`
- **Optimization Strategies**: See `architecture/optimization.md`
- **Troubleshooting**: See `troubleshooting/` directory






