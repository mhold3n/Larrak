# Cam-Ring System Refactoring Plan for Maximum Adaptability

## Current State Analysis

The current implementation has several areas that need refactoring for maximum adaptability:

### Issues with Current Architecture:
1. **Monolithic Classes**: `CamRingMapper` is a large class with multiple responsibilities
2. **Tight Coupling**: Processing functions are tightly coupled to specific data structures
3. **Limited Extensibility**: Hard to add new system types or optimization strategies
4. **Mixed Concerns**: Physics, optimization, and data processing are mixed together
5. **Fixed Interfaces**: Hard to modify system parameters or add new components

## Refactoring Strategy: Modular Component Architecture

### 1. Core Physics Components (Separation of Concerns)

```
campro/physics/
├── base/
│   ├── __init__.py
│   ├── component.py          # Base component interface
│   ├── system.py            # Base system interface
│   └── result.py            # Standardized result types
├── geometry/
│   ├── __init__.py
│   ├── curves.py            # Cam curve computation
│   ├── curvature.py         # Curvature analysis
│   └── transformations.py   # Coordinate transformations
├── kinematics/
│   ├── __init__.py
│   ├── meshing_law.py       # Rolling kinematics
│   ├── time_kinematics.py   # Time-based kinematics
│   └── constraints.py       # Kinematic constraints
├── systems/
│   ├── __init__.py
│   ├── cam_ring_system.py   # Cam-ring system implementation
│   ├── connecting_rod.py    # Connecting rod component
│   └── ring_follower.py     # Ring follower component
└── validation/
    ├── __init__.py
    ├── design_validator.py  # Design validation
    └── constraints.py       # Physical constraints
```

### 2. Optimization Framework (Strategy Pattern)

```
campro/optimization/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py     # Base optimization strategy
│   ├── motion_strategy.py   # Motion law optimization
│   ├── geometry_strategy.py # Geometry optimization
│   └── multi_objective.py   # Multi-objective optimization
├── objectives/
│   ├── __init__.py
│   ├── base_objective.py    # Base objective function
│   ├── motion_objectives.py # Motion-based objectives
│   ├── geometry_objectives.py # Geometry-based objectives
│   └── performance_objectives.py # Performance objectives
└── constraints/
    ├── __init__.py
    ├── base_constraint.py   # Base constraint interface
    ├── physical_constraints.py # Physical constraints
    └── design_constraints.py # Design constraints
```

### 3. Configuration System (Builder Pattern)

```
campro/config/
├── __init__.py
├── system_builder.py        # System configuration builder
├── parameter_manager.py     # Parameter management
└── validation.py           # Configuration validation
```

### 4. Data Pipeline (Pipeline Pattern)

```
campro/pipeline/
├── __init__.py
├── base_pipeline.py         # Base pipeline interface
├── motion_pipeline.py       # Motion law processing
├── geometry_pipeline.py     # Geometry processing
└── optimization_pipeline.py # Optimization processing
```

## Implementation Plan

### Phase 1: Core Component Architecture

#### 1.1 Base Component Interface
```python
# campro/physics/base/component.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class BaseComponent(ABC):
    """Base interface for all physics components."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self._validate_parameters()
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate component parameters."""
        pass
    
    @abstractmethod
    def compute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute component outputs from inputs."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get component parameters."""
        pass
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update component parameters."""
        self.parameters.update(parameters)
        self._validate_parameters()
```

#### 1.2 System Architecture
```python
# campro/physics/base/system.py
class BaseSystem(ABC):
    """Base interface for complete systems."""
    
    def __init__(self, components: Dict[str, BaseComponent]):
        self.components = components
        self._validate_system()
    
    @abstractmethod
    def _validate_system(self) -> None:
        """Validate system configuration."""
        pass
    
    @abstractmethod
    def solve(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Solve the complete system."""
        pass
    
    def add_component(self, name: str, component: BaseComponent) -> None:
        """Add a component to the system."""
        self.components[name] = component
        self._validate_system()
    
    def remove_component(self, name: str) -> None:
        """Remove a component from the system."""
        if name in self.components:
            del self.components[name]
            self._validate_system()
```

### Phase 2: Modular Physics Components

#### 2.1 Cam Curve Component
```python
# campro/physics/geometry/curves.py
class CamCurveComponent(BaseComponent):
    """Component for computing cam curves."""
    
    def _validate_parameters(self) -> None:
        required = ['base_radius', 'connecting_rod_length']
        for param in required:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")
    
    def compute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        theta = inputs['theta']
        x_theta = inputs['x_theta']
        
        # Compute cam curves
        pitch_radius = self.parameters['base_radius'] + x_theta
        profile_radius = pitch_radius  # Direct contact
        
        return {
            'pitch_radius': pitch_radius,
            'profile_radius': profile_radius,
            'contact_radius': profile_radius
        }
```

#### 2.2 Meshing Law Component
```python
# campro/physics/kinematics/meshing_law.py
class MeshingLawComponent(BaseComponent):
    """Component for solving the meshing law."""
    
    def compute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        theta = inputs['theta']
        rho_c = inputs['rho_c']
        psi_initial = inputs['psi_initial']
        R_psi = inputs['R_psi']
        
        # Solve meshing law: ρ_c(θ)dθ = R(ψ)dψ
        psi = self._solve_meshing_law(theta, rho_c, psi_initial, R_psi)
        
        return {'psi': psi}
```

### Phase 3: Strategy-Based Optimization

#### 3.1 Base Optimization Strategy
```python
# campro/optimization/strategies/base_strategy.py
class BaseOptimizationStrategy(ABC):
    """Base interface for optimization strategies."""
    
    def __init__(self, objectives: List[BaseObjective], 
                 constraints: List[BaseConstraint]):
        self.objectives = objectives
        self.constraints = constraints
    
    @abstractmethod
    def optimize(self, system: BaseSystem, 
                initial_guess: Dict[str, Any]) -> OptimizationResult:
        """Perform optimization."""
        pass
```

#### 3.2 Multi-Objective Strategy
```python
# campro/optimization/strategies/multi_objective.py
class MultiObjectiveStrategy(BaseOptimizationStrategy):
    """Multi-objective optimization strategy."""
    
    def optimize(self, system: BaseSystem, 
                initial_guess: Dict[str, Any]) -> OptimizationResult:
        # Implement Pareto optimization
        pass
```

### Phase 4: Configuration Builder

#### 4.1 System Builder
```python
# campro/config/system_builder.py
class SystemBuilder:
    """Builder for creating configurable systems."""
    
    def __init__(self):
        self.components = {}
        self.connections = {}
    
    def add_component(self, name: str, component_type: str, 
                     parameters: Dict[str, Any]) -> 'SystemBuilder':
        """Add a component to the system."""
        component = self._create_component(component_type, parameters)
        self.components[name] = component
        return self
    
    def connect_components(self, from_component: str, to_component: str,
                          connection_type: str) -> 'SystemBuilder':
        """Connect components."""
        self.connections[(from_component, to_component)] = connection_type
        return self
    
    def build(self) -> BaseSystem:
        """Build the configured system."""
        return self._create_system()
```

## Benefits of This Architecture

### 1. **Maximum Adaptability**
- Easy to add new components (cam types, ring designs, etc.)
- Pluggable optimization strategies
- Configurable system topologies

### 2. **Separation of Concerns**
- Physics components are independent
- Optimization logic is separate from physics
- Clear interfaces between components

### 3. **Extensibility**
- New system types can be added easily
- New optimization objectives can be plugged in
- New constraints can be added without modifying existing code

### 4. **Testability**
- Each component can be tested independently
- Mock components for testing
- Clear input/output contracts

### 5. **Maintainability**
- Single responsibility principle
- Clear dependencies
- Easy to modify individual components

## Migration Strategy

### Step 1: Create Base Interfaces
- Implement base component and system interfaces
- Create result and parameter management classes

### Step 2: Extract Components
- Break down `CamRingMapper` into individual components
- Create geometry, kinematics, and validation components

### Step 3: Implement Strategy Pattern
- Create optimization strategy interfaces
- Implement specific strategies (motion, geometry, multi-objective)

### Step 4: Build Configuration System
- Implement system builder
- Create parameter management system

### Step 5: Update GUI
- Modify GUI to use new architecture
- Add configuration capabilities

### Step 6: Testing and Validation
- Comprehensive testing of all components
- Performance validation
- Integration testing

This refactoring will provide the foundation for complex optimizations while maintaining clean, adaptable code.

