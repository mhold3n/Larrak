# CasADi Optimization Architecture

## Overview

This document describes the architecture of the CasADi-based optimization framework for Phase 1 motion law optimization with thermal efficiency objectives. The framework implements direct collocation using CasADi's Opti stack with warm-starting capabilities and physics-based constraints from free-piston engine literature.

## System Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  CasADi Optimization Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Problem Spec] ──> [WarmStart Manager] ──> [Initial Guess]│
│        │                                           │        │
│        │                                           ▼        │
│        └──────> [CasADi Opti Stack] <──────────────        │
│                        │                                    │
│                        ├─> [Direct Collocation]            │
│                        ├─> [Thermal Physics Model]         │
│                        ├─> [Motion Constraints]            │
│                        └─> [IPOPT Solver]                  │
│                                │                            │
│                                ▼                            │
│                        [Solution Storage] ──> History       │
│                                │                            │
│                                ▼                            │
│                        [Phase 2/3 Interface]               │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CasADiUnifiedFlow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ CasADiMotion    │    │ WarmStart       │                │
│  │ Optimizer       │    │ Manager         │                │
│  │                 │    │                 │                │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                │
│  │ │ CasADi Opti │ │    │ │ Solution    │ │                │
│  │ │ Stack       │ │    │ │ History     │ │                │
│  │ └─────────────┘ │    │ └─────────────┘ │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Simplified      │    │ CasADiMotion    │                │
│  │ Thermal Model   │    │ Problem         │                │
│  │                 │    │                 │                │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                │
│  │ │ Otto Cycle  │ │    │ │ Constraints │ │                │
│  │ │ Efficiency  │ │    │ │ Objectives  │ │                │
│  │ └─────────────┘ │    │ └─────────────┘ │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. CasADiMotionOptimizer

**Purpose**: Implements Phase 1 motion law optimization using CasADi Opti stack with direct collocation.

**Key Features**:
- Direct collocation with Legendre/Radau polynomials
- State variables: position, velocity, acceleration
- Control variables: jerk
- Boundary conditions from stroke/cycle time constraints
- IPOPT solver with MA57 linear solver

**Class Diagram**:
```
CasADiMotionOptimizer
├── opti: Opti
├── n_segments: int
├── poly_order: int
├── collocation_method: str
├── setup_collocation()
├── add_boundary_conditions()
├── add_motion_constraints()
├── add_physics_constraints()
├── add_thermal_efficiency_objective()
└── solve()
```

### 2. WarmStartManager

**Purpose**: Manages solution history and generates initial guesses for warm-starting.

**Key Features**:
- Solution history storage with metadata
- Parameter-based solution matching
- Linear interpolation between solutions
- Fallback to polynomial interpolation
- Persistent storage support

**Class Diagram**:
```
WarmStartManager
├── solution_history: List[SolutionRecord]
├── max_history: int
├── tolerance: float
├── storage_path: Optional[str]
├── store_solution()
├── get_initial_guess()
├── _find_closest_solution()
├── _interpolate_solution()
└── _generate_fallback_guess()
```

### 3. SimplifiedThermalModel

**Purpose**: Implements physics-based thermal efficiency calculations with key constraints from FPE literature.

**Key Features**:
- Otto cycle efficiency with variable compression ratio
- Heat loss modeling (simplified Woschni correlation)
- Mechanical losses
- Pressure rate constraints
- Temperature constraints

**Class Diagram**:
```
SimplifiedThermalModel
├── config: ThermalEfficiencyConfig
├── compute_compression_ratio()
├── compute_otto_efficiency()
├── compute_heat_loss_penalty()
├── compute_mechanical_losses()
├── compute_thermal_efficiency()
├── add_compression_ratio_constraints()
├── add_pressure_rate_constraints()
└── add_temperature_constraints()
```

### 4. CasADiUnifiedFlow

**Purpose**: Orchestrates the complete optimization flow including problem setup, warm-starting, optimization, and solution storage.

**Key Features**:
- Problem setup and validation
- Warm-starting from previous solutions
- CasADi Opti stack optimization
- Thermal efficiency evaluation
- Solution storage for future warm-starts
- Benchmarking capabilities

**Class Diagram**:
```
CasADiUnifiedFlow
├── motion_optimizer: CasADiMotionOptimizer
├── warmstart_mgr: WarmStartManager
├── thermal_model: SimplifiedThermalModel
├── settings: CasADiOptimizationSettings
├── optimize_phase1()
├── _create_problem_from_constraints()
├── _store_solution()
└── benchmark_optimization()
```

## Data Flow

### 1. Problem Setup Flow

```
Input Constraints/Targets
    ↓
CasADiMotionProblem Creation
    ↓
Parameter Validation
    ↓
Problem Specification
```

### 2. Warm-Start Flow

```
Problem Parameters
    ↓
WarmStartManager.get_initial_guess()
    ↓
Strategy 1: Find Closest Solution
    ↓ (if no match)
Strategy 2: Interpolate Between Solutions
    ↓ (if no bracketing)
Strategy 3: Generate Fallback Guess
    ↓
Initial Guess Variables
```

### 3. Optimization Flow

```
Initial Guess + Problem Spec
    ↓
CasADi Opti Stack Setup
    ↓
Direct Collocation Discretization
    ↓
Add Boundary Conditions
    ↓
Add Motion Constraints
    ↓
Add Physics Constraints
    ↓
Add Thermal Efficiency Objective
    ↓
IPOPT Solver
    ↓
Solution Variables
```

### 4. Solution Storage Flow

```
Optimization Result
    ↓
Extract Variables (x, v, a, j)
    ↓
Evaluate Thermal Efficiency
    ↓
Create SolutionRecord
    ↓
WarmStartManager.store_solution()
    ↓
Persistent Storage
```

## Integration Points

### 1. Unified Framework Integration

The CasADi optimization framework integrates with the existing unified optimization framework through:

- **Replacement of MotionOptimizer**: `unified_framework.py` line 292
- **Warm-start logic**: Added before line 482 (primary optimization call)
- **Solution storage**: After line 483 for warm-starting

### 2. GUI Integration

The framework integrates with the GUI through:

- **CasADi optimization option**: Checkbox to enable CasADi Opti stack
- **Warm-start toggle**: Enable/disable warm-starting
- **Thermal efficiency display**: Show efficiency in results
- **Convergence history plot**: Display optimization progress

### 3. Phase 2/3 Interface

The framework provides clean interfaces for Phase 2/3 integration:

- **Motion profile output**: Standardized format for cam-ring optimization
- **Thermal efficiency metrics**: For crank center optimization
- **Solution metadata**: For warm-starting subsequent phases

## Performance Characteristics

### 1. Computational Complexity

- **Direct Collocation**: O(N³) where N is number of segments
- **Warm-starting**: O(M) where M is history size
- **Thermal efficiency**: O(N) per evaluation

### 2. Memory Requirements

- **Solution history**: O(M × N) where M is history size, N is segments
- **Collocation matrices**: O(N × d²) where d is polynomial order
- **Sparse matrices**: CasADi uses compressed column storage

### 3. Convergence Properties

- **Direct collocation**: Typically 3-5x faster with warm-starting
- **IPOPT solver**: Robust convergence with MA57 linear solver
- **Thermal efficiency**: 55% target achievable with simplified model

## Error Handling

### 1. Optimization Failures

- **Solver failures**: Caught and logged with error details
- **Constraint violations**: Validated before optimization
- **Numerical issues**: Handled by IPOPT with appropriate tolerances

### 2. Warm-start Failures

- **No history**: Falls back to default initial guess
- **Parameter mismatch**: Uses interpolation or fallback
- **Storage errors**: Logged but doesn't stop optimization

### 3. Physics Model Failures

- **Invalid parameters**: Validated before use
- **Numerical overflow**: Handled with appropriate bounds
- **Convergence issues**: Logged with diagnostic information

## Future Extensions

### 1. Multiple Shooting Fallback

- **Stub implementation**: `setup_multiple_shooting_fallback()`
- **Future development**: For stiff cases where direct collocation fails
- **Integration**: Seamless fallback in optimization flow

### 2. Advanced Physics Models

- **1D gas dynamics**: Stubs for future implementation
- **Heat transfer**: More detailed Woschni correlation
- **Combustion modeling**: HCCI/SI/CI mode support

### 3. Machine Learning Integration

- **Initial guess prediction**: ML-based parameter matching
- **Efficiency prediction**: Neural network models
- **Adaptive weights**: Dynamic objective weighting

## Dependencies

### 1. Core Dependencies

- **CasADi ≥ 3.6.0**: Optimization framework
- **IPOPT with MA57**: Nonlinear solver
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing

### 2. Optional Dependencies

- **Jupyter**: For interactive notebooks
- **Matplotlib**: For visualization
- **Pandas**: For data analysis

### 3. Development Dependencies

- **Pytest**: Testing framework
- **Black**: Code formatting
- **MyPy**: Type checking

