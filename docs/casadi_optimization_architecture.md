# CasADi Optimization Architecture

## Overview

This document describes the architecture of the CasADi-based optimization framework for Phase 1 motion law optimization with thermal efficiency objectives. The framework implements direct collocation using CasADi's Opti stack with deterministic seeding/polish capabilities and physics-based constraints from free-piston engine literature.

## System Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  CasADi Optimization Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Problem Spec] ──> [Initial Guess Builder] ──> [Seed/Polish]│
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
│                        [Result Metadata] ──> Phase Summary  │
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
│  ┌─────────────────┐    ┌───────────────────────────────┐ │
│  │ CasADiMotion    │    │ Initial Guess Builder          │ │
│  │ Optimizer       │    │                               │ │
│  │                 │    │ ┌─────────────┐  ┌──────────┐ │ │
│  │ ┌─────────────┐ │    │ │ build_seed │  │ polish   │ │ │
│  │ │ CasADi Opti │ │    │ └─────────────┘  └──────────┘ │ │
│  │ │ Stack       │ │    └───────────────────────────────┘ │
│  │ └─────────────┘ │                                      │
│  └─────────────────┘                                      │
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

### 2. InitialGuessBuilder

**Purpose**: Generates deterministic seeds (with optional polishing) for the CasADi optimizer without relying on historical solutions.

**Key Features**:
- Analytic 5th-order S-curve that satisfies boundary conditions on any grid
- Optional smoothing/polishing pass that enforces velocity/acceleration/jerk limits
- Automatic rescaling when GUI updates the collocation segment count
- Stateless design: no persistent history required

**Class Diagram**:
```
InitialGuessBuilder
├── n_segments: int
├── build_seed(problem)
├── polish_seed(problem, guess, smoothing_passes=2)
└── update_segments(n_segments)
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

**Purpose**: Orchestrates the complete optimization flow including problem setup, deterministic seeding/polishing, optimization, and thermal analysis.

**Key Features**:
- Problem setup and validation
- Deterministic initial guess with optional polishing
- CasADi Opti stack optimization
- Thermal efficiency evaluation
- Adaptive collocation resolution ladder (coarse -> fine angle) with interpolation between levels
- Benchmarking capabilities

**Class Diagram**:
```
CasADiUnifiedFlow
├── motion_optimizer: CasADiMotionOptimizer
├── initial_guess_builder: InitialGuessBuilder
├── thermal_model: SimplifiedThermalModel
├── settings: CasADiOptimizationSettings
├── optimize_phase1()
├── _create_problem_from_constraints()
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

### 2. Initial Guess Flow

```
Problem Parameters
    ↓
InitialGuessBuilder.build_seed()
    ↓ (if enabled)
InitialGuessBuilder.polish_seed()
    ↓
Seeded Variables (x, v, a, j)
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
Update Metadata / GUI Display
    ↓
Archive result (optional analytics)
```

## Integration Points

### 1. Unified Framework Integration

The CasADi optimization framework integrates with the existing unified optimization framework through:

- **Replacement of MotionOptimizer**: `unified_framework.py` line 292
- **Deterministic seeding**: InitialGuessBuilder invoked inside CasADiUnifiedFlow
- **Result metadata updates**: Efficiency metrics injected for downstream phases

### 2. GUI Integration

The framework integrates with the GUI through:

- **CasADi optimization option**: Checkbox to enable CasADi Opti stack
- **Seed polish toggle**: Reuses the existing “Enable warm-start” checkbox to control the polish pass
- **Thermal efficiency display**: Show efficiency in results
- **Convergence history plot**: Display optimization progress

### 3. Phase 2/3 Interface

The framework provides clean interfaces for Phase 2/3 integration:

- **Motion profile output**: Standardized format for cam-ring optimization
- **Thermal efficiency metrics**: For crank center optimization
- **Solution metadata**: For reporting/diagnostics (no persistent warm-start history)

## Performance Characteristics

### 1. Computational Complexity

- **Direct Collocation**: O(N³) where N is number of segments
- **Deterministic seed/polish**: O(N) smoothing and clipping
- **Thermal efficiency**: O(N) per evaluation

### 2. Memory Requirements

- **Solution history**: O(M × N) where M is history size, N is segments
- **Collocation matrices**: O(N × d²) where d is polynomial order
- **Sparse matrices**: CasADi uses compressed column storage

### 3. Convergence Properties

- **Deterministic seed**: Provides consistent boundary-satisfying initial iterate
- **Polish pass**: Keeps derivatives within bounds, improving IPOPT restoration behavior
- **IPOPT solver**: Robust convergence with MA57 linear solver
- **Thermal efficiency**: 55% target achievable with simplified model

## Error Handling

### 1. Optimization Failures

- **Solver failures**: Caught and logged with error details
- **Constraint violations**: Validated before optimization
- **Numerical issues**: Handled by IPOPT with appropriate tolerances

### 2. Seed/Polish Behavior

- **Polish disabled**: Solver proceeds with raw S-curve seed
- **Constraint clipping**: Polish clamps derivatives to avoid infeasible starts
- **Failure handling**: Any polish error is logged and the raw seed is used

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
