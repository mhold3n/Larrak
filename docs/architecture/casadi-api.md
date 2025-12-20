# CasADi Architecture and API Reference

This document consolidates CasADi optimization architecture, physics integration, and API documentation for the Larrak system.

## Overview

The CasADi-based optimization framework implements direct collocation with thermal efficiency objectives for free-piston engine motion law optimization. The framework uses CasADi's Opti stack with IPOPT solver and supports automatic differentiation for physics constraints.

### Architectural Shift: Hybrid → Full CasADi

**Current (Hybrid)**:
- Optimizer wraps Python physics in objective/constraint callbacks
- Each evaluation: Python loop → extract scalar → return to IPOPT
- Gradients via finite differences (slow, inaccurate ~1e-6)

**Target (Full CasADi)**:
- Physics as symbolic MX graphs built once
- NLP contains full expression tree
- Automatic differentiation: exact gradients at zero extra cost
- IPOPT exploits sparsity patterns
- Expected 2-3x speedup + improved convergence

**Toggle**: `campro.constants.USE_CASADI_PHYSICS` (default `False` until validated)

---

## System Architecture

### High-Level Flow

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
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `CasADiMotionOptimizer` | Direct collocation with Legendre/Radau polynomials |
| `InitialGuessBuilder` | Deterministic seeds with optional polishing |
| `SimplifiedThermalModel` | Otto cycle efficiency with physics constraints |
| `CasADiUnifiedFlow` | Orchestrates problem setup, optimization, and analysis |

---

## Mathematical Formulation

### Objective

Minimize the weighted sum of jerk and thermal efficiency:

```
minimize: w_jerk * ∫j² dt + w_thermal * (1 - η_thermal)
```

### State Variables

| Variable | Symbol | Units |
|----------|--------|-------|
| Position | x(t) | m |
| Velocity | v(t) | m/s |
| Acceleration | a(t) | m/s² |
| Jerk (control) | j(t) | m/s³ |

### System Dynamics

```
ẋ = v
v̇ = a  
ȧ = j
```

### Direct Collocation

The time horizon [0, T] is divided into N finite elements with d+1 Legendre-Gauss-Radau collocation points per element.

State discretization using Lagrange polynomials:
```
x_k(τ) = Σᵢ₌₀ᵈ x_{k,i} * L_i(τ)
```

Collocation constraints:
```
x_{k,j} = x_{k,0} + (Δt/2) * Σᵢ₌₀ᵈ A_{j,i} * v_{k,i}
```

---

## CasADi Physics API

### Design Principles

1. **CasADi MX Exclusively**: All symbolic expressions use `casadi.MX` type
2. **Units Convention**: 
   - Angles: radians
   - Linear: millimeters (mm)
   - Forces: Newtons (N)
   - Torque: Newton-meters (N⋅m)
3. **Domain Safety**: Guard all singular operations with epsilon thresholds
4. **Vectorization**: Accept vector inputs `MX(n,1)` with elementwise operations

### Constants

```python
USE_CASADI_PHYSICS: bool = False  # Feature toggle
CASADI_PHYSICS_EPSILON: float = 1e-12  # Domain guard
CASADI_PHYSICS_ASIN_CLAMP: float = 0.999999  # arcsin clamp
```

### Kinematics Module

```python
from campro.physics.casadi.kinematics import create_crank_piston_kinematics

kin_fn = create_crank_piston_kinematics()
x, v, a, rod_angle, r_eff = kin_fn(theta, r, l, x_off, y_off)
```

**Inputs**:
- `theta: MX(n,1)` - Crank angles (radians)
- `r: MX` - Crank radius (mm)
- `l: MX` - Connecting rod length (mm)
- `x_off, y_off: MX` - Crank center offset (mm)

**Outputs**:
- `x: MX(n,1)` - Piston displacement (mm)
- `v: MX(n,1)` - Piston velocity (mm/rad)
- `a: MX(n,1)` - Piston acceleration (mm/rad²)

---

## API Usage Examples

### Basic Optimization

```python
from campro.optimization.casadi_motion_optimizer import (
    CasADiMotionOptimizer, CasADiMotionProblem
)

# Create problem (per-degree units)
problem = CasADiMotionProblem(
    stroke=0.100,  # 100mm in meters
    duration_angle_deg=360.0,
    cycle_time=0.0385,  # 26 Hz
    max_velocity=0.00019,  # m/deg
    max_acceleration=0.0019,  # m/deg²
    max_jerk=0.019,  # m/deg³
    compression_ratio_limits=(20.0, 70.0),
    weights={'jerk': 1.0, 'thermal_efficiency': 0.1}
)

# Initialize and solve
optimizer = CasADiMotionOptimizer(
    n_segments=50,
    poly_order=3,
    collocation_method="legendre"
)
result = optimizer.solve(problem)

if result.successful:
    position = result.variables['position']
    velocity = result.variables['velocity']
```

### Deterministic Seeding

```python
from campro.optimization.initial_guess import InitialGuessBuilder

builder = InitialGuessBuilder(n_segments=50)
seed = builder.build_seed(problem)
polished_seed = builder.polish_seed(problem, seed)
result = optimizer.solve(problem, polished_seed)
```

### Thermal Efficiency Integration

```python
from campro.optimization.casadi_thermal import ThermalEfficiencyConfig

thermal_config = ThermalEfficiencyConfig(
    compression_ratio_min=20.0,
    compression_ratio_max=70.0,
    heat_loss_coefficient=0.05,
    max_pressure_rate=1e8
)

problem = CasADiMotionProblem(
    ...,
    thermal_config=thermal_config,
    maximize_thermal_efficiency=True
)
```

---

## Known Issues and Resolutions

### NaN in Jacobian (Row 5, Col 20)

**Problem**: NaN detected in temperature collocation residual during symbolic differentiation.

**Root Cause**: Complex derivative chain through:
- `p * dV_dt` where `p = rho * R * T`
- Division by `m_safe * cv`
- Temperature-dependent specific heat

**CasADi AD is NOT the problem** - it's correctly computing derivatives. The issue is the formulation creates a derivative chain that produces NaN.

**Solutions**:
1. **Reformulate energy balance** as direct constraint (recommended)
2. Simplify to constant properties
3. Add regularization at constraint level

See `troubleshooting/nan-diagnosis.md` for full analysis.

---

## Integration Roadmap

### Phase 1: Torque Calculation Port (Complete)
- Crank-piston kinematics in CasADi
- Piston force calculation
- Torque integration

### Phase 2: Side Loading Analysis
- Connecting rod force decomposition
- Cylinder wall loading
- Journal bearing loads

### Phase 3: Litvin Metrics
- Slip integral calculation
- Contact length evaluation
- Hertzian stress constraints

### Phase 4: Full NLP Integration
- Replace hybrid callbacks
- Enable exact gradients
- Exploit sparsity patterns






