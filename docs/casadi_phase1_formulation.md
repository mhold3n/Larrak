# CasADi Phase 1 Optimization Formulation

## Mathematical Formulation

This document provides the mathematical formulation for Phase 1 motion law optimization using CasADi's direct collocation method with thermal efficiency objectives.

## Problem Statement

### Objective

Minimize the weighted sum of jerk and thermal efficiency objectives:

```
minimize: w_jerk * ∫j² dt + w_thermal * (1 - η_thermal)
```

where:
- `j(t)` is the jerk (third derivative of position)
- `η_thermal(t)` is the thermal efficiency
- `w_jerk` and `w_thermal` are objective weights

### State Variables

The state vector consists of:
- `x(t)`: Position (m)
- `v(t)`: Velocity (m/s)  
- `a(t)`: Acceleration (m/s²)

### Control Variables

- `j(t)`: Jerk (m/s³)

### System Dynamics

The system dynamics are:
```
ẋ = v
v̇ = a  
ȧ = j
```

## Direct Collocation Discretization

### Finite Elements

The time horizon `[0, T]` is divided into `N` finite elements:

```
t₀ = 0 < t₁ < t₂ < ... < t_N = T
```

Each element has length `Δt = T/N`.

### Collocation Points

Within each element, we use `d+1` collocation points based on Legendre-Gauss-Radau (LGR) polynomials:

```
τ₀ = -1, τ₁, τ₂, ..., τ_d = 1
```

where `τ_j` are the roots of the Legendre polynomial of degree `d`.

### State Discretization

Within each element `k`, the state is approximated as:

```
x_k(τ) = Σᵢ₌₀ᵈ x_{k,i} * L_i(τ)
v_k(τ) = Σᵢ₌₀ᵈ v_{k,i} * L_i(τ)  
a_k(τ) = Σᵢ₌₀ᵈ a_{k,i} * L_i(τ)
```

where `L_i(τ)` are Lagrange polynomials:

```
L_i(τ) = Πⱼ₌₀,ⱼ≠ᵢᵈ (τ - τ_j) / (τ_i - τ_j)
```

### Collocation Constraints

The collocation constraints enforce the system dynamics at each collocation point:

```
x_{k,j} = x_{k,0} + (Δt/2) * Σᵢ₌₀ᵈ A_{j,i} * v_{k,i}
v_{k,j} = v_{k,0} + (Δt/2) * Σᵢ₌₀ᵈ A_{j,i} * a_{k,i}
a_{k,j} = a_{k,0} + (Δt/2) * Σᵢ₌₀ᵈ A_{j,i} * j_{k,i}
```

where `A_{j,i}` are the collocation matrix elements.

### Collocation Matrix

The collocation matrix `A` is constructed as:

```
A_{j,i} = ∫₋₁ᵗⱼ L_i(τ) dτ
```

For Legendre-Gauss-Radau collocation with `d = 3`:

```
A = [0.0, 0.0, 0.0, 0.0]
    [0.5, 0.5, 0.0, 0.0]
    [0.5, 0.5, 0.5, 0.0]
    [0.5, 0.5, 0.5, 0.5]
```

## Boundary Conditions

### Position Constraints

```
x(0) = 0          # Start at zero position
x(T) = stroke      # End at stroke position
```

### Velocity Constraints

```
v(0) = 0          # Start at rest
v(T) = 0          # End at rest
```

### Upstroke/Downstroke Constraints

```
v(t) ≥ 0          for t ∈ [0, t_upstroke]
v(t) ≤ 0          for t ∈ [t_upstroke, T]
```

where `t_upstroke = T * upstroke_percent / 100`.

## Motion Constraints

### Velocity Limits

```
|v(t)| ≤ v_max
```

### Acceleration Limits

```
|a(t)| ≤ a_max
```

### Jerk Limits

```
|j(t)| ≤ j_max
```

## Physics Constraints

### Compression Ratio Constraints

The compression ratio is defined as:

```
CR(t) = (x(t) + clearance) / clearance
```

where `clearance` is the clearance distance (typically 2mm).

The compression ratio must satisfy:

```
CR_min ≤ CR(t) ≤ CR_max
```

From FPE literature:
- `CR_min = 20` (minimum for auto-ignition)
- `CR_max = 70` (maximum practical limit)

### Pressure Rate Constraints

To avoid diesel knock, the pressure rate is limited:

```
|dP/dt| ≤ P_rate_max
```

This translates to an acceleration rate constraint:

```
|a(t+1) - a(t)| / Δt ≤ a_rate_max
```

### Temperature Constraints

Temperature rise is limited by velocity constraints:

```
|v(t)| ≤ v_temp_max
```

where `v_temp_max` is derived from maximum temperature rise.

## Thermal Efficiency Model

### Otto Cycle Efficiency

The thermal efficiency is based on the Otto cycle:

```
η_otto = 1 - 1/CR^(γ-1)
```

where `γ = 1.4` is the specific heat ratio.

### Heat Loss Penalty

Heat loss is modeled using a simplified Woschni correlation:

```
Q_loss = h * v²
```

where `h` is the heat transfer coefficient.

### Mechanical Losses

Mechanical losses include friction and viscous damping:

```
P_mech = f * v² + c * a²
```

where `f` is the friction coefficient and `c` is the viscous damping coefficient.

### Total Thermal Efficiency

```
η_thermal = η_otto - Q_loss - P_mech
```

## CasADi Implementation

### Opti Stack Setup

```python
from casadi import *

# Create Opti instance
opti = Opti()

# Variables
x = opti.variable(N+1)      # Position
v = opti.variable(N+1)      # Velocity  
a = opti.variable(N+1)      # Acceleration
j = opti.variable(N)        # Jerk

# Parameters
T = opti.parameter()        # Cycle time
stroke = opti.parameter()   # Stroke length
```

### Collocation Implementation

```python
# Collocation matrix (example for d=3)
A = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.0],
    [0.5, 0.5, 0.5, 0.5]
])

# Collocation constraints
for k in range(N):
    for j in range(1, d+1):
        # Position constraint
        opti.subject_to(
            x[k+1] == x[k] + (T/N) * sum(A[j,i] * v[k,i] for i in range(d+1))
        )
        # Velocity constraint  
        opti.subject_to(
            v[k+1] == v[k] + (T/N) * sum(A[j,i] * a[k,i] for i in range(d+1))
        )
        # Acceleration constraint
        opti.subject_to(
            a[k+1] == a[k] + (T/N) * sum(A[j,i] * j[k,i] for i in range(d+1))
        )
```

### Boundary Conditions

```python
# Position boundary conditions
opti.subject_to(x[0] == 0)
opti.subject_to(x[-1] == stroke)

# Velocity boundary conditions  
opti.subject_to(v[0] == 0)
opti.subject_to(v[-1] == 0)
```

### Motion Constraints

```python
# Velocity limits
for i in range(N+1):
    opti.subject_to(opti.bounded(-v_max, v[i], v_max))

# Acceleration limits
for i in range(N+1):
    opti.subject_to(opti.bounded(-a_max, a[i], a_max))

# Jerk limits
for i in range(N):
    opti.subject_to(opti.bounded(-j_max, j[i], j_max))
```

### Physics Constraints

```python
# Compression ratio constraints
clearance = 0.002  # 2mm clearance
for i in range(N+1):
    CR = (x[i] + clearance) / clearance
    opti.subject_to(CR >= CR_min)
    opti.subject_to(CR <= CR_max)

# Pressure rate constraints
for i in range(N-1):
    pressure_rate = abs(a[i+1] - a[i]) / (T/N)
    opti.subject_to(pressure_rate <= P_rate_max)
```

### Thermal Efficiency Objective

```python
# Otto cycle efficiency
gamma = 1.4
CR = (x + clearance) / clearance
eta_otto = 1 - 1 / (CR ** (gamma - 1))

# Heat loss penalty
h = 0.1  # Heat transfer coefficient
Q_loss = h * v**2

# Mechanical losses
f = 0.01  # Friction coefficient
c = 0.05  # Viscous damping
P_mech = f * v**2 + c * a**2

# Total thermal efficiency
eta_thermal = eta_otto - Q_loss - P_mech

# Objective function
J_jerk = sum(j**2) * (T/N)  # Jerk squared integral
J_thermal = -sum(eta_thermal) / (N+1)  # Negative efficiency

opti.minimize(w_jerk * J_jerk + w_thermal * J_thermal)
```

### Solver Configuration

```python
# Set solver options
opti.solver('ipopt', {
    'ipopt.linear_solver': 'ma57',
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6,
    'ipopt.print_level': 0,
    'ipopt.warm_start_init_point': 'yes'
})

# Solve
sol = opti.solve()
```

## Solution Extraction

### State Variables

```python
# Extract solution
x_opt = sol.value(x)
v_opt = sol.value(v)  
a_opt = sol.value(a)
j_opt = sol.value(j)
```

### Time Vector

```python
# Create time vector
t = np.linspace(0, T, N+1)
```

### Thermal Efficiency Evaluation

```python
# Evaluate thermal efficiency
CR_opt = (x_opt + clearance) / clearance
eta_otto_opt = 1 - 1 / (CR_opt ** (gamma - 1))
Q_loss_opt = h * v_opt**2
P_mech_opt = f * v_opt**2 + c * a_opt**2
eta_thermal_opt = eta_otto_opt - Q_loss_opt - P_mech_opt

# Average thermal efficiency
eta_avg = np.mean(eta_thermal_opt)
```

## Performance Metrics

### Convergence Criteria

- **Optimality tolerance**: `1e-6`
- **Feasibility tolerance**: `1e-6`
- **Maximum iterations**: `1000`

### Expected Performance

- **Convergence time**: 3-5x faster with warm-starting
- **Thermal efficiency**: 55% target achievable
- **Solution quality**: Smooth motion profiles with minimal jerk

### Solver Statistics

```python
# Get solver statistics
stats = sol.stats()
print(f"Solve time: {stats['t_wall_total']:.3f}s")
print(f"Iterations: {stats['iter_count']}")
print(f"Objective value: {sol.value(opti.f):.6f}")
```

## Validation

### Solution Verification

1. **Boundary conditions**: Verify `x(0) = 0`, `x(T) = stroke`, `v(0) = v(T) = 0`
2. **Constraint satisfaction**: Check all motion and physics constraints
3. **Thermal efficiency**: Verify efficiency is within target range
4. **Smoothness**: Check jerk profile for excessive oscillations

### Benchmarking

1. **Convergence time**: Compare with analytical solutions
2. **Solution quality**: Compare with reference trajectories
3. **Thermal efficiency**: Compare with literature values
4. **Robustness**: Test across parameter ranges

## Future Extensions

### Multiple Shooting

For stiff cases, multiple shooting can be implemented as a fallback:

```python
# Multiple shooting setup (future implementation)
def setup_multiple_shooting():
    # Divide time horizon into shooting intervals
    # Add continuity constraints at interval boundaries
    # Use sensitivity analysis for gradients
    pass
```

### Advanced Physics Models

1. **1D gas dynamics**: Detailed compressible flow modeling
2. **Heat transfer**: Full Woschni correlation implementation
3. **Combustion**: HCCI/SI/CI mode-specific models
4. **Mechanical losses**: Detailed friction and damping models

### Machine Learning Integration

1. **Initial guess prediction**: ML-based parameter matching
2. **Efficiency prediction**: Neural network models
3. **Adaptive weights**: Dynamic objective weighting based on performance

