# CasADi Physics API Documentation

## Overview

This document specifies the complete API contract for the CasADi physics port. All implementations must conform to these specifications to ensure consistency and prevent goal/implementation drift across development sessions.

### Purpose

Port core physics computations (torque, side loading, Litvin gear metrics) from Python/NumPy to CasADi symbolic expressions with automatic differentiation support. This enables direct integration into NLP optimizers (Ipopt) with exact gradients, eliminating hybrid validation overhead and improving performance.

### Design Principles

1. **CasADi MX Exclusively**: All symbolic expressions use `casadi.MX` type (optimal for large-scale NLP with Ipopt)
2. **Units Convention**: 
   - Angles: radians
   - Linear dimensions: millimeters (mm) - explicit conversion where needed
   - Forces: Newtons (N)
   - Torque: Newton-meters (N⋅m)
   - Pressure: Pascals (Pa)
3. **Domain Safety**: Guard all potentially singular operations with epsilon thresholds
4. **Vectorization**: Accept vector inputs (`MX(n,1)`) and use elementwise operations
5. **Type Annotations**: All public functions have complete type hints
6. **Logging**: Use `campro.logging.get_logger(__name__)` exclusively

### Architectural Shift: Hybrid → Full CasADi

**Current (Hybrid)**:
- Optimizer wraps Python physics in objective/constraint callbacks
- Each evaluation: Python loop → extract scalar → return to Ipopt
- Gradients via finite differences (slow, inaccurate ~1e-6)
- No expression graph visibility

**Proposed (Full CasADi)**:
- Physics as symbolic MX graphs built once
- NLP contains full expression tree
- Automatic differentiation: exact gradients at zero extra cost
- Ipopt exploits sparsity patterns
- Expected 2-3x speedup + improved convergence

**Toggle**: `campro.constants.USE_CASADI_PHYSICS` (default `False` until validated)

---

## Constants

Defined in `campro/constants.py`:

```python
# CasADi physics integration
USE_CASADI_PHYSICS: bool = False
    # Feature toggle: enable CasADi physics in optimizers (default disabled)

CASADI_PHYSICS_EPSILON: float = 1e-12
    # Domain guard epsilon for sqrt, division, etc.
    # Usage: sqrt(fmax(value, CASADI_PHYSICS_EPSILON))

CASADI_PHYSICS_ASIN_CLAMP: float = 0.999999
    # Clamp value for arcsin input to avoid domain errors
    # Usage: arcsin(fmin(fmax(value, -clamp), clamp))
```

---

## Module: `campro/physics/casadi/kinematics.py`

### `create_crank_piston_kinematics() -> ca.Function`

Compute crank-piston kinematics with crank center offset effects.

**Signature**:
```python
kinematics_fn = create_crank_piston_kinematics()
x, v, a, rod_angle, r_eff = kinematics_fn(theta, r, l, x_off, y_off)
```

**Inputs**:
- `theta: MX(n,1)` - Crank angles (radians)
- `r: MX` - Crank radius (mm)
- `l: MX` - Connecting rod length (mm)
- `x_off: MX` - Crank center x-offset from gear center (mm)
- `y_off: MX` - Crank center y-offset from gear center (mm)

**Outputs**:
- `x: MX(n,1)` - Piston displacement (mm)
- `v: MX(n,1)` - Piston velocity (mm/rad)
- `a: MX(n,1)` - Piston acceleration (mm/rad²)
- `rod_angle: MX(n,1)` - Connecting rod angle (radians)
- `r_eff: MX(n,1)` - Effective crank radius accounting for offset (mm)

**Formulas**:
```
# Rod angle with offset correction
effective_crank_angle = theta + arctan2(y_off, x_off)
sin_arg = (r / l) * sin(effective_crank_angle)
sin_arg_clamped = fmin(fmax(sin_arg, -ASIN_CLAMP), ASIN_CLAMP)
rod_angle = arcsin(sin_arg_clamped)

# Position with offset correction
offset_correction = x_off * cos(theta) + y_off * sin(theta)
radicand = l² - (r * sin(theta))²
radicand_safe = fmax(radicand, EPSILON)
x = offset_correction + r * cos(theta) + sqrt(radicand_safe)

# Velocity and acceleration via automatic differentiation
v = jacobian(x, theta)
a = jacobian(v, theta)

# Effective radius (simplified - mark for future enhancement)
r_eff = r  # TODO: account for offset effects on effective radius
```

**Domain Guards**:
- `|r·sin(θ)/l| ≤ ASIN_CLAMP` to avoid arcsin domain errors
- `l² - (r·sin(θ))² ≥ EPSILON` to avoid sqrt of negative

**Usage Example**:
```python
import casadi as ca
from campro.physics.casadi.kinematics import create_crank_piston_kinematics

kin_fn = create_crank_piston_kinematics()

# Scalar case
theta = ca.DM([0.0, ca.pi/4, ca.pi/2])
r, l = 50.0, 150.0  # mm
x_off, y_off = 5.0, 2.0  # mm offset
x, v, a, rod_angle, r_eff = kin_fn(theta, r, l, x_off, y_off)
```

---

### `create_phase_masks() -> ca.Function`

Detect expansion and compression phases from displacement profile.

**Signature**:
```python
mask_fn = create_phase_masks()
expansion_mask, compression_mask = mask_fn(displacement)
```

**Inputs**:
- `displacement: MX(n,1)` - Radial position of piston/planet (mm)

**Outputs**:
- `expansion_mask: MX(n,1)` - 1 where displacement increasing, 0 otherwise
- `compression_mask: MX(n,1)` - 1 where displacement decreasing, 0 otherwise

**Logic**:
```
# Compute differences (expand adds leading 0)
diff = displacement[1:] - displacement[:-1]
expansion = (diff > 0)  # boolean
compression = (diff < 0)  # boolean

# Convert to numeric 0/1 and pad to original length
expansion_mask = vertcat(0, if_else(expansion, 1, 0))
compression_mask = vertcat(0, if_else(compression, 1, 0))
```

**Usage Example**:
```python
mask_fn = create_phase_masks()
displacement = ca.DM([100, 110, 120, 115, 105])  # mm
exp_mask, comp_mask = mask_fn(displacement)
# exp_mask ≈ [0, 1, 1, 0, 0]
# comp_mask ≈ [0, 0, 0, 1, 1]
```

---

## Module: `campro/physics/casadi/forces.py`

### `create_piston_force_simple() -> ca.Function`

Compute piston force from gas pressure and bore diameter.

**Signature**:
```python
force_fn = create_piston_force_simple()
F = force_fn(pressure, bore)
```

**Inputs**:
- `pressure: MX(n,1)` - Gas pressure (Pa)
- `bore: MX` - Cylinder bore diameter (mm)

**Outputs**:
- `F: MX(n,1)` - Piston force (N)

**Formula**:
```
area_mm2 = pi * (bore / 2)²
area_m2 = area_mm2 * 1e-6  # mm² to m²
F = pressure * area_m2
```

**Units Note**: Pressure in Pa, bore in mm → Force in N (SI consistent)

**Usage Example**:
```python
from campro.physics.casadi.forces import create_piston_force_simple

force_fn = create_piston_force_simple()
pressure = ca.DM([1e5, 2e5, 3e5])  # Pa
bore = 100.0  # mm
F = force_fn(pressure, bore)
```

---

## Module: `campro/physics/casadi/torque.py`

### `create_torque_pointwise() -> ca.Function`

Compute instantaneous torque at single crank angle.

**Signature**:
```python
torque_fn = create_torque_pointwise()
T = torque_fn(theta, F, r, rod_angle, pressure_angle)
```

**Inputs**:
- `theta: MX` - Crank angle (radians)
- `F: MX` - Piston force (N)
- `r: MX` - Effective crank radius (mm)
- `rod_angle: MX` - Connecting rod angle (radians)
- `pressure_angle: MX` - Gear pressure angle (radians)

**Outputs**:
- `T: MX` - Instantaneous torque (N⋅m)

**Formula** (from `torque_analysis.py:186-187`):
```
r_m = r * 1e-3  # mm to m
T = F * r_m * sin(theta + rod_angle) * cos(pressure_angle)
```

**Usage Example**:
```python
from campro.physics.casadi.torque import create_torque_pointwise

T_fn = create_torque_pointwise()
T = T_fn(ca.pi/4, 1000.0, 50.0, 0.1, ca.pi/9)
```

---

### `create_torque_profile() -> ca.Function`

Compute torque profile with statistics over full cycle.

**Signature**:
```python
profile_fn = create_torque_profile()
T_vec, T_avg, T_max, T_min, ripple = profile_fn(theta_vec, F_vec, r, l, x_off, y_off, pressure_angle)
```

**Inputs**:
- `theta_vec: MX(n,1)` - Crank angles (radians)
- `F_vec: MX(n,1)` - Piston forces (N)
- `r: MX` - Crank radius (mm)
- `l: MX` - Connecting rod length (mm)
- `x_off: MX` - Crank center x-offset (mm)
- `y_off: MX` - Crank center y-offset (mm)
- `pressure_angle: MX` - Gear pressure angle (radians)

**Outputs**:
- `T_vec: MX(n,1)` - Torque at each angle (N⋅m)
- `T_avg: MX` - Cycle-averaged torque via trapezoidal integration (N⋅m)
- `T_max: MX` - Maximum torque (N⋅m)
- `T_min: MX` - Minimum torque (N⋅m)
- `ripple: MX` - Torque variation coefficient (dimensionless)

**Formulas**:
```
# Compute kinematics for all points
_, _, _, rod_angle_vec, r_eff_vec = create_crank_piston_kinematics()(theta_vec, r, l, x_off, y_off)

# Torque at each point
T_vec[i] = create_torque_pointwise()(theta_vec[i], F_vec[i], r_eff_vec[i], rod_angle_vec[i], pressure_angle)

# Trapezoidal integration for average
dtheta = theta_vec[1:] - theta_vec[:-1]
T_mid = 0.5 * (T_vec[:-1] + T_vec[1:])
T_avg = sum1(T_mid * dtheta) / (theta_vec[-1] - theta_vec[0])

# Statistics
T_max = mmax(T_vec)
T_min = mmin(T_vec)
T_mean = sum1(T_vec) / n
T_std = sqrt(sum1((T_vec - T_mean)²) / n)
ripple = T_std / fmax(fabs(T_mean), EPSILON)
```

**Usage Example**:
```python
from campro.physics.casadi.torque import create_torque_profile

profile_fn = create_torque_profile()
theta = ca.DM(np.linspace(0, 2*np.pi, 100))
F = ca.DM(1000.0 * np.ones(100))
T_vec, T_avg, T_max, T_min, ripple = profile_fn(theta, F, 50.0, 150.0, 5.0, 2.0, np.radians(20))
```

---

## Module: `campro/physics/casadi/side_loading.py`

### `create_side_load_pointwise() -> ca.Function`

Compute instantaneous side load at single crank angle.

**Signature**:
```python
side_load_fn = create_side_load_pointwise()
F_side = side_load_fn(theta, F, r, l, x_off, y_off)
```

**Inputs**:
- `theta: MX` - Crank angle (radians)
- `F: MX` - Piston force (N)
- `r: MX` - Crank radius (mm)
- `l: MX` - Connecting rod length (mm)
- `x_off: MX` - Crank center x-offset (mm)
- `y_off: MX` - Crank center y-offset (mm)

**Outputs**:
- `F_side: MX` - Lateral force on piston (N)

**Formula** (from `side_loading.py:306-316`):
```
# Compute rod angle
effective_crank_angle = theta + arctan2(y_off, x_off)
sin_arg = (r / l) * sin(effective_crank_angle)
sin_arg_clamped = fmin(fmax(sin_arg, -ASIN_CLAMP), ASIN_CLAMP)
rod_angle = arcsin(sin_arg_clamped)

# Side load (simplified - empirical factors marked TODO)
F_side = F * tan(rod_angle)

# TODO: Investigate and potentially add physics-based offset/clearance corrections
# Current Python implementation uses empirical factors that lack justification
```

**Usage Example**:
```python
from campro.physics.casadi.side_loading import create_side_load_pointwise

F_side_fn = create_side_load_pointwise()
F_side = F_side_fn(ca.pi/4, 1000.0, 50.0, 150.0, 5.0, 2.0)
```

---

### `create_side_load_profile() -> ca.Function`

Compute side load profile with statistics.

**Signature**:
```python
profile_fn = create_side_load_profile()
F_side_vec, F_side_max, F_side_avg, ripple = profile_fn(theta_vec, F_vec, r, l, x_off, y_off)
```

**Inputs**:
- `theta_vec: MX(n,1)` - Crank angles (radians)
- `F_vec: MX(n,1)` - Piston forces (N)
- `r: MX` - Crank radius (mm)
- `l: MX` - Connecting rod length (mm)
- `x_off: MX` - Crank center x-offset (mm)
- `y_off: MX` - Crank center y-offset (mm)

**Outputs**:
- `F_side_vec: MX(n,1)` - Side load at each angle (N)
- `F_side_max: MX` - Maximum absolute side load (N)
- `F_side_avg: MX` - Average absolute side load (N)
- `ripple: MX` - Side load variation coefficient (dimensionless)

**Formulas**:
```
# Compute side load at each point
F_side_vec = [create_side_load_pointwise()(theta_vec[i], F_vec[i], r, l, x_off, y_off) for i in range(n)]

# Statistics
F_side_abs = fabs(F_side_vec)
F_side_max = mmax(F_side_abs)
F_side_avg = sum1(F_side_abs) / n
F_side_std = sqrt(sum1((F_side_vec - sum1(F_side_vec)/n)²) / n)
ripple = F_side_std / fmax(F_side_avg, EPSILON)
```

---

### `create_side_load_penalty() -> ca.Function`

Compute smooth penalty for side loading with phase-specific weighting.

**Signature**:
```python
penalty_fn = create_side_load_penalty()
penalty = penalty_fn(F_side_vec, max_threshold, compression_mask, combustion_mask)
```

**Inputs**:
- `F_side_vec: MX(n,1)` - Side load profile (N)
- `max_threshold: MX` - Maximum allowable side load (N)
- `compression_mask: MX(n,1)` - 0/1 mask for compression phases
- `combustion_mask: MX(n,1)` - 0/1 mask for combustion phases (if different from compression)

**Outputs**:
- `penalty: MX` - Combined side load penalty (dimensionless)

**Formulas**:
```
# Smooth ReLU (differentiable everywhere)
def relu_smooth(x):
    return 0.5 * (x + sqrt(x² + EPSILON))

# Exceedance
F_side_abs = fabs(F_side_vec)
exceedance = relu_smooth(F_side_abs - max_threshold)

# Phase weights (from side_loading.py:205-207)
compression_weight = 1.2
combustion_weight = 1.5
general_weight = 1.0

# Weighted penalties
general_penalty = general_weight * sum1(exceedance²) / n
compression_penalty = compression_weight * sum1(exceedance² * compression_mask) / fmax(sum1(compression_mask), 1)
combustion_penalty = combustion_weight * sum1(exceedance² * combustion_mask) / fmax(sum1(combustion_mask), 1)

penalty = general_penalty + compression_penalty + combustion_penalty
```

**Usage Example**:
```python
from campro.physics.casadi.side_loading import create_side_load_penalty

penalty_fn = create_side_load_penalty()
F_side_vec = ca.DM([100, 200, 300, 400, 500])  # N
max_threshold = 250.0  # N
comp_mask = ca.DM([0, 0, 1, 1, 0])
comb_mask = ca.DM([0, 0, 0, 1, 1])
penalty = penalty_fn(F_side_vec, max_threshold, comp_mask, comb_mask)
```

---

## Module: `campro/physics/casadi/litvin.py`

**Note**: This module is the most complex due to Litvin gear contact kinematics and Newton solver requirements.

### `create_internal_flank_sampler() -> ca.Function`

Sample involute internal gear flank coordinates.

**Signature**:
```python
flank_fn = create_internal_flank_sampler(n_samples=256)
phi_vec, x_vec, y_vec = flank_fn(z_r, module, alpha_deg, addendum_factor)
```

**Inputs**:
- `z_r: MX` - Ring gear teeth count
- `module: MX` - Gear module (mm)
- `alpha_deg: MX` - Pressure angle (degrees)
- `addendum_factor: MX` - Addendum coefficient

**Outputs**:
- `phi_vec: MX(n_samples,1)` - Flank parameter values (radians)
- `x_vec: MX(n_samples,1)` - Flank x-coordinates (mm)
- `y_vec: MX(n_samples,1)` - Flank y-coordinates (mm)

**Based on**: `campro/litvin/involute_internal.py:sample_internal_flank`

---

### `create_planet_transform() -> ca.Function`

Transform planet gear coordinates to ring gear frame.

**Signature**:
```python
transform_fn = create_planet_transform()
x_planet, y_planet, tangent_x, tangent_y = transform_fn(phi, theta_r, R0, motion_params)
```

**Inputs**:
- `phi: MX` - Flank parameter (radians)
- `theta_r: MX` - Ring gear rotation angle (radians)
- `R0: MX` - Base center radius (mm)
- `motion_params: Dict` - Motion law coefficients

**Outputs**:
- `x_planet: MX` - Planet x-coordinate in ring frame (mm)
- `y_planet: MX` - Planet y-coordinate in ring frame (mm)
- `tangent_x: MX` - Tangent vector x-component
- `tangent_y: MX` - Tangent vector y-component

**Based on**: `campro/litvin/kinematics.py:PlanetKinematics`

---

### `create_contact_phi_solver() -> ca.Function`

Solve for contact point phi using Newton iteration.

**Signature**:
```python
solver_fn = create_contact_phi_solver()
phi_contact = solver_fn(flank_data, theta_r, phi_seed, R0, motion_params)
```

**Inputs**:
- `flank_data: Dict[str, MX]` - Flank sampling data (phi, x, y)
- `theta_r: MX` - Ring gear angle (radians)
- `phi_seed: MX` - Initial guess for phi (radians)
- `R0: MX` - Base center radius (mm)
- `motion_params: Dict` - Motion law coefficients

**Outputs**:
- `phi_contact: MX` - Contact point phi (radians)

**Implementation Note**: Use `ca.rootfinder` with Newton method. If unstable, fallback to interpolation over fixed phi grid.

**Based on**: `campro/litvin/planetary_synthesis.py:_newton_solve_phi`

---

### `create_litvin_metrics() -> ca.Function`

Compute Litvin gear metrics: slip integral, contact length, closure residual.

**Signature**:
```python
metrics_fn = create_litvin_metrics()
slip_integral, contact_length, closure, objective = metrics_fn(config_params, theta_vec)
```

**Inputs**:
- `config_params: Dict` - Configuration containing:
  - `ring_teeth: MX`
  - `planet_teeth: MX`
  - `module: MX`
  - `pressure_angle_deg: MX`
  - `base_center_radius: MX`
  - `motion_coeffs: Dict`
  - `addendum_factor: MX`
- `theta_vec: MX(n,1)` - Ring gear angles to evaluate (radians)

**Outputs**:
- `slip_integral: MX` - Accumulated sliding metric (dimensionless)
- `contact_length: MX` - Total contact path length (mm)
- `closure: MX` - Path closure residual (mm)
- `objective: MX` - Combined objective = slip - 0.1*length + penalty

**Formulas** (from `metrics.py:99-108`):
```
# Sample flank
phi_vec, x_flank, y_flank = create_internal_flank_sampler()(z_r, module, alpha_deg, addendum_factor)

# For each theta_r in theta_vec:
for i, theta_r in enumerate(theta_vec):
    # Solve contact point
    phi_contact[i] = create_contact_phi_solver()(flank_data, theta_r, phi_seed, R0, motion_params)
    
    # Compute contact coordinates and tangent
    x_planet[i], y_planet[i], tx[i], ty[i] = create_planet_transform()(phi_contact[i], theta_r, R0, motion_params)
    
    # Compute slip via finite difference
    dtheta_vec = partial derivative of (x_planet, y_planet) w.r.t. theta_r
    t_norm = sqrt(tx²+ty² + EPSILON)
    slip[i] = |dtheta_vec[0]*tx/t_norm + dtheta_vec[1]*ty/t_norm|

# Integrate slip
dtheta = 2*pi / n
slip_integral = sum(slip) * dtheta

# Contact length (polyline)
contact_length = sum(sqrt((x_planet[i+1]-x_planet[i])² + (y_planet[i+1]-y_planet[i])²))

# Closure residual
closure = sqrt((x_planet[0]-x_planet[-1])² + (y_planet[0]-y_planet[-1])²)

# Objective with penalty
closure_penalty = 1e6 * relu_smooth(closure - PROFILE_CLOSURE_TOL)²
objective = slip_integral - 0.1*contact_length + closure_penalty
```

**Based on**: `campro/litvin/metrics.py:evaluate_order0_metrics`

---

## Module: `campro/physics/casadi/unified.py`

### `create_unified_physics() -> ca.Function`

Unified physics function combining torque, side loading, and optional Litvin metrics.

**Signature**:
```python
unified_fn = create_unified_physics()
results = unified_fn(theta_vec, input_vec, input_type, params)
```

**Inputs**:
- `theta_vec: MX(n,1)` - Crank angles (radians)
- `input_vec: MX(n,1)` - Either pressure (Pa) or force (N), depending on `input_type`
- `input_type: str` - "pressure" or "force"
- `params: Dict` - Contains:
  - `r: MX` - Crank radius (mm)
  - `l: MX` - Rod length (mm)
  - `x_off: MX` - Crank x-offset (mm)
  - `y_off: MX` - Crank y-offset (mm)
  - `bore: MX` - Bore diameter (mm) [if input_type="pressure"]
  - `pressure_angle: MX` - Gear pressure angle (radians)
  - `max_side_load: MX` - Side load threshold (N)
  - `litvin_config: Optional[Dict]` - Litvin parameters (if metrics desired)

**Outputs** (as Dict or struct):
- `torque_avg: MX` - Average torque (N⋅m)
- `torque_ripple: MX` - Torque variation coefficient
- `side_load_penalty: MX` - Combined side load penalty
- `litvin_objective: MX` - Litvin objective (if `litvin_config` provided, else 0)

**Logic**:
```python
# Convert pressure to force if needed
if input_type == "pressure":
    F_vec = create_piston_force_simple()(input_vec, params['bore'])
else:
    F_vec = input_vec

# Compute torque profile
_, T_avg, _, _, T_ripple = create_torque_profile()(
    theta_vec, F_vec, params['r'], params['l'], 
    params['x_off'], params['y_off'], params['pressure_angle']
)

# Compute side load profile and penalty
F_side_vec, _, _, _ = create_side_load_profile()(
    theta_vec, F_vec, params['r'], params['l'], 
    params['x_off'], params['y_off']
)

# Generate phase masks
x_vec, _, _, _, _ = create_crank_piston_kinematics()(
    theta_vec, params['r'], params['l'], params['x_off'], params['y_off']
)
expansion_mask, compression_mask = create_phase_masks()(x_vec)

# Compute side load penalty
side_load_penalty = create_side_load_penalty()(
    F_side_vec, params['max_side_load'], compression_mask, compression_mask  # using compression for combustion
)

# Optionally compute Litvin metrics
if params['litvin_config'] is not None:
    _, _, _, litvin_objective = create_litvin_metrics()(params['litvin_config'], theta_vec)
else:
    litvin_objective = 0.0

return {
    'torque_avg': T_avg,
    'torque_ripple': T_ripple,
    'side_load_penalty': side_load_penalty,
    'litvin_objective': litvin_objective
}
```

**Usage in Optimizer**:
```python
from campro import constants
from campro.physics.casadi.unified import create_unified_physics

if constants.USE_CASADI_PHYSICS:
    # Build symbolic NLP
    unified_fn = create_unified_physics()
    
    # Define symbolic variables
    r = ca.MX.sym('r')
    l = ca.MX.sym('l')
    x_off = ca.MX.sym('x_off')
    y_off = ca.MX.sym('y_off')
    
    params = {
        'r': r, 'l': l, 'x_off': x_off, 'y_off': y_off,
        'bore': 100.0, 'pressure_angle': ca.pi/9,
        'max_side_load': 500.0, 'litvin_config': None
    }
    
    results = unified_fn(theta_vec_data, pressure_vec_data, "pressure", params)
    
    # Build NLP objective
    objective = -results['torque_avg'] + 0.1*results['side_load_penalty']
    
    # Pass symbolic expressions to Ipopt...
else:
    # Use existing Python physics
    ...
```

---

## Integration Points

### `campro/optimization/crank_center_optimizer.py`

**Modification**: In `_define_objective` and `_define_constraints` methods:

```python
from campro import constants
from campro.physics.casadi import unified

def _define_objective(self, ...):
    if constants.USE_CASADI_PHYSICS:
        # Build symbolic CasADi NLP
        unified_fn = unified.create_unified_physics()
        # ... construct MX variables for (r, l, x_off, y_off)
        # ... call unified_fn with symbolic inputs
        # ... return symbolic objective expression
    else:
        # Existing Python callback approach
        def objective_callback(x):
            # ... Python simulation loop
            return scalar_value
        return objective_callback
```

**Key Difference**: CasADi mode returns **symbolic expression**, Python mode returns **function callback**.

---

## Testing Strategy

### Phase 1 Tests (`tests/test_casadi_phase1.py`)
- Parity: CasADi vs Python `PistonTorqueCalculator` (rtol=1e-6)
- Gradients: AD vs finite differences (rtol=1e-5)
- Domain safety: no NaN/Inf near singularities

### Phase 2 Tests (`tests/test_casadi_phase2.py`)
- Parity: CasADi vs Python `SideLoadAnalyzer` (rtol=1e-6)
- Penalty monotonicity and differentiability

### Phase 3 Tests (`tests/test_casadi_phase3.py`)
- Parity: CasADi vs Python `evaluate_order0_metrics` (rtol=1e-4)
- Gradient existence (no errors)

### Phase 4 Tests (`tests/test_casadi_phase4.py`)
- Toy NLP convergence
- CasADi ON vs OFF parity (rtol=1e-3)
- Performance benchmark (>2x speedup)

---

## Performance Expectations

- **Objective Evaluation**: 2-3x speedup over Python loops
- **Gradient Computation**: Near-zero cost via AD (vs. expensive finite differences)
- **Convergence**: Improved due to exact gradients
- **Memory**: Slightly higher (expression graphs), but negligible for this problem size

---

## Future Enhancements

1. **Offset Effects** (TODO in kinematics/side_loading):
   - Investigate physics basis for empirical offset/clearance factors
   - Implement validated corrections for effective radius and side load

2. **Litvin Solver Robustness**:
   - Improve Newton rootfinder initialization strategy
   - Add fallback to collocation-based contact solver

3. **Inertia Forces**:
   - Extend force calculation to include piston/rod inertia
   - Port from `campro/freepiston/core/piston.py:piston_force_balance`

4. **Multi-Piston Support**:
   - Vectorize over multiple pistons for inline engines
   - Sum torques accounting for phase offsets

---

## References

- Python implementations:
  - `campro/physics/mechanics/torque_analysis.py`
  - `campro/physics/mechanics/side_loading.py`
  - `campro/physics/kinematics/crank_kinematics.py`
  - `campro/litvin/metrics.py`
- CasADi documentation: https://web.casadi.org/docs/
- Ipopt solver: https://coin-or.github.io/Ipopt/

