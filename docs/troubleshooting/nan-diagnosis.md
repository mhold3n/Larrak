# NaN Diagnosis and Resolution Guide

This document consolidates all NaN-related analysis and troubleshooting for the Larrak optimization system.

## Problem Summary

### Symptoms
- Test failed with `Invalid_Number_Detected`
- Condition number was `nan`
- Optimization failed immediately at 0 iterations
- CasADi reports: `"NaN detected for output jac_g_x, at nonzero index 219 (row 5, col 20)"`

### Root Cause
The NaN originates from **CasADi NLP Jacobian evaluation**, not from scaling code:
- **Row 5**: Temperature collocation residual constraint (`T_c - rhs_T`)
- **Col 20**: Velocity variable (from velocities group)
- **Constraint Type**: `collocation_residuals`

---

## Technical Analysis

### Constraint Structure
- Row 5: `T_c - rhs_T = 0` (temperature collocation residual)
- `rhs_T = T_k + h * sum(grid.a[c][j] * dT_dt)` (accumulated over collocation points)
- `dT_dt` comes from `gas_energy_balance()` function

### Derivative Chain
When differentiating `d(T_c - rhs_T)/d(v)`, CasADi must compute:
```
d(rhs_T)/d(v) = d(T_k)/d(v) + h * sum(grid.a[c][j] * d(dT_dt)/d(v))
```

The problematic term is `d(dT_dt)/d(v)`, which involves:
- `dT_dt = (numerator) / (m_safe * cv)`
- `numerator = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV_dt - e*dm_dt`

### Circular Dependency
- `dT_dt` depends on `mdot_in` and `mdot_out`
- `mdot_in` and `mdot_out` depend on `T_c`
- `T_c` is constrained by `T_c - rhs_T = 0` where `rhs_T` depends on `dT_dt`
- When differentiating this constraint w.r.t. velocity, the circular dependency creates NaN

### Potential NaN Sources
1. **Division by `m_safe * cv`**: If this becomes zero/small during differentiation
2. **Chain rule in `p * dV_dt`**: `d(p * dV_dt)/d(v) = p * d(dV_dt)/d(v) + dV_dt * d(p)/d(v)`
3. **Chain rule in `e * dm_dt`**: Similar issue with internal energy
4. **Temperature-dependent `cv`**: `cv = cp/gamma` where `cp` depends on `T`

---

## Resolution Status

### Completed Fixes

| Component | Status | Description |
|-----------|--------|-------------|
| **Scaling Implementation** | ✅ COMPLETE | All planned features implemented |
| **NaN Detection** | ✅ COMPLETE | Comprehensive checks in `_verify_scaling_quality()` |
| **NaN Sanitization** | ✅ COMPLETE | Clean NaN values in Jacobian before scaling |
| **Condition Number** | ✅ FIXED | Now computed correctly (4.750e+08) |
| **cv Protection** | ✅ DONE | Clamped T, bounded cp/cv |
| **dT_dt Protection** | ✅ DONE | Clamped numerator to prevent Inf |
| **drho_dt Protection** | ✅ DONE | Protected computations |

### Current Results
- **Condition Number**: 4.750e+08 (consistently computed)
- **Quality Score**: 0.667
- **Underscaled Entries**: 453 (11.2%)
- **Overscaled Entries**: 0 (0.0%)

---

## Troubleshooting Steps

### Step 1: Verify Scaling
```python
# Check if scaling is working
from campro.diagnostics import verify_scaling_quality
result = verify_scaling_quality(nlp)
print(f"Condition number: {result.condition_number}")
```

### Step 2: Identify NaN Location
```python
# The NaN is at:
# - Row: constraint index 5 (temperature collocation residual)
# - Col: variable index 20 (velocity variable)
# Check build_collocation_nlp for this constraint
```

### Step 3: Add Term-by-Term Diagnostics
Terms to check in `dT_dt`:
- `Q_combustion`
- `Q_heat_transfer`
- `mdot_in * h_in`
- `mdot_out * h_out`
- `p * dV_dt`
- `e * dm_dt`
- `m_safe * cv` (denominator)

---

## Fix Options

### Option A: Simplify Mass Flow Derivatives
If mass flow derivatives are problematic:
- Use constant mass flow rates for derivative purposes
- Add stronger regularization to pressure ratio computations

### Option B: Break Circular Dependency
If circular dependency is the issue:
- Reformulate constraint to break the dependency
- Use implicit formulation: `energy_balance = 0` instead of `T_c - rhs_T = 0`

### Option C: Stronger Denominator Protection
If denominator derivative is the issue:
```python
# Use stronger regularization
m_cv_safe = m_safe * cv + epsilon_reg  # where epsilon_reg is larger
# Or use:
ca.fmax(m_safe * cv, epsilon_min)  # with larger epsilon_min
```

### Option D: Alternative Constraint Formulation
Instead of:
```python
rhs_T = T_k + h * sum(grid.a[c][j] * dT_dt)
constraint: T_c - rhs_T = 0
```

Use:
```python
# Direct energy balance constraint
energy_residual = m * cv * (T_c - T_k) / h - sum(grid.a[c][j] * dT_dt)
constraint: energy_residual = 0
```

---

## Comparison Table

| Metric | Baseline (1e-6) | Tight (1e-8) | Current |
|--------|----------------|--------------|---------|
| **Condition Number** | ~4.75e8 | ~4.75e8 | 4.750e+08 ✅ |
| **Computation** | Valid | Valid | Valid ✅ |
| **NaN in Scaling** | None | None | None ✅ |
| **NaN in NLP** | Unknown | Unknown | Documented ✅ |
| **Optimization** | Success | Failed | Failed (NLP issue) |

---

## Next Steps

### For NLP Issue
The NaN in the NLP Jacobian needs to be fixed in the NLP formulation:
1. Investigate constraint at row 5, col 20 in `build_collocation_nlp`
2. Check for division by zero or invalid operations in constraint definitions
3. Add bounds checking or regularization to prevent NaN generation

### Success Criteria
- [ ] No NaN in Jacobian evaluation
- [ ] Optimization can start (at least 1 iteration)
- [ ] Condition number remains computable
- [ ] Optimization converges at tol=1e-8 (ultimate goal)






