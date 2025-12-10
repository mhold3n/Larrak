# CasADi Symbolic Differentiation Analysis

## Problem Statement
NaN detected in Jacobian at row 5 (temperature collocation residual) with respect to col 20 (velocity variable) during CasADi's symbolic differentiation.

## Root Cause Analysis

### The Problematic Expression
The constraint is: `T_c - rhs_T = 0`

Where `rhs_T` is accumulated as:
```python
rhs_T = T_k
for j in range(C):
    rhs_T += h * grid.a[c][j] * dT_dt
```

And `dT_dt` comes from `gas_energy_balance()`:
```python
dT_dt = (
    Q_combustion - Q_heat_transfer
    + mdot_in * h_in - mdot_out * h_out
    - p * dV_dt
    - e * dm_dt
) / (m_safe * cv)
```

### Why Symbolic Differentiation Fails
When CasADi differentiates `d(T_c - rhs_T)/d(v)`, it must compute:
```
d(rhs_T)/d(v) = d(T_k)/d(v) + h * sum(grid.a[c][j] * d(dT_dt)/d(v))
```

The problematic term is `d(dT_dt)/d(v)`, which involves:
1. **Chain rule through `p * dV_dt`**: 
   - `d(p * dV_dt)/d(v) = p * d(dV_dt)/d(v) + dV_dt * d(p)/d(v)`
   - `p = rho * R * T` depends on `T`, which may have complex derivatives
   
2. **Chain rule through `e * dm_dt`**:
   - `d(e * dm_dt)/d(v) = e * d(dm_dt)/d(v) + dm_dt * d(e)/d(v)`
   - `e = cv * T` where `cv` depends on `T` (temperature-dependent specific heat)
   
3. **Division by `m_safe * cv`**:
   - When differentiating `f(x)/g(x)`, we get: `(f'*g - f*g')/g²`
   - If `g` becomes very small or has problematic derivatives, this can produce NaN

4. **Temperature-dependent `cv`**:
   - `cv = cp/gamma` where `cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))`
   - Differentiating through this creates: `dcv/dT = cp_ref * 0.0001 / gamma`
   - When this appears in the chain rule for `d(dT_dt)/d(v)`, it can create complex nested derivatives

## Is This a CasADi Limitation?

### CasADi's Approach
CasADi uses **automatic differentiation (AD)** which is:
- **Exact** (not approximate like finite differences)
- **Efficient** (computes derivatives at similar cost to function evaluation)
- **Symbolic** (builds expression graphs, then evaluates)

### Known Limitations
1. **Expression Complexity**: Very complex expressions with many nested operations can produce large expression graphs
2. **Numerical Stability**: AD preserves numerical issues - if the original expression can produce NaN, so can its derivative
3. **Division by Zero**: AD will propagate division-by-zero issues through the derivative chain

### Is This Inherent?
**No, this is NOT an inherent limitation of CasADi's AD.** The issue is that:
- The **original expression** (`dT_dt`) can produce problematic values during differentiation
- The **formulation** creates a complex derivative chain that amplifies numerical issues
- The **safeguards** we added protect numerical evaluation but not symbolic differentiation

## Alternative Approaches

### Option 1: Reformulate the Constraint (RECOMMENDED)
Instead of accumulating `dT_dt` in a loop, use a more direct formulation:

**Current (problematic)**:
```python
rhs_T = T_k
for j in range(C):
    rhs_T += h * grid.a[c][j] * dT_dt  # dT_dt depends on velocity through dV_dt
```

**Alternative 1a: Separate velocity dependence**
```python
# Compute dT_dt without velocity dependence first
dT_dt_base = (Q_combustion - Q_heat_transfer + ...) / (m_safe * cv)
# Then add velocity-dependent term separately with protection
dT_dt_vel = -p * dV_dt / (m_safe * cv)
# Protect the division
dT_dt_vel = ca.fmax(ca.fmin(dT_dt_vel, dT_dt_max), -dT_dt_max)
dT_dt = dT_dt_base + dT_dt_vel
```

**Alternative 1b: Use implicit formulation**
Instead of `T_c - rhs_T = 0`, use a formulation that avoids the accumulation:
```python
# Direct constraint: T_c should satisfy energy balance
# This avoids the complex derivative chain through rhs_T accumulation
```

### Option 2: Use Numerical Differentiation (NOT RECOMMENDED)
IPOPT can use finite differences instead of exact gradients:
- **Pros**: Avoids symbolic differentiation issues
- **Cons**: 
  - Much slower (O(n) function evaluations per iteration)
  - Less accurate (~1e-6 vs machine precision)
  - Doesn't solve the root cause

### Option 3: Simplify the Energy Balance
Reduce the complexity of `dT_dt`:
- Use constant `cv` instead of temperature-dependent
- Simplify the pressure term
- Use simpler mass flow models

### Option 4: Use CasADi's Forward/Reverse Mode
CasADi supports forward and reverse mode AD, but `jacobian()` already uses the most efficient approach. This won't help.

### Option 5: Regularize the Expression
Add small regularization terms to prevent problematic divisions:
```python
# Instead of: dT_dt = numerator / (m_safe * cv)
# Use: dT_dt = numerator / (m_safe * cv + epsilon_reg)
# where epsilon_reg is a small regularization term
```

## Recommended Solution

**Reformulate the constraint to avoid the problematic derivative chain.**

The issue is that `rhs_T` accumulates `dT_dt` which has a complex derivative with respect to velocity. We should:

1. **Protect the division in `dT_dt` at the symbolic level**:
   ```python
   # Ensure denominator is always positive and bounded
   denominator = ca.fmax(m_safe * cv, CASADI_PHYSICS_EPSILON)
   # Use regularized division
   dT_dt = numerator_dT_dt / (denominator + 1e-10)
   ```

2. **Simplify the temperature-dependent `cv`**:
   ```python
   # Use constant cv for now, or use a smoother function
   cv = cv_ref  # Constant, or use ca.fmax(cv, cv_min)
   ```

3. **Protect the accumulation**:
   ```python
   # After accumulation, clamp rhs_T to prevent extreme values
   rhs_T = ca.fmax(ca.fmin(rhs_T, rhs_T_max), rhs_T_min)
   ```

## Conclusion

This is **NOT a fundamental limitation of CasADi's symbolic differentiation**. The issue is in the **formulation** - the expression creates a complex derivative chain that produces NaN. The solution is to **reformulate the constraint** to avoid problematic derivative chains, not to abandon CasADi.

CasADi's AD is the right tool for this problem - we just need to formulate the physics in a way that's numerically stable for differentiation.
