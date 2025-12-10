# NaN Investigation and Implementation Plan

## Current Situation

### Problem
- NaN detected in Jacobian at row 5 (temperature collocation residual) with respect to col 20 (velocity variable)
- Occurs during CasADi's symbolic differentiation
- All numerical safeguards have been added but NaN persists

### Key Findings

1. **Constraint Structure**:
   - Row 5: `T_c - rhs_T = 0` (temperature collocation residual)
   - `rhs_T = T_k + h * sum(grid.a[c][j] * dT_dt)` (accumulated over collocation points)
   - `dT_dt` comes from `gas_energy_balance()` function

2. **Derivative Chain**:
   When differentiating `d(T_c - rhs_T)/d(v)`, CasADi must compute:
   ```
   d(rhs_T)/d(v) = d(T_k)/d(v) + h * sum(grid.a[c][j] * d(dT_dt)/d(v))
   ```
   
   The problematic term is `d(dT_dt)/d(v)`, which involves:
   - `dT_dt = (numerator) / (m_safe * cv)`
   - `numerator = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV_dt - e*dm_dt`
   - `dm_dt = mdot_in - mdot_out`
   - `mdot_in` and `mdot_out` depend on `T_c` and `rho_c_safe` (not directly on velocity)

3. **Critical Insight**:
   - `mdot_in` and `mdot_out` depend on `p_c_safe = rho_c_safe * R * T_c`
   - When differentiating `dT_dt` w.r.t. velocity, we get terms like:
     - `d(p * dV_dt)/d(v) = p * d(dV_dt)/d(v) + dV_dt * d(p)/d(v)`
     - `d(p)/d(v) = d(rho * R * T)/d(v) = R * (rho * d(T)/d(v) + T * d(rho)/d(v))`
   - The term `d(T)/d(v)` creates a **circular dependency**: T depends on dT_dt, which depends on mdot, which depends on T

4. **Mass Flow Dependencies**:
   - `mdot_in` depends on `p_down = p_c_safe = rho_c_safe * R * T_c`
   - `mdot_out` depends on `p_up = p_c_safe = rho_c_safe * R * T_c` and `rho_up = rho_c_safe`
   - These don't directly depend on velocity, BUT:
     - The constraint `T_c - rhs_T = 0` creates an implicit coupling
     - When differentiating this constraint w.r.t. velocity, we get the circular dependency

## Root Cause Hypothesis

The NaN is likely caused by one of these scenarios:

1. **Circular Derivative Dependency**:
   - `dT_dt` depends on `mdot_in` and `mdot_out`
   - `mdot_in` and `mdot_out` depend on `T_c`
   - `T_c` is constrained by `T_c - rhs_T = 0` where `rhs_T` depends on `dT_dt`
   - When differentiating this constraint w.r.t. velocity, the circular dependency creates a 0/0 or Inf-Inf situation

2. **Division by Zero in Derivative**:
   - The derivative of `f(x)/g(x)` is `(f'*g - f*g')/g²`
   - If `g = m_safe * cv` becomes very small during differentiation, `g²` can become extremely small
   - Even with regularization, if `g'` (derivative of denominator) is large, the term `f*g'` can dominate and create issues

3. **Pressure Ratio in Mass Flow**:
   - `mdot_in` and `mdot_out` use pressure ratios `pr = p_down/p_up`
   - These involve fractional powers: `pr^(2/gamma)`, `pr^((gamma+1)/gamma)`
   - When differentiating these w.r.t. velocity (through the T and rho dependencies), the fractional powers can create problematic derivatives

## Implementation Plan

### Phase 1: Deep Diagnostics (IMMEDIATE)

**Goal**: Identify the exact term producing NaN

1. **Add Term-by-Term Diagnostics**:
   - Create a diagnostic function that evaluates each term in `dT_dt` separately
   - Check which term produces NaN when differentiated w.r.t. velocity
   - Terms to check:
     - `Q_combustion`
     - `Q_heat_transfer`
     - `mdot_in * h_in`
     - `mdot_out * h_out`
     - `p * dV_dt`
     - `e * dm_dt`
     - `m_safe * cv` (denominator)

2. **Check Mass Flow Derivatives**:
   - Evaluate `d(mdot_in)/d(T_c)` and `d(mdot_out)/d(T_c)`
   - Check if these produce NaN or extreme values
   - Verify pressure ratio derivatives are stable

3. **Check Accumulation Loop**:
   - Verify that `rhs_T += h * grid.a[c][j] * dT_dt` doesn't create issues
   - Check if the accumulation itself is causing the problem

### Phase 2: Targeted Fixes (Based on Diagnostics)

**Option A: If Mass Flow Derivatives are Problematic**
- Simplify mass flow model for derivative computation
- Use constant mass flow rates for derivative purposes
- Add stronger regularization to pressure ratio computations

**Option B: If Circular Dependency is the Issue**
- Reformulate constraint to break the circular dependency
- Use implicit formulation: `energy_balance = 0` instead of `T_c - rhs_T = 0`
- Separate temperature update from energy balance

**Option C: If Denominator Derivative is the Issue**
- Use stronger regularization: `m_cv_safe = m_safe * cv + epsilon_reg` where `epsilon_reg` is larger
- Consider using `ca.fmax(m_safe * cv, epsilon_min)` with larger `epsilon_min`
- Add protection to `d(m_safe)/d(v)` term

**Option D: If Pressure-Dependent Terms are the Issue**
- Decouple pressure from velocity in the derivative chain
- Use pressure as an intermediate variable with explicit bounds
- Add regularization to pressure-dependent mass flow terms

### Phase 3: Reformulation (If Targeted Fixes Don't Work)

**Alternative Constraint Formulation**:
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

This avoids the accumulation loop and may break the circular dependency.

### Phase 4: Fallback Options

If reformulation doesn't work:

1. **Use Numerical Differentiation for This Constraint**:
   - Mark the temperature constraint to use finite differences
   - IPOPT supports mixed exact/numerical gradients
   - Trade-off: Slower but may work

2. **Simplify Physics Model**:
   - Use constant mass flow rates
   - Use simpler heat transfer model
   - Use constant gas properties

3. **Document as Known Limitation**:
   - Document the issue and workaround
   - Consider alternative optimization formulations

## Success Criteria

1. ✅ No NaN in Jacobian evaluation
2. ✅ Optimization can start (at least 1 iteration)
3. ✅ Condition number remains computable
4. ✅ Optimization converges at tol=1e-8 (ultimate goal)

## Next Steps

1. **Immediate**: Add term-by-term diagnostics to identify exact NaN source
2. **Short-term**: Apply targeted fix based on diagnostics
3. **Medium-term**: If needed, reformulate constraint
4. **Long-term**: Validate convergence at tol=1e-8
