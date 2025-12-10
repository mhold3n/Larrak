# CasADi Symbolic Differentiation Investigation - Conclusion

## Investigation Summary

### Problem
NaN detected in Jacobian at row 5 (temperature collocation residual) with respect to col 20 (velocity variable) during CasADi's symbolic differentiation.

### Root Cause
The NaN occurs when CasADi differentiates the temperature collocation residual:
```
d(T_c - rhs_T)/d(v) = -d(rhs_T)/d(v)
```

Where `rhs_T` accumulates `dT_dt` which has a complex derivative chain involving:
- `p * dV_dt` where `p = rho * R * T` and `dV_dt` is linear in velocity
- `e * dm_dt` where `e = cv * T` and `dm_dt` may depend on velocity
- Division by `m_safe * cv`

### Is This a CasADi Limitation?

**No, this is NOT an inherent limitation of CasADi's automatic differentiation.**

CasADi's AD is:
- **Exact** (not approximate)
- **Efficient** (similar cost to function evaluation)
- **Robust** (widely used in production)

The issue is in the **formulation**, not the differentiation method.

### Fixes Attempted

1. ✅ Simplified temperature-dependent `cv` to constant
2. ✅ Added regularization to denominator (`m_cv_safe + epsilon_reg`)
3. ✅ Protected `p * dV_dt` and `e * dm_dt` terms with clamping
4. ✅ Protected pressure computation
5. ✅ Added safeguards to all derivative computations

**Result**: NaN persists, indicating the issue is deeper in the derivative chain.

### Why Fixes Haven't Worked

The problem is that **symbolic differentiation happens at expression construction time**, not evaluation time. When CasADi builds the derivative expression:
```
d(p * dV_dt)/d(v) = p * d(dV_dt)/d(v) + dV_dt * d(p)/d(v)
```

The term `d(p)/d(v) = d(rho * R * T)/d(v)` creates a chain:
- `d(rho)/d(v)` - density derivative w.r.t. velocity
- `d(T)/d(v)` - temperature derivative w.r.t. velocity

If either of these produces NaN (or creates a 0/0 or Inf-Inf situation), the entire derivative becomes NaN.

### Alternative Approaches

#### Option 1: Reformulate the Energy Balance (RECOMMENDED)
Instead of using `dT_dt` directly in the collocation residual, use an implicit formulation:

**Current**:
```python
rhs_T = T_k + h * sum(grid.a[c][j] * dT_dt)
constraint: T_c - rhs_T = 0
```

**Alternative**: Use energy balance directly as constraint:
```python
# Direct energy balance constraint (no accumulation)
energy_balance = m * cv * dT_dt + e * dm_dt - (Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV_dt)
constraint: energy_balance = 0
```

This avoids the accumulation loop that creates the complex derivative chain.

#### Option 2: Use Numerical Differentiation (NOT RECOMMENDED)
- IPOPT can use finite differences via `ipopt.derivative_test`
- **Pros**: Avoids symbolic differentiation
- **Cons**: 
  - 10-100x slower (O(n) function evaluations per iteration)
  - Less accurate (~1e-6 vs machine precision)
  - Doesn't solve root cause

#### Option 3: Simplify Physics Model
- Use constant properties (cv, cp) - **Already tried**
- Use simpler mass flow models
- Use simpler heat transfer models
- **Trade-off**: Less physical accuracy

#### Option 4: Use Different Collocation Method
- Try different collocation schemes (Gauss-Legendre, Chebyshev)
- May have different numerical properties
- **Uncertain**: May not fix the underlying issue

#### Option 5: Regularize at Constraint Level
Add small regularization to the constraint itself:
```python
# Instead of: T_c - rhs_T = 0
# Use: (T_c - rhs_T) / (1 + epsilon * |T_c|) = 0
# This regularizes the constraint and its derivatives
```

## Recommended Next Steps

1. **Investigate `d(rho)/d(v)` and `d(T)/d(v)`**: 
   - Check if these derivatives are producing NaN
   - Add diagnostics to identify which term in the chain is problematic

2. **Try Implicit Formulation**:
   - Reformulate energy balance as direct constraint
   - Avoids accumulation loop that creates derivative chain

3. **Consider Alternative Collocation Schemes**:
   - Different schemes may have better numerical properties
   - May avoid the problematic derivative structure

4. **Document as Known Limitation**:
   - If fixes don't work, document the issue
   - Consider workaround (e.g., use numerical differentiation for this specific constraint)

## Conclusion

**CasADi's symbolic differentiation is not the problem** - it's correctly computing the derivative of the expression we provided. The issue is that **our expression creates a derivative chain that produces NaN**.

The solution is to **reformulate the physics** to avoid problematic derivative chains, not to abandon CasADi. However, this may require significant changes to the collocation formulation.

**Recommendation**: Continue investigating the specific derivative terms (`d(rho)/d(v)`, `d(T)/d(v)`) to identify the exact source, then reformulate that specific part of the physics model.
