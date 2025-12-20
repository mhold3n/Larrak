# NaN Source Analysis

## Identified Location
- **Row 5**: Temperature collocation residual constraint (`T_c - rhs_T`)
- **Col 20**: Velocity variable (from velocities group)
- **Constraint Type**: `collocation_residuals`
- **Variable Group**: `velocities`

## Root Cause Hypothesis
The NaN occurs when CasADi differentiates the temperature collocation residual with respect to a velocity variable:

```
d(T_c - rhs_T)/d(v) = -d(rhs_T)/d(v)
```

where `rhs_T = T_k + h * sum(grid.a[c][j] * dT_dt)`.

The derivative `d(rhs_T)/d(v)` requires differentiating `dT_dt` with respect to velocity, which involves:
1. `dT_dt` depends on `dV_dt` (linear in velocities: `dV_dt = const * (vR - vL)`)
2. `dT_dt` has terms like `p * dV_dt` and `e * dm_dt`
3. `dT_dt` is divided by `m_safe * cv`

## Potential NaN Sources
1. **Division by `m_safe * cv`**: If this becomes zero or very small during differentiation
2. **Chain rule in `p * dV_dt`**: `d(p * dV_dt)/d(v) = p * d(dV_dt)/d(v) + dV_dt * d(p)/d(v)`
3. **Chain rule in `e * dm_dt`**: Similar issue with internal energy
4. **Temperature-dependent `cv`**: `cv = cp/gamma` where `cp` depends on `T`, creating complex derivatives

## Fixes Applied
1. ✅ Protected `cv` computation (clamped T, bounded cp/cv)
2. ✅ Protected `dT_dt` numerator (clamped to prevent Inf)
3. ✅ Protected `drho_dt` and `drho_log_dt` computations
4. ✅ Protected `dyF_dt` computation

## Remaining Issue
NaN still persists, suggesting the issue is in the symbolic differentiation itself, not just numerical evaluation. The problem may be:
- A division that produces NaN during differentiation (not just evaluation)
- An expression that becomes 0/0 or Inf-Inf during differentiation
- A dependency chain that creates problematic derivatives

## Next Steps
1. Add symbolic-level protection using `ca.fmax`/`ca.fmin` in all constraint expressions
2. Consider using `ca.if_else` or conditional expressions to handle edge cases
3. Review all divisions in `gas_energy_balance` and ensure denominators are always protected
4. Check if the issue is in the accumulation loop (`rhs_T += h * grid.a[c][j] * dT_dt`)
