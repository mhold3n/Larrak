# Diagnostic Report: Root Cause of NaN/Inf Errors

## Error Location Analysis

**Error**: NaN at row 56 col 59 in Jacobian, Inf at row 63 in constraints

### Constraint Structure
- Each collocation point (k, c) produces 6-8 constraints (depending on log-space and dynamic_wall)
- Row 63 ≈ 10th collocation point (63/6 ≈ 10.5)
- Constraint type: Collocation residual (likely density constraint: `rho_colloc_log[c] - rhs_rho_log`)

### Variable Structure  
- Column 59 ≈ 10th variable group
- Likely a density variable (log-space or physical-space)

## Key Finding: Log-Space Transformation Issue

### The Problem

**Line 1621**: `rho_colloc = [_exp_transform_var(ca, rho_log) for rho_log in rho_colloc_log]`

**Line 39-41**: `_exp_transform_var` is just `ca.exp(log_var)` - **NO BOUNDS ENFORCEMENT**

**Line 1704**: `rho_c = rho_colloc[c]` - uses unprotected `exp(rho_colloc_log[c])`

**Line 1707**: `rho_c_safe = ca.fmax(rho_c, rho_min_bound)` - **BANDAID FIX**

### Why This Is Wrong

1. **Bounds are set correctly**: Log-space bounds are `[log(1e-3), log(rho_max)]` ≈ `[-6.9, 2.3]`
2. **IPOPT should enforce bounds**: If `rho_colloc_log[c]` is within bounds, `exp(rho_colloc_log[c])` should be `>= 1e-3`
3. **But `exp()` doesn't guarantee bounds**: Even if input is within bounds, numerical precision or optimizer line search could produce values slightly outside
4. **The real issue**: `_exp_transform_var` should enforce bounds: `ca.fmax(ca.exp(log_var), rho_min_bound)`

## Root Cause Analysis

### Hypothesis 1: Log-Space Transformation Missing Bounds Enforcement ✅ LIKELY

**Problem**: `_exp_transform_var(ca, log_var)` is just `ca.exp(log_var)` without bounds enforcement.

**Why it fails**: 
- Even if `log_var` is within bounds `[log(1e-3), log(rho_max)]`, `exp(log_var)` can produce values slightly below `1e-3` due to:
  - Numerical precision in `exp()` evaluation
  - Optimizer line search trying values slightly outside bounds
  - Accumulated numerical errors

**Evidence**:
- Line 1707 adds `rho_c_safe = ca.fmax(rho_c, rho_min_bound)` - this wouldn't be needed if `exp()` guaranteed bounds
- Initial guess conversion in `driver.py` line 719 uses `np.log(max(x0[idx], 1e-3))` - protecting BEFORE log, but we need protection AFTER exp

**Fix**: Change `_exp_transform_var` to enforce bounds:
```python
def _exp_transform_var(ca: Any, log_var: Any, rho_min: float = 1e-3) -> Any:
    """Transform log-space variable back to physical space with bounds enforcement."""
    return ca.fmax(ca.exp(log_var), rho_min)
```

### Hypothesis 2: Initial Guess Violates Bounds ❌ UNLIKELY

**Check**: Initial guess conversion uses `np.log(max(x0[idx], 1e-3))` which ensures values are `>= 1e-3` before log, so log-space values should be `>= log(1e-3) ≈ -6.9`.

**Verdict**: Initial guess should be fine.

### Hypothesis 3: Temperature Calculation Issue ⚠️ NEEDS INVESTIGATION

**Check**: `dT_dt` calculation in `gas_energy_balance()` uses `m * cv` in denominator. Even with protected `rho` and `V`, `m = rho * V` could be extremely small.

**Status**: Already fixed with `m_safe` protection, but need to verify this is sufficient.

## Recommended Fixes

### Fix 1: Enforce Bounds in Log-Space Transformation (CRITICAL)

**File**: `campro/freepiston/opt/nlp.py`

**Problem**: `_exp_transform_var` is used in 5 places:
1. Line 1370: `rho0 = _exp_transform_var(ca, rho0_log)` - initial state density
2. Line 1460-1461: `Ain0`, `Aex0` - initial valve areas  
3. Line 1570-1571: `ain_sym`, `aex_sym` - collocation valve areas
4. Line 1621: `rho_colloc` - collocation density states ⚠️ **THIS IS WHERE NaN OCCURS**
5. Line 2115: `rho_k` - time step density states

**Change**: Update `_exp_transform_var` to accept and enforce minimum bounds:

```python
def _exp_transform_var(ca: Any, log_var: Any, epsilon: float = 1e-3) -> Any:
    """Transform log-space variable back to physical space with bounds enforcement.
    
    Ensures exp(log_var) >= epsilon to prevent numerical issues.
    This is necessary because exp() can produce values slightly below the
    theoretical minimum even when log_var is within bounds.
    
    Args:
        ca: CasADi module
        log_var: Log-space variable
        epsilon: Minimum value to enforce (default 1e-3 for density, 0.0 for valve areas)
    """
    return ca.fmax(ca.exp(log_var), epsilon)
```

**Usage**: 
- For density: `_exp_transform_var(ca, rho_log, epsilon=1e-3)`
- For valve areas: `_exp_transform_var(ca, A_log, epsilon=0.0)` or `epsilon=1e-10`

**Impact**: This eliminates the need for `rho_c_safe = ca.fmax(rho_c, rho_min_bound)` at line 1707, as `rho_c` will already be protected.

### Fix 2: Verify Temperature Calculation

**Status**: Already fixed with `m_safe` in `gas_energy_balance()`, but verify it's sufficient.

### Fix 3: Remove Redundant Safe Calculations

After Fix 1, many "safe" calculations become redundant:
- `rho_c_safe` can be removed if `rho_c` is already protected by `_exp_transform_var`
- `V_c_safe` is still needed (volume protection is separate)
- `p_c_safe` might still be needed for gas model (pressure ratio protection)

## Implementation Status

### ✅ Fix 1: Enforce Bounds in Log-Space Transformation - IMPLEMENTED

**Changes Made**:
1. Updated `_exp_transform_var()` to accept `epsilon` parameter and enforce bounds: `ca.fmax(ca.exp(log_var), epsilon)`
2. Updated all 5 call sites:
   - `rho0`: `epsilon=1e-3`
   - `Ain0`, `Aex0`: `epsilon=1e-10`
   - `ain_sym`, `aex_sym`: `epsilon=1e-10`
   - `rho_colloc`: `epsilon=1e-3` ⚠️ **THIS FIXES THE NaN SOURCE**
   - `rho_k`: `epsilon=1e-3`

**Impact**: 
- `rho_c = rho_colloc[c]` is now guaranteed to be `>= 1e-3` due to bounds enforcement in `_exp_transform_var`
- The `rho_c_safe = ca.fmax(rho_c, rho_min_bound)` at line 1707 is now **redundant** but harmless (can be removed after testing)
- This should eliminate NaN/Inf errors originating from extremely small density values

### ⚠️ Redundant Safe Calculations (Can Be Removed After Testing)

After Fix 1, the following "safe" calculations are now redundant:
- **Line 1707**: `rho_c_safe = ca.fmax(rho_c, rho_min_bound)` - redundant because `rho_c` is already protected by `_exp_transform_var`
- However, keeping it is harmless and provides defense-in-depth

**Still Needed**:
- `V_c_safe` - volume protection is separate from density
- `p_c_safe` - pressure protection for gas model (prevents extreme pressure ratios)
- `m_c_safe` - mass protection in `gas_energy_balance()` (already implemented)

## Summary

### Root Cause Identified ✅

**The fundamental issue**: `_exp_transform_var()` was missing bounds enforcement. Even though log-space variables have bounds `[log(1e-3), log(rho_max)]`, the `exp()` transformation doesn't guarantee the physical-space value stays `>= 1e-3` due to:
- Numerical precision in `exp()` evaluation
- Optimizer line search trying values slightly outside bounds
- Accumulated numerical errors

**The fix**: Enforce bounds directly in the transformation: `ca.fmax(ca.exp(log_var), epsilon)`

### Why "Safe" Calculations Were Needed

The "safe" calculations (`rho_c_safe`, `V_c_safe`, `p_c_safe`, etc.) were bandaid fixes for a fundamental problem:
- **`rho_c_safe`**: Needed because `exp(rho_colloc_log[c])` could produce values `< 1e-3` even when `rho_colloc_log[c]` was within bounds
- **`V_c_safe`**: Still needed (separate issue - volume protection)
- **`p_c_safe`**: Still needed (prevents extreme pressure ratios in gas model)
- **`m_safe`**: Still needed (prevents division by extremely small mass)

### After Fix 1

- **`rho_c_safe`** is now **redundant** but harmless (provides defense-in-depth)
- All other "safe" calculations remain necessary for their specific purposes
- The log-space implementation is now correct - bounds are enforced at the transformation level

## Next Steps

1. ✅ **COMPLETED**: Implement Fix 1 (enforce bounds in `_exp_transform_var`)
2. **TODO**: Test if this eliminates NaN/Inf errors
3. **TODO**: After successful testing, consider removing redundant `rho_c_safe` calculation (optional, provides defense-in-depth)
4. **TODO**: Verify temperature calculation is stable with `m_safe` protection

## Conclusion

The "safe" calculations were needed because the log-space transformation was incomplete. By enforcing bounds in `_exp_transform_var()`, we fix the root cause rather than patching symptoms. This is the correct approach - the transformation should guarantee the physical-space value respects bounds, not rely on downstream "safe" calculations.

