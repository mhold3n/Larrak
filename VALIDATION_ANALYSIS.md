# Phase 1 Combustion Validation - Output Analysis

## Summary

The validation script ran successfully, but the output reveals **critical issues** with pressure ratio calculations:

### 🔴 **Critical Issue: Pressure Ratio Values Are Massively Incorrect**

**Combustion-aware PR stats:**
- `pi_mean: 38,869.95` (should be ~1-2)
- `pi_peak: 308,725.60` (should be ~1-3)
- `pi_std: 59,728.91` (should be ~0.1-0.5)

**Root Cause Analysis:**

The denominator in the pressure ratio calculation is missing the workload-aligned `p_load_kpa` component. Looking at line 155-159 in the validation script:

```python
combustion_stats, combustion_pi = pressure_ratio_summary(
    np.asarray(p_cyl, dtype=float),
    np.asarray(combustion_out.get("p_bounce"), dtype=float),
    thermo.p_atm_kpa,
    # Missing: p_load_kpa parameter!
)
```

Since `p_load_kpa` defaults to `0.0`, the denominator is:
```
denom = 0.0 + 0.0 + p_env_kpa + p_bounce ≈ 101 kPa + small bounce
```

If `p_cyl` values are reasonable (e.g., 100-500 kPa), PR should be ~1-5. But we're seeing PR ~38,000, which suggests:

1. **Either `p_cyl` is in wrong units** (e.g., Pa instead of kPa)
2. **Or `p_bounce` is very small/negative**, making denominator tiny
3. **Or combustion model is returning unreasonably high pressures**

### ✅ **What's Working Correctly**

1. **CA Markers**: Perfect consistency (all 14.57° across workload steps)
2. **Workload Scaling**: Load pressure scales correctly (1:2:3:4 ratio)
3. **Π Invariance**: Variation is 0.0284 (within 0.10 tolerance) - **BUT values are wrong!**
4. **PR Template**: Correctly differs from seed-derived (decoupled design)

### ⚠️ **Issues to Fix**

1. **Missing `p_load_kpa` in combustion-aware PR calculation**
   - The baseline combustion test doesn't compute workload-aligned load pressure
   - Should compute `p_load_kpa` from a reference workload (e.g., 100J)

2. **Unit verification needed**
   - Verify `p_cyl` from combustion model is in kPa (not Pa)
   - Verify `p_bounce` is reasonable (not near-zero or negative)

3. **Guardrail loss is huge** (50,185,297) - this is a symptom of the PR issue

### 🔧 **Recommended Fixes**

1. **Update validation script** to compute `p_load_kpa` for combustion-aware baseline:
   ```python
   # Compute workload-aligned p_load for baseline (e.g., 100J reference)
   baseline_workload_j = 100.0
   p_load_kpa_baseline = compute_workload_aligned_p_load(
       baseline_workload_j, geom.area_mm2, stroke_mm
   )
   
   combustion_stats, combustion_pi = pressure_ratio_summary(
       np.asarray(p_cyl, dtype=float),
       np.asarray(combustion_out.get("p_bounce"), dtype=float),
       thermo.p_atm_kpa,
       p_load_kpa=p_load_kpa_baseline,  # Add this!
       p_cc_kpa=0.0,
   )
   ```

2. **Add diagnostic output** to verify units:
   - Print `p_cyl` min/max values
   - Print `p_bounce` min/max values
   - Print denominator min/max values
   - This will help identify unit mismatches

3. **Verify combustion model output units**:
   - Check `campro/physics/simple_cycle_adapter.py` line 412: `p_cyl_kpa = p_cyl_pa / 1e3`
   - Ensure this conversion is correct

### 📊 **Expected vs Actual Values**

| Metric | Expected | Actual | Status |
|--------|-----------|--------|--------|
| Baseline PR mean | 1.0-2.0 | 1.29 | ✅ OK |
| Combustion PR mean | 1.0-2.0 | 38,870 | 🔴 **WRONG** |
| Workload PR mean | 1.0-2.0 | 732-753 | 🔴 **WRONG** |
| CA50 consistency | < 5° | 0.00° | ✅ Perfect |
| Workload scaling | Linear | Linear | ✅ Perfect |
| PR invariance | < 0.10 | 0.0284 | ✅ OK (but values wrong) |

### 🎯 **Next Steps**

1. Fix the missing `p_load_kpa` in baseline combustion PR calculation
2. Add unit diagnostics to identify any unit mismatches
3. Re-run validation to verify PR values are now reasonable
4. If PR values are still wrong, investigate:
   - Combustion model pressure units
   - `p_bounce` calculation
   - Initial pressure conditions in combustion model




