# Phase 1 Optimization GUI Testing Checklist

## Prerequisites

### 1. Environment Setup
- ✅ **CasADi installed** (tested: version 3.7.0)
- ✅ **IPOPT solver available** (via CasADi)
- ✅ **Free-piston solver imports** (verified working)
- ✅ **Python dependencies**: numpy, scipy, matplotlib
- ✅ **All project modules importable**

### 2. Required GUI Input Fields (All Exposed)

#### Core System Parameters
- **Stroke (mm)**: Default 20.0 mm
- **Cycle Time (s)**: Default 1.0 s
- **Upstroke Duration (%)**: Default 60%
- **Motion Type**: Default "minimum_jerk"

#### Combustion Parameters (All Required for Phase 1)
- ✅ **AFR** (Air-Fuel Ratio): Default 18.0
- ✅ **Fuel Mass (kg)**: Default 5e-4 kg
- ✅ **Ignition (deg rel TDC)**: Default -5.0°
- ✅ **Injector Delay (deg)**: Default 0.0° (NEW - just added)

#### Combustion Targets (Optional but Recommended)
- **CA50 Target (deg ATDC)**: Default 5.0°
- **CA50 Weight**: Default 0.0 (set > 0 to enable)
- **CA90-CA10 Target (deg)**: Default 20.0°
- **CA90-CA10 Weight**: Default 0.0 (set > 0 to enable)

#### CasADi Optimization Options
- **Use CasADi Optimizer**: Should be checked (required for Phase 1)
- **Collocation Method**: "legendre" or "radau" (default: legendre)
- **Universal Points**: Default ~360 (controls grid resolution)
- **Segments**: Number of collocation segments
- **Poly Order**: Polynomial order for collocation

### 3. Configuration Settings (Automatic via GUI)

The GUI automatically configures:
- `FreePistonPhase1Adapter` as primary optimizer
- Combustion model enabled (`disable_combustion=False`)
- Pressure invariance settings (if enabled)
- Workload target computation
- PR template generation (if enabled)

## Testing Steps

### Step 1: Launch GUI
```bash
python cam_motion_gui.py
```

### Step 2: Verify Input Fields
- Confirm all combustion parameters are visible in GUI
- Check that "Injector Delay (deg)" field exists (row 3, column 2)

### Step 3: Set Basic Phase 1 Parameters

**Minimum Required:**
1. Stroke: 20.0 mm
2. Cycle Time: 0.02 s (or 1.0 s for slower test)
3. AFR: 18.0
4. Fuel Mass: 5e-4 kg
5. Ignition: -5.0°
6. Injector Delay: 0.0° (or try -5° to +5° range)

**Recommended for Full Test:**
7. Enable "Use CasADi Optimizer" checkbox
8. CA50 Target: 5.0° (if testing CA tuning)
9. CA50 Weight: 1.0 (to enable CA50 constraint)
10. Collocation Method: "radau" (more stable for combustion)

### Step 4: Run Optimization

Click "Optimize" button and monitor:
- **Status bar**: Should show "Running cascaded optimization..."
- **Progress logger**: Should show steps:
  1. Framework configuration
  2. Input preparation
  3. Running cascaded optimization
  4. Processing results
  5. Updating plots

### Step 5: Verify Results

**Check Diagnostics Tab:**
- CA markers should be displayed (CA10, CA50, CA90, CA100)
- Pressure ratio statistics should show reasonable values (Π_mean ~1-10, not 30,000+)
- Workload-aligned metrics should be present

**Check Plots:**
- Motion law plots (position, velocity, acceleration)
- Pressure traces (if available)
- CA marker overlays (if implemented)

**Expected Behavior:**
- Optimization should converge (status: CONVERGED)
- CA markers should be within reasonable ranges:
  - CA10: ~5-15° ATDC
  - CA50: ~10-20° ATDC
  - CA90: ~20-30° ATDC
- Pressure ratios should be single-digit (1-10 range)
- No crashes or unhandled exceptions

### Step 6: Test Injector Delay

1. Run baseline with injector_delay_deg = 0.0°
2. Note CA10, CA50, CA90 values
3. Change injector_delay_deg to +5.0°
4. Re-run optimization
5. Verify CA markers shift earlier (more negative/less positive)
6. Try -5.0° and verify CA markers shift later

**Expected:**
- CA markers should shift by approximately the injector delay amount
- Shift direction: positive delay → earlier combustion → earlier CA markers
- Negative delay → later combustion → later CA markers

### Step 7: Test CA10/CA90 Tuning (If Targets Set)

1. Set CA50 Target: 10.0°
2. Set CA50 Weight: 1.0
3. Set CA90-CA10 Target: 20.0°
4. Set CA90-CA10 Weight: 1.0
5. Run optimization

**Expected:**
- Stage-A controller should adjust fuel/air/ignition to meet targets
- Injector delay should adjust if CA10/CA90 are out of bounds
- Final CA markers should be close to targets

## Troubleshooting

### Issue: Pressure ratios are still very high (30,000+)
**Possible Causes:**
- First-law fix may not be fully applied
- Denominator calculation issue
- Unit mismatch

**Check:**
- Verify validation script shows reasonable values
- Check `campro/physics/simple_cycle_adapter.py` line 409 uses `(gamma_comb - 1)`

### Issue: CA markers don't change with injector delay
**Possible Causes:**
- Injector delay not passed through to CombustionModel
- Delay parameter not in sim_inputs

**Check:**
- Verify `campro/physics/simple_cycle_adapter.py` lines 454-460 include injector delay
- Check CombustionModel.simulate() receives injector_delay_s or injector_delay_deg

### Issue: Optimization fails to converge
**Possible Causes:**
- Collocation grid too coarse
- Initial guess too far from solution
- Constraints too tight

**Try:**
- Increase Universal Points (e.g., 720)
- Reduce collocation segments (coarser grid)
- Relax CA50/CA90 targets
- Check IPOPT log in diagnostics tab

### Issue: GUI freezes during optimization
**Expected:**
- Optimization runs in separate thread
- GUI should remain responsive
- Progress updates in status bar

**If frozen:**
- Check for exceptions in terminal/console
- Verify threading is working
- Check for infinite loops in optimization

## Success Criteria

✅ **All tests pass if:**
1. GUI launches without errors
2. All input fields are visible and editable
3. Optimization runs to completion (converges or fails gracefully)
4. CA markers appear in diagnostics
5. Pressure ratios are in reasonable range (1-10)
6. Injector delay affects CA markers as expected
7. No crashes or unhandled exceptions
8. Results are displayed in plots/diagnostics

## Next Steps After Successful Test

1. **Verify PR template alignment**: Check that PR template matches corrected pressure traces
2. **Test workload sweeps**: Verify Π invariance across different workload values
3. **Test gain scheduling**: Enable pressure invariance and verify gain table populates
4. **Performance testing**: Measure solve times for different grid resolutions
5. **Integration testing**: Verify Phase 1 → Phase 2 → Phase 3 cascading works



