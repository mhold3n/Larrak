# Phase 1 Combustion Integration - Test Summary

## Implementation Status: ✅ COMPLETE

All planned features have been implemented:

1. ✅ **Piston-speed fallback fixed** - Real velocity computed from `v_mm_per_theta`
2. ✅ **Workload-to-load-pressure translator** - Converts `workload_target` to `p_load_kpa`
3. ✅ **Workload-aligned pressure ratios** - Denominators scale with workload across sweeps
4. ✅ **Expanded combustion controller** - Co-tunes fuel, air, AFR, ignition
5. ✅ **Combustion outputs threaded** - CA markers and pressure ratios flow through data structures
6. ✅ **FreePistonPhase1Adapter updated** - Fuel/load sweeps with workload-aligned denominators
7. ✅ **PR template implementation** - Geometry-informed, efficiency-optimized template

## Phase-Specific Collocation Validation

Golden tests for the GUI workflow now run each collocation solver directly:

- `tests/test_phase1_collocation_targets.py` exercises minimum-jerk/time, efficiency-focused `pcurve_te`, and acceleration-limited cases.
- `tests/test_phase2_profile_generator.py` feeds the cam-ring mapper with cycloidal motion and Litvin-style base radius changes to emulate crank center offsets.

Run them with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_phase1_collocation_targets.py tests/test_phase2_profile_generator.py
```

These tests keep the phase boundaries verifiable without capturing GUI output data.

## Test Instructions

### 1. Run Validation Script

```bash
cd /Users/maxholden/Documents/GitHub/Larrak
export PYTHONPATH=/Users/maxholden/Documents/GitHub/Larrak:$PYTHONPATH
python scripts/phase1_combustion_validation.py
```

**Expected Output:**
- Baseline (legacy) pressure ratio stats
- Combustion-aware pressure ratio stats
- CA markers (CA10, CA50, CA90, CA100)
- Guardrail loss for 10% fuel reduction
- Workload steps test (50J, 100J, 150J, 200J)
- Π invariance verification (variation < 10%)
- CA50 marker tolerance check (variation < 5°)
- Workload-derived load pressure scaling validation
- PR template vs seed-derived comparison

### 2. Test PR Template Function Directly

```bash
python test_pr_template.py
```

**Expected Output:**
- Template computed successfully
- Mean, peak, min, std statistics
- Comparison between small and large engine templates
- Templates should differ based on geometry

### 3. GUI Testing

1. **Launch GUI:**
   ```bash
   python cam_motion_gui.py
   ```

2. **Navigate to Phase 1 Optimization**

3. **Set Combustion Inputs:**
   - AFR: 18.0
   - Fuel mass: 5e-4 kg
   - Ignition timing: -5° (or 0.005s)
   - Workload target: 100J (try different values)

4. **Enable PR Template (if available in settings):**
   - `pr_template_use_template: True`
   - `pr_template_expansion_efficiency: 0.85`
   - `pr_template_peak_scale: 1.5`

5. **Run Optimization and Check:**
   - **Diagnostics Tab:**
     - CA markers should appear (CA10, CA50, CA90)
     - Pressure ratio statistics should show workload-aligned values
     - `pressure_invariance` metadata should include:
       - `denominator_base.p_load_kpa` (workload-derived)
       - `work_target_j`
       - `cases` array with CA markers for each fuel/load combination

6. **Test Workload Sweeps:**
   - Change workload target (50J → 100J → 150J → 200J)
   - Verify Π invariance: pressure ratios should stay relatively constant
   - Verify CA markers stay within tolerance (< 5° variation)

### 4. Verify PR Template is Geometry-Informed

The PR template should:
- **Scale with compression ratio**: Higher CR → higher PR peak
- **Account for bore/stroke**: Different geometries → different template shapes
- **Show expansion efficiency**: Template should reflect efficiency target (85% default)

You can verify by:
1. Running optimization with different stroke/bore values
2. Comparing PR template shapes in metadata
3. Checking that template differs from seed-derived PR

## Key Verification Points

### ✅ Π Invariance
- Pressure ratios should vary < 10% across workload steps
- Check `pressure_ratio.cases` in metadata - all cases should have similar `pi_mean`

### ✅ CA Marker Consistency
- CA50 should vary < 5° across workload steps
- CA markers should appear in all fuel/load cases
- Check `ratio_cases` and `work_cases` in metadata

### ✅ Workload Scaling
- `p_load_kpa` should scale linearly with `workload_target`
- Formula: `p_load = w / (A·stroke)`
- Check `denominator_base.p_load_kpa` in metadata

### ✅ PR Template Physics
- Template should show compression phase (increasing PR)
- Template should show combustion peak (scaled by `pr_peak_scale`)
- Template should show flat expansion (ideal isentropic)
- Template should taper near EVO

## Troubleshooting

If validation script fails:
1. Check PYTHONPATH is set correctly
2. Verify all dependencies are installed
3. Check that `campro` package can be imported

If GUI doesn't show CA markers:
1. Verify combustion inputs are provided (AFR, fuel_mass)
2. Check that `use_pressure_invariance` is enabled
3. Verify `pr_template_use_template` setting

If optimization doesn't converge:
1. Try disabling PR template (`pr_template_use_template: False`) to use seed-derived
2. Check that combustion inputs are valid
3. Verify workload target is reasonable for engine geometry


