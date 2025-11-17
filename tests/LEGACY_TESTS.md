# Legacy Tests Documentation

This document tracks deprecated and legacy tests in the test suite.

## Deprecated Tests

### Placeholder Tests
- `tests/property/test_bounded_jerk.py` - Placeholder for jerk bounds enforcement (DEPRECATED)
- `tests/golden/test_motion_law_plot_hash.py` - Placeholder for golden SVG hash test (DEPRECATED)

### Phase 3 Tests (Out of Scope)
The following tests are marked as OUT OF SCOPE for current refactoring:
- `tests/test_casadi_phase3.py` - Phase 3 functionality not in scope
- `tests/test_casadi_phase4.py` - Phase 3 functionality not in scope
- `tests/test_phase3_unified_framework_integration.py` - Phase 3 functionality not in scope
- `tests/test_three_run_optimization_integration.py` - Phase 3 functionality not in scope

### Class-Based Tests (Legacy Style)
The following files use class-based tests and should be refactored to function-based tests:
- `tests/test_thermal_efficiency_analysis.py` - Uses class-based tests
- `tests/test_environment_validation.py` - Uses class-based tests
- `tests/test_casadi_phase1.py` - Uses class-based tests
- `tests/test_casadi_phase2.py` - Uses class-based tests
- `tests/test_casadi_phase1_integration.py` - Uses class-based tests
- `tests/test_thermal_efficiency_integration.py` - Uses class-based tests
- `tests/test_unified_framework_validation_mode.py` - Uses class-based tests
- `tests/test_validation_statistics_collection.py` - Uses class-based tests
- `tests/test_thermal_efficiency_optimizer_integration.py` - Uses class-based tests
- `tests/test_phase2_crank_center_optimizer.py` - Uses class-based tests
- `tests/test_gui_validation_mode_integration.py` - Uses class-based tests
- `tests/test_crank_center_physics_integration.py` - Uses class-based tests
- `tests/test_casadi_validation_mode.py` - Uses class-based tests
- `tests/test_casadi_performance.py` - Uses class-based tests
- `tests/test_casadi_optimizer_integration.py` - Uses class-based tests

**Note**: These files are not deprecated but should be refactored to match the main test file style.

## Test Style Guidelines

All tests should follow the main test file style:
- Standard header with path setup (lines 1-9)
- Helper functions prefixed with `_`
- Function-based tests (no classes unless necessary)
- Compact, focused tests
- Clear docstrings

## Main Test Files (Reference Style)
- `tests/test_gear_profile_generation.py`
- `tests/test_phase2_profile_generator.py`
- `tests/test_phase1_collocation_targets.py`

## Refactoring Status

### Completed Refactoring
- ✅ All small test files have standard headers
- ✅ `test_logging.py` - Converted to function-based
- ✅ `test_cam_ring_mapping.py` - Converted to function-based
- ✅ `test_phase1_physics_foundation.py` - Converted to function-based
- ✅ `test_litvin_physics_integration.py` - Converted to function-based
- ✅ `test_gain_scheduler.py` - Converted to function-based
- ✅ `test_adaptive_solver_selection.py` - Converted to function-based
- ✅ `test_optimal_motion.py` - Major refactoring completed

### Pending Refactoring
- ⏳ Large Phase 1/2 files still need refactoring (class-based → function-based)
- ⏳ Medium files need consolidation and style updates

