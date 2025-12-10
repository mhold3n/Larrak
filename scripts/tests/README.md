# Test Suite Documentation

## Overview

This directory contains the test suite for the Larrak project. Tests are organized to match the main test file style for consistency and maintainability.

## Test Style Guidelines

All tests should follow the style of the main test files:
- **Standard header** with path setup (lines 1-9)
- **Helper functions** prefixed with `_`
- **Function-based tests** (avoid classes unless necessary)
- **Compact, focused tests** with clear docstrings
- **Type hints** for better code clarity

### Reference Files (Main Test Style)
- `test_gear_profile_generation.py`
- `test_phase2_profile_generator.py`
- `test_phase1_collocation_targets.py`

## Deprecated/Legacy Tests

See `LEGACY_TESTS.md` for a complete list of deprecated tests.

### Phase 3 Tests (Out of Scope)
The following tests are marked as OUT OF SCOPE:
- `test_casadi_phase3.py`
- `test_casadi_phase4.py`
- `test_phase3_unified_framework_integration.py`
- `test_three_run_optimization_integration.py`

### Placeholder Tests
- `property/test_bounded_jerk.py` - Marked as deprecated
- `golden/test_motion_law_plot_hash.py` - Marked as deprecated

## Test Organization

### Phase 1 Tests (Motion Law Creation)
Tests for Phase 1 functionality including:
- Motion law optimization
- Collocation-based optimization
- Motion law constraints and validation

### Phase 2 Tests (Motion Law → Profiles + Gear Meshes)
Tests for Phase 2 functionality including:
- Cam-ring mapping
- Litvin synthesis
- Gear geometry generation
- Profile generation

### GUI Tests
Tests for GUI plotting and results feedback functionality.

### Supporting Tests
- Environment validation
- Solver selection
- Error recovery
- Logging

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_gear_profile_generation.py

# Run with verbose output
pytest tests/ -v

# Run only non-deprecated tests
pytest tests/ -m "not deprecated"

# Run IPOPT-heavy integration tests (requires CAMPRO_RUN_HEAVY_TESTS=1)
pytest tests/heavy -m ipopt_phase1
pytest tests/heavy -m ipopt_phase2

# Refresh golden references (requires working IPOPT/HSL install)
python tests/heavy/phase1_ipopt_integration.py --update
python tests/heavy/phase2_ipopt_integration.py --update
```

### Heavy IPOPT Integration Tests

- Located in `tests/heavy`. These scripts exercise the full IPOPT/HSL pipeline for Phase 1 and Phase 2 and compare results against JSON references in `tests/golden`.
- Opt-in by setting `CAMPRO_RUN_HEAVY_TESTS=1`. Without it, pytest will skip the heavy markers.
- Refresh the golden data after validating a known-good environment via the `--update` flag shown above.
- The stored payloads capture the core arrays (θ, displacement, ring radii) and optimized gear configuration to detect regressions.
- CI/Nightly suggestion: `CAMPRO_RUN_HEAVY_TESTS=1 pytest tests/heavy -m "ipopt_phase1 or ipopt_phase2"` (disabled by default to keep main pipelines fast).

## Coverage Requirements

- **Phase 1**: 97% coverage required
- **Phase 2**: 97% coverage required
- **GUI**: Coverage for plotting/results feedback required
- **Phase 3**: Not in scope (no coverage requirement)

