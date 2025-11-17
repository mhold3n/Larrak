# Silent Fallback Fix Implementation Plan

**Created**: 2024  
**Based on**: Comprehensive Silent Fallback Audit Report  
**Priority**: CRITICAL fixes must be completed immediately

---

## Overview

This plan provides step-by-step implementation guidance for fixing all identified silent fallbacks, prioritized by severity and impact.

---

## Phase 1: CRITICAL Fixes (Week 1)

### Fix 1.1: HSL Path Detection Silent Fallbacks

**File**: `campro/constants.py`  
**Lines**: 51-101  
**Priority**: 🔴 **CRITICAL**

#### Current Problem
- Multiple silent `except Exception: pass` blocks
- Returns empty string `""` on failure
- No logging of which detection method failed
- System continues with invalid configuration

#### Implementation Steps

1. **Add logging at each priority level**:
```python
def _detect_hsl_path() -> str:
    """
    Detect HSL library path with priority:
    1. Local conda environment (OS-specific)
    2. Active conda environment
    3. Environment variable HSLLIB_PATH
    4. Project CoinHSL archive (OS-specific) - auto-detected
    
    Raises:
        RuntimeError: If all detection methods fail
    """
    from campro.logging import get_logger
    log = get_logger(__name__)
    
    detection_attempts = []
    last_error = None
    
    try:
        project_root = _Path(__file__).resolve().parents[1]
        
        # Priority 1: Check local conda environment using env_manager
        try:
            from campro.environment.env_manager import find_hsl_library
            hsl_path = find_hsl_library()
            if hsl_path and hsl_path.exists():
                log.debug(f"HSL library found via Priority 1 (conda env): {hsl_path}")
                return str(hsl_path)
            detection_attempts.append("Priority 1: HSL library not found in conda env")
        except ImportError:
            detection_attempts.append("Priority 1: env_manager not available (skipped)")
        except Exception as e:
            detection_attempts.append(f"Priority 1: Error - {e}")
            last_error = e
        
        # Priority 2: Check environment variable
        hsl_env = _os.environ.get("HSLLIB_PATH", "")
        if hsl_env:
            env_path = _Path(hsl_env)
            if env_path.exists():
                log.debug(f"HSL library found via Priority 2 (env var): {env_path}")
                return str(env_path)
            detection_attempts.append(f"Priority 2: HSLLIB_PATH set but file not found: {hsl_env}")
        else:
            detection_attempts.append("Priority 2: HSLLIB_PATH not set")
        
        # Priority 3: Auto-detect CoinHSL directory using hsl_detector
        try:
            from campro.environment.hsl_detector import get_hsl_library_path
            hsl_lib_path = get_hsl_library_path()
            if hsl_lib_path and hsl_lib_path.exists():
                log.debug(f"HSL library found via Priority 3 (auto-detect): {hsl_lib_path}")
                return str(hsl_lib_path)
            detection_attempts.append("Priority 3: HSL library not found in project directory")
        except ImportError:
            detection_attempts.append("Priority 3: hsl_detector not available (skipped)")
        except Exception as e:
            detection_attempts.append(f"Priority 3: Error - {e}")
            last_error = e
        
    except Exception as e:
        detection_attempts.append(f"Outer exception: {e}")
        last_error = e
    
    # All detection methods failed - raise exception with details
    error_msg = "HSL library path detection failed. Attempted methods:\n"
    error_msg += "\n".join(f"  - {attempt}" for attempt in detection_attempts)
    if last_error:
        error_msg += f"\nLast error: {last_error}"
    
    log.error(error_msg)
    raise RuntimeError(error_msg)
```

#### Testing
- Test with HSL library present in conda env
- Test with HSL library in project directory
- Test with HSLLIB_PATH set
- Test with no HSL library available (should raise RuntimeError)
- Verify logging appears at appropriate levels

#### Success Criteria
- ✅ All detection attempts are logged
- ✅ Exception raised if all methods fail
- ✅ Clear error message with attempted methods
- ✅ No silent failures

---

### Fix 1.2: HSL Library Detection Silent Exception Handling

**File**: `campro/environment/env_manager.py`  
**Lines**: 224-237  
**Priority**: 🔴 **CRITICAL**

#### Current Problem
- Silent `except Exception: pass` in fallback path
- No logging of errors
- Errors in HSL detection are masked

#### Implementation Steps

```python
# Fallback: Check project CoinHSL directory using hsl_detector
try:
    from campro.environment.hsl_detector import get_hsl_library_path
    hsl_lib_path = get_hsl_library_path()
    if hsl_lib_path and hsl_lib_path.exists():
        log.info(f"Found HSL library in project CoinHSL directory: {hsl_lib_path}")
        return hsl_lib_path
except ImportError:
    # If hsl_detector not available, skip this check
    log.debug("hsl_detector not available; skipping project directory check")
    pass
except Exception as e:
    # Log error but continue (this is a fallback path)
    log.warning(f"Error checking project CoinHSL directory: {e}")
    # Continue to return None (not found)
    pass
```

#### Testing
- Test with hsl_detector available
- Test with hsl_detector import error
- Test with hsl_detector raising exception
- Verify warnings are logged

#### Success Criteria
- ✅ Exceptions are logged with warning level
- ✅ ImportError is handled gracefully (debug level)
- ✅ No silent failures

---

### Fix 1.3: IPOPT Factory Silent Solver Fallback

**File**: `campro/optimization/ipopt_factory.py`  
**Lines**: 28-51  
**Priority**: 🔴 **CRITICAL**

#### Current Problem
- Silent fallback to MA27 when detection fails
- No logging of detection failure
- No indication that fallback was used

#### Implementation Steps

```python
def _get_default_linear_solver() -> str:
    """Get default linear solver, detecting available solvers if needed."""
    global _DEFAULT_LINEAR_SOLVER
    
    if _DEFAULT_LINEAR_SOLVER is not None:
        return _DEFAULT_LINEAR_SOLVER
    
    # Detect available solvers and use MA27 if available, otherwise first available
    try:
        from campro.environment.hsl_detector import detect_available_solvers
        
        available = detect_available_solvers(test_runtime=False)
        if available:
            # Prefer MA27 as default, otherwise use first available
            if "ma27" in available:
                _DEFAULT_LINEAR_SOLVER = "ma27"
                log.debug(f"Default linear solver set to MA27 (available: {', '.join(available)})")
            else:
                _DEFAULT_LINEAR_SOLVER = available[0]
                log.info(f"Default linear solver set to {available[0]} (MA27 not available, found: {', '.join(available)})")
        else:
            _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
            log.warning("No HSL solvers detected; defaulting to MA27 (fallback)")
    except ImportError as e:
        _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
        log.warning(f"hsl_detector not available ({e}); defaulting to MA27 (fallback)")
    except Exception as e:
        _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
        log.warning(f"Error detecting available solvers ({e}); defaulting to MA27 (fallback)")
    
    return _DEFAULT_LINEAR_SOLVER
```

#### Testing
- Test with solvers detected successfully
- Test with no solvers detected
- Test with ImportError
- Test with other exceptions
- Verify warnings are logged for fallback cases

#### Success Criteria
- ✅ All fallback cases log warnings
- ✅ Detected solvers are logged
- ✅ Exception details are included in warnings
- ✅ No silent fallbacks

---

## Phase 2: HIGH Priority Fixes (Week 2)

### Fix 2.1: Unified Framework Legacy PR Calculation

**File**: `campro/optimization/unified_framework.py`  
**Lines**: 1462  
**Priority**: 🟡 **HIGH**

#### Implementation Steps

Add deprecation warning when legacy path is used:

```python
else:
    # Fallback to seed-derived PR (legacy behavior)
    log.warning(
        "Using legacy seed-derived PR calculation. "
        "This method is deprecated and will be removed in a future version. "
        "Enable pr_template_use_template=True to use the modern template-based method."
    )
    p_cyl_seed_raw__ = out0__.get("p_cyl")
    # ... rest of legacy code
```

#### Success Criteria
- ✅ Warning logged when legacy path used
- ✅ Clear deprecation message
- ✅ Guidance on using modern method

---

### Fix 2.2: Unified Framework Workload Target Fallback

**File**: `campro/optimization/unified_framework.py`  
**Lines**: 1414-1416  
**Priority**: 🟡 **HIGH**

#### Implementation Steps

Add logging when fallback is used:

```python
elif workload_target_j__ and area_m2__ > 0.0:
    # Fallback to workload target if cycle work not available
    log.debug(
        f"Using workload target fallback for load pressure calculation "
        f"(cycle work unavailable, workload_target={workload_target_j__}J)"
    )
    p_load_pa_ref = workload_target_j__ / max(area_m2__ * stroke_m__, 1e-12)
    p_load_kpa_ref__ = p_load_pa_ref / 1000.0
```

#### Success Criteria
- ✅ Debug log when fallback used
- ✅ Reason for fallback documented

---

## Phase 3: MEDIUM Priority Fixes (Week 3)

### Fix 3.1: HSL Detector bin Directory Fallback

**File**: `campro/environment/hsl_detector.py`  
**Lines**: 137-142  
**Priority**: 🟠 **MEDIUM**

#### Implementation Steps

Add logging when fallback is used:

```python
if IS_MACOS:
    # Try lib directory first (standard macOS location)
    lib_path = coinhsl_dir / "lib" / "libcoinhsl.dylib"
    if not lib_path.exists():
        # Fallback to bin directory
        lib_path = coinhsl_dir / "bin" / "libcoinhsl.dylib"
        log.debug(f"HSL library not found in lib/, checking bin/: {lib_path}")
```

#### Success Criteria
- ✅ Debug log when fallback path used

---

### Fix 3.2: Diagnostics Feasibility Heuristic Fallback

**File**: `campro/diagnostics/feasibility.py`  
**Lines**: 420-422  
**Priority**: 🟠 **MEDIUM**

#### Implementation Steps

Add logging when fallback is used:

```python
except Exception as e:
    # Fallback to heuristic in case of solver error
    log.warning(
        f"Solver-based feasibility check failed ({e}); "
        f"falling back to heuristic check"
    )
    return check_feasibility(constraints, bounds)
```

#### Success Criteria
- ✅ Warning logged with exception details
- ✅ Clear indication of fallback behavior

---

### Fix 3.3: System Builder Placeholder

**File**: `campro/config/system_builder.py`  
**Lines**: 243-244  
**Priority**: 🟠 **MEDIUM**

#### Implementation Steps

Either:
1. **Implement properly**, or
2. **Remove if not needed**, or
3. **Raise NotImplementedError** if called

Recommendation: Raise NotImplementedError if this is not actually used:

```python
# Create system (this would be implemented in a specific system class)
raise NotImplementedError(
    "System creation is not yet implemented. "
    "This functionality is planned for a future release."
)
```

#### Success Criteria
- ✅ Either implemented or raises NotImplementedError
- ✅ No placeholder code in production

---

## Testing Strategy

### Unit Tests
- Test each detection method independently
- Test fallback paths
- Test exception handling
- Verify logging output

### Integration Tests
- Test full HSL detection chain
- Test solver selection with various configurations
- Test optimization with fallback paths

### Manual Testing
- Run optimization with HSL library present
- Run optimization without HSL library (should fail with clear error)
- Verify all warnings appear in logs

---

## Rollout Plan

1. **Week 1**: Implement CRITICAL fixes
2. **Week 2**: Test CRITICAL fixes, implement HIGH priority fixes
3. **Week 3**: Test HIGH priority fixes, implement MEDIUM priority fixes
4. **Week 4**: Final testing and documentation updates

---

## Success Metrics

- ✅ Zero silent fallbacks in CRITICAL paths
- ✅ All fallbacks log appropriate warnings
- ✅ All failures raise exceptions with clear messages
- ✅ Documentation updated with current status
- ✅ All tests pass

---

## Related Documentation

- `docs/silent_fallback_audit_comprehensive.md` - Full audit report
- `docs/backwards_compatibility_stopgap_analysis.md` - Previous analysis
- `docs/silent_fallback_audit_report.md` - Raw audit findings


