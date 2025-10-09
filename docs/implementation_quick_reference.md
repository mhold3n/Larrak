# Implementation Quick Reference

## 🚨 **CRITICAL FIXES NEEDED** (Phase 2 - Weeks 1-2)

### **1. Thermal Efficiency Adapter** (3 days)
**File**: `campro/optimization/thermal_efficiency_adapter.py`

**Lines 254-262**: Replace placeholder motion law extraction
```python
# REPLACE THIS:
for i in range(360):
    x[i] = 0.0  # Placeholder
    v[i] = 0.0  # Placeholder
    a[i] = 0.0  # Placeholder
    j[i] = 0.0  # Placeholder

# WITH THIS:
x, v, a, j = self._extract_motion_law_from_complex_result(complex_result, constraints)
```

**Line 391**: Replace placeholder objective value
```python
# REPLACE THIS:
objective_value=0.0,  # Placeholder

# WITH THIS:
objective_value=self._extract_objective_value(complex_result),
```

### **2. Collocation Optimization** (4 days)
**File**: `campro/optimization/collocation.py`

**Lines 162-164**: Replace NotImplementedError
```python
# REPLACE THIS:
raise NotImplementedError(
    f"Collocation optimization for problem type {type(constraints).__name__} "
    f"is not yet implemented. Only motion law problems are supported."
)

# WITH THIS:
if isinstance(constraints, MotionLawConstraints):
    return self._setup_motion_law_problem(constraints, objective)
elif hasattr(constraints, 'cam_ring'):
    return self._setup_cam_ring_problem(constraints, objective)
elif hasattr(constraints, 'sun_gear'):
    return self._setup_sun_gear_problem(constraints, objective)
else:
    raise ValueError(f"Unsupported constraint type: {type(constraints)}")
```

**Lines 418-419**: Replace placeholder objective calculation
```python
# REPLACE THIS:
# For now, return a placeholder value
return 0.0

# WITH THIS:
return self._calculate_objective_value(solution, objective_func)
```

### **3. Configuration Values** (1 day)
**File**: `cfg/thermal_efficiency_config.yaml`

**Replace placeholder values with realistic engine parameters**:
```yaml
# REPLACE THESE:
T_wall: 400.0  # K - Wall temperature
T_intake: 300.0  # K - Intake temperature
T_min: 200.0   # K - Minimum gas temperature
T_max: 2000.0  # K - Maximum gas temperature
max_temperature_limit: 2600.0   # K - Maximum temperature limit

# WITH THESE:
T_wall: 450.0  # K - Typical cylinder wall temperature
T_intake: 320.0  # K - Intake air temperature
T_min: 250.0   # K - Minimum gas temperature
T_max: 2200.0  # K - Maximum gas temperature
max_temperature_limit: 2400.0   # K - Realistic limit
```

---

## 🎯 **IMPLEMENTATION ORDER**

### **Week 1**:
1. **Day 1-2**: Fix thermal efficiency adapter motion law extraction
2. **Day 3**: Fix thermal efficiency adapter objective value
3. **Day 4-5**: Fix collocation optimization NotImplementedError

### **Week 2**:
1. **Day 1-2**: Fix collocation optimization objective calculation
2. **Day 3**: Update configuration values
3. **Day 4-5**: Integration testing and validation

---

## 🧪 **TESTING CHECKLIST**

### **After Each Fix**:
- [ ] Run unit tests: `pytest tests/test_thermal_efficiency_integration.py`
- [ ] Run integration test: `python scripts/test_thermal_efficiency_integration.py`
- [ ] Verify no placeholder values in output
- [ ] Check performance (should be < 1 second)

### **End-to-End Testing**:
- [ ] Create unified framework with thermal efficiency enabled
- [ ] Run optimization
- [ ] Verify motion law data is not all zeros
- [ ] Check objective value is realistic
- [ ] Validate performance metrics

---

## 📊 **SUCCESS METRICS**

### **Thermal Efficiency Adapter**:
- [ ] Motion law extraction returns non-zero values
- [ ] Objective value reflects actual thermal efficiency
- [ ] No placeholder values in output

### **Collocation Optimization**:
- [ ] No NotImplementedError for supported constraint types
- [ ] Objective calculation returns real values
- [ ] All constraint types work

### **Configuration**:
- [ ] Temperature/pressure values are realistic
- [ ] Engine parameters are within operating ranges
- [ ] Configuration validation passes

---

## 🚨 **CRITICAL FILES TO MODIFY**

1. **`campro/optimization/thermal_efficiency_adapter.py`** (Lines 254-262, 391)
2. **`campro/optimization/collocation.py`** (Lines 162-164, 418-419)
3. **`cfg/thermal_efficiency_config.yaml`** (Temperature/pressure values)

---

## 📞 **IMMEDIATE NEXT STEPS**

1. **Create development branch**: `git checkout -b feature/placeholder-implementation`
2. **Start with thermal efficiency adapter**: Focus on motion law extraction
3. **Test incrementally**: After each fix, run tests to verify
4. **Document changes**: Update comments and docstrings

**Target**: Complete critical fixes by end of Week 2
**Priority**: Thermal efficiency adapter is the most critical
