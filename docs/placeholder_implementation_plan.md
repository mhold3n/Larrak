# Mock and Placeholder Implementation Plan

## Executive Summary

This plan addresses **103 mock and placeholder elements** found in the Larrak codebase, prioritizing **8 critical elements** that must be fixed for Phase 2 completion. The plan is structured in 4 phases over 8 weeks, with clear deliverables and success criteria.

---

## 🎯 **Phase 2: Critical Placeholder Resolution** (Weeks 1-2)

### **Priority 1: Thermal Efficiency Adapter** (Week 1)

#### **Task 2.1.1: Motion Law Data Extraction** 
**File**: `campro/optimization/thermal_efficiency_adapter.py` (Lines 254-262)

**Current Issue**:
```python
# Placeholder: would extract actual motion law from complex result
for i in range(360):
    x[i] = 0.0  # Placeholder
    v[i] = 0.0  # Placeholder
    a[i] = 0.0  # Placeholder
    j[i] = 0.0  # Placeholder
```

**Implementation Plan**:
1. **Analyze complex optimizer output structure**
   - Study `OP-dynamic-gas-simulator/campro/freepiston/opt/optimization_lib.py`
   - Map solution data structure to motion law format
   - Identify piston position/velocity data extraction points

2. **Implement data extraction logic**
   ```python
   def _extract_motion_law_from_complex_result(self, complex_result, constraints):
       """Extract motion law data from complex optimizer result."""
       # Extract piston positions (x_L, x_R) from complex result
       # Convert to single linear follower motion
       # Interpolate to 360 cam angle points
       # Calculate derivatives (velocity, acceleration, jerk)
   ```

3. **Add data validation**
   - Verify motion law continuity
   - Check constraint satisfaction
   - Validate physical feasibility

**Deliverable**: Real motion law extraction from complex optimizer
**Success Criteria**: Motion law data matches complex optimizer output
**Effort**: 3 days

#### **Task 2.1.2: Objective Value Calculation**
**File**: `campro/optimization/thermal_efficiency_adapter.py` (Line 391)

**Current Issue**:
```python
objective_value=0.0,  # Placeholder
```

**Implementation Plan**:
1. **Extract thermal efficiency from complex result**
   ```python
   def _extract_objective_value(self, complex_result):
       """Extract objective value from complex optimizer result."""
       return complex_result.performance_metrics.get("thermal_efficiency", 0.0)
   ```

2. **Add performance metrics extraction**
   - Thermal efficiency
   - Indicated work
   - Pressure/temperature limits
   - Short-circuit losses

**Deliverable**: Real objective value from complex optimizer
**Success Criteria**: Objective value reflects actual thermal efficiency
**Effort**: 1 day

### **Priority 2: Collocation Optimization** (Week 1-2)

#### **Task 2.2.1: Non-Motion Law Problem Support**
**File**: `campro/optimization/collocation.py` (Lines 162-164)

**Current Issue**:
```python
raise NotImplementedError(
    f"Collocation optimization for problem type {type(constraints).__name__} "
    f"is not yet implemented. Only motion law problems are supported."
)
```

**Implementation Plan**:
1. **Extend constraint type support**
   ```python
   def _setup_optimization_problem(self, constraints, objective):
       """Setup optimization problem based on constraint type."""
       if isinstance(constraints, MotionLawConstraints):
           return self._setup_motion_law_problem(constraints, objective)
       elif isinstance(constraints, CamRingConstraints):
           return self._setup_cam_ring_problem(constraints, objective)
       elif isinstance(constraints, SunGearConstraints):
           return self._setup_sun_gear_problem(constraints, objective)
       else:
           raise ValueError(f"Unsupported constraint type: {type(constraints)}")
   ```

2. **Implement cam-ring optimization setup**
3. **Implement sun gear optimization setup**

**Deliverable**: Support for all constraint types
**Success Criteria**: No NotImplementedError for supported constraint types
**Effort**: 4 days

#### **Task 2.2.2: Objective Value Calculation**
**File**: `campro/optimization/collocation.py` (Lines 418-419)

**Current Issue**:
```python
# For now, return a placeholder value
return 0.0
```

**Implementation Plan**:
1. **Implement proper objective calculation**
   ```python
   def _calculate_objective_value(self, solution, objective_func):
       """Calculate objective value from solution."""
       # Extract solution data
       # Apply objective function
       # Return calculated value
   ```

**Deliverable**: Real objective value calculation
**Success Criteria**: Objective values match expected optimization results
**Effort**: 2 days

### **Priority 3: Configuration Values** (Week 2)

#### **Task 2.3.1: Realistic Engine Parameters**
**File**: `cfg/thermal_efficiency_config.yaml`

**Current Issue**: Placeholder temperature/pressure values

**Implementation Plan**:
1. **Research realistic engine parameters**
   - Diesel engine specifications
   - Opposed-piston engine data
   - Thermal efficiency targets

2. **Update configuration values**
   ```yaml
   # Realistic values based on research
   T_wall: 450.0  # K - Typical cylinder wall temperature
   T_intake: 320.0  # K - Intake air temperature
   T_min: 250.0   # K - Minimum gas temperature
   T_max: 2200.0  # K - Maximum gas temperature
   max_temperature_limit: 2400.0   # K - Realistic limit
   ```

**Deliverable**: Realistic engine configuration
**Success Criteria**: Values within realistic engine operating ranges
**Effort**: 1 day

---

## 🎯 **Phase 3: Physics Components** (Weeks 3-4)

### **Task 3.1: Time Kinematics Component**
**File**: `campro/physics/kinematics/time_kinematics.py` (Lines 43-50)

**Current Issue**:
```python
# Placeholder implementation
log.info("Time kinematics component - placeholder implementation")
return ComponentResult(
    success=True,
    outputs={},
    metadata={'note': 'placeholder implementation'}
)
```

**Implementation Plan**:
1. **Implement time-based kinematics**
   ```python
   def solve(self, inputs):
       """Solve time-based kinematics problem."""
       # Extract input parameters
       # Calculate time-based motion
       # Return kinematic results
   ```

2. **Add time integration methods**
   - Euler integration
   - Runge-Kutta methods
   - Adaptive time stepping

**Deliverable**: Functional time kinematics component
**Success Criteria**: Accurate time-based motion calculations
**Effort**: 5 days

### **Task 3.2: Kinematic Constraints Component**
**File**: `campro/physics/kinematics/constraints.py` (Lines 43-50)

**Implementation Plan**:
1. **Implement kinematic constraint checking**
   ```python
   def solve(self, inputs):
       """Check kinematic constraints."""
       # Validate motion constraints
       # Check boundary conditions
       # Return constraint satisfaction status
   ```

**Deliverable**: Functional kinematic constraints component
**Success Criteria**: Accurate constraint validation
**Effort**: 3 days

### **Task 3.3: System Builder**
**File**: `campro/config/system_builder.py` (Lines 232-233)

**Implementation Plan**:
1. **Implement system creation logic**
   ```python
   def build_system(self):
       """Build complete system from configuration."""
       # Instantiate components
       # Configure connections
       # Validate system integrity
       # Return system instance
   ```

**Deliverable**: Functional system builder
**Success Criteria**: Creates valid system instances
**Effort**: 4 days

---

## 🎯 **Phase 4: Complex Gas Optimizer Enhancement** (Weeks 5-6)

### **Task 4.1: Objective Function Placeholders**
**File**: `OP-dynamic-gas-simulator/campro/freepiston/opt/obj.py`

**Implementation Plan**:
1. **Implement scavenging quality calculation**
   ```python
   def scavenging_uniformity(p_series, V_series, weights):
       """Calculate scavenging uniformity from 1D model."""
       # Implement 1D scavenging analysis
       # Calculate uniformity metrics
       # Return weighted score
   ```

2. **Add missing objective functions**
   - Combustion efficiency
   - Heat transfer optimization
   - Flow uniformity

**Deliverable**: Complete objective function suite
**Success Criteria**: All objectives return meaningful values
**Effort**: 6 days

### **Task 4.2: NLP Constraint Placeholders**
**File**: `OP-dynamic-gas-simulator/campro/freepiston/opt/nlp.py`

**Implementation Plan**:
1. **Implement combustion timing constraints**
   ```python
   def add_combustion_timing_constraints(nlp, K):
       """Add combustion timing constraints to NLP."""
       # Implement timing constraint logic
       # Add to NLP structure
   ```

**Deliverable**: Complete constraint set
**Success Criteria**: All constraints properly implemented
**Effort**: 4 days

### **Task 4.3: Collocation Method Limitations**
**File**: `OP-dynamic-gas-simulator/campro/freepiston/opt/colloc.py`

**Implementation Plan**:
1. **Extend Radau IIA implementation**
   ```python
   def make_radau_grid(K, C, kind="radau"):
       """Create Radau IIA grid for arbitrary C."""
       # Implement general Radau IIA
       # Support C > 1
   ```

2. **Extend Gauss-Legendre implementation**
   ```python
   def make_gauss_grid(K, C, kind="gauss"):
       """Create Gauss-Legendre grid for arbitrary C."""
       # Implement general Gauss-Legendre
       # Support C > 2
   ```

**Deliverable**: Complete collocation method support
**Success Criteria**: All collocation methods work for arbitrary C
**Effort**: 5 days

---

## 🎯 **Phase 5: Integration and Testing** (Weeks 7-8)

### **Task 5.1: Integration Testing**
**Implementation Plan**:
1. **End-to-end testing**
   - Test thermal efficiency adapter with real complex optimizer
   - Validate motion law extraction
   - Verify objective value calculation

2. **Performance testing**
   - Benchmark optimization performance
   - Validate convergence behavior
   - Test error handling

**Deliverable**: Fully integrated system
**Success Criteria**: All tests pass with real implementations
**Effort**: 3 days

### **Task 5.2: Documentation Update**
**Implementation Plan**:
1. **Update API documentation**
2. **Create usage examples**
3. **Update configuration guides**

**Deliverable**: Complete documentation
**Success Criteria**: Documentation matches implementation
**Effort**: 2 days

---

## 📊 **Resource Allocation**

### **Team Structure**
- **Lead Developer**: Thermal efficiency adapter, collocation optimization
- **Physics Engineer**: Physics components, system builder
- **Optimization Engineer**: Complex gas optimizer enhancements
- **QA Engineer**: Testing and validation

### **Timeline Summary**
- **Week 1**: Thermal efficiency adapter + collocation optimization
- **Week 2**: Configuration values + collocation completion
- **Week 3**: Time kinematics + kinematic constraints
- **Week 4**: System builder + physics integration
- **Week 5**: Objective functions + NLP constraints
- **Week 6**: Collocation methods + complex optimizer
- **Week 7**: Integration testing + performance validation
- **Week 8**: Documentation + final testing

---

## 🎯 **Success Metrics**

### **Phase 2 Success Criteria**
- [ ] Thermal efficiency adapter extracts real motion law data
- [ ] Objective values reflect actual thermal efficiency
- [ ] Collocation optimization supports all constraint types
- [ ] Configuration values are realistic and validated

### **Phase 3 Success Criteria**
- [ ] Physics components return real calculations
- [ ] System builder creates functional systems
- [ ] All components integrate properly

### **Phase 4 Success Criteria**
- [ ] Complex gas optimizer has complete objective functions
- [ ] All NLP constraints are implemented
- [ ] Collocation methods support arbitrary C values

### **Phase 5 Success Criteria**
- [ ] End-to-end system works with real implementations
- [ ] All tests pass
- [ ] Documentation is complete and accurate

---

## 🚨 **Risk Mitigation**

### **Technical Risks**
1. **Complex optimizer integration complexity**
   - **Mitigation**: Start with simple data extraction, iterate
   - **Fallback**: Maintain fallback mode for testing

2. **Performance degradation**
   - **Mitigation**: Benchmark at each phase
   - **Fallback**: Optimize critical paths

3. **Integration issues**
   - **Mitigation**: Continuous integration testing
   - **Fallback**: Modular testing approach

### **Schedule Risks**
1. **Scope creep**
   - **Mitigation**: Strict phase boundaries
   - **Fallback**: Defer non-critical features

2. **Resource constraints**
   - **Mitigation**: Parallel development where possible
   - **Fallback**: Prioritize critical path items

---

## 📋 **Implementation Checklist**

### **Phase 2 Checklist**
- [ ] **Task 2.1.1**: Motion law data extraction implemented
- [ ] **Task 2.1.2**: Objective value calculation implemented
- [ ] **Task 2.2.1**: Non-motion law problem support added
- [ ] **Task 2.2.2**: Objective value calculation completed
- [ ] **Task 2.3.1**: Realistic configuration values set

### **Phase 3 Checklist**
- [ ] **Task 3.1**: Time kinematics component implemented
- [ ] **Task 3.2**: Kinematic constraints component implemented
- [ ] **Task 3.3**: System builder implemented

### **Phase 4 Checklist**
- [ ] **Task 4.1**: Objective function placeholders resolved
- [ ] **Task 4.2**: NLP constraint placeholders resolved
- [ ] **Task 4.3**: Collocation method limitations resolved

### **Phase 5 Checklist**
- [ ] **Task 5.1**: Integration testing completed
- [ ] **Task 5.2**: Documentation updated

---

## 🎯 **Next Steps**

1. **Immediate (This Week)**:
   - Start Task 2.1.1: Motion law data extraction
   - Begin Task 2.1.2: Objective value calculation
   - Set up development environment for complex optimizer integration

2. **Week 1**:
   - Complete thermal efficiency adapter implementation
   - Start collocation optimization enhancements
   - Begin configuration value research

3. **Week 2**:
   - Complete collocation optimization
   - Finalize configuration values
   - Begin physics component planning

**Total Effort**: 8 weeks, 4 developers
**Critical Path**: Thermal efficiency adapter → Collocation optimization → Physics components → Integration testing
