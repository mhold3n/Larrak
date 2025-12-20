# Development Roadmap

This document provides the implementation roadmap and integration plans for the Larrak/CamPro system.

## Current Status

The system has evolved from a simple motion law optimizer to a comprehensive thermal efficiency-focused optimization framework with:

- ✅ Modular library architecture
- ✅ CasADi-based direct collocation
- ✅ Three-layer cascaded optimization
- ✅ Shell-based extensible optimizer design
- 🔄 Full CasADi physics integration (in progress)
- 🔄 Complex gas optimizer integration (in progress)

---

## Complex Gas Optimizer Integration

### Overview

Integration of the complex gas optimizer from `OP-dynamic-gas-simulator/` replaces simple optimization with thermal efficiency-focused acceleration zone optimization.

### Architecture

```
Integrated System:
├── campro/optimization/
│   ├── motion_law_optimizer.py     # Updated with thermal efficiency support
│   ├── thermal_efficiency_adapter.py  # Bridge to complex optimizer
│   ├── unified_framework.py        # Updated framework
│   └── motion_law.py               # Data structures
├── cfg/
│   └── thermal_efficiency_config.yaml  # Configuration
└── OP-dynamic-gas-simulator/       # Complex gas optimizer
    └── campro/freepiston/opt/
        ├── optimization_lib.py     # Main complex optimizer
        ├── config_factory.py       # Configuration management
        ├── nlp.py                  # NLP construction
        └── obj.py                  # Objective functions
```

### Key Components

#### Thermal Efficiency Adapter

**File**: `campro/optimization/thermal_efficiency_adapter.py`

**Purpose**: Bridge between existing system and complex gas optimizer.

```python
@dataclass
class ThermalEfficiencyConfig:
    # Engine geometry
    bore: float = 0.082  # m
    stroke: float = 0.180  # m
    compression_ratio: float = 12.0
    
    # Thermodynamics
    gamma: float = 1.34
    R: float = 287.0  # J/(kg K)
    
    # Optimization parameters
    collocation_points: int = 30
    collocation_degree: int = 3
    
    # Thermal efficiency weights
    thermal_efficiency_weight: float = 1.0
    smoothness_weight: float = 0.01
```

#### Motion Law Optimizer Updates

**Key Changes**:
- Add `use_thermal_efficiency` parameter
- Integrate thermal efficiency adapter
- Maintain backward compatibility
- Route optimization based on configuration

```python
class MotionLawOptimizer(BaseOptimizer):
    def __init__(self, use_thermal_efficiency: bool = False):
        if use_thermal_efficiency:
            from .thermal_efficiency_adapter import ThermalEfficiencyAdapter
            self.thermal_adapter = ThermalEfficiencyAdapter()
```

---

## Implementation Phases

### Phase 1: Core Integration (Weeks 1-2)

| Step | Task | Priority | Effort |
|------|------|----------|--------|
| 1.1 | Create Thermal Efficiency Adapter | High | 3 days |
| 1.2 | Update Motion Law Optimizer | High | 2 days |
| 1.3 | Update Unified Framework | High | 2 days |
| 1.4 | Create Configuration System | Medium | 1 day |

**Deliverables**:
- Thermal efficiency adapter with complex optimizer integration
- Updated motion law optimizer with backward compatibility
- Configuration management system

### Phase 2: GUI Integration (Weeks 3-4)

| Step | Task | Priority | Effort |
|------|------|----------|--------|
| 2.1 | Add Thermal Efficiency Tab to GUI | High | 3 days |
| 2.2 | Implement Config Panel | Medium | 2 days |
| 2.3 | Add Results Visualization | Medium | 2 days |
| 2.4 | Update Comparison Features | Low | 1 day |

**Deliverables**:
- New thermal efficiency tab in GUI
- Configuration panel for thermal parameters
- Results visualization with thermal metrics

### Phase 3: Testing & Validation (Weeks 5-6)

| Step | Task | Priority | Effort |
|------|------|----------|--------|
| 3.1 | Unit Tests for Adapter | High | 2 days |
| 3.2 | Integration Tests | High | 2 days |
| 3.3 | Performance Tests | Medium | 2 days |
| 3.4 | Validation Against Reference | High | 2 days |

**Deliverables**:
- Comprehensive test suite
- Performance benchmarks
- Validation results

### Phase 4: Documentation & Polish (Weeks 7-8)

| Step | Task | Priority | Effort |
|------|------|----------|--------|
| 4.1 | API Documentation | High | 2 days |
| 4.2 | User Guide Updates | High | 2 days |
| 4.3 | Code Cleanup | Medium | 2 days |
| 4.4 | Final Review | High | 2 days |

---

## CasADi Physics Integration

### Current State

**Hybrid Approach**:
- CasADi provides smoothness objectives
- Python physics models used for validation
- No automatic differentiation for physics

### Target State

**Full CasADi Integration**:
- Physics as symbolic MX graphs
- Automatic differentiation for exact gradients
- IPOPT exploits sparsity patterns
- Expected 2-3x speedup

### Integration Phases

#### Phase 1: Torque Calculation Port (Complete)
- ✅ Crank-piston kinematics in CasADi
- ✅ Piston force calculation
- ✅ Torque integration

#### Phase 2: Side Loading Analysis
- [ ] Connecting rod force decomposition
- [ ] Cylinder wall loading
- [ ] Journal bearing loads

#### Phase 3: Litvin Metrics
- [ ] Slip integral calculation
- [ ] Contact length evaluation
- [ ] Hertzian stress constraints

#### Phase 4: Full NLP Integration
- [ ] Replace hybrid callbacks
- [ ] Enable exact gradients
- [ ] Exploit sparsity patterns

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Complex optimizer compatibility | Medium | High | Thorough testing, fallback modes |
| Performance regression | Low | Medium | Benchmarking, profiling |
| CasADi NaN issues | Medium | High | Careful formulation, diagnostics |
| GUI integration issues | Low | Low | Incremental integration |

### Mitigation Strategies

1. **Backward Compatibility**: Maintain simple optimization as fallback
2. **Feature Flags**: `use_thermal_efficiency` toggle for gradual rollout
3. **Incremental Testing**: Test each phase before proceeding
4. **Performance Monitoring**: Benchmark at each integration stage

---

## Success Criteria

### Phase 1 (Core Integration)
- [ ] Thermal adapter passes unit tests
- [ ] Simple and thermal modes work independently
- [ ] No regression in existing functionality

### Phase 2 (GUI Integration)
- [ ] Thermal efficiency tab functional
- [ ] Configuration updates work correctly
- [ ] Results display properly

### Phase 3 (Testing)
- [ ] 90%+ test coverage for new code
- [ ] Performance within 20% of baseline
- [ ] Validation results match reference

### Phase 4 (Documentation)
- [ ] API documentation complete
- [ ] User guide updated
- [ ] Code review passed

---

## Related Documentation

- **System Architecture**: See `architecture/overview.md`
- **Optimization Architecture**: See `architecture/optimization.md`
- **CasADi API**: See `architecture/casadi-api.md`
- **Project Status**: See `development/project-status.md`






