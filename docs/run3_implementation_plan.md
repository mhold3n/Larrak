# Run 3 Implementation Plan: Crank Center Optimization

## Overview

This plan addresses the critical gap in the three-run optimization system where Run 3 (crank center optimization for torque maximization and side-loading minimization) is not currently achievable. The analysis revealed that while Runs 1 and 2 are fully implemented, Run 3 lacks the necessary physics models, optimization variables, and objective functions.

## Current State Analysis

### ✅ What Works (Runs 1 & 2)
- **Run 1**: Motion law optimization with free-piston idealization, constant temperature/load assumptions
- **Run 2**: Litvin profile synthesis from motion law with complete gear geometry

### ❌ What's Missing (Run 3)
- Crank center position variable relative to Litvin gear center
- Piston torque computation from motion law and gear geometry
- Cylinder side-loading model integration
- Physics-based objective function for torque/side-load optimization
- Proper tertiary optimizer integration in unified framework

## Implementation Phases

### Phase 1: Physics Foundation
**Objective**: Establish the fundamental physics models needed for crank center optimization

#### 1.1 Torque Computation Module
**File**: `campro/physics/mechanics/torque_analysis.py`

**Components**:
- `PistonTorqueCalculator` class
- Methods for computing instantaneous torque from:
  - Piston force (from motion law + load profile)
  - Connecting rod geometry and kinematics
  - Litvin gear pressure angles and contact ratios
  - Crank center offset effects

**Key Functions**:
```python
def compute_instantaneous_torque(
    piston_force: float,
    crank_angle: float,
    crank_radius: float,
    rod_length: float,
    crank_center_offset: Tuple[float, float],
    pressure_angle: float
) -> float

def compute_cycle_average_torque(
    motion_law: Dict[str, np.ndarray],
    load_profile: np.ndarray,
    gear_geometry: LitvinGearGeometry,
    crank_center_offset: Tuple[float, float]
) -> float
```

#### 1.2 Side-Loading Model
**File**: `campro/physics/mechanics/side_loading.py`

**Components**:
- `SideLoadAnalyzer` class
- Integration with existing `piston_slap_force` from OP-dynamic-gas-simulator
- Enhanced lateral force computation considering:
  - Crank center offset effects
  - Connecting rod angle variations
  - Piston clearance dynamics

**Key Functions**:
```python
def compute_side_load_profile(
    motion_law: Dict[str, np.ndarray],
    crank_center_offset: Tuple[float, float],
    piston_geometry: Dict[str, float]
) -> np.ndarray

def compute_side_load_penalty(
    side_load_profile: np.ndarray,
    compression_phases: np.ndarray,
    combustion_phases: np.ndarray
) -> float
```

#### 1.3 Kinematic Analysis
**File**: `campro/physics/kinematics/crank_kinematics.py`

**Components**:
- `CrankKinematics` class
- Methods for computing connecting rod angles and velocities
- Crank center offset effects on piston motion
- Integration with existing motion law data

### Phase 2: Enhanced Tertiary Optimizer
**Objective**: Create a physics-aware tertiary optimizer that can optimize crank center position

#### 2.1 Crank Center Optimizer
**File**: `campro/optimization/crank_center_optimizer.py`

**Components**:
- `CrankCenterOptimizer` class inheriting from `BaseOptimizer`
- Integration with torque and side-loading models
- Proper handling of Litvin gear geometry from secondary optimization
- Multi-objective optimization for torque maximization and side-loading minimization

**Key Features**:
```python
@dataclass
class CrankCenterOptimizationConstraints:
    crank_center_x_min: float = -50.0
    crank_center_x_max: float = 50.0
    crank_center_y_min: float = -50.0
    crank_center_y_max: float = 50.0
    min_torque_output: float = 100.0  # N⋅m
    max_side_load: float = 1000.0     # N

@dataclass
class CrankCenterOptimizationTargets:
    maximize_torque: bool = True
    minimize_side_loading: bool = True
    minimize_side_loading_during_compression: bool = True
    minimize_side_loading_during_combustion: bool = True
    torque_weight: float = 1.0
    side_load_weight: float = 0.8
    compression_side_load_weight: float = 1.2
    combustion_side_load_weight: float = 1.5
```

#### 2.2 Litvin-Centric Design
**Integration Points**:
- Use actual synthesized `R_psi` and `psi` from secondary optimization
- Compute Litvin gear center as reference point for crank center offset
- Incorporate pressure angles and contact ratios in torque calculations
- Ensure crank center optimization respects gear geometry constraints

### Phase 3: Unified Framework Integration
**Objective**: Integrate the new crank center optimizer into the unified optimization framework

#### 3.1 Framework Modification
**File**: `campro/optimization/unified_framework.py`

**Changes**:
- Replace `SunGearOptimizer` with `CrankCenterOptimizer` in tertiary stage
- Update `_optimize_tertiary()` method to use crank center optimization
- Add crank center parameters to `UnifiedOptimizationData`
- Update constraints and targets to include torque/side-loading objectives

**Key Modifications**:
```python
# In UnifiedOptimizationData
tertiary_crank_center_x: Optional[float] = None
tertiary_crank_center_y: Optional[float] = None
tertiary_torque_output: Optional[float] = None
tertiary_side_load_penalty: Optional[float] = None

# In UnifiedOptimizationConstraints
crank_center_x_min: float = -50.0
crank_center_x_max: float = 50.0
crank_center_y_min: float = -50.0
crank_center_y_max: float = 50.0

# In UnifiedOptimizationTargets
maximize_torque_output: bool = True
minimize_side_loading: bool = True
```

#### 3.2 Data Flow Enhancement
**Components**:
- Ensure secondary optimization results (Litvin geometry) are properly passed to tertiary
- Add torque and side-loading metrics to optimization summary
- Update convergence tracking for physics-based objectives

### Phase 4: Testing and Validation
**Objective**: Ensure the complete three-run optimization system works correctly

#### 4.1 Unit Tests
**Files**: `tests/test_crank_center_optimization.py`

**Test Coverage**:
- Torque computation accuracy
- Side-loading model validation
- Crank center optimization convergence
- Integration with existing motion law and Litvin synthesis
- Multi-objective optimization behavior

#### 4.2 Integration Tests
**Files**: `tests/test_three_run_optimization.py`

**Test Scenarios**:
- Complete three-run optimization pipeline
- Verification that Run 3 objectives are met
- Performance comparison with baseline (no crank center optimization)
- Edge cases and constraint handling

#### 4.3 Demo Scripts
**Files**: `scripts/three_run_optimization_demo.py`

**Features**:
- End-to-end demonstration of all three runs
- Visualization of torque and side-loading results
- Comparison plots showing optimization improvements
- Performance metrics and convergence analysis

### Phase 5: Documentation and Examples
**Objective**: Provide comprehensive documentation and usage examples

#### 5.1 API Documentation
**Files**: 
- `docs/crank_center_optimization_api.md`
- `docs/three_run_optimization_guide.md`

**Content**:
- Complete API reference for new modules
- Usage examples and best practices
- Parameter tuning guidelines
- Troubleshooting guide

#### 5.2 Example Configurations
**Files**: `examples/crank_center_optimization/`

**Examples**:
- Basic three-run optimization
- High-torque configuration
- Low-side-loading configuration
- Balanced optimization setup

## Implementation Dependencies

### Phase Dependencies
- **Phase 1** → **Phase 2**: Physics models must be complete before optimizer implementation
- **Phase 2** → **Phase 3**: Optimizer must be functional before framework integration
- **Phase 3** → **Phase 4**: Framework integration must be complete before testing
- **Phase 4** → **Phase 5**: Testing must pass before documentation

### External Dependencies
- Integration with existing `LitvinGearGeometry` from secondary optimization
- Compatibility with `MotionLawResult` from primary optimization
- Proper handling of `OptimizationResult` data structures
- Integration with existing logging and storage systems

## Success Criteria

### Phase 1 Success
- [ ] Torque computation module produces physically reasonable results
- [ ] Side-loading model integrates with existing piston dynamics
- [ ] Kinematic analysis handles crank center offsets correctly

### Phase 2 Success
- [ ] Crank center optimizer converges to optimal solutions
- [ ] Multi-objective optimization balances torque and side-loading
- [ ] Integration with Litvin geometry is seamless

### Phase 3 Success
- [ ] Unified framework runs complete three-run optimization
- [ ] Tertiary stage produces meaningful crank center offsets
- [ ] All optimization data flows correctly between stages

### Phase 4 Success
- [ ] All unit tests pass
- [ ] Integration tests demonstrate end-to-end functionality
- [ ] Demo scripts show clear optimization improvements

### Phase 5 Success
- [ ] Documentation is complete and accurate
- [ ] Examples are functional and educational
- [ ] API is well-documented and user-friendly

## Risk Mitigation

### Technical Risks
- **Physics Model Accuracy**: Validate against known analytical solutions and literature
- **Optimization Convergence**: Use robust optimization methods and proper initialization
- **Integration Complexity**: Maintain backward compatibility and clear interfaces

### Implementation Risks
- **Scope Creep**: Focus on core functionality first, add features incrementally
- **Performance Impact**: Profile optimization performance and optimize critical paths
- **Testing Coverage**: Ensure comprehensive test coverage for all new functionality

## Future Enhancements

### Advanced Features (Post-Phase 5)
- Dynamic load profiles (beyond constant load assumption)
- Thermal effects on side-loading
- Multi-cylinder optimization
- Real-time optimization capabilities
- Advanced visualization tools

### Integration Opportunities
- Combustion physics integration
- Advanced material models
- Manufacturing constraint integration
- Cost optimization objectives

## Conclusion

This implementation plan provides a structured approach to achieving the missing Run 3 functionality. By following the phased approach, we can systematically build the physics models, optimization capabilities, and integration points needed to make crank center optimization for torque maximization and side-loading minimization a reality.

The plan ensures that the existing Runs 1 and 2 functionality remains intact while adding the sophisticated physics-based optimization capabilities required for Run 3. Upon completion, the Larrak project will have a complete three-run optimization system that can optimize motion laws, generate Litvin profiles, and optimize crank center placement for optimal engine performance.