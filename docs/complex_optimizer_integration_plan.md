# Complex Gas Optimizer Integration Plan

## Overview

This document outlines the comprehensive plan to integrate the complex gas optimizer system from `OP-dynamic-gas-simulator/` into the existing Larrak architecture, replacing the simple phase 1 optimization with a thermal efficiency-focused system for acceleration zone optimization.

## Current State Analysis

### Existing Simple Optimization System
- **Location**: `CamPro_OptimalMotion.py`, `campro/optimization/motion_law_optimizer.py`
- **Method**: Fake analytical solutions with basic collocation
- **Focus**: Simple motion law generation (minimum jerk, time, energy)
- **Physics**: No gas dynamics, heat transfer, or thermal efficiency
- **Integration**: Used in `cam_motion_gui.py` and `unified_framework.py`

### Complex Gas Optimizer System
- **Location**: `OP-dynamic-gas-simulator/campro/freepiston/opt/`
- **Method**: Real collocation-based optimization with full physics
- **Focus**: Thermal efficiency, indicated work, scavenging efficiency
- **Physics**: Full 1D gas dynamics, heat transfer, mechanical dynamics
- **Integration**: Standalone system with comprehensive API

## Integration Strategy

### Phase 1: Core Integration (Thermal Efficiency Focus)

#### 1.1 Create Thermal Efficiency Adapter
**File**: `campro/optimization/thermal_efficiency_adapter.py`

```python
"""
Thermal efficiency adapter for integrating complex gas optimizer.

This adapter provides a bridge between the existing motion law optimization
system and the complex gas optimizer, focusing specifically on thermal
efficiency optimization for acceleration zones.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from campro.logging import get_logger
from campro.optimization.motion_law import MotionLawConstraints, MotionLawResult, MotionType
from campro.optimization.base import BaseOptimizer, OptimizationResult, OptimizationStatus

# Import complex optimizer components
from campro.freepiston.opt.optimization_lib import (
    MotionLawOptimizer as ComplexMotionLawOptimizer,
    OptimizationConfig,
    ConfigFactory
)
from campro.freepiston.opt.config_factory import create_optimization_scenario

log = get_logger(__name__)


@dataclass
class ThermalEfficiencyConfig:
    """Configuration for thermal efficiency optimization."""
    
    # Engine geometry
    bore: float = 0.082  # m
    stroke: float = 0.180  # m
    compression_ratio: float = 12.0
    clearance_volume: float = 3.2e-5  # m^3
    
    # Thermodynamics
    gamma: float = 1.34
    R: float = 287.0  # J/(kg K)
    cp: float = 1005.0  # J/(kg K)
    
    # Optimization parameters
    collocation_points: int = 30
    collocation_degree: int = 3
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Thermal efficiency weights
    thermal_efficiency_weight: float = 1.0
    smoothness_weight: float = 0.01
    short_circuit_weight: float = 2.0
    
    # Model configuration
    use_1d_gas_model: bool = True
    n_cells: int = 50


class ThermalEfficiencyAdapter(BaseOptimizer):
    """
    Adapter for thermal efficiency optimization using complex gas optimizer.
    
    This adapter bridges the existing motion law optimization system with the
    complex gas optimizer, focusing specifically on thermal efficiency for
    acceleration zone optimization.
    """
    
    def __init__(self, config: Optional[ThermalEfficiencyConfig] = None):
        super().__init__("ThermalEfficiencyAdapter")
        self.config = config or ThermalEfficiencyConfig()
        self.complex_optimizer: Optional[ComplexMotionLawOptimizer] = None
        self._setup_complex_optimizer()
    
    def _setup_complex_optimizer(self) -> None:
        """Setup the complex gas optimizer with thermal efficiency focus."""
        # Create thermal efficiency scenario configuration
        complex_config = create_optimization_scenario("efficiency")
        
        # Override with our specific configuration
        complex_config.geometry.update({
            "bore": self.config.bore,
            "stroke": self.config.stroke,
            "compression_ratio": self.config.compression_ratio,
            "clearance_volume": self.config.clearance_volume,
        })
        
        complex_config.thermodynamics.update({
            "gamma": self.config.gamma,
            "R": self.config.R,
            "cp": self.config.cp,
        })
        
        complex_config.num = {
            "K": self.config.collocation_points,
            "C": self.config.collocation_degree
        }
        
        complex_config.objective.update({
            "method": "thermal_efficiency",
            "w": {
                "smooth": self.config.smoothness_weight,
                "short_circuit": self.config.short_circuit_weight,
                "eta_th": self.config.thermal_efficiency_weight,
            }
        })
        
        complex_config.solver["ipopt"].update({
            "max_iter": self.config.max_iterations,
            "tol": self.config.tolerance,
        })
        
        # Enable 1D gas model if requested
        if self.config.use_1d_gas_model:
            complex_config.model_type = "1d"
            complex_config.use_1d_gas = True
            complex_config.n_cells = self.config.n_cells
        
        # Create the complex optimizer
        self.complex_optimizer = ComplexMotionLawOptimizer(complex_config)
        
        log.info("Thermal efficiency adapter configured with complex gas optimizer")
    
    def optimize(self, objective, constraints: MotionLawConstraints, 
                initial_guess: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> OptimizationResult:
        """
        Optimize motion law for thermal efficiency.
        
        Args:
            objective: Objective function (ignored, uses thermal efficiency)
            constraints: Motion law constraints
            initial_guess: Initial guess (optional)
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult with thermal efficiency optimization
        """
        log.info("Starting thermal efficiency optimization for acceleration zones")
        
        try:
            # Run complex optimization
            complex_result = self.complex_optimizer.optimize_with_validation(validate=True)
            
            # Convert to standard OptimizationResult
            if complex_result.success:
                # Extract motion law data from complex result
                motion_law_data = self._extract_motion_law_data(complex_result, constraints)
                
                return OptimizationResult(
                    status=OptimizationStatus.CONVERGED,
                    objective_value=complex_result.objective_value,
                    solution=motion_law_data,
                    iterations=complex_result.iterations,
                    solve_time=complex_result.cpu_time,
                    metadata={
                        "thermal_efficiency": complex_result.performance_metrics.get("thermal_efficiency", 0.0),
                        "indicated_work": complex_result.performance_metrics.get("indicated_work", 0.0),
                        "max_pressure": complex_result.performance_metrics.get("max_pressure", 0.0),
                        "max_temperature": complex_result.performance_metrics.get("max_temperature", 0.0),
                        "optimization_method": "thermal_efficiency",
                        "complex_optimizer": True
                    }
                )
            else:
                log.warning(f"Complex optimization failed: {complex_result.message}")
                return OptimizationResult(
                    status=OptimizationStatus.FAILED,
                    objective_value=float('inf'),
                    solution=None,
                    iterations=complex_result.iterations,
                    solve_time=complex_result.cpu_time,
                    metadata={"error": complex_result.message}
                )
                
        except Exception as e:
            log.error(f"Thermal efficiency optimization failed: {e}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution=None,
                iterations=0,
                solve_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _extract_motion_law_data(self, complex_result, constraints: MotionLawConstraints) -> Dict[str, Any]:
        """Extract motion law data from complex optimization result."""
        # This would extract the optimized motion law from the complex result
        # and format it for the existing system
        
        # For now, return a placeholder structure
        # In practice, this would extract the actual optimized motion law
        return {
            "theta": np.linspace(0, 2*np.pi, 360),  # Cam angle
            "x": np.zeros(360),  # Position (to be filled from complex result)
            "v": np.zeros(360),  # Velocity (to be filled from complex result)
            "a": np.zeros(360),  # Acceleration (to be filled from complex result)
            "j": np.zeros(360),  # Jerk (to be filled from complex result)
            "constraints": constraints.to_dict(),
            "optimization_type": "thermal_efficiency",
            "thermal_efficiency": complex_result.performance_metrics.get("thermal_efficiency", 0.0)
        }
    
    def solve_motion_law(self, constraints: MotionLawConstraints, 
                        motion_type: MotionType) -> MotionLawResult:
        """
        Solve motion law optimization with thermal efficiency focus.
        
        Args:
            constraints: Motion law constraints
            motion_type: Motion type (ignored, always uses thermal efficiency)
            
        Returns:
            MotionLawResult with thermal efficiency optimization
        """
        log.info(f"Solving thermal efficiency motion law (ignoring motion_type: {motion_type})")
        
        # Run optimization
        result = self.optimize(None, constraints)
        
        if result.status == OptimizationStatus.CONVERGED:
            # Convert to MotionLawResult
            return MotionLawResult(
                theta=result.solution["theta"],
                x=result.solution["x"],
                v=result.solution["v"],
                a=result.solution["a"],
                j=result.solution["j"],
                objective_value=result.objective_value,
                convergence_status="converged",
                iterations=result.iterations,
                solve_time=result.solve_time,
                constraints=constraints,
                motion_type=motion_type,
                metadata=result.metadata
            )
        else:
            # Return failed result
            return MotionLawResult(
                theta=np.linspace(0, 2*np.pi, 360),
                x=np.zeros(360),
                v=np.zeros(360),
                a=np.zeros(360),
                j=np.zeros(360),
                objective_value=float('inf'),
                convergence_status="failed",
                iterations=result.iterations,
                solve_time=result.solve_time,
                constraints=constraints,
                motion_type=motion_type,
                metadata=result.metadata
            )
```

#### 1.2 Update Motion Law Optimizer
**File**: `campro/optimization/motion_law_optimizer.py` (modify existing)

```python
# Add thermal efficiency support to existing MotionLawOptimizer

class MotionLawOptimizer(BaseOptimizer):
    def __init__(self, name: str = "MotionLawOptimizer", use_thermal_efficiency: bool = False):
        super().__init__(name)
        self.use_thermal_efficiency = use_thermal_efficiency
        
        if use_thermal_efficiency:
            # Use complex gas optimizer for thermal efficiency
            from .thermal_efficiency_adapter import ThermalEfficiencyAdapter
            self.thermal_adapter = ThermalEfficiencyAdapter()
        else:
            # Use existing simple optimization
            self.collocation_method = "legendre"
            self.degree = 3
            self.n_points = 100
            self.tolerance = 1e-6
            self.max_iterations = 1000
        
        self._is_configured = True
    
    def solve_motion_law(self, constraints: MotionLawConstraints, 
                        motion_type: MotionType) -> MotionLawResult:
        """Solve motion law with optional thermal efficiency optimization."""
        
        if self.use_thermal_efficiency:
            # Use complex gas optimizer for thermal efficiency
            log.info("Using thermal efficiency optimization")
            return self.thermal_adapter.solve_motion_law(constraints, motion_type)
        else:
            # Use existing simple optimization
            log.info("Using simple motion law optimization")
            return self._solve_simple_motion_law(constraints, motion_type)
```

#### 1.3 Update Unified Framework
**File**: `campro/optimization/unified_framework.py` (modify existing)

```python
# Add thermal efficiency option to UnifiedOptimizationSettings

@dataclass
class UnifiedOptimizationSettings:
    # ... existing fields ...
    
    # Thermal efficiency optimization
    use_thermal_efficiency: bool = False
    thermal_efficiency_config: Optional[Dict[str, Any]] = None

# Update UnifiedOptimizationFramework to support thermal efficiency

class UnifiedOptimizationFramework:
    def __init__(self, name: str, settings: Optional[UnifiedOptimizationSettings] = None):
        # ... existing initialization ...
        
        # Setup thermal efficiency optimization if requested
        if settings and settings.use_thermal_efficiency:
            from .thermal_efficiency_adapter import ThermalEfficiencyAdapter
            self.thermal_adapter = ThermalEfficiencyAdapter()
            log.info("Thermal efficiency optimization enabled")
    
    def optimize_primary(self, constraints: UnifiedOptimizationConstraints, 
                        targets: UnifiedOptimizationTargets) -> OptimizationResult:
        """Optimize primary motion law with optional thermal efficiency focus."""
        
        if self.settings and self.settings.use_thermal_efficiency:
            # Use thermal efficiency optimization
            log.info("Running primary optimization with thermal efficiency focus")
            return self.thermal_adapter.optimize(None, constraints.primary_constraints)
        else:
            # Use existing simple optimization
            return self._optimize_primary_simple(constraints, targets)
```

### Phase 2: GUI Integration

#### 2.1 Update Main GUI
**File**: `cam_motion_gui.py` (modify existing)

```python
# Add thermal efficiency option to GUI

class CamMotionGUI:
    def _create_variables(self):
        """Create Tkinter variables for input fields."""
        variables = {
            # ... existing variables ...
            
            # Thermal efficiency optimization
            'use_thermal_efficiency': tk.BooleanVar(value=False),
            'thermal_efficiency_weight': tk.DoubleVar(value=1.0),
            'use_1d_gas_model': tk.BooleanVar(value=True),
            'n_cells': tk.IntVar(value=50),
        }
        return variables
    
    def _create_control_panel(self):
        """Create control panel with thermal efficiency options."""
        # ... existing control panel code ...
        
        # Add thermal efficiency section
        thermal_frame = ttk.LabelFrame(self.control_panel, text="Thermal Efficiency Optimization", padding="5")
        thermal_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Thermal efficiency checkbox
        ttk.Checkbutton(
            thermal_frame, 
            text="Use Thermal Efficiency Optimization",
            variable=self.variables['use_thermal_efficiency']
        ).grid(row=0, column=0, sticky="w")
        
        # Thermal efficiency weight
        ttk.Label(thermal_frame, text="Thermal Efficiency Weight:").grid(row=1, column=0, sticky="w")
        ttk.Scale(
            thermal_frame, 
            from_=0.1, to=5.0, 
            variable=self.variables['thermal_efficiency_weight'],
            orient="horizontal"
        ).grid(row=1, column=1, sticky="ew")
        
        # 1D gas model options
        ttk.Checkbutton(
            thermal_frame, 
            text="Use 1D Gas Model",
            variable=self.variables['use_1d_gas_model']
        ).grid(row=2, column=0, sticky="w")
        
        ttk.Label(thermal_frame, text="Number of Cells:").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(
            thermal_frame, 
            from_=20, to=100, 
            textvariable=self.variables['n_cells']
        ).grid(row=3, column=1, sticky="ew")
    
    def _run_optimization(self):
        """Run optimization with thermal efficiency support."""
        # ... existing optimization code ...
        
        # Check if thermal efficiency optimization is enabled
        if self.variables['use_thermal_efficiency'].get():
            log.info("Running thermal efficiency optimization")
            
            # Configure thermal efficiency settings
            settings = UnifiedOptimizationSettings(
                use_thermal_efficiency=True,
                thermal_efficiency_config={
                    "thermal_efficiency_weight": self.variables['thermal_efficiency_weight'].get(),
                    "use_1d_gas_model": self.variables['use_1d_gas_model'].get(),
                    "n_cells": self.variables['n_cells'].get(),
                }
            )
            
            # Update framework settings
            self.unified_framework.settings = settings
            
            # Run optimization
            result = self.unified_framework.optimize_primary(constraints, targets)
            
            if result.status == OptimizationStatus.CONVERGED:
                log.info(f"Thermal efficiency optimization successful: {result.metadata.get('thermal_efficiency', 0.0):.3f}")
            else:
                log.warning("Thermal efficiency optimization failed")
        else:
            # Use existing simple optimization
            result = self.unified_framework.optimize_primary(constraints, targets)
```

### Phase 3: Configuration and Testing

#### 3.1 Create Configuration Files
**File**: `cfg/thermal_efficiency_config.yaml`

```yaml
# Thermal efficiency optimization configuration

# Engine geometry
geometry:
  bore: 0.082  # m
  stroke: 0.180  # m
  compression_ratio: 12.0
  clearance_volume: 3.2e-5  # m^3
  mass: 1.0  # kg
  rod_mass: 0.5  # kg
  rod_length: 0.15  # m

# Thermodynamics
thermodynamics:
  gamma: 1.34
  R: 287.0  # J/(kg K)
  cp: 1005.0  # J/(kg K)
  cv: 717.5  # J/(kg K)

# Optimization parameters
optimization:
  method: thermal_efficiency
  collocation_points: 30
  collocation_degree: 3
  max_iterations: 1000
  tolerance: 1e-6
  
  # Objective weights
  weights:
    thermal_efficiency: 1.0
    smoothness: 0.01
    short_circuit: 2.0

# Model configuration
model:
  use_1d_gas: true
  n_cells: 50
  mesh_refinement: true

# Solver settings
solver:
  ipopt:
    max_iter: 1000
    tol: 1e-6
    linear_solver: ma57
    hessian_approximation: limited-memory

# Validation
validation:
  check_convergence: true
  check_physics: true
  check_constraints: true
  thermal_efficiency_min: 0.3
  max_pressure_limit: 12e6  # Pa
  max_temperature_limit: 2600.0  # K
```

#### 3.2 Create Integration Tests
**File**: `tests/test_thermal_efficiency_integration.py`

```python
"""
Integration tests for thermal efficiency optimization.

Tests the integration between the existing motion law system and the
complex gas optimizer for thermal efficiency optimization.
"""

import pytest
import numpy as np
from pathlib import Path

from campro.optimization.thermal_efficiency_adapter import (
    ThermalEfficiencyAdapter, 
    ThermalEfficiencyConfig
)
from campro.optimization.motion_law import MotionLawConstraints, MotionType
from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
    UnifiedOptimizationConstraints
)


class TestThermalEfficiencyIntegration:
    """Test thermal efficiency integration."""
    
    def test_thermal_efficiency_adapter_creation(self):
        """Test thermal efficiency adapter creation."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        
        assert adapter is not None
        assert adapter.complex_optimizer is not None
        assert adapter.config.bore == 0.082
        assert adapter.config.thermal_efficiency_weight == 1.0
    
    def test_thermal_efficiency_optimization(self):
        """Test thermal efficiency optimization."""
        config = ThermalEfficiencyConfig()
        config.collocation_points = 10  # Small for testing
        config.max_iterations = 100  # Small for testing
        
        adapter = ThermalEfficiencyAdapter(config)
        
        # Create test constraints
        constraints = MotionLawConstraints(
            stroke=20.0,  # mm
            upstroke_duration_percent=60.0,
            zero_accel_duration_percent=0.0
        )
        
        # Run optimization
        result = adapter.solve_motion_law(constraints, MotionType.MINIMUM_JERK)
        
        # Check results
        assert result is not None
        assert len(result.theta) == 360
        assert len(result.x) == 360
        assert len(result.v) == 360
        assert len(result.a) == 360
        assert len(result.j) == 360
    
    def test_unified_framework_integration(self):
        """Test unified framework integration with thermal efficiency."""
        settings = UnifiedOptimizationSettings(
            use_thermal_efficiency=True,
            thermal_efficiency_config={
                "thermal_efficiency_weight": 1.0,
                "use_1d_gas_model": True,
                "n_cells": 30
            }
        )
        
        framework = UnifiedOptimizationFramework("TestFramework", settings)
        
        assert framework.settings.use_thermal_efficiency is True
        assert framework.thermal_adapter is not None
    
    def test_configuration_loading(self):
        """Test loading thermal efficiency configuration from file."""
        config_path = Path("cfg/thermal_efficiency_config.yaml")
        
        if config_path.exists():
            # Test loading configuration
            # This would test the configuration loading functionality
            pass
    
    def test_thermal_efficiency_metrics(self):
        """Test thermal efficiency metrics calculation."""
        config = ThermalEfficiencyConfig()
        adapter = ThermalEfficiencyAdapter(config)
        
        # Test metrics calculation
        # This would test the thermal efficiency metrics
        pass
```

### Phase 4: Migration Strategy

#### 4.1 Backward Compatibility
- Maintain existing API compatibility
- Add thermal efficiency as optional feature
- Provide fallback to simple optimization
- Gradual migration path

#### 4.2 Migration Steps
1. **Step 1**: Add thermal efficiency adapter (Phase 1)
2. **Step 2**: Update motion law optimizer (Phase 1)
3. **Step 3**: Update unified framework (Phase 1)
4. **Step 4**: Update GUI (Phase 2)
5. **Step 5**: Add configuration files (Phase 3)
6. **Step 6**: Add integration tests (Phase 3)
7. **Step 7**: Performance testing and optimization
8. **Step 8**: Documentation updates
9. **Step 9**: Gradual rollout to users

#### 4.3 Performance Considerations
- **Memory Usage**: 1D gas model uses more memory
- **CPU Time**: Complex optimization takes longer
- **Convergence**: May need robust solver settings
- **Validation**: Comprehensive physics validation

### Phase 5: Advanced Features

#### 5.1 Multi-Objective Optimization
- Thermal efficiency + smoothness
- Thermal efficiency + power
- Thermal efficiency + emissions

#### 5.2 Adaptive Refinement
- Automatic 0D → 1D switching
- Dynamic mesh refinement
- Error-based refinement

#### 5.3 Real-Time Optimization
- Fast thermal efficiency estimation
- Online optimization updates
- Performance monitoring

## Implementation Timeline

### Week 1-2: Core Integration
- [ ] Create thermal efficiency adapter
- [ ] Update motion law optimizer
- [ ] Update unified framework
- [ ] Basic integration tests

### Week 3-4: GUI Integration
- [ ] Update main GUI
- [ ] Add thermal efficiency controls
- [ ] Update visualization
- [ ] User interface testing

### Week 5-6: Configuration and Testing
- [ ] Create configuration files
- [ ] Comprehensive integration tests
- [ ] Performance testing
- [ ] Documentation updates

### Week 7-8: Advanced Features
- [ ] Multi-objective optimization
- [ ] Adaptive refinement
- [ ] Performance optimization
- [ ] Final testing and validation

## Success Criteria

1. **Functional Integration**: Thermal efficiency optimization works seamlessly
2. **Performance**: Acceptable optimization times (< 5 minutes for typical problems)
3. **Accuracy**: Thermal efficiency calculations are physically accurate
4. **Usability**: GUI integration is intuitive and user-friendly
5. **Reliability**: Robust convergence and error handling
6. **Compatibility**: Backward compatibility maintained
7. **Documentation**: Comprehensive documentation and examples

## Risk Mitigation

1. **Convergence Issues**: Use robust solver settings and fallback options
2. **Performance Problems**: Implement adaptive refinement and caching
3. **Integration Complexity**: Maintain clear separation of concerns
4. **User Adoption**: Provide gradual migration path and training
5. **Testing Coverage**: Comprehensive test suite with edge cases

## Conclusion

This integration plan provides a comprehensive approach to replacing the simple phase 1 optimization with the complex gas optimizer system, focusing specifically on thermal efficiency for acceleration zone optimization. The plan maintains backward compatibility while providing significant new functionality and physical accuracy improvements.

The phased approach ensures minimal disruption to existing users while providing a clear migration path to the more sophisticated optimization system. The focus on thermal efficiency addresses the specific requirement for acceleration zone optimization while maintaining the flexibility to extend to other optimization objectives in the future.
