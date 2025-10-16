# CasADi Physics Integration Roadmap

## Overview

This document outlines the long-term plan to integrate Python physics models directly into CasADi for automatic differentiation and improved optimization performance. The current implementation uses a hybrid approach where CasADi provides smoothness objectives while Python physics models are used for validation. This roadmap details the path to full CasADi integration.

## Current State

### Hybrid Approach (Implemented)
- **Phase 2A (Litvin)**: CasADi smoothness penalty + Python physics validation
- **Phase 2B (Crank Center)**: CasADi quadratic proxy + Python physics evaluation
- **Benefits**: Maintains physics accuracy while providing CasADi optimization benefits
- **Limitations**: No automatic differentiation, limited optimization efficiency

### Physics Models Status
- **Torque Calculation**: Python-based `PistonTorqueCalculator`
- **Side Loading Analysis**: Python-based `SideLoadAnalyzer`
- **Litvin Metrics**: Python-based slip integral and contact length calculations
- **Kinematics**: Python-based crank and planetary kinematics

## Phase 1: Torque Calculation CasADi Port (Months 1-2)

### 1.1 Crank-Piston Kinematics in CasADi
**Objective**: Convert crank-piston kinematics to CasADi expressions

**Implementation**:
```python
def create_crank_piston_kinematics_casadi():
    """Create CasADi function for crank-piston kinematics."""
    import casadi as ca
    
    # Variables
    theta = ca.SX.sym('theta')  # Crank angle
    r = ca.SX.sym('r')          # Crank radius
    l = ca.SX.sym('l')          # Connecting rod length
    x_offset = ca.SX.sym('x_offset')  # Crank center x offset
    y_offset = ca.SX.sym('y_offset')  # Crank center y offset
    
    # Kinematics
    x = x_offset + r * ca.cos(theta) + ca.sqrt(l**2 - (r * ca.sin(theta))**2)
    v = ca.jacobian(x, theta)  # Velocity
    a = ca.jacobian(v, theta)  # Acceleration
    
    return ca.Function('crank_kinematics', 
                      [theta, r, l, x_offset, y_offset], 
                      [x, v, a])
```

**Testing**:
- Validate against Python reference implementation
- Test gradient computation accuracy
- Benchmark performance vs Python

### 1.2 Piston Force Calculation in CasADi
**Objective**: Convert piston force calculations to CasADi

**Implementation**:
```python
def create_piston_force_casadi():
    """Create CasADi function for piston force calculation."""
    import casadi as ca
    
    # Variables
    pressure = ca.SX.sym('pressure')  # Cylinder pressure
    bore_diameter = ca.SX.sym('bore_diameter')
    piston_clearance = ca.SX.sym('piston_clearance')
    
    # Force calculation
    piston_area = ca.pi * (bore_diameter / 2)**2
    force = pressure * piston_area
    
    return ca.Function('piston_force', 
                      [pressure, bore_diameter, piston_clearance], 
                      [force])
```

### 1.3 Torque Calculation Integration
**Objective**: Combine kinematics and force calculation for torque

**Implementation**:
```python
def create_torque_calculation_casadi():
    """Create CasADi function for torque calculation."""
    import casadi as ca
    
    # Get kinematics and force functions
    kinematics = create_crank_piston_kinematics_casadi()
    force_calc = create_piston_force_casadi()
    
    # Variables
    theta = ca.SX.sym('theta')
    r = ca.SX.sym('r')
    l = ca.SX.sym('l')
    x_offset = ca.SX.sym('x_offset')
    y_offset = ca.SX.sym('y_offset')
    pressure = ca.SX.sym('pressure')
    bore_diameter = ca.SX.sym('bore_diameter')
    piston_clearance = ca.SX.sym('piston_clearance')
    
    # Calculate kinematics
    x, v, a = kinematics(theta, r, l, x_offset, y_offset)
    
    # Calculate force
    force = force_calc(pressure, bore_diameter, piston_clearance)
    
    # Calculate torque
    torque = force * r * ca.sin(theta)
    
    return ca.Function('torque_calculation',
                      [theta, r, l, x_offset, y_offset, pressure, bore_diameter, piston_clearance],
                      [torque])
```

## Phase 2: Side Loading CasADi Port (Months 3-4)

### 2.1 Force Decomposition in CasADi
**Objective**: Convert side loading force decomposition to CasADi

**Implementation**:
```python
def create_side_load_analysis_casadi():
    """Create CasADi function for side load analysis."""
    import casadi as ca
    
    # Variables
    theta = ca.SX.sym('theta')  # Crank angle
    r = ca.SX.sym('r')          # Crank radius
    l = ca.SX.sym('l')          # Connecting rod length
    force = ca.SX.sym('force')  # Piston force
    
    # Side load calculation
    beta = ca.asin(r * ca.sin(theta) / l)  # Connecting rod angle
    side_force = force * ca.tan(beta)
    
    # Penalty calculation
    max_side_force = ca.SX.sym('max_side_force')
    penalty = ca.fmax(0, side_force - max_side_force)**2
    
    return ca.Function('side_load_analysis',
                      [theta, r, l, force, max_side_force],
                      [side_force, penalty])
```

### 2.2 Integration with Torque Calculation
**Objective**: Combine torque and side loading in unified CasADi function

**Implementation**:
```python
def create_unified_physics_casadi():
    """Create unified CasADi function for torque and side loading."""
    import casadi as ca
    
    # Get component functions
    torque_calc = create_torque_calculation_casadi()
    side_load_calc = create_side_load_analysis_casadi()
    
    # Variables
    theta = ca.SX.sym('theta')
    r = ca.SX.sym('r')
    l = ca.SX.sym('l')
    x_offset = ca.SX.sym('x_offset')
    y_offset = ca.SX.sym('y_offset')
    pressure = ca.SX.sym('pressure')
    bore_diameter = ca.SX.sym('bore_diameter')
    piston_clearance = ca.SX.sym('piston_clearance')
    max_side_force = ca.SX.sym('max_side_force')
    
    # Calculate torque
    torque = torque_calc(theta, r, l, x_offset, y_offset, pressure, bore_diameter, piston_clearance)
    
    # Calculate force for side loading
    force_calc = create_piston_force_casadi()
    force = force_calc(pressure, bore_diameter, piston_clearance)
    
    # Calculate side loading
    side_force, side_penalty = side_load_calc(theta, r, l, force, max_side_force)
    
    return ca.Function('unified_physics',
                      [theta, r, l, x_offset, y_offset, pressure, bore_diameter, piston_clearance, max_side_force],
                      [torque, side_force, side_penalty])
```

## Phase 3: Litvin Metrics CasADi Port (Months 5-6)

### 3.1 Slip Integral Calculation in CasADi
**Objective**: Convert slip integral calculation to CasADi

**Implementation**:
```python
def create_slip_integral_casadi():
    """Create CasADi function for slip integral calculation."""
    import casadi as ca
    
    # Variables
    phi = ca.SX.sym('phi')  # Contact parameter
    theta = ca.SX.sym('theta')  # Ring angle
    ring_teeth = ca.SX.sym('ring_teeth')
    planet_teeth = ca.SX.sym('planet_teeth')
    pressure_angle = ca.SX.sym('pressure_angle')
    
    # Slip calculation
    # This is a simplified version - full implementation would include
    # complex involute gear geometry calculations
    slip = ca.fabs(phi - theta * (ring_teeth - planet_teeth) / ring_teeth)
    
    # Integral approximation (using trapezoidal rule)
    slip_integral = slip  # Simplified for single point
    
    return ca.Function('slip_integral',
                      [phi, theta, ring_teeth, planet_teeth, pressure_angle],
                      [slip_integral])
```

### 3.2 Contact Length Computation in CasADi
**Objective**: Convert contact length calculation to CasADi

**Implementation**:
```python
def create_contact_length_casadi():
    """Create CasADi function for contact length calculation."""
    import casadi as ca
    
    # Variables
    phi = ca.SX.sym('phi')
    ring_teeth = ca.SX.sym('ring_teeth')
    planet_teeth = ca.SX.sym('planet_teeth')
    module = ca.SX.sym('module')
    
    # Contact length calculation
    # Simplified version - full implementation would include
    # complex gear geometry calculations
    contact_length = module * ca.pi * (ring_teeth - planet_teeth) / ring_teeth
    
    return ca.Function('contact_length',
                      [phi, ring_teeth, planet_teeth, module],
                      [contact_length])
```

### 3.3 Litvin Metrics Integration
**Objective**: Combine slip integral and contact length in unified function

**Implementation**:
```python
def create_litvin_metrics_casadi():
    """Create CasADi function for Litvin metrics."""
    import casadi as ca
    
    # Get component functions
    slip_calc = create_slip_integral_casadi()
    contact_calc = create_contact_length_casadi()
    
    # Variables
    phi = ca.SX.sym('phi')
    theta = ca.SX.sym('theta')
    ring_teeth = ca.SX.sym('ring_teeth')
    planet_teeth = ca.SX.sym('planet_teeth')
    pressure_angle = ca.SX.sym('pressure_angle')
    module = ca.SX.sym('module')
    
    # Calculate metrics
    slip_integral = slip_calc(phi, theta, ring_teeth, planet_teeth, pressure_angle)
    contact_length = contact_calc(phi, ring_teeth, planet_teeth, module)
    
    # Combined objective
    objective = slip_integral - 0.1 * contact_length
    
    return ca.Function('litvin_metrics',
                      [phi, theta, ring_teeth, planet_teeth, pressure_angle, module],
                      [slip_integral, contact_length, objective])
```

## Phase 4: Integration Testing (Months 7-8)

### 4.1 Validation Against Python Reference
**Objective**: Ensure CasADi implementations match Python reference

**Testing Strategy**:
```python
def validate_casadi_against_python():
    """Validate CasADi implementations against Python reference."""
    
    # Test cases
    test_cases = [
        {"theta": 0.0, "r": 50.0, "l": 150.0, "pressure": 1e6},
        {"theta": ca.pi/4, "r": 50.0, "l": 150.0, "pressure": 1e6},
        {"theta": ca.pi/2, "r": 50.0, "l": 150.0, "pressure": 1e6},
    ]
    
    for case in test_cases:
        # Python reference
        python_result = python_physics_function(case)
        
        # CasADi implementation
        casadi_result = casadi_physics_function(case)
        
        # Validate
        assert ca.fabs(python_result - casadi_result) < 1e-6
```

### 4.2 Performance Benchmarking
**Objective**: Measure performance improvements

**Benchmarks**:
- Function evaluation time
- Gradient computation time
- Memory usage
- Optimization convergence rate

### 4.3 Gradient Verification
**Objective**: Verify automatic differentiation accuracy

**Implementation**:
```python
def verify_gradients():
    """Verify CasADi gradients against finite differences."""
    import numpy as np
    
    # Test function
    f = create_unified_physics_casadi()
    
    # Test point
    x0 = np.array([0.0, 50.0, 150.0, 0.0, 0.0, 1e6, 100.0, 0.1, 1000.0])
    
    # CasADi gradient
    grad_casadi = f.jacobian()(x0)
    
    # Finite difference gradient
    eps = 1e-8
    grad_fd = np.zeros_like(x0)
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_plus[i] += eps
        x_minus = x0.copy()
        x_minus[i] -= eps
        
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
        grad_fd[i] = (f_plus - f_minus) / (2 * eps)
    
    # Verify
    assert np.allclose(grad_casadi, grad_fd, rtol=1e-6)
```

## Implementation Timeline

### Month 1-2: Torque Calculation
- [ ] Crank-piston kinematics in CasADi
- [ ] Piston force calculation in CasADi
- [ ] Torque calculation integration
- [ ] Unit tests and validation

### Month 3-4: Side Loading
- [ ] Force decomposition in CasADi
- [ ] Side load penalty calculation
- [ ] Integration with torque calculation
- [ ] Unit tests and validation

### Month 5-6: Litvin Metrics
- [ ] Slip integral calculation in CasADi
- [ ] Contact length computation in CasADi
- [ ] Litvin metrics integration
- [ ] Unit tests and validation

### Month 7-8: Integration Testing
- [ ] Validation against Python reference
- [ ] Performance benchmarking
- [ ] Gradient verification
- [ ] End-to-end testing

## Benefits of Full CasADi Integration

### Performance Improvements
- **Automatic Differentiation**: Exact gradients for faster convergence
- **Optimization Efficiency**: Better solver performance with exact derivatives
- **Memory Efficiency**: Reduced memory usage compared to finite differences

### Code Quality
- **Unified Framework**: Single physics implementation for all optimization phases
- **Maintainability**: Reduced code duplication between Python and CasADi
- **Extensibility**: Easier to add new physics models

### Optimization Benefits
- **Faster Convergence**: Exact gradients lead to better optimization performance
- **Better Scaling**: Improved performance for larger problems
- **Robustness**: More reliable optimization with exact derivatives

## Migration Strategy

### Phase 1: Parallel Implementation
- Implement CasADi versions alongside Python versions
- Use feature flags to switch between implementations
- Maintain backward compatibility

### Phase 2: Validation and Testing
- Comprehensive testing against Python reference
- Performance benchmarking
- Gradient verification

### Phase 3: Gradual Migration
- Migrate one optimization phase at a time
- Monitor performance and accuracy
- Rollback capability if issues arise

### Phase 4: Full Migration
- Remove Python physics implementations
- Update all optimization phases to use CasADi
- Final performance optimization

## Risk Mitigation

### Technical Risks
- **Complexity**: CasADi implementation may be more complex than Python
- **Accuracy**: Risk of numerical differences between implementations
- **Performance**: CasADi may not always be faster for simple calculations

### Mitigation Strategies
- **Incremental Development**: Implement and test one component at a time
- **Comprehensive Testing**: Extensive validation against Python reference
- **Performance Monitoring**: Continuous benchmarking during development
- **Rollback Plan**: Maintain Python implementations until CasADi is fully validated

## Success Metrics

### Accuracy Metrics
- Function evaluation accuracy: < 1e-6 relative error
- Gradient accuracy: < 1e-6 relative error vs finite differences
- Optimization convergence: Same or better than Python implementation

### Performance Metrics
- Function evaluation speed: 2-5x faster than Python
- Gradient computation speed: 10-20x faster than finite differences
- Memory usage: 50% reduction compared to Python + finite differences

### Integration Metrics
- All three optimization phases using CasADi physics
- No regression in optimization quality
- Improved convergence rates across all phases

## Conclusion

The CasADi physics integration roadmap provides a structured approach to migrating from the current hybrid implementation to full CasADi integration. This will significantly improve optimization performance while maintaining physics accuracy. The phased approach minimizes risk while providing incremental benefits throughout the development process.

The timeline of 8 months is realistic for a comprehensive implementation, with each phase building on the previous one. The focus on validation and testing ensures that the migration maintains the high quality standards of the current implementation while providing the performance benefits of automatic differentiation.
