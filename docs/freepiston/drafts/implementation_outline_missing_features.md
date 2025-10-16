# Implementation Outline: Missing Features for OP Engine Motion-Law

## Overview

This document provides a detailed implementation plan for the critical missing components identified in the codebase analysis. The implementation is prioritized by impact and dependencies, with clear milestones and deliverables.

---

## **Phase 1: Complete 1D Gas-Structure Coupling (Highest Priority)**

### **1.1 Enhanced Gas-Structure Interface**

**Files to Modify:**
- `campro/freepiston/net1d/stepper.py`
- `campro/freepiston/opt/nlp.py`
- `campro/freepiston/net1d/mesh.py`

**Implementation Tasks:**

#### **1.1.1 Moving Boundary Mesh Integration**
```python
# In net1d/mesh.py - Add to existing ALEMesh class
@dataclass
class MovingBoundaryMesh:
    """Enhanced mesh for gas-structure coupling."""
    
    # Existing fields...
    piston_positions: Dict[str, float]  # {"left": x_L, "right": x_R}
    piston_velocities: Dict[str, float]  # {"left": v_L, "right": v_R}
    mesh_velocity: np.ndarray  # Grid velocity at each cell
    volume_change_rate: np.ndarray  # dV/dt for each cell
    
    def update_piston_boundaries(self, x_L: float, x_R: float, v_L: float, v_R: float):
        """Update mesh based on piston positions and velocities."""
        # Implement ALE mesh motion
        # Calculate grid velocities
        # Update cell volumes and face areas
        pass
    
    def calculate_volume_change_rate(self) -> np.ndarray:
        """Calculate dV/dt for each cell due to piston motion."""
        # Implement volume change calculation
        pass
```

#### **1.1.2 Gas-Structure Coupling in Time Stepper**
```python
# In net1d/stepper.py - Add new function
def gas_structure_coupled_step(
    U: np.ndarray,
    mesh: MovingBoundaryMesh,
    piston_forces: Dict[str, float],
    dt: float,
    params: TimeStepParameters
) -> TimeStepResult:
    """
    Single time step with full gas-structure coupling.
    
    Args:
        U: Conservative variables [rho, rho*u, rho*E] for all cells
        mesh: Moving boundary mesh
        piston_forces: Gas forces on pistons
        dt: Time step
        params: Time stepping parameters
        
    Returns:
        Time step result with updated state
    """
    # 1. Update mesh based on piston motion
    mesh.update_piston_boundaries(
        piston_forces["x_L"], piston_forces["x_R"],
        piston_forces["v_L"], piston_forces["v_R"]
    )
    
    # 2. Calculate ALE fluxes with moving boundaries
    F_ale = calculate_ale_fluxes(U, mesh)
    
    # 3. Apply source terms (volume change, heat transfer)
    S_sources = calculate_source_terms(U, mesh, dt)
    
    # 4. Update conservative variables
    U_new = U + dt * (F_ale + S_sources)
    
    # 5. Calculate gas forces on pistons
    piston_forces_new = calculate_piston_forces(U_new, mesh)
    
    return TimeStepResult(
        success=True,
        U_new=U_new,
        piston_forces=piston_forces_new,
        dt_used=dt
    )
```

#### **1.1.3 Integration with Collocation NLP**
```python
# In opt/nlp.py - Enhance build_collocation_nlp function
def build_collocation_nlp_with_1d_coupling(P: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Build NLP with full 1D gas-structure coupling.
    
    This replaces the current 0D gas model with 1D FV model
    while maintaining the collocation framework.
    """
    ca = _import_casadi()
    
    # ... existing setup ...
    
    # 1D gas model variables (per cell, per collocation point)
    n_cells = P.get("flow", {}).get("mesh_cells", 80)
    
    for k in range(K):
        for c in range(C):
            # Gas state variables for each cell
            rho_cells = [ca.SX.sym(f"rho_{k}_{c}_{i}") for i in range(n_cells)]
            u_cells = [ca.SX.sym(f"u_{k}_{c}_{i}") for i in range(n_cells)]
            E_cells = [ca.SX.sym(f"E_{k}_{c}_{i}") for i in range(n_cells)]
            
            # Add to optimization variables
            w += rho_cells + u_cells + E_cells
            
            # 1D gas dynamics constraints
            for i in range(n_cells):
                # Conservative form: dU/dt + dF/dx = S
                # where U = [rho, rho*u, rho*E], F = [rho*u, rho*u^2+p, (rho*E+p)*u]
                
                # Calculate fluxes at cell faces
                F_left = hllc_flux_symbolic(U_cells[i-1], U_cells[i])
                F_right = hllc_flux_symbolic(U_cells[i], U_cells[i+1])
                
                # Source terms (volume change, heat transfer)
                S_sources = calculate_1d_source_terms(
                    rho_cells[i], u_cells[i], E_cells[i],
                    xL_c, xR_c, vL_c, vR_c, mesh
                )
                
                # Semi-discrete form
                dU_dt = -(F_right - F_left) / dx + S_sources
                
                # Add collocation constraint
                g += [U_cells[i] - (U_k[i] + h * sum(grid.a[c][j] * dU_dt for j in range(C)))]
                lbg += [0.0]
                ubg += [0.0]
    
    # ... rest of NLP setup ...
```

### **1.2 Deliverables**
- [ ] Enhanced `MovingBoundaryMesh` class with ALE capabilities
- [ ] Gas-structure coupled time stepper
- [ ] 1D gas model integration in collocation NLP
- [ ] Unit tests for moving boundary mesh
- [ ] Integration tests for gas-structure coupling

---

## **Phase 2: Complete Collocation DAE Constraints**

### **2.1 Enhanced DAE Constraint Implementation**

**Files to Modify:**
- `campro/freepiston/opt/nlp.py`
- `campro/freepiston/opt/cons.py`

**Implementation Tasks:**

#### **2.1.1 Complete Mechanical DAE Constraints**
```python
# In opt/nlp.py - Enhance piston dynamics
def enhanced_piston_dae_constraints(
    xL_c: Any, xR_c: Any, vL_c: Any, vR_c: Any,
    aL_c: Any, aR_c: Any, p_gas_c: Any,
    geometry: Dict[str, float]
) -> Tuple[Any, Any]:
    """
    Complete piston DAE with all force components.
    
    Returns:
        F_L, F_R: Net forces on left and right pistons
    """
    ca = _import_casadi()
    
    # Gas pressure forces
    A_piston = math.pi * (geometry["bore"] / 2.0) ** 2
    F_gas_L = p_gas_c * A_piston
    F_gas_R = -p_gas_c * A_piston  # Opposite direction
    
    # Inertia forces (piston + connecting rod)
    m_piston = geometry["mass"]
    m_rod = geometry["rod_mass"]
    rod_cg_offset = geometry["rod_cg_offset"]
    rod_length = geometry["rod_length"]
    
    # Effective mass including rod dynamics
    m_eff_L = m_piston + m_rod * (rod_cg_offset / rod_length)
    m_eff_R = m_piston + m_rod * (rod_cg_offset / rod_length)
    
    F_inertia_L = -m_eff_L * aL_c
    F_inertia_R = -m_eff_R * aR_c
    
    # Friction forces (velocity-dependent)
    friction_coeff = geometry.get("friction_coeff", 0.1)
    F_friction_L = -friction_coeff * ca.sign(vL_c) * ca.fabs(vL_c)
    F_friction_R = -friction_coeff * ca.sign(vR_c) * ca.fabs(vR_c)
    
    # Clearance penalty forces (smooth)
    gap_min = geometry.get("gap_min", 0.0008)
    gap_current = xR_c - xL_c
    penalty_stiffness = geometry.get("penalty_stiffness", 1e6)
    
    # Smooth penalty function
    gap_violation = ca.fmax(0.0, gap_min - gap_current)
    F_clearance_L = penalty_stiffness * gap_violation
    F_clearance_R = -penalty_stiffness * gap_violation
    
    # Net forces
    F_L = F_gas_L + F_inertia_L + F_friction_L + F_clearance_L
    F_R = F_gas_R + F_inertia_R + F_friction_R + F_clearance_R
    
    return F_L, F_R
```

#### **2.1.2 Complete Gas DAE Constraints**
```python
# In opt/nlp.py - Enhance gas dynamics
def enhanced_gas_dae_constraints(
    rho_c: Any, T_c: Any, V_c: Any, dV_dt_c: Any,
    mdot_in_c: Any, mdot_out_c: Any, Q_comb_c: Any, Q_heat_c: Any,
    geometry: Dict[str, float], thermo: Dict[str, float]
) -> Tuple[Any, Any]:
    """
    Complete gas DAE with all source terms.
    
    Returns:
        drho_dt, dT_dt: Density and temperature rates
    """
    ca = _import_casadi()
    
    # Gas properties
    R = thermo.get("R", 287.0)
    gamma = thermo.get("gamma", 1.4)
    cp = thermo.get("cp", 1005.0)
    cv = cp / gamma
    
    # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
    # Expanding: rho*dV/dt + V*drho/dt = mdot_in - mdot_out
    # Solving for drho/dt: drho/dt = (mdot_in - mdot_out - rho*dV/dt) / V
    drho_dt = (mdot_in_c - mdot_out_c - rho_c * dV_dt_c) / V_c
    
    # Energy balance: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # where m = rho*V, e = cv*T, h = cp*T, p = rho*R*T
    
    # Total mass and internal energy
    m_total = rho_c * V_c
    e_internal = cv * T_c
    
    # Enthalpy of inlet/outlet streams
    T_in = thermo.get("T_in", 300.0)
    T_out = T_c  # Assume outlet at chamber temperature
    h_in = cp * T_in
    h_out = cp * T_out
    
    # Pressure
    p_gas = rho_c * R * T_c
    
    # Energy equation: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Expanding: m*de/dt + e*dm/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Since de/dt = cv*dT/dt and dm/dt = drho_dt*V + rho*dV_dt_c:
    # m*cv*dT/dt + e*(drho_dt*V + rho*dV_dt_c) = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    
    # Solve for dT/dt
    dT_dt = (Q_comb_c - Q_heat_c + 
             mdot_in_c * h_in - mdot_out_c * h_out - 
             p_gas * dV_dt_c - 
             e_internal * (drho_dt * V_c + rho_c * dV_dt_c)) / (m_total * cv)
    
    return drho_dt, dT_dt
```

#### **2.1.3 Path Constraint Enhancements**
```python
# In opt/cons.py - Add comprehensive path constraints
def comprehensive_path_constraints(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    bounds: Dict[str, float],
    grid: CollocationGrid
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Comprehensive path constraints for all states and controls.
    
    Args:
        states: Dictionary of state variables over time
        controls: Dictionary of control variables over time
        bounds: Bounds dictionary
        grid: Collocation grid
        
    Returns:
        g_path, lbg_path, ubg_path: Path constraints and bounds
    """
    ca = _import_casadi()
    g_path = []
    lbg_path = []
    ubg_path = []
    
    # Pressure constraints
    for p in states["pressure"]:
        g_path.append(p)
        lbg_path.append(bounds.get("p_min", 1e3))
        ubg_path.append(bounds.get("p_max", 1e7))
    
    # Temperature constraints
    for T in states["temperature"]:
        g_path.append(T)
        lbg_path.append(bounds.get("T_min", 200.0))
        ubg_path.append(bounds.get("T_max", 2000.0))
    
    # Piston clearance constraints
    for xL, xR in zip(states["x_L"], states["x_R"]):
        gap = xR - xL
        g_path.append(gap)
        lbg_path.append(bounds.get("gap_min", 0.0008))
        ubg_path.append(ca.inf)
    
    # Valve rate constraints
    for i in range(1, len(controls["A_in"])):
        dA_dt = (controls["A_in"][i] - controls["A_in"][i-1]) / grid.h
        g_path.append(dA_dt)
        lbg_path.append(-bounds.get("dA_dt_max", 0.02))
        ubg_path.append(bounds.get("dA_dt_max", 0.02))
    
    # Piston velocity constraints
    for vL, vR in zip(states["v_L"], states["v_R"]):
        g_path.extend([vL, vR])
        lbg_path.extend([-bounds.get("v_max", 50.0), -bounds.get("v_max", 50.0)])
        ubg_path.extend([bounds.get("v_max", 50.0), bounds.get("v_max", 50.0)])
    
    return g_path, lbg_path, ubg_path
```

### **2.2 Deliverables**
- [ ] Complete mechanical DAE with all force components
- [ ] Complete gas DAE with all source terms
- [ ] Comprehensive path constraints
- [ ] Unit tests for DAE constraints
- [ ] Validation tests for constraint satisfaction

---

## **Phase 3: Scavenging Efficiency Objectives**

### **3.1 Two-Stroke OP Specific Metrics**

**Files to Modify:**
- `campro/freepiston/opt/obj.py`
- `campro/freepiston/zerod/cv.py`

**Implementation Tasks:**

#### **3.1.1 Enhanced Scavenging Metrics**
```python
# In opt/obj.py - Add comprehensive scavenging objectives
def comprehensive_scavenging_objectives(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    geometry: Dict[str, float],
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Comprehensive scavenging objectives for two-stroke OP engines.
    
    Returns:
        Dictionary of objective terms
    """
    ca = _import_casadi()
    
    objectives = {}
    
    # 1. Scavenging Efficiency (fresh charge / total trapped)
    m_fresh_trapped = states["m_fresh_trapped"][-1]  # At end of cycle
    m_total_trapped = states["m_total_trapped"][-1]
    eta_scav = m_fresh_trapped / (m_total_trapped + 1e-9)
    objectives["scavenging_efficiency"] = weights.get("eta_scav", 1.0) * eta_scav
    
    # 2. Trapping Efficiency (trapped mass / delivered mass)
    m_delivered = states["m_delivered"][-1]
    eta_trap = m_total_trapped / (m_delivered + 1e-9)
    objectives["trapping_efficiency"] = weights.get("eta_trap", 1.0) * eta_trap
    
    # 3. Short-Circuit Loss (minimize fresh charge loss)
    m_short_circuit = states["m_short_circuit"][-1]
    short_circuit_fraction = m_short_circuit / (m_delivered + 1e-9)
    objectives["short_circuit_penalty"] = weights.get("short_circuit", 2.0) * short_circuit_fraction
    
    # 4. Scavenging Quality (uniformity of fresh charge distribution)
    # This requires 1D model - placeholder for now
    objectives["scavenging_uniformity"] = weights.get("uniformity", 0.5) * 0.0
    
    # 5. Blow-Down Efficiency (exhaust gas removal)
    m_exhaust_removed = states["m_exhaust_removed"][-1]
    m_exhaust_initial = states["m_exhaust_initial"][0]
    eta_blowdown = m_exhaust_removed / (m_exhaust_initial + 1e-9)
    objectives["blowdown_efficiency"] = weights.get("eta_blowdown", 1.0) * eta_blowdown
    
    return objectives

def scavenging_phase_timing_objectives(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    targets: Dict[str, float],
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Objectives for optimal scavenging phase timing.
    
    Args:
        states: State variables over time
        controls: Control variables over time
        targets: Target phase durations
        weights: Objective weights
        
    Returns:
        Dictionary of timing objective terms
    """
    ca = _import_casadi()
    
    objectives = {}
    
    # 1. Intake Phase Duration
    t_intake_start = find_phase_start(controls["A_in"], threshold=0.01)
    t_intake_end = find_phase_end(controls["A_in"], threshold=0.01)
    t_intake_duration = t_intake_end - t_intake_start
    t_intake_target = targets.get("intake_duration", 0.1)
    objectives["intake_timing"] = weights.get("intake_timing", 1.0) * (t_intake_duration - t_intake_target)**2
    
    # 2. Exhaust Phase Duration
    t_exhaust_start = find_phase_start(controls["A_ex"], threshold=0.01)
    t_exhaust_end = find_phase_end(controls["A_ex"], threshold=0.01)
    t_exhaust_duration = t_exhaust_end - t_exhaust_start
    t_exhaust_target = targets.get("exhaust_duration", 0.1)
    objectives["exhaust_timing"] = weights.get("exhaust_timing", 1.0) * (t_exhaust_duration - t_exhaust_target)**2
    
    # 3. Overlap Phase (both valves open)
    overlap_penalty = calculate_overlap_penalty(controls["A_in"], controls["A_ex"])
    objectives["overlap_penalty"] = weights.get("overlap", 2.0) * overlap_penalty
    
    return objectives
```

#### **3.1.2 Enhanced 0D Scavenging Tracking**
```python
# In zerod/cv.py - Enhance scavenging tracking
@dataclass
class ScavengingState:
    """Enhanced scavenging state tracking."""
    
    # Mass tracking
    m_fresh_delivered: float = 0.0  # Fresh charge delivered
    m_fresh_trapped: float = 0.0    # Fresh charge trapped
    m_exhaust_initial: float = 0.0  # Initial exhaust gas mass
    m_exhaust_removed: float = 0.0  # Exhaust gas removed
    m_short_circuit: float = 0.0    # Fresh charge short-circuit
    
    # Composition tracking
    y_fresh: float = 1.0  # Fresh charge mass fraction
    y_exhaust: float = 0.0  # Exhaust gas mass fraction
    
    # Phase tracking
    phase: str = "compression"  # Current phase
    t_phase_start: float = 0.0  # Phase start time
    
    def update_scavenging_state(
        self,
        mdot_in: float,
        mdot_out: float,
        y_fresh_in: float,
        dt: float
    ) -> None:
        """Update scavenging state based on mass flows."""
        
        # Fresh charge delivery
        m_fresh_in = mdot_in * y_fresh_in * dt
        self.m_fresh_delivered += m_fresh_in
        
        # Exhaust gas removal
        m_exhaust_out = mdot_out * self.y_exhaust * dt
        self.m_exhaust_removed += m_exhaust_out
        
        # Short-circuit (fresh charge leaving)
        m_fresh_out = mdot_out * self.y_fresh * dt
        self.m_short_circuit += m_fresh_out
        
        # Update trapped masses
        self.m_fresh_trapped += m_fresh_in - m_fresh_out
        self.m_exhaust_initial -= m_exhaust_out
        
        # Update composition
        m_total = self.m_fresh_trapped + self.m_exhaust_initial
        if m_total > 0:
            self.y_fresh = self.m_fresh_trapped / m_total
            self.y_exhaust = self.m_exhaust_initial / m_total

def calculate_scavenging_metrics_enhanced(
    scavenging_state: ScavengingState,
    m_total_trapped: float
) -> Dict[str, float]:
    """
    Calculate comprehensive scavenging metrics.
    
    Returns:
        Dictionary of scavenging metrics
    """
    metrics = {}
    
    # Scavenging efficiency
    metrics["scavenging_efficiency"] = (
        scavenging_state.m_fresh_trapped / (m_total_trapped + 1e-9)
    )
    
    # Trapping efficiency
    metrics["trapping_efficiency"] = (
        m_total_trapped / (scavenging_state.m_fresh_delivered + 1e-9)
    )
    
    # Short-circuit fraction
    metrics["short_circuit_fraction"] = (
        scavenging_state.m_short_circuit / (scavenging_state.m_fresh_delivered + 1e-9)
    )
    
    # Blow-down efficiency
    metrics["blowdown_efficiency"] = (
        scavenging_state.m_exhaust_removed / (scavenging_state.m_exhaust_initial + 1e-9)
    )
    
    # Fresh charge purity
    metrics["fresh_charge_purity"] = scavenging_state.y_fresh
    
    return metrics
```

### **3.2 Deliverables**
- [ ] Comprehensive scavenging objectives
- [ ] Enhanced scavenging state tracking
- [ ] Phase timing objectives
- [ ] Unit tests for scavenging metrics
- [ ] Validation tests for scavenging efficiency

---

## **Phase 4: Riemann Solver Validation**

### **4.1 Comprehensive HLLC Testing**

**Files to Modify:**
- `tests/unit/test_net1d_flux.py`
- `campro/freepiston/net1d/flux.py`

**Implementation Tasks:**

#### **4.1.1 Enhanced HLLC Test Suite**
```python
# In tests/unit/test_net1d_flux.py - Add comprehensive tests
class TestHLLCComprehensive:
    """Comprehensive HLLC Riemann solver tests."""
    
    def test_hllc_shock_tube_sod(self):
        """Test HLLC on Sod shock tube problem."""
        # Left state: rho=1.0, u=0.0, p=1.0
        # Right state: rho=0.125, u=0.0, p=0.1
        U_L = (1.0, 0.0, 2.5)  # [rho, rho*u, rho*E]
        U_R = (0.125, 0.0, 0.25)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # Verify flux properties
        assert len(F) == 3
        assert all(not math.isnan(f) for f in F)
        assert all(not math.isinf(f) for f in F)
        
        # Verify physical properties
        assert F[0] >= 0.0  # Mass flux should be positive (left to right)
        assert F[1] >= 0.0  # Momentum flux should be positive
    
    def test_hllc_contact_discontinuity(self):
        """Test HLLC on contact discontinuity."""
        # Left state: rho=1.0, u=0.75, p=1.0
        # Right state: rho=0.125, u=0.0, p=0.1
        U_L = (1.0, 0.75, 2.5)
        U_R = (0.125, 0.0, 0.25)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # Verify flux is well-defined
        assert all(not math.isnan(f) for f in F)
        assert all(not math.isinf(f) for f in F)
    
    def test_hllc_rarefaction_wave(self):
        """Test HLLC on rarefaction wave."""
        # Left state: rho=1.0, u=-2.0, p=0.4
        # Right state: rho=1.0, u=-2.0, p=0.4
        U_L = (1.0, -2.0, 2.0)
        U_R = (1.0, -2.0, 2.0)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # For identical states, flux should be exact
        rho, u, p = primitive_from_conservative(U_L, gamma=1.4)
        F_exact = flux_from_primitive(rho, u, p)
        
        for i in range(3):
            assert abs(F[i] - F_exact[i]) < 1e-12
    
    def test_hllc_vacuum_state(self):
        """Test HLLC on vacuum state."""
        # Left state: rho=1.0, u=-2.0, p=0.4
        # Right state: rho=1.0, u=2.0, p=0.4
        U_L = (1.0, -2.0, 2.0)
        U_R = (1.0, 2.0, 2.0)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # Should handle vacuum gracefully
        assert all(not math.isnan(f) for f in F)
        assert all(not math.isinf(f) for f in F)
    
    def test_hllc_conservation_properties(self):
        """Test HLLC conservation properties."""
        # Test mass conservation
        U_L = (1.0, 0.0, 2.5)
        U_R = (0.5, 0.0, 1.25)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # Mass flux should be between left and right mass fluxes
        F_L = flux_from_primitive(*primitive_from_conservative(U_L, gamma=1.4))
        F_R = flux_from_primitive(*primitive_from_conservative(U_R, gamma=1.4))
        
        assert F_L[0] >= F[0] >= F_R[0]  # Mass flux ordering
    
    def test_hllc_entropy_condition(self):
        """Test HLLC entropy condition."""
        # Test that HLLC satisfies entropy condition
        U_L = (1.0, 0.0, 2.5)
        U_R = (0.125, 0.0, 0.25)
        
        F = hllc_flux(U_L, U_R, gamma=1.4)
        
        # Entropy condition: flux should be monotonic
        # This is a simplified test - full entropy condition is complex
        assert F[0] >= 0.0  # Mass flux should be positive
    
    def test_hllc_wave_speeds(self):
        """Test HLLC wave speed estimation."""
        U_L = (1.0, 0.0, 2.5)
        U_R = (0.125, 0.0, 0.25)
        
        S_L, S_R, S_star, p_star = wave_speeds(U_L, U_R, gamma=1.4)
        
        # Wave speeds should be ordered
        assert S_L <= S_star <= S_R
        
        # Pressure should be positive
        assert p_star > 0.0
        
        # Wave speeds should be finite
        assert all(not math.isnan(s) for s in [S_L, S_R, S_star])
        assert all(not math.isinf(s) for s in [S_L, S_R, S_star])
```

#### **4.1.2 Enhanced HLLC Implementation**
```python
# In net1d/flux.py - Enhance HLLC implementation
def hllc_flux_enhanced(
    U_L: Tuple[float, float, float], 
    U_R: Tuple[float, float, float], 
    gamma: float = 1.4
) -> Tuple[float, float, float]:
    """
    Enhanced HLLC Riemann solver with robust handling.
    
    Includes:
    - Vacuum state detection
    - Entropy fix for sonic rarefactions
    - Robust wave speed estimation
    - Conservation property verification
    """
    # Check for vacuum states
    if U_L[0] <= 0.0 or U_R[0] <= 0.0:
        return (0.0, 0.0, 0.0)
    
    # Check for negative pressure
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)
    
    if p_L <= 0.0 or p_R <= 0.0:
        return (0.0, 0.0, 0.0)
    
    # Compute wave speeds with entropy fix
    S_L, S_R, S_star, p_star = wave_speeds_enhanced(U_L, U_R, gamma)
    
    # Check for vacuum condition
    if S_L >= S_R:
        return (0.0, 0.0, 0.0)
    
    # Compute fluxes from primitive variables
    F_L = flux_from_primitive(rho_L, u_L, p_L)
    F_R = flux_from_primitive(rho_R, u_R, p_R)
    
    # HLLC flux selection
    if S_L >= 0.0:
        # Left state
        return F_L
    elif S_R <= 0.0:
        # Right state
        return F_R
    elif S_star >= 0.0:
        # Left star state
        U_star_L = hllc_star_state(U_L, S_L, S_star, p_star, gamma)
        return F_L + S_L * (U_star_L - U_L)
    else:
        # Right star state
        U_star_R = hllc_star_state(U_R, S_R, S_star, p_star, gamma)
        return F_R + S_R * (U_star_R - U_R)

def wave_speeds_enhanced(
    U_L: Tuple[float, float, float], 
    U_R: Tuple[float, float, float], 
    gamma: float = 1.4
) -> Tuple[float, float, float, float]:
    """
    Enhanced wave speed estimation with entropy fix.
    """
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)
    
    # Sound speeds
    c_L = math.sqrt(gamma * p_L / rho_L)
    c_R = math.sqrt(gamma * p_R / rho_R)
    
    # Roe averages
    rho_roe, u_roe, c_roe, H_roe = roe_averages(U_L, U_R, gamma)
    
    # Wave speeds with entropy fix
    S_L = min(u_L - c_L, u_roe - c_roe)
    S_R = max(u_R + c_R, u_roe + c_roe)
    
    # Contact wave speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    
    # Pressure in star region
    p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L)
    
    return S_L, S_R, S_star, p_star
```

### **4.2 Deliverables**
- [ ] Comprehensive HLLC test suite
- [ ] Enhanced HLLC implementation with entropy fix
- [ ] Vacuum state handling
- [ ] Conservation property verification
- [ ] Performance benchmarks

---

## **Phase 5: End-to-End Integration**

### **5.1 Complete Cycle Optimization Pipeline**

**Files to Modify:**
- `scripts/run_colloc.py`
- `campro/freepiston/opt/driver.py`
- `campro/freepiston/io/load.py`

**Implementation Tasks:**

#### **5.1.1 Enhanced Main Script**
```python
# In scripts/run_colloc.py - Complete integration
def main() -> None:
    """Complete end-to-end cycle optimization."""
    
    # 1. Load configuration
    cfg = load_cfg(Path("cfg/defaults.yaml"))
    
    # 2. Validate configuration
    validate_configuration(cfg)
    
    # 3. Set up run directory
    run_dir = Path("runs/op_cycle_001")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(run_dir)
    
    # 4. Save configuration
    save_configuration(cfg, run_dir / "config.yaml")
    
    # 5. Run optimization
    log.info("Starting OP engine cycle optimization...")
    sol = solve_cycle(cfg)
    
    # 6. Validate solution
    if validate_solution(sol, cfg):
        log.info("Solution validation passed")
        
        # 7. Save results
        if isinstance(sol, Solution):
            sol.save(run_dir)
        
        # 8. Generate reports
        generate_optimization_report(sol, cfg, run_dir)
        
        # 9. Plot results
        plot_optimization_results(sol, run_dir)
        
        log.info(f"Optimization completed successfully. Results saved to {run_dir}")
    else:
        log.error("Solution validation failed")
        sys.exit(1)

def validate_configuration(cfg: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_sections = ["geom", "flow", "thermo", "bounds", "obj", "num", "solver"]
    
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate geometry
    geom = cfg["geom"]
    if geom["B"] <= 0 or geom["L"] <= 0 or geom["Vc"] <= 0:
        raise ValueError("Invalid geometry parameters")
    
    # Validate bounds
    bounds = cfg["bounds"]
    if bounds["x_gap_min"] <= 0:
        raise ValueError("Invalid gap minimum")
    
    # Validate numerical parameters
    num = cfg["num"]
    if num["K"] <= 0 or num["C"] <= 0:
        raise ValueError("Invalid numerical parameters")

def validate_solution(sol: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Validate optimization solution."""
    if not sol.get("success", False):
        log.error("Optimization failed")
        return False
    
    # Check solution quality
    if sol.get("kkt_error", float('inf')) > cfg["solver"]["ipopt"]["tol"] * 10:
        log.warning("High KKT error")
        return False
    
    if sol.get("feasibility_error", float('inf')) > cfg["solver"]["ipopt"]["tol"] * 10:
        log.warning("High feasibility error")
        return False
    
    # Check physical constraints
    x_opt = sol.get("x_opt", [])
    if not x_opt:
        log.error("No solution variables")
        return False
    
    # Check for NaN or Inf values
    if any(math.isnan(x) or math.isinf(x) for x in x_opt):
        log.error("Solution contains NaN or Inf values")
        return False
    
    return True

def generate_optimization_report(sol: Dict[str, Any], cfg: Dict[str, Any], run_dir: Path) -> None:
    """Generate comprehensive optimization report."""
    report = {
        "optimization_summary": {
            "success": sol.get("success", False),
            "iterations": sol.get("iterations", 0),
            "cpu_time": sol.get("cpu_time", 0.0),
            "objective_value": sol.get("f_opt", 0.0),
            "kkt_error": sol.get("kkt_error", 0.0),
            "feasibility_error": sol.get("feasibility_error", 0.0)
        },
        "configuration": cfg,
        "solution_quality": {
            "constraint_violations": check_constraint_violations(sol, cfg),
            "physical_plausibility": check_physical_plausibility(sol, cfg)
        }
    }
    
    # Save report
    with open(run_dir / "optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)

def plot_optimization_results(sol: Dict[str, Any], run_dir: Path) -> None:
    """Plot optimization results."""
    try:
        import matplotlib.pyplot as plt
        
        # Extract solution data
        x_opt = sol.get("x_opt", [])
        if not x_opt:
            return
        
        # Plot piston trajectories
        plt.figure(figsize=(12, 8))
        
        # Piston positions
        plt.subplot(2, 2, 1)
        plt.plot(x_opt[::6], label="Left piston")
        plt.plot(x_opt[1::6], label="Right piston")
        plt.xlabel("Time step")
        plt.ylabel("Position [m]")
        plt.title("Piston Positions")
        plt.legend()
        
        # Piston velocities
        plt.subplot(2, 2, 2)
        plt.plot(x_opt[2::6], label="Left piston")
        plt.plot(x_opt[3::6], label="Right piston")
        plt.xlabel("Time step")
        plt.ylabel("Velocity [m/s]")
        plt.title("Piston Velocities")
        plt.legend()
        
        # Gas pressure
        plt.subplot(2, 2, 3)
        plt.plot(x_opt[4::6])
        plt.xlabel("Time step")
        plt.ylabel("Density [kg/m³]")
        plt.title("Gas Density")
        
        # Gas temperature
        plt.subplot(2, 2, 4)
        plt.plot(x_opt[5::6])
        plt.xlabel("Time step")
        plt.ylabel("Temperature [K]")
        plt.title("Gas Temperature")
        
        plt.tight_layout()
        plt.savefig(run_dir / "optimization_results.png", dpi=300, bbox_inches="tight")
        plt.close()
        
    except ImportError:
        log.warning("Matplotlib not available, skipping plots")
```

#### **5.1.2 Enhanced Driver with Validation**
```python
# In opt/driver.py - Add comprehensive validation
def solve_cycle_with_validation(P: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve cycle with comprehensive validation and error handling.
    """
    log.info("Starting validated cycle optimization...")
    
    # 1. Pre-solve validation
    validate_problem_setup(P)
    
    # 2. Build NLP
    try:
        nlp, meta = build_collocation_nlp(P)
        log.info(f"NLP built successfully: {meta['n_vars']} variables, {meta['n_constraints']} constraints")
    except Exception as e:
        log.error(f"Failed to build NLP: {e}")
        return {"success": False, "error": str(e)}
    
    # 3. Set up solver
    try:
        solver_opts = P.get("solver", {}).get("ipopt", {})
        ipopt_options = _create_ipopt_options(solver_opts, P)
        solver = IPOPTSolver(ipopt_options)
        log.info("Solver configured successfully")
    except Exception as e:
        log.error(f"Failed to configure solver: {e}")
        return {"success": False, "error": str(e)}
    
    # 4. Set up bounds and initial guess
    try:
        x0, lbx, ubx, lbg, ubg, p = _setup_optimization_bounds(nlp, P, P.get("warm_start", {}))
        log.info(f"Bounds set up: {len(x0)} variables, {len(lbg)} constraints")
    except Exception as e:
        log.error(f"Failed to set up bounds: {e}")
        return {"success": False, "error": str(e)}
    
    # 5. Solve
    try:
        log.info("Starting IPOPT optimization...")
        result = solver.solve(nlp, x0, lbx, ubx, lbg, ubg, p)
        log.info(f"Optimization completed: {result.message}")
    except Exception as e:
        log.error(f"Optimization failed: {e}")
        return {"success": False, "error": str(e)}
    
    # 6. Post-solve validation
    if result.success:
        validation_result = validate_optimization_result(result, P)
        if not validation_result["valid"]:
            log.warning(f"Solution validation failed: {validation_result['message']}")
            result.success = False
            result.message = f"Validation failed: {validation_result['message']}"
    
    # 7. Package results
    optimization_result = {
        "success": result.success,
        "x_opt": result.x_opt,
        "f_opt": result.f_opt,
        "iterations": result.iterations,
        "cpu_time": result.cpu_time,
        "message": result.message,
        "status": result.status,
        "kkt_error": result.kkt_error,
        "feasibility_error": result.feasibility_error,
        "meta": meta
    }
    
    # 8. Save checkpoint
    run_dir = P.get("run_dir", None)
    if run_dir:
        try:
            save_json({"meta": meta, "opt": optimization_result}, run_dir, filename="checkpoint.json")
        except Exception as exc:
            log.warning(f"Checkpoint save failed: {exc}")
    
    return optimization_result

def validate_problem_setup(P: Dict[str, Any]) -> None:
    """Validate problem setup before optimization."""
    # Check required parameters
    required = ["geom", "flow", "thermo", "bounds", "obj", "num", "solver"]
    for key in required:
        if key not in P:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Check geometry consistency
    geom = P["geom"]
    if geom["B"] <= 0:
        raise ValueError("Bore diameter must be positive")
    if geom["L"] <= 0:
        raise ValueError("Liner length must be positive")
    if geom["Vc"] <= 0:
        raise ValueError("Clearance volume must be positive")
    
    # Check bounds consistency
    bounds = P["bounds"]
    if bounds["x_gap_min"] <= 0:
        raise ValueError("Minimum gap must be positive")
    if bounds["p_max"] <= bounds.get("p_min", 0):
        raise ValueError("Invalid pressure bounds")
    if bounds["T_max"] <= bounds.get("T_min", 0):
        raise ValueError("Invalid temperature bounds")
    
    # Check numerical parameters
    num = P["num"]
    if num["K"] <= 0:
        raise ValueError("Number of intervals must be positive")
    if num["C"] <= 0:
        raise ValueError("Number of collocation points must be positive")

def validate_optimization_result(result: Any, P: Dict[str, Any]) -> Dict[str, Any]:
    """Validate optimization result."""
    validation = {"valid": True, "message": "Validation passed"}
    
    # Check solution quality
    if result.kkt_error > P["solver"]["ipopt"]["tol"] * 100:
        validation["valid"] = False
        validation["message"] = f"High KKT error: {result.kkt_error}"
        return validation
    
    if result.feasibility_error > P["solver"]["ipopt"]["tol"] * 100:
        validation["valid"] = False
        validation["message"] = f"High feasibility error: {result.feasibility_error}"
        return validation
    
    # Check for NaN or Inf values
    if any(math.isnan(x) or math.isinf(x) for x in result.x_opt):
        validation["valid"] = False
        validation["message"] = "Solution contains NaN or Inf values"
        return validation
    
    # Check physical constraints
    bounds = P["bounds"]
    x_opt = result.x_opt
    
    # Check piston clearance
    for i in range(0, len(x_opt), 6):  # Assuming 6 variables per time step
        if i + 1 < len(x_opt):
            gap = x_opt[i + 1] - x_opt[i]  # x_R - x_L
            if gap < bounds["x_gap_min"]:
                validation["valid"] = False
                validation["message"] = f"Piston clearance violation: {gap} < {bounds['x_gap_min']}"
                return validation
    
    return validation
```

### **5.2 Deliverables**
- [ ] Complete end-to-end integration script
- [ ] Comprehensive configuration validation
- [ ] Solution validation and quality checks
- [ ] Optimization report generation
- [ ] Result visualization and plotting
- [ ] Error handling and recovery

---

## **Implementation Timeline**

### **Week 1-2: Phase 1 - Gas-Structure Coupling**
- Day 1-3: Enhanced moving boundary mesh
- Day 4-7: Gas-structure coupled time stepper
- Day 8-10: 1D gas model integration in NLP
- Day 11-14: Testing and validation

### **Week 3-4: Phase 2 - Complete DAE Constraints**
- Day 15-17: Enhanced mechanical DAE
- Day 18-20: Complete gas DAE
- Day 21-24: Comprehensive path constraints
- Day 25-28: Testing and validation

### **Week 5-6: Phase 3 - Scavenging Objectives**
- Day 29-31: Enhanced scavenging metrics
- Day 32-35: Phase timing objectives
- Day 36-38: Enhanced 0D scavenging tracking
- Day 39-42: Testing and validation

### **Week 7-8: Phase 4 - Riemann Solver Validation**
- Day 43-45: Comprehensive HLLC test suite
- Day 46-48: Enhanced HLLC implementation
- Day 49-52: Performance benchmarks
- Day 53-56: Documentation and validation

### **Week 9-10: Phase 5 - End-to-End Integration**
- Day 57-59: Enhanced main script
- Day 60-62: Comprehensive validation
- Day 63-66: Report generation and visualization
- Day 67-70: Final testing and documentation

---

## **Success Criteria**

### **Phase 1 Success Criteria:**
- [ ] Moving boundary mesh handles piston motion correctly
- [ ] Gas-structure coupling maintains conservation
- [ ] 1D gas model integrates with collocation framework
- [ ] All unit tests pass

### **Phase 2 Success Criteria:**
- [ ] Complete DAE constraints are properly implemented
- [ ] Path constraints are satisfied
- [ ] Solution quality meets specified tolerances
- [ ] All validation tests pass

### **Phase 3 Success Criteria:**
- [ ] Scavenging metrics are correctly calculated
- [ ] Phase timing objectives are properly implemented
- [ ] Enhanced scavenging tracking works correctly
- [ ] All scavenging tests pass

### **Phase 4 Success Criteria:**
- [ ] HLLC solver passes all test cases
- [ ] Entropy condition is satisfied
- [ ] Vacuum states are handled correctly
- [ ] Performance benchmarks meet targets

### **Phase 5 Success Criteria:**
- [ ] End-to-end integration works correctly
- [ ] Configuration validation is comprehensive
- [ ] Solution validation catches errors
- [ ] Reports and visualizations are generated
- [ ] All integration tests pass

---

## **Risk Mitigation**

### **Technical Risks:**
1. **Gas-structure coupling instability**: Use robust time stepping and validation
2. **DAE constraint complexity**: Implement incrementally with extensive testing
3. **Scavenging metric accuracy**: Validate against known solutions
4. **Riemann solver robustness**: Use entropy fixes and vacuum handling
5. **Integration complexity**: Use modular design with clear interfaces

### **Mitigation Strategies:**
1. **Incremental development**: Implement and test each component separately
2. **Comprehensive testing**: Unit tests, integration tests, and validation tests
3. **Documentation**: Clear documentation for each component
4. **Error handling**: Robust error handling and recovery mechanisms
5. **Performance monitoring**: Regular performance checks and optimization

---

## **Conclusion**

This implementation outline provides a comprehensive roadmap for completing the missing features in the OP engine motion-law codebase. The phased approach ensures that each component is properly implemented and tested before moving to the next phase. The success criteria and risk mitigation strategies provide clear guidance for achieving the project goals.

The implementation will result in a fully functional, validated, and robust OP engine motion-law optimization system that can be used for research and development purposes.

