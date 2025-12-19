"""
Friction Solver Module for Phase 4.
Implements Stribeck Curve modeling for Piston Rings + Skirt.
"""

import numpy as np
from typing import Dict, Any, List
from Simulations.common.simulation import BaseSimulation, SimulationConfig
from Simulations.common.io_schema import SimulationInput, SimulationOutput

class FrictionConfig(SimulationConfig):
    """Configuration for Friction Solvers."""
    method: str = "Stribeck" 
    # Ring Parameters
    ring_tension: float = 10.0 # [N]
    ring_height: float = 0.0015 # [m]
    # Scaling factors for calibration
    asperity_coeff: float = 0.1 # dry friction
    hydro_coeff: float = 0.05 # viscous scalar
    
    # Defaults
    dt: float = 1.0 
    t_end: float = 1.0

class StribeckSolver:
    """
    Computes Instantaneous Friction Force using Stribeck Logic.
    F_f = F_asperity + F_hydrodynamic
    Depends on Hersey Number (mu * U / P).
    """
    
    def solve_cycle(self, 
                    theta_arr: np.ndarray,
                    p_gas_arr: np.ndarray,
                    u_piston_arr: np.ndarray,
                    T_oil: float,
                    geo: Any,
                    cfg: FrictionConfig) -> Dict[str, Any]:
        
        # 1. Oil Viscosity (Vogel Equation approx for SAE 30)
        # mu = A * exp(B / (T + C))
        # T in Celsius
        T_c = T_oil - 273.15
        mu_oil = 0.0001 * np.exp(800.0 / (T_c + 80.0)) # rough approx [Pa.s]
        
        # 2. Ring Normal Load
        # F_normal = P_gas * Area_back + F_tension
        # Area_back = pi * B * height (approx line contact length * height)
        # Actually P_gas acts behind the ring pushing it out.
        L_circ = np.pi * geo.bore
        A_contact = L_circ * cfg.ring_height
        
        # Gas pressure helps seal the ring (blow-by effect)
        # Assume 80% of cylinder pressure gets behind the top ring
        p_contact = 0.8 * p_gas_arr + (cfg.ring_tension / A_contact)
        f_normal = p_contact * A_contact
        
        # 3. Sliding Speed (Absolute)
        u_abs = np.abs(u_piston_arr) + 1e-6 # Avoid zero div
        
        # 4. Stribeck Calculation
        # Stribeck Parameter S = mu * U / P_contact
        S = mu_oil * u_abs / (p_contact + 1e-3)
        
        # Friction Coefficient (Schematic Stribeck)
        # COF = c_boundary / (1 + C1 * S) + c_hydro * S
        # This blends boundary (low S) to mixed to hydro (high S)
        mu_boundary = 0.15
        C1 = 1.0e6 # Transition hardness
        
        cof_trace = mu_boundary / (1.0 + C1 * S**2)**0.5 + 20.0 * S
        
        # 5. Friction Force
        f_friction = cof_trace * f_normal * np.sign(u_piston_arr)
        
        # 6. Work Calculation
        # W = Integral(F_f * dx) = Integral(F_f * v * dt)
        # Or more simply: F_f * v is Power. Integrate Power over cycle.
        
        power_friction = np.abs(f_friction * u_piston_arr)
        
        # Cycle Integration (Trapezoidal)
        # We need time steps. theta is likely uniform or passed in.
        # dt = dtheta / omega?
        # Let's use mean power -> Work per cycle
        
        return {
            "friction_force": f_friction,
            "power_friction": power_friction,
            "mu_oil": mu_oil
        }

class FrictionSimulation(BaseSimulation):
    """Main Friction Simulation Wrapper."""
    
    def __init__(self, name: str, config: FrictionConfig):
        super().__init__(name, config)
        self.input_data: SimulationInput = None
        self.solver = StribeckSolver()
        self.config = config # Pydantic model
        
    def load_input(self, input_data: SimulationInput):
        self.input_data = input_data
        
    def setup(self):
        if not self.input_data:
            raise ValueError("No Input Data Loaded")
        
    def step(self, dt: float):
        pass
        
    def solve_steady_state(self) -> SimulationOutput:
        """Run the cycle-averaged friction calculation."""
        self.setup()
        
        inp = self.input_data
        bcs = inp.boundary_conditions
        
        theta = np.array(bcs.crank_angle)
        p_gas = np.array(bcs.pressure_gas)
        u_piston = np.array(bcs.piston_speed)
        
        if u_piston is None:
            # Reconstruct speed from theta/RPM if missing?
            # Schema says optional, but for friction we need it.
            # Assuming it's provided by Exporter.
            raise ValueError("Piston Speed (u_piston) required for friction solver")
            
        res = self.solver.solve_cycle(
            theta, p_gas, u_piston, 
            inp.operating_point.T_oil,
            inp.geometry,
            self.config
        )
        
        # Calculate FMEP
        # Power [W]
        power = res["power_friction"]
        # Cycle Work [J] = Integral Power dt
        # Map theta to time: t = theta / (6N)
        rpm = inp.operating_point.rpm
        omega = rpm * 2 * np.pi / 60.0
        
        # dTheta array
        # Handle wrap around? theta usually monotonic 0->720
        d_theta = np.gradient(theta) * (np.pi/180.0) # Deg to Rad
        dt_arr = d_theta / omega
        
        work_j = np.sum(power * dt_arr) # Only valid for full cycle?
        # NOTE: DOE is usually Expansion Stroke Only (0->180).
        # We must normalized carefully.
        # FMEP = Work / V_disp
        
        V_disp = (np.pi * inp.geometry.bore**2 / 4.0) * inp.geometry.stroke
        fmep_pa = work_j / V_disp 
        # If we only integrated Expansion (1/4 cycle), this FMEP is "During Expansion".
        # But global FMEP is usually cycle-averaged. 
        # Let's report what we calculated. 180 deg integration is partial FMEP.
        # But better to just report the scalar.
        
        # Scaling FMEP to bar
        fmep_bar = fmep_pa / 1e5
        
        # Fit A, B coeffs?
        # A + B * P_max.
        # We can pass P_max.
        p_max = np.max(p_gas)
        
        out = SimulationOutput(
            run_id=inp.run_id,
            success=True,
            friction_fmep=fmep_bar,
            # Pass back "Calibration Points" for the surrogate fitter
            calibration_params={
                "p_max_bar": float(p_max / 1e5),
                "fmep_bar": float(fmep_bar),
                "rpm": float(rpm)
            }
        )
        return out

    def post_process(self) -> Dict[str, Any]:
        return {}
