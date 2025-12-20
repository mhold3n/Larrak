"""
Transient Drive Cycle Simulation (Quasi-Steady).
Simulates a vehicle driving a simplified WLTP cycle using Larrak Engine Maps.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from campro.optimization.calibration.registry import get_calibration

class VehicleModel:
    def __init__(self):
        self.mass = 1500.0 # kg
        self.Cd = 0.30
        self.A_front = 2.2 # m2
        self.rho_air = 1.2 # kg/m3
        self.r_wheel = 0.3 # m
        self.crr = 0.01 # Rolling resistance
        
        # 6-Speed Transmission
        self.gear_ratios = [15.0, 10.0, 7.0, 5.0, 4.0, 3.0] # Incl Final Drive
        self.final_drive = 1.0 
        
    def power_req(self, v: float, a: float, grade: float = 0.0) -> float:
        """Calculate Power Requirement at Wheels [W]."""
        # Aerodynamic Drag
        F_aero = 0.5 * self.rho_air * self.Cd * self.A_front * v**2
        
        # Rolling Resistance
        F_roll = self.mass * 9.81 * self.crr
        
        # Inertia / Grade
        F_inert = self.mass * a
        F_grade = self.mass * 9.81 * np.sin(grade)
        
        F_total = F_aero + F_roll + F_inert + F_grade
        P_wheel = F_total * v
        
        return max(0.0, P_wheel) # No regen for ICE logic yet
        
    def get_gear_and_rpm(self, v: float) -> tuple[int, float]:
        """Simple Shift Logic based on RPM limits."""
        # Target RPM range: 1500 - 3000 cruising
        
        if v < 0.1: return 1, 800.0 # Idle
        
        for i, ratio in enumerate(self.gear_ratios):
            # Omega Wheel = v / r
            w_wheel = v / self.r_wheel
            w_eng = w_wheel * ratio
            rpm = w_eng * 60.0 / (2 * np.pi)
            
            if rpm > 1200 and rpm < 3500:
                return i+1, rpm
            if i == len(self.gear_ratios)-1: # Top gear
                return i+1, max(800.0, rpm)
                
        return 1, 800.0 # Fallback

def generate_wltp_cycle() -> tuple[np.ndarray, np.ndarray]:
    """Generate simplified time-velocity profile [s, m/s]."""
    # Create a synthetic cycle with Urban, Rural, Highway phases
    t = np.arange(0, 600, 1.0) # 10 mins short test
    v = np.zeros_like(t)
    
    # 0-50: Idle
    
    # 50-100: Accel to 30 kph (8.3 m/s)
    v[50:100] = np.linspace(0, 8.3, 50)
    
    # 100-200: Cruise 30 kph
    v[100:200] = 8.3
    
    # 200-250: Accel to 100 kph (27.7 m/s)
    v[200:250] = np.linspace(8.3, 27.7, 50)
    
    # 250-400: Cruise 100 kph
    v[250:400] = 27.7
    
    # 400-500: Decel
    v[400:500] = np.linspace(27.7, 0, 100)
    
    return t, v

def run_simulation():
    print("=== Transient Cycle Simulation ===")
    
    # Setup
    veh = VehicleModel()
    t_arr, v_arr = generate_wltp_cycle()
    
    # Results
    fuel_cum_g = 0.0
    results = []
    
    for i in range(len(t_arr)-1):
        t = t_arr[i]
        v = v_arr[i]
        
        # Calculate Accel (Forward Diff)
        a = (v_arr[i+1] - v) / 1.0
        
        # Power
        P_req = veh.power_req(v, a)
        
        # Gear / RPM
        gear, rpm = veh.get_gear_and_rpm(v)
        
        # Load (BMEP/P_max proxy)
        # Power = Torque * w
        # Torque = Power / w
        w_eng = rpm * 2 * np.pi / 60.0
        if w_eng > 1.0:
            torque = P_req / w_eng
        else:
            torque = 0.1 # Idle torque
            P_req = 1000.0 # Idle power overhead
            
        # P_int calculation (Simplified Map)
        # P_int ~ Torque / Vd ... 
        # For Registry, we need Features: 'rpm', 'p_max_bar', etc.
        # We need to map Torque -> P_max.
        # P_max ~ 50 + 2 * BMEP_bar approx?
        # Let's use a rough proxy for now since we don't have the inverse map yet.
        # We have Friction Map (fmep = f(p_max, rpm)).
        
        p_max_proxy = 50.0 + (torque / 500.0) * 100.0 # Rough scaling
        
        # Query Registry for Map Data
        # Friction
        fmep_res = get_calibration("friction", {"p_max_bar": p_max_proxy, "rpm": rpm})
        # If returns dict
        fmep = 1.0 # Default
        if isinstance(fmep_res, dict): fmep = fmep_res.get("value", 1.0)
        elif isinstance(fmep_res, float): fmep = fmep_res
        
        # Combustion (Wiebe) - Not directly used for fuel here, but affects efficiency
        # Eff = Eff_ind * (1 - Friction_Loss)
        # Eff_ind ~ 0.45 (Ideal)
        eff_ind = 0.45
        P_frict = (fmep * 1e5) * (0.0015) * (rpm/120.0) # Vd * N/2
        
        # P_fuel = (P_brake + P_frict) / Eff_ind
        P_fuel = (P_req + P_frict) / eff_ind
        
        # Fuel Rate
        # LHV = 44 MJ/kg
        fuel_rate_kg_s = P_fuel / 44.0e6
        fuel_cum_g += fuel_rate_kg_s * 1000.0 * 1.0 # 1 sec
        
        results.append({
            "t": t, "v": v, "rpm": rpm, "gear": gear, 
            "power": P_req, "fuel_rate": fuel_rate_kg_s*1000
        })
        
    print(f"Cycle Complete.")
    print(f"Total Fuel Consumed: {fuel_cum_g:.2f} grams")
    print(f"Avg Speed: {np.mean(v_arr)*3.6:.1f} kph")
    
    # Save Results
    import csv
    with open("tests/wltp_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("Results saved to tests/wltp_results.csv")

if __name__ == "__main__":
    run_simulation()
