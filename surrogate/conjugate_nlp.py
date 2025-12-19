import casadi as ca
import numpy as np
from surrogate.gear_config import GEAR_CONFIG

def build_gear_nlp(
    target_profile_x: np.ndarray, 
    n_points: int = 360,
    debug: bool = False
) -> tuple[dict, dict]:
    """
    Builds the Conjugate Gear NLP.
    Goal: Find Pitch Curves Rp(theta), Rr(theta) that reproduce target_profile_x(theta).
    
    Variables:
    - R_planet(theta): Planar radius of planet gear.
    - R_ring(theta): Planar radius of ring gear.
    
    Constraints:
    - Conjugacy: V_rel dot N = 0 (Implicit in kinematic generation or imposed).
    - Toplogy: R_ring > R_planet.
    - Ratio: Integral(R_planet) / Integral(R_ring) = 1/2 (2:1 Ratio).
    
    Note: 'target_profile_x' drives the required ratio.
    Actually, transmission function 'i(theta)' is derived from 'target_profile_x'.
    i(theta) = d(psi_planet)/d(theta_ring).
    
    So the NLP optimizes R_planet to minimize error against 'i(theta)' 
    while satisfying R_ring = R_planet * i + C? 
    No, R_ring and R_planet are coupled by center distance C.
    
    Let's stick to the 'Closed Loop' philosophy:
    The NLP finds the optimal Geometry (Rp, Rr, C) to match the Target Motion.
    """
    
    # 0. Setup
    opti = ca.Opti()
    
    # Grid
    N = n_points
    theta = np.linspace(0, 2*np.pi, N)
    
    # 1. Variables
    # Planet Radius Curve (Polar)
    Rp = opti.variable(N)
    opti.set_initial(Rp, 40.0) # Start within 20-80mm range
    
    # Center Distance (Variable Profile for Breathing Gear)
    # Allows non-harmonic motion generation.
    C_var = opti.variable(N)
    opti.set_initial(C_var, 60.0) 
    opti.subject_to(C_var >= 10.0) 
    
    # 2. Derived Variables
    # Ring Radius (from Center Distance and Pitch Point)
    # Rr = C + Rp (Standard Internal Gear geometry at each theta)
    Rr = C_var + Rp
    
    # Transmission Ratio (Instantaneous)
    # i = Rr / Rp 
    i_ratio = Rr / Rp 
    
    # 3. Parameters
    Target_X = opti.parameter(N)
    opti.set_value(Target_X, target_profile_x)
    
    # 4. Objective
    J = 0
    
    # Smoothness (dRp/dtheta)
    dRp = Rp[1:] - Rp[:-1]
    J += GEAR_CONFIG.gear.w_smooth * ca.sumsqr(dRp)
    
    # Breathing Smoothness (dC/dtheta)
    dC = C_var[1:] - C_var[:-1]
    J += GEAR_CONFIG.gear.w_smooth * ca.sumsqr(dC) # Reuse same weight or add w_breathing
    
    # Tracking Error Objectve (Core Kinematics)
    # 1. Integrate Gear Ratio to get Psi (Planet Angle)
    # psi[k] = sum(i_ratio[0]...i_ratio[k]) * dtheta
    # Use Matrix Multiplication for cumulative sum which works with CasADi symbols
    
    # Create Lower Triangular Matrix of 1s (N x N)
    Tril = ca.MX(np.tril(np.ones((N, N))))
    
    # dtheta
    theta_total = 2 * np.pi
    dtheta = theta_total / (N - 1) # or N depending on endpoint
    
    # Psi profile
    # psi = Tril @ i_ratio * dtheta
    psi_kin = ca.mtimes(Tril, i_ratio) * dtheta
    
    # 2. Calculate Piston Position Model
    # Consistent with Validator: x = C - Rp * cos(psi)
    # (Assuming 0 start matches Target Minimum)
    Model_X = C_var - Rp * ca.cos(psi_kin)
    
    # 3. Add to Objective
    # Normalize by scale?
    J += GEAR_CONFIG.gear.w_tracking * ca.sumsqr(Model_X - Target_X)
    
    # Periodicity
    opti.subject_to(Rp[0] == Rp[-1])
    opti.subject_to(C_var[0] == C_var[-1])
    
    # Bounds (From Config)
    opti.subject_to(Rp >= GEAR_CONFIG.gear.min_radius)
    opti.subject_to(Rp <= GEAR_CONFIG.gear.max_radius)
    
    # 2:1 Mean Ratio Constraint
    # Integral(i) dtheta = 4 pi (Planet rotates 720 for 360 ring?) 
    # Or 2:1 means Ring=360, Planet=180?
    # Config says ratio=2.0
    mean_ratio = ca.sum1(i_ratio)/N
    opti.subject_to(mean_ratio == GEAR_CONFIG.gear.ratio)
    
    opti.minimize(J)
    
    # Return raw Opti for flexibility in the Runner
    return opti, {"Rp": Rp, "Rr": Rr, "C": C_var, "Target": Target_X}
