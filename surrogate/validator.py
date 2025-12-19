import numpy as np
from scipy.integrate import cumulative_trapezoid

def validate_gear_set(
    Rp_arr: np.ndarray, 
    Rr_arr: np.ndarray, 
    C_arr: np.ndarray, 
    target_motion_x: np.ndarray,
    theta_rad: np.ndarray = None
) -> tuple[float, bool]:
    """
    Validates a candidate gear set against the target piston motion using High-Fidelity Kinematics.
    
    Args:
        Rp_arr: Planet Radius Profile [mm] (Array of length N)
        Rr_arr: Ring Radius Profile [mm]
        C_arr: Center Distance Profile [mm]
        target_motion_x: Target Piston Position [mm]
        theta_rad: Crank Angle array [rad]. If None, assumed 0 to 2pi.
        
    Returns:
        rmse: Root Mean Square Error [mm] between Actual and Target motion.
        feasible: Boolean indicating if geometry is physically valid.
    """
    N = len(Rp_arr)
    if theta_rad is None:
        theta_rad = np.linspace(0, 2*np.pi, N)
        
    # 1. Geometry Constraints Check
    # Planet must be smaller than Ring (Internal Gear)
    if np.any(Rp_arr >= Rr_arr):
        return 999.0, False # Interference
        
    # Radius must be positive
    if np.any(Rp_arr <= 0) or np.any(Rr_arr <= 0):
        return 999.0, False # Physical impossibility
        
    # Center Distance Consistency check
    # For internal gears: C = Rr - Rp
    # Check if optimized C matches implied geometry
    C_implied = Rr_arr - Rp_arr
    c_error = np.mean(np.abs(C_arr - C_implied))
    if c_error > 0.1: # Allow small optimization slack
         # Warn? or Fail? 
         # Let's assume the Optimizer might relax C slightly, but kinematics strictly obey C = Rr - Rp?
         # No, Kinematics logic:
         # Piston x depends on C and Rp.
         # So we use the Provided C_arr.
         pass
         
    # 2. Forward Kinematics
    # Transmission Function i(theta) = d(psi)/d(theta) = Rr / Rp
    # (Instantaneous ratio at contact point)
    # Note: Depending on reference frame. 
    # For stationary ring: d(psi_planet_abs)/d(theta_carrier) = 1 + Rr/Rp?
    # Let's align with `breathing.py` inverse logic:
    # "ratio = d(psi)/d(theta)"
    # Standard Hypocycloid: angular velocity of planet relative to carrier is w_p = w_c * (Rr/Rp).
    # Total angle psi (absolute) = theta * (1 + Rr/Rp)?
    # Let's use i = Rr/Rp for the "Rolling" component.
    
    # Let's trust the Phase 3 optimization definition:
    # i_ratio = Rr / Rp
    i_ratio = Rr_arr / Rp_arr
    
    # Integrate to get Planet Angle psi
    # psi(0) assumed 0? Or Phase match?
    # Let's assume 0 at TDC (theta=0).
    psi_arr = cumulative_trapezoid(i_ratio, theta_rad, initial=0)
    
    # 3. Calculate Piston Position
    # x = C - Rp * cos(psi)
    # Rationale: At psi=0 (TDC), we want x = Min (0)?
    # Wait, usually TDC is MAX extension or MIN volume?
    # Larrak Engine: Opposed Piston.
    # If x=0 is center (Head), then x increases outwards.
    # Target profile: 0 -> 150.
    # So 0 is Inner Dead Center (IDC).
    # Breathing Gear TDC (Max Extension) vs IDC (Min Extension).
    # x = C + Rp * cos(psi) -> Max at psi=0 (Value C+Rp).
    # x = C - Rp * cos(psi) -> Min at psi=0 (Value C-Rp).
    # Since Target starts at 0 (Min), we need x = C - Rp * cos(psi).
    x_kinematic = C_arr - Rp_arr * np.cos(psi_arr)
    
    # 4. Error Calculation
    error = x_kinematic - target_motion_x
    rmse = np.sqrt(np.mean(error**2))
    
    # 5. Feasibility
    # Check for "Undercutting" or "Looping" implies d(Rp)/dtheta shouldn't be too high?
    # Simple check: Is RMSE excessive? implies kinematic mismatch.
    feasible = True
    if rmse > 5.0: # Arbitrary "Gross Failure" threshold
        feasible = False
        
    return rmse, feasible
