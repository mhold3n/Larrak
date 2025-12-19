import numpy as np
from scipy.integrate import cumulative_trapezoid

def analyze_gear_set(
    Rp_arr: np.ndarray, 
    Rr_arr: np.ndarray, 
    C_arr: np.ndarray, 
    target_motion_x: np.ndarray,
    theta_rad: np.ndarray = None
) -> tuple[bool, dict]:
    """
    Performs Geometric FEA (Contact Analysis) on the gear set profiles.
    
    Args:
        Rp_arr, Rr_arr, C_arr: Geometry profiles [mm]
        target_motion_x: Target Piston Position [mm]
        
    Returns:
        feasible: Overall pass/fail.
        metrics: Dictionary of scores for recalibration logic.
            - 'interference': bool (True if binding detected)
            - 'curvature_score': float (Max curvature check)
            - 'tracking_rmse': float (mm)
            - 'jerk_score': float (Proxy for NVH)
    """
    if theta_rad is None:
        N = len(Rp_arr)
        theta_rad = np.linspace(0, 2*np.pi, N)
        
    metrics = {
        'interference': False,
        'curvature_score': 0.0,
        'tracking_rmse': 0.0,
        'jerk_score': 0.0
    }
    
    # 1. Kinematic Validation (Motion Error)
    # i = Rr / Rp
    i_ratio = Rr_arr / Rp_arr
    psi_arr = cumulative_trapezoid(i_ratio, theta_rad, initial=0)
    
    # x = C - Rp * cos(psi) (Using verified phase)
    x_kinematic = C_arr - Rp_arr * np.cos(psi_arr)
    
    error = x_kinematic - target_motion_x
    rmse = np.sqrt(np.mean(error**2))
    metrics['tracking_rmse'] = rmse
    
    # 2. NVH / Jerk Analysis
    # Jerk ~ d3x/dt3. Proportional to d3x/dtheta3 if constant RPM.
    # Numerical differentiation
    dx = np.gradient(x_kinematic, theta_rad)
    ddx = np.gradient(dx, theta_rad)
    dddx = np.gradient(ddx, theta_rad)
    
    metrics['jerk_score'] = np.max(np.abs(dddx))
    
    # 3. Interference / Binding (Curvature Analysis)
    # Check Planet Curvature
    # Metric: Minimum Radius of Curvature.
    # Approx for localized curve y(x) in polar? 
    # Use simple derivative check: huge spikes in dRp usually mean cusp/undercut.
    dRp = np.gradient(Rp_arr, theta_rad)
    ddRp = np.gradient(dRp, theta_rad)
    
    # Curvature score: Sum of squared 2nd derivative (smoothness proxy)
    metrics['curvature_score'] = np.mean(ddRp**2)
    
    # Binding Detection
    # 1. Geometry Overlap
    if np.any(Rp_arr >= Rr_arr):
        metrics['interference'] = True
    
    # 2. Negative Radius (Impossible)
    if np.any(Rp_arr <= 0):
        metrics['interference'] = True
        
    # 3. Undercutting Check (Trochoidal Loop)
    # If d(psi)/d(theta) reverses or Rp changes too fast relative to curvature limit.
    # Heuristic: If Curvature Score is massive (> Threshold), flag binding.
    if metrics['curvature_score'] > 500.0: # Arbitrary high threshold
         metrics['interference'] = True
         
    # Feasibility Decision
    # Fail if Interference OR RMSE > 0.5mm
    feasible = not metrics['interference'] and metrics['tracking_rmse'] <= 0.5
    
    return feasible, metrics
