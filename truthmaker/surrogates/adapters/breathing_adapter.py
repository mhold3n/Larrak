from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from campro.litvin.motion import RadialSlotMotion

class BreathingAdapter(RadialSlotMotion):
    """
    Adapts Phase 2 Breathing Kinematics data to CamPro's RadialSlotMotion interface.
    """
    def __init__(self, theta_arr: np.ndarray, C_arr: np.ndarray, psi_arr: np.ndarray):
        """
        Args:
            theta_arr: Reference Angle (Radians). Assumed monotonic 0-2pi.
            C_arr: Center Distance trajectory [m].
            psi_arr: Planet Angle trajectory [rad].
        """
        # Ensure periodicity/wrapping for interpolation? 
        # For now assume theta 0 to 2pi.
        
        # Base Center Radius R0 (Mean C)
        self.R0 = np.mean(C_arr)
        
        # Force Closure for C (Center Distance must be periodic)
        C_arr[-1] = C_arr[0] 
        # Create Offset Array
        offset_arr = C_arr - self.R0
        
        # Handle Psi Wrapping (Spin)
        # psi(2pi) = psi(0) + 2*pi*k + periodic_error
        # We model psi(t) = slope * t + periodic(t)
        # slope = (psi[-1] - psi[0]) / (2*pi) (Approximation if theta 0->2pi)
        
        # Ensure theta is 0 to 2pi
        span = theta_arr[-1] - theta_arr[0]
        slope = (psi_arr[-1] - psi_arr[0]) / span
        
        psi_periodic = psi_arr - slope * theta_arr
        psi_periodic[-1] = psi_periodic[0] # Force closure on residual
        
        # Create Splines
        # Use bc_type='periodic' to ensure smooth closure
        self.spline_offset = CubicSpline(theta_arr, offset_arr, bc_type='periodic')
        self.spline_psi_periodic = CubicSpline(theta_arr, psi_periodic, bc_type='periodic')
        
        # Define Functions
        center_offset_fn = lambda t: float(self.spline_offset(t % (2*np.pi)))
        
        # Re-add slope
        planet_angle_fn = lambda t: float(self.spline_psi_periodic(t % (2*np.pi))) + slope * t
        
        d_center_offset_fn = lambda t: float(self.spline_offset(t % (2*np.pi), 1))
        # d2_center_offset_fn = lambda t: float(self.spline_offset(t % (2*np.pi), 2)) # Not heavily used
        
        super().__init__(
            center_offset_fn=center_offset_fn,
            planet_angle_fn=planet_angle_fn,
            d_center_offset_fn=d_center_offset_fn
        )
