import numpy as np

class LitvinVerifier:
    """
    Verifies feasibility of Non-Circular Gear (NCG) generation based on Litvin's theory.
    """
    def __init__(self, center_distance: float = 0.1):
        self.C = center_distance
        
    def verify_ratio_profile(self, theta_arr: np.ndarray, ratio_arr: np.ndarray) -> dict:
        """
        Analyzes the Gear Ratio function i(theta) = d_phi_out / d_phi_in.
        
        Criteria:
        1. Sign Consistency: Ratio should not cross 0 (Reversal) unless intended (rare for efficient power).
           If Ratio < 0, it means output spins opposite to input.
           If Ratio crosses 0, it means zero velocity (infinite torque?).
           
        2. Singularity: If using Internal Gears, Ratio=1 is a singularity (Radii -> inf).
        
        3. Closure: Integral(Ratio) dTheta over 2pi must be 2pi * k.
        """
        results = {}
        
        # 1. Sign Check
        # Valid if all positive OR all negative.
        min_r = np.min(ratio_arr)
        max_r = np.max(ratio_arr)
        
        if (min_r < 0 and max_r > 0):
            results["sign_consistent"] = False
            results["sign_status"] = "mixed"
        else:
            results["sign_consistent"] = True
            results["sign_status"] = "positive" if min_r >= 0 else "negative"
            
        # 2. Internal Gear Singularity Check (Ratio approx 1.0)
        # Avoid 0.95 to 1.05
        # If min < 1 < max, we have a singularity for internal gears.
        if min_r < 1.0 and max_r > 1.0:
            results["singularity_internal"] = True
        else:
            results["singularity_internal"] = False
            
        # 3. Closure
        # Integrate ratio using Trapezoidal
        # theta is in radians
        integral = np.trapz(ratio_arr, theta_arr)
        target = 2 * np.pi
        
        closure_err = abs(integral - target)
        results["closure_error_deg"] = np.degrees(closure_err)
        # Allow small numerical error (e.g. 1 degree)
        results["is_closed"] = results["closure_error_deg"] < 1.0
        
        results["min_ratio"] = min_r
        results["max_ratio"] = max_r
        
        return results
