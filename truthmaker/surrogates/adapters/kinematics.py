import numpy as np

class InverseKinematics:
    """
    Solves the Inverse Kinematics for a Slider-Crank mechanism.
    Given Piston Position x, find Crank Angle phi.
    """
    def __init__(self, stroke: float, conrod: float, offset: float = 0.0):
        """
        Args:
            stroke: Full stroke length [m] (2 * crank_radius)
            conrod: Connecting rod length [m]
            offset: Piston pin offset [m] (Default 0)
        """
        self.r = stroke / 2.0
        self.l = conrod
        self.offset = offset
        
    def get_crank_angle(self, x_target: float, branch: int = 1) -> float:
        """
        Calculate Crank Angle phi for a given Piston Position x.
        
        Equation for Slider Crank Position x (from center of crank):
        x = r * cos(phi) + sqrt(l^2 - (r * sin(phi) - offset)^2)
        
        This is a transcendental equation if solving for phi directly is hard, 
        but usually we solve:
        l^2 = (x - r*cos(phi))^2 + (r*sin(phi) - offset)^2
        
        Actually, standard derivation finds x(phi).
        Here we need phi(x).
        
        Is it unique? No, there are two solutions (up/down stroke) for a given x.
        Since we are processing a trajectory x(t), we need to track the continuous phase.
        
        Numerical inversion is robust here given the initial guess from previous step.
        """
        # Simplification: Assume Center-Line (offset=0)
        # x = r cos(phi) + sqrt(l^2 - r^2 sin^2(phi))
        # Let's use simple geometric relation with Arccos.
        # But we need full 0-360 handling.
        pass

    def solve_trajectory(self, x_arr: np.ndarray, theta_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves for phi(theta) and ratio(theta) for a full cycle.
        
        Args:
            x_arr: Phase 1 Piston Positions [m] (0=TDC, Increasing towards BDC)
            theta_arr: Cycle clock angles [rad] (For calculating derivative)
            
        Returns:
            phi_arr: Physical crank angles [rad] (accumulated/unwrapped)
            ratio_arr: Gear ratio d_phi/d_theta
        """
        # Transform Phase 1 x to Geometric x (Distance from Crank Center)
        # Phase 1: 0 at TDC
        # Geometric: r+l at TDC
        x_geo = (self.r + self.l) - x_arr
        
        # 1. Algebraic Solution for Principal Angle alpha in [0, pi]
        # cos(alpha) = (x^2 + r^2 - l^2) / (2xr)
        arg = (x_geo**2 + self.r**2 - self.l**2) / (2 * x_geo * self.r)
        arg = np.clip(arg, -1.0, 1.0)
        alpha = np.arccos(arg)
        
        # 2. Determine Branch (0-pi vs pi-2pi)
        # We need to look at velocity v = dx_phase1 / dt
        # If x_phase1 is INCREASING (moving to BDC), piston is moving AWAY from head.
        # In Standard Crank (CW?): 
        # theta 0 -> pi moves TDC (r+l) -> BDC (l-r). x_geo DECREASES.
        # So x_phase1 INCREASES.
        # So if dx_phase1 > 0, we are in 0->pi.
        # If dx_phase1 < 0, we are in pi->2pi.
        
        # BUT Phase 1 loops might have multiple strokes (4 stroke?? No, 2 stroke OP).
        # We just need continuity.
        # Let's derive velocity to check direction.
        grad_x = np.gradient(x_arr, theta_arr) # dx/dtheta check
        
        phi_sol = np.zeros_like(alpha)
        
        # Logic:
        # If grad_x > 0 (Moving to BDC), we are in 0 to pi branch. (phi = alpha)
        # If grad_x < 0 (Moving to TDC), we are in pi to 2pi branch. (phi = 2pi - alpha)
        # What if grad_x is 0? (At Dead Centers). Continuity handles it.
        
        for i in range(len(alpha)):
            if grad_x[i] >= 0:
                phi_sol[i] = alpha[i]
            else:
                phi_sol[i] = 2 * np.pi - alpha[i]
                
        # 3. Unwrap Phase (Handle multicycle or wrap-around)
        phi_unwrapped = np.unwrap(phi_sol)
        
        # 4. Calculate Ratio (d_phi / d_theta)
        # This is the "Transmission Function"
        ratio = np.gradient(phi_unwrapped, theta_arr)
        
        return phi_unwrapped, ratio

    def verify_forward_kinematics(self, phi_arr: np.ndarray) -> np.ndarray:
        """
        Calculates Piston Position x from Crank Angle phi.
        x_geo = r cos(phi) + sqrt(l^2 - r^2 sin^2(phi))
        x_phase1 = (r+l) - x_geo
        """
        # Geometric (Distance from center)
        # Note: This formula assumes offset=0
        # Check for imaginary sqrt (geometric violation)
        term = self.l**2 - (self.r * np.sin(phi_arr))**2
        # Clip to 0
        term = np.maximum(term, 0.0)
        
        x_geo = self.r * np.cos(phi_arr) + np.sqrt(term)
        
        # Transform to Phase 1 (0 at TDC)
        x_phase1 = (self.r + self.l) - x_geo
        return x_phase1

