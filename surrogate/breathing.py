import numpy as np

class BreathingKinematics:
    """
    Solves Kinematics for a Breathing Gear Mechanism (Radial Displacement + Planetary Rotation).
    Topic: Variable Center Distance Hypocycloid.
    
    Model:
    Piston Position x(theta) is vector sum of Carrier Center C and Planet Radius Vector r_p.
    Assuming vertical piston motion:
    x = C_y + r_p * cos(psi)
    
    But usually Carrier rotates too.
    Standard Cardan:
    Carrier Angle = theta
    Center C = (R - r) * [cos(theta), sin(theta)]
    But here, "Breathing" implies C changes length magnitude.
    Let C_vec = C(theta) * [cos(theta), sin(theta)] ?
    
    User says "linear displacement of planet gear center... equal to stroke".
    This implies the Planet Center itself moves linearly? (Or radially in a slot?).
    If Planet Center moves Radially:
    C_vec = C(theta) * [0, 1] ? (Vertical breathing?)
    
    Let's assume the user's specific simplified equation:
    x = C_breathing(theta) + x_gear(psi)
    
    Interpretation:
    C_breathing(theta) provides the "Bulk" displacement (Position).
    x_gear(psi) provides the "Fine" control (Accel).
    
    Let's model:
    x_target(theta) = C(theta) + r_p * cos(psi)
    
    We need to define C(theta).
    User: "Radial displacement... equal to stroke."
    Strict interpretation: C_max - C_min = S.
    Let's try: C(theta) = x_target(theta) - Offset?
    Then r_p * cos(psi) = Const? No.
    
    Let's try the hint: "TDC/BDC 1:1".
    This usually means d_psi/d_theta = 1.
    
    Algorithm:
    1. Define C(theta) as the 'Ideal Sinusoidal Stroke' for the given engine geometry.
       C_ideal = S/2 * (1 - cos(theta))
    2. But x_target is 'Optimal' (Non-Sinusoidal).
    3. The difference must be made up by the Gear Angle psi.
    4. x_target = C_ideal + Correction(psi).
       Wait, simple addition?
       
    Let's try a Vector Closure:
    x_target = C(theta) + r_p * cos(psi)
    
    If we fix C(theta) to be a pure Sinusoid (Breathing), then:
    r_p * cos(psi) = x_target - C_sinusoid.
    cos(psi) = (x_target - C_sinusoid) / r_p.
    psi = arccos(...)
    
    This works IF (x - C) is within [-r_p, r_p].
    Constraint: The 'Residual' must be bounded by Planet Radius.
    
    Parameter r_p: needs to be defined. user didn't specify.
    We can optimize r_p or assume a standard size (e.g. 1/4 stroke?).
    """
    def __init__(self, stroke: float):
        self.stroke = stroke
        # Guess Planet Radius = 1/4 Stroke (Cardan-ish)
        # Or maybe smaller?
        # If C moves by S, and x moves by S.
        # Then x ~ C.
        # So r_p * cos(psi) is a small correction?
        self.r_p = stroke / 4.0 
        
    def solve_trajectory(self, x_arr: np.ndarray, theta_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            x_arr: Target Position (0 to S).
            theta_arr: Cycle Angle.
        
        Returns:
            C_arr: Breathing Profile.
            psi_arr: Planet Angle.
            ratio_arr: Gear Ratio d_psi/d_theta.
        """
        # 1. Define Breathing Profile C(theta)
        # Try Pure Sinusoid matching the Stroke
        # x goes 0 to S.
        # C goes 0 to S.
        # But x is non-harmonic.
        # Let's align C with the fundamental frequency of x.
        
        # Simple attempt: C is the "Sine Wave Equivalent"
        # x_sin = S/2 * (1 - cos(theta))
        # Note: Check phase of x_arr. 0 at theta=0?
        
        # C_breathing = ideal slider crank?
        # Let's use cosine approximation.
        S = self.stroke
        # C(theta) = S/2 * (1 - cos(theta)) + Offset
        # Actually user says "Radial displacement... equal to stroke".
        # So Delta C = S.
        
        # Let's set C to be the "Smoothed" x?
        # Or just the sinusoidal component.
        C_arr = S/2.0 * (1.0 - np.cos(theta_arr))
        
        # 2. Solve for psi
        # x = C + component?
        # Wait, if x ~ C, then component ~ 0.
        # x_target = C_breathing + r_p * cos(psi)
        # If x_target == C_breathing, then r_p * cos(psi) = 0 -> psi = pi/2.
        # If x_target deviates, psi changes.
        
        # Check bounds
        # residual = x_target - C_arr
        # We need residual / r_p in [-1, 1].
        # If residual > r_p, we need larger r_p.
        
        residual = x_arr - C_arr
        
        # Auto-scale r_p to fit residual with margin
        max_res = np.max(np.abs(residual))
        if max_res > self.r_p:
            print(f"  [Info] Increasing Planet Radius to fit deviation: {max_res:.4f} m")
            self.r_p = max_res * 1.1 # 10% margin
            
        arg = residual / self.r_p
        
        # Invert to find psi
        # Branch ambiguity?
        # Try to maintain continuity near pi/2.
        psi = np.arccos(arg)
        
        # Unwrapping?
        # If psi oscillates around pi/2, arccos is fine (0 to pi).
        
        # 3. Calculate Ratio
        psi_unwrap = np.unwrap(psi)
        ratio = np.gradient(psi_unwrap, theta_arr)
        
        return C_arr, psi_unwrap, ratio
