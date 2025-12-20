import numpy as np
import casadi as ca

class ConjugateOptimizer:
    """
    Optimizes Ring and Planet pitch curves (Rr, Rp) to match target piston motion
    while satisfying internal gear constraints.
    """
    def __init__(self, n_points: int = 360):
        self.N = n_points
        
    def solve(self, theta_arr: np.ndarray, x_target_arr: np.ndarray, 
              omega_ring_ratio: float = 1.0, target_psi_total: float = None,
              symmetry_blocks: int = 1):
        """
        Solve using Scipy SLSQP (Robust fallback).
        Args:
            target_psi_total: Optional total rotation of Planet (radians).
            symmetry_blocks: Number of identical repeating sectors (e.g. 2 for 2-lobe symmetry).
        """
        from scipy.optimize import minimize
        
        N = self.N
        dt = theta_arr[1] - theta_arr[0]
        
        # State vector: [Rr_0...Rr_N-1, Rp_0...Rp_N-1, slip, offset]
        
        def unpack(x_vec):
            Rr = x_vec[0:N]
            Rp = x_vec[N:2*N]
            slip = x_vec[2*N]
            offset = x_vec[2*N+1]
            return Rr, Rp, slip, offset
            
        def obj_fn(x_vec):
            Rr, Rp, slip, offset = unpack(x_vec)
            
            dpsi = slip * (Rr / Rp)
            Psi = np.cumsum(dpsi) * dt 
            Psi = np.insert(Psi, 0, 0.0)[:-1] 
            
            C = Rr - Rp
            # X_mech = C + Rp * cos(Psi)
            X_mech = C + Rp * np.cos(Psi)
            
            residuals = (offset - X_mech) - x_target_arr
            J_track = np.sum(residuals**2)
            
            # Regularization
            dRr = np.diff(Rr)
            dRp = np.diff(Rp)
            dRr_wrap = Rr[0] - Rr[-1]
            dRp_wrap = Rp[0] - Rp[-1]
            
            J_reg = np.sum(dRr**2) + np.sum(dRp**2) + dRr_wrap**2 + dRp_wrap**2
            
            return J_track + 10.0 * J_reg
            
        # Constraints
        # 1. Rr > Rp + 0.005
        def const_overlap(x_vec):
            Rr, Rp, _, _ = unpack(x_vec)
            return Rr - Rp - 0.005
            
        def const_closure_Rr(x_vec):
            Rr, _, _, _ = unpack(x_vec)
            return Rr[0] - Rr[-1]
            
        def const_closure_Rp(x_vec):
            _, Rp, _, _ = unpack(x_vec)
            return Rp[0] - Rp[-1]
            
        constraints = [
            {'type': 'ineq', 'fun': const_overlap},
            {'type': 'eq', 'fun': const_closure_Rr},
            {'type': 'eq', 'fun': const_closure_Rp}
        ]
        
        # Symmetry Constraints
        if symmetry_blocks > 1:
            block_size = N // symmetry_blocks
            
            def const_symmetry_Rr(x_vec):
                Rr, _, _, _ = unpack(x_vec)
                # Ensure Rr[i] == Rr[i + block_size]
                # Return array of diffs
                diffs = Rr[:N - block_size] - Rr[block_size:]
                # Scipy constraints must return scalar or array. 
                # SLSQP treats array return as multiple constraints? Yes usually.
                # But 'eq' expects 0.
                return diffs
                
            def const_symmetry_Rp(x_vec):
                _, Rp, _, _ = unpack(x_vec)
                return Rp[:N - block_size] - Rp[block_size:]

            # IMPORTANT: Scipy 'fun' needs to handle array output carefully or we loop.
            # SLSQP allows vector constraints.
            constraints.append({'type': 'eq', 'fun': const_symmetry_Rr})
            constraints.append({'type': 'eq', 'fun': const_symmetry_Rp})

        # Target Psi Constraint
        if target_psi_total is not None:
             def const_psi_total(x_vec):
                Rr, Rp, slip, _ = unpack(x_vec)
                dpsi = slip * (Rr / Rp)
                Psi_total = np.sum(dpsi) * dt
                return Psi_total - target_psi_total
             
             constraints.append({'type': 'eq', 'fun': const_psi_total})
        
        # Bounds
        bounds = []
        for _ in range(N): bounds.append((0.05, 0.5)) 
        for _ in range(N): bounds.append((0.02, 0.4)) 
        bounds.append((0.5, 4.0)) # Slip > 0.5 enforced strictly
        bounds.append((-0.5, 0.5)) 
        
        # Initial Guess (Symmetric)
        x0 = np.zeros(2*N + 2)
        x0[0:N] = 0.10 
        x0[N:2*N] = 0.05 
        x0[2*N] = 2.0 
        x0[2*N+1] = 0.15 
        
        res = minimize(obj_fn, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 250, 'ftol': 1e-4, 'disp': True})
        
        if res.success or res.message:
            print(f"Scipy Opt Result: {res.message}")
            Rr, Rp, slip, offset = unpack(res.x)
            C = Rr - Rp
            dpsi = slip * (Rr / Rp)
            Psi = np.cumsum(dpsi) * dt
            Psi = np.insert(Psi, 0, 0.0)[:-1]
            
            return {
                "Rr": Rr,
                "Rp": Rp,
                "Psi": Psi,
                "C": C,
                "slip": slip,
                "success": True
            }
        else:
            print("Scipy Failed.")
            return {
                "Rr": x0[0:N], 
                "Rp": x0[N:2*N],
                "Psi": np.zeros(N),
                "C": np.full(N, 0.1),
                "slip": 1.0,
                "success": False
            }
