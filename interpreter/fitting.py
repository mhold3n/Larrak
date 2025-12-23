"""
Parametric NLP for Gear Ratio Curve Fitting.
"""

import casadi as ca
import numpy as np


class RatioCurveFitter:
    """
    Fits a smooth B-Spline gear ratio function to ideal kinematic data.
    """

    def __init__(self, n_knots=50, degree=3):
        self.n_knots = n_knots
        self.degree = degree

    def fit(
        self,
        phi_grid: np.ndarray,
        i_ideal: np.ndarray,
        mean_ratio_target: float = 2.0,
        weights: dict = None,
    ) -> dict:
        """
        Run the optimization.

        Args:
            phi_grid: Ring angle array (must be monotonic).
            i_ideal: Ideal ratio array.
            mean_ratio_target: Target mean ratio.
            weights: Dictionary of cost weights (track, smooth, etc).

        Returns:
            Dictionary with optimized 'weights', 'i_fitted', 'status'.
        """
        weights = weights or {}
        w_track = weights.get("track", 1.0)
        w_smooth = weights.get("smooth", 1.0)

        N = len(phi_grid)
        if len(i_ideal) != N:
            raise ValueError("phi_grid and i_ideal must have same length.")

        # 1. Setup B-Spline Basis
        # Knots over the domain [phi_min, phi_max]
        phi_min, phi_max = phi_grid[0], phi_grid[-1]

        # CasADi BSpline setup
        # We optimize Control Points (Coefficients) 'C'
        # i(phi) = Basis(phi) @ C

        # Simple uniform knots
        knots = np.linspace(phi_min, phi_max, self.n_knots - self.degree + 1)
        # Pad knots for clamped/periodic?
        # Let's use CasADi's bspline convenience or just fit to values?
        # A simpler parametric approach for 1D fit:
        # Define variable C (coefficients)
        # Precompute Basis Matrix M (N x n_coeffs)
        # i_vec = M @ C

        # Scipy for basis generation
        try:
            from scipy.interpolate import BSpline

            n_coeffs = self.n_knots  # Number of control points
            # Construct knot vector with padding
            # t: knot vector
            # interior knots:
            n_interior = n_coeffs - self.degree
            # We want total n_coeffs basis functions.
            # len(t) = n_coeffs + degree + 1

            # Interior knots evenly spaced
            dt = (phi_max - phi_min) / (n_coeffs - self.degree)
            t_interior = np.linspace(
                phi_min, phi_max, n_coeffs - self.degree + 1
            )  # This has n_interior + 1 points?

            # Robust knot vector generation
            # Let's use linspace for all knots including padding to be safe/simple
            # Greville abscissae style?

            # Simply:
            t = np.concatenate(
                [
                    [phi_min] * (self.degree),
                    np.linspace(phi_min, phi_max, n_coeffs - self.degree + 1),
                    [phi_max] * (self.degree),
                ]
            )

            # Generate Basis Matrix M
            # Row k has [B_0(phi_k), ..., B_n(phi_k)]
            M = np.zeros((N, n_coeffs))

            # Identity weights
            eye = np.eye(n_coeffs)
            for j in range(n_coeffs):
                spl = BSpline(t, eye[j], self.degree)
                M[:, j] = spl(phi_grid)

            # Derivative Basis M_prime
            M_prime = np.zeros((N, n_coeffs))
            for j in range(n_coeffs):
                spl = BSpline(t, eye[j], self.degree).derivative()
                M_prime[:, j] = spl(phi_grid)

        except ImportError:
            raise ImportError("Scipy required for B-spline basis construction")

        # 2. CasADi Optimization
        opti = ca.Opti()

        # Decision Variables: Control Points
        coeffs = opti.variable(n_coeffs)

        # Evaluate Spline
        # i_fit = M @ coeffs
        # In CasADi, M must be DM or numpy
        i_fit = ca.mtimes(M, coeffs)
        i_prime = ca.mtimes(M_prime, coeffs)

        # Objective
        J_track = ca.sumsqr(i_fit - i_ideal)
        J_smooth = ca.sumsqr(i_prime)
        # J_smooth_2nd? (Curvature)

        opti.minimize(w_track * J_track + w_smooth * J_smooth)

        # Constraints
        # 1. Bounds on ratio
        opti.subject_to(
            i_fit >= 1.0
        )  # Cannot be less than 1 (physically > 0, 1 implies no relative motion if diff?)
        # Wait, i = R/r. internal gear -> i > 1.
        opti.subject_to(i_fit <= 5.0)

        # 2. Mean Ratio
        # Integral average approx by sum if uniform grid
        opti.subject_to(ca.sum1(i_fit) / N == mean_ratio_target)

        # 3. Periodicity (Start = End)
        opti.subject_to(coeffs[0] == coeffs[-1])  # C0 continuity
        # C1 continuity for periodic
        # coeffs[0]-coeffs[1] ? No, BSpline periodicity links first/last k coeffs.
        # For now, explicit constraints on values/derivs
        opti.subject_to(i_fit[0] == i_fit[-1])
        opti.subject_to(i_prime[0] == i_prime[-1])

        # Initial guess: flat mean
        opti.set_initial(coeffs, mean_ratio_target)

        # Solve
        # Combine options into a single dictionary
        # CasADi Opti.solver(name, opts) is safer than splitting plugin/solver opts
        solver_opts = {
            "expand": True,
            "ipopt.print_level": 5,
            "ipopt.tol": 1e-4,
            "print_time": False,
        }
        opti.solver("ipopt", solver_opts)

        try:
            sol = opti.solve()
            coeffs_opt = sol.value(coeffs)
            status = "Optimal"
        except Exception as e:
            # If solve crashes (e.g. bad options), we might not be able to get debug values
            # Check if e is unrelated to infeasibility
            print(f"Fit failed with error: {e}")
            try:
                # opti.debug.value only works if solver attempted at least one step?
                # or if we provide initial guess to debug.value?
                # CasADi 3.5+: opti.debug.value(x) returns current value
                coeffs_opt = opti.debug.value(coeffs)
                status = "Failed (Debug Value)"
            except Exception:
                # Fallback
                coeffs_opt = np.full(n_coeffs, mean_ratio_target)
                status = "Failed (Fallback)"

        # Re-evaluate
        i_final = M @ coeffs_opt

        return {
            "status": status,
            "weights": coeffs_opt.tolist(),
            "i_fitted": i_final.tolist(),
            "knots": t.tolist(),
            "phi_grid": phi_grid.tolist(),
            "mean_ratio": float(np.mean(i_final)),
        }
