import logging

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import minimize

log = logging.getLogger(__name__)


class GeometrySolver:
    """Base class for geometry solvers."""

    def solve(self, inputs: dict) -> dict:
        raise NotImplementedError


class KinematicOptimizer(GeometrySolver):
    """
    Optimizes a B-Spline trajectory x(t) to minimize acceleration/jerk
    subject to stroke and velocity constraints.
    """

    def __init__(self, num_control_points=12, degree=3):
        self.num_control_points = num_control_points
        self.degree = degree

    def solve(self, inputs: dict) -> dict:
        stroke = inputs.get("stroke", 0.1)
        cycle_time = inputs.get("cycle_time", 0.02)
        v_max = inputs.get("v_max", 50.0)

        # Time grid for evaluation
        t_eval = np.linspace(0, cycle_time, 100)
        dt = t_eval[1] - t_eval[0]

        # Initial guess: Cosine profile (Simple Harmonic Motion)
        # x(t) = (S/2) * cos(omega * t)
        omega = 2 * np.pi / cycle_time
        amplitude = stroke / 2.0
        x_guess = amplitude * np.cos(omega * t_eval)

        # Fit initial B-Spline to get coefficients
        spl = make_interp_spline(t_eval, x_guess, k=self.degree)
        # We optimize the internal control points.
        # For a periodic spline, we might need special handling, but here we'll
        # fit a standard spline and enforce periodic constraints.

        # Simplified approach: Optimize control points c directly
        # Knot vector is fixed (uniform for now)
        t_knots = np.linspace(0, cycle_time, self.num_control_points - self.degree + 1)
        # Make it periodic-ish by padding knots?
        # Actually, let's just use scipy.interpolate.BSpline with periodic boundary conditions?
        # Scipy's make_interp_spline supports bc_type='periodic'

        spl_periodic = make_interp_spline(
            t_eval, x_guess, k=self.degree, bc_type="periodic"
        )
        c0 = spl_periodic.c[: -self.degree - 1]  # Independent coeffs for periodic

        def objective(c):
            # Reconstruct spline
            # Pad coefficients for periodicity
            c_full = np.concatenate([c, c[: self.degree + 1]])
            spl = BSpline(spl_periodic.t, c_full, self.degree, extrapolate="periodic")

            # Evaluate derivatives
            # acc = spl(t_eval, nu=2)
            jerk = spl(t_eval, nu=3)

            # Minimize Integral of Jerk^2
            return np.sum(jerk**2) * dt

        def constraint_stroke(c):
            # Max - Min >= Stroke (approximate on grid)
            c_full = np.concatenate([c, c[: self.degree + 1]])
            spl = BSpline(spl_periodic.t, c_full, self.degree, extrapolate="periodic")
            x = spl(t_eval)
            return (np.max(x) - np.min(x)) - stroke

        def constraint_velocity(c):
            # |v| <= v_max
            c_full = np.concatenate([c, c[: self.degree + 1]])
            spl = BSpline(spl_periodic.t, c_full, self.degree, extrapolate="periodic")
            v = spl(t_eval, nu=1)
            # We want v_max - |v| >= 0
            return v_max - np.max(np.abs(v))

        # Constraints
        cons = [
            {
                "type": "eq",
                "fun": constraint_stroke,
            },  # Equality for stroke? Or inequality? usually we want Exact stroke.
            {"type": "ineq", "fun": constraint_velocity},
        ]

        # Run optimization
        res = minimize(
            objective,
            c0,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 100, "ftol": 1e-4},
        )

        if not res.success:
            log.warning(f"KinematicOptimizer failed: {res.message}")

        # Build solution object
        c_opt = res.x
        c_full = np.concatenate([c_opt, c_opt[: self.degree + 1]])
        spl_opt = BSpline(spl_periodic.t, c_full, self.degree, extrapolate="periodic")

        return {
            "success": res.success,
            "t": t_eval,
            "x": spl_opt(t_eval),
            "v": spl_opt(t_eval, nu=1),
            "a": spl_opt(t_eval, nu=2),
            "spline": spl_opt,
            "method": "kinematic_opt_bspline",
        }


class FourierProfileGenerator(GeometrySolver):
    """
    Generates a profile using a truncated Fourier series.
    Naturally periodic and smooth.
    """

    def __init__(self, num_harmonics=3):
        self.num_harmonics = num_harmonics

    def solve(self, inputs: dict) -> dict:
        stroke = inputs.get("stroke", 0.1)
        cycle_time = inputs.get("cycle_time", 0.02)

        t_eval = np.linspace(0, cycle_time, 100)
        omega = 2 * np.pi / cycle_time

        # Simple implementation: optimized flattened harmonic to maximize time at TDC?
        # x(t) = A1*cos(wt) + A3*cos(3wt)...
        # For now, just generate the fundamental + modest 3rd harmonic (squaring)

        # Optimize coefficients [A1, A3, ...] to flatten TDC but keep stroke
        # Objective: Flatness? Or just providing variational candidates?
        # Let's provide a "Square-ish" candidate (high compression dwell)

        A1 = stroke / 2.0 * 1.15  # Boost fundamental
        A3 = -stroke / 2.0 * 0.15  # Subtract 3rd to flatten peak

        x = A1 * np.cos(omega * t_eval) + A3 * np.cos(3 * omega * t_eval)

        # Rescale exactly to stroke
        current_stroke = np.max(x) - np.min(x)
        scale = stroke / current_stroke
        x *= scale

        # Derivatives (analytical)
        v = (
            -(
                A1 * omega * np.sin(omega * t_eval)
                + A3 * 3 * omega * np.sin(3 * omega * t_eval)
            )
            * scale
        )
        a = (
            -(
                A1 * omega**2 * np.cos(omega * t_eval)
                + A3 * (3 * omega) ** 2 * np.cos(3 * omega * t_eval)
            )
            * scale
        )

        return {
            "success": True,
            "t": t_eval,
            "x": x,
            "v": v,
            "a": a,
            "coeffs": {"A1": A1 * scale, "A3": A3 * scale},
            "method": "fourier_harmonic",
        }
