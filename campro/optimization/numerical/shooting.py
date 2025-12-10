import logging

import numpy as np
from campro.optimization.physics_trajectory import system_dynamics
from scipy.integrate import solve_ivp
from scipy.optimize import root

from campro.optimization.core.polar_geometry import PolarCamGeometry

log = logging.getLogger(__name__)


class ShootingSolver:
    """
    Shooting method solver for Planet-Ring engine cycle.

    Uses `PolarCamGeometry` to define kinematic constraints (piston motion)
    and solves for the thermodynamic limit cycle (periodicity of gas states).
    """

    def __init__(self, params: dict):
        self.params = params
        self.geometry = None
        self.omega = 0.0

        # Extract shooting-specific settings
        self.cycle_time = float(params.get("combustion", {}).get("cycle_time_s", 0.05))
        # Approximate omega from cycle time (2pi / T)
        self.omega = 2 * np.pi / self.cycle_time

        # Construct simulation parameters expected by system_dynamics
        # system_dynamics expects: geometry, flow, thermo, bounds, cycle_time
        self.sim_params = {
            "geometry": params.get("geometry", {}),
            "flow": params.get("flow", {}),
            "thermo": params.get(
                "thermo", {"R": 287.0, "gamma": 1.4, "cv": 718.0, "cp": 1005.0}
            ),
            "bounds": params.get("bounds", {}),
            "cycle_time": self.cycle_time,
        }

    def setup_geometry(
        self,
        stroke: float,
        outer_diameter: float,
        theta_bdc: float,
        ratios: tuple[float, float],
        inflections: tuple[float, float] = (0.5, 0.5),
        gen_mode: str = "spline",
        r_drive: float | None = None,
    ):
        """Initialize the PolarCamGeometry."""
        self.geometry = PolarCamGeometry(
            stroke=stroke,
            outer_diameter=outer_diameter,
            theta_bdc=theta_bdc,
            ratios=ratios,
            inflections=inflections,
            gen_mode=gen_mode,
            r_drive=r_drive,
        )
        # Update params geometry dict to match (important for physics func)
        # The physics_trajectory.system_dynamics expects 'geometry' dict
        # We need to ensure it's consistent, though 'system_dynamics' calculates
        # forces based on xL/xR state.
        # In the constraint-based shooting (kinematic driver),
        # we might OVERRIDE the motion integration with the cam profile.
        pass

    def _dynamics_wrapper(self, t, y):
        """
        Dynamics wrapper that enforces Cam kinematics.

        The system_dynamics function in physics_trajectory normally calculates
        accelerations (dv/dt) from forces.
        Here, we want to PRESCRIBE motion x(t) = Cam(theta).
        So:
        1. Calculate theta = omega * t
        2. Get x, v, a from Cam.
        3. Override the derivative of position/velocity in the state vector.
           - dx/dt = v_cam
           - dv/dt = a_cam
        4. Use the underlying thermo dynamics for rho, T, but with the
           PRESCRIBED volume change.

        # State y is [xL, vL, rho, T]
        """
        xL, vL, rho, T = y

        # 1. Enforce Kinematics
        x_cam, v_cam, a_cam = self.geometry.get_kinematics(t, self.omega)

        # Map cam 'x' to piston coordinates
        # xL = -x_cam (Symmetric)
        xL = -x_cam
        vL = -v_cam

        # Override state in y for the thermo calc
        # y_forced = [xL, vL, rho, T]
        y_forced = np.array([xL, vL, rho, T])

        # 2. Call standard Physics
        # Returns [vL, dvL_dt, drho_dt, dT_dt]
        dydt = system_dynamics(t, y_forced, self.sim_params)

        # 3. Override Mechanical Derivatives
        # dxL/dt = vL_cam
        dydt[0] = vL
        # dvL/dt = aL_cam
        dydt[1] = -a_cam

        return dydt

    def solve_limit_cycle(self, p0_guess: float = 1e5, T0_guess: float = 300.0):
        """
        Find the initial state [rho0, T0] that results in a periodic cycle.

        Since kinematics are fixed by the Cam, x(T) == x(0) is guaranteed.
        We only need to ensure rho(T) == rho(0) and T(T) == T(0).
        (Or p(T) == p(0)).
        """
        R = self.params["thermo"]["R"]

        def residual(z):
            """z = [rho0, T0]. Return [rho_end - rho0, T_end - T0]."""
            rho0, T0 = z

            # Initial full state
            # Kinematics at t=0
            x_start, v_start, _ = self.geometry.get_kinematics(0, self.omega)
            xL0 = -x_start
            vL0 = -v_start

            y0 = np.array([xL0, vL0, rho0, T0])

            # Integrate
            sol = solve_ivp(
                self._dynamics_wrapper,
                (0, self.cycle_time),
                y0,
                method="LSODA",  # Auto-switching stiff/non-stiff
                rtol=1e-2,
                atol=1e-4,
            )

            if not sol.success:
                return np.array([1e6, 1e6])  # Penalty

            y_end = sol.y[:, -1]
            rho_end = y_end[2]
            T_end = y_end[3]

            return np.array([rho_end - rho0, T_end - T0])

        # Initial guess from args
        rho0_guess = p0_guess / (R * T0_guess)
        z0 = np.array([rho0_guess, T0_guess])

        log.info("Starting Shooting Method root finding...")
        res = root(residual, z0, method="hybr", tol=1e-3)

        if not res.success:
            log.warning(f"Shooting method did not converge: {res.message}")

        # Final Run to get trajectory
        z_opt = res.x
        rho0, T0 = z_opt

        x_start, v_start, _ = self.geometry.get_kinematics(0, self.omega)
        y0 = np.array([-x_start, -v_start, rho0, T0])

        sol = solve_ivp(
            self._dynamics_wrapper,
            (0, self.cycle_time),
            y0,
            method="Radau",
            dense_output=True,
        )

        return {
            "success": res.success,
            "z_initial": z_opt,
            "trajectory": sol,
            "geometry": self.geometry,
        }
