"""Turbulent Flamefront Propagation Model.

Implements flame propagation using either:
- 0D burned mass fraction (Wiebe-like)
- G-equation level set (for spatial resolution)

Reference: Peters, "Turbulent Combustion"
"""

from typing import Any

import numpy as np

from Simulations.common.io_schema import SimulationInput, SimulationOutput
from Simulations.common.simulation import BaseSimulation, SimulationConfig


class FlamefrontConfig(SimulationConfig):
    """Configuration for flamefront model."""

    # Combustion chamber
    bore: float = 0.085  # [m]
    stroke: float = 0.090  # [m]

    # Flame parameters
    laminar_flame_speed: float = 0.4  # S_L [m/s]
    markstein_length: float = 0.5e-3  # [m]

    # Turbulence
    integral_length_scale: float = 0.005  # [m]
    turbulent_intensity: float = 3.0  # u'/S_L

    # TJI specific
    n_ignition_points: int = 6  # Number of jet ignition sites
    jet_penetration: float = 0.02  # Jet penetration depth [m]

    # Simulation
    dt: float = 1e-5  # Time step [s]
    t_end: float = 0.003  # 3ms burn duration


class FlamefrontModel(BaseSimulation):
    """
    0D/Quasi-1D Turbulent Flame Propagation Model.

    Models:
    - Multiple ignition sites (from TJI jets)
    - Flame kernel growth
    - Flame-flame interaction
    - Wall quenching

    Output:
    - burned_fraction(t)
    - flame_surface_area(t)
    - heat_release_rate(t)

    Source: Peters + TJI literature (Attard, Toulson)
    """

    def __init__(self, name: str, config: FlamefrontConfig):
        super().__init__(name, config)
        self.config: FlamefrontConfig = config
        self.input_data: SimulationInput | None = None

        # State: multiple flame kernels
        self.kernels = []  # List of {center, radius, active}

        # Bulk state
        self.state = {
            "burned_fraction": 0.0,
            "flame_area": 0.0,
            "heat_release_rate": 0.0,
        }

        # History
        self.history = {
            "time": [],
            "burned_fraction": [],
            "flame_area": [],
            "hrr": [],
        }

        # Physical
        self.gamma = 1.35
        self.lhv = 44e6  # [J/kg]

    def load_input(self, input_data: SimulationInput):
        """Load simulation input bundle."""
        self.input_data = input_data

    def setup(self):
        """
        Initialize flame kernels at jet ignition sites.
        """
        n_jets = self.config.n_ignition_points
        r_chamber = self.config.bore / 2

        # Place kernels in a ring near the periphery (typical TJI layout)
        self.kernels = []
        for i in range(n_jets):
            theta = 2 * np.pi * i / n_jets
            x = (r_chamber - self.config.jet_penetration) * np.cos(theta)
            y = (r_chamber - self.config.jet_penetration) * np.sin(theta)
            self.kernels.append(
                {
                    "center": np.array([x, y]),
                    "radius": 0.001,  # 1mm initial kernel
                    "active": True,
                }
            )

        self.state = {
            "burned_fraction": 0.0,
            "flame_area": 0.0,
            "heat_release_rate": 0.0,
        }

        for key in self.history:
            self.history[key] = []

        self.t = 0.0
        print(f"[{self.name}] Setup Complete: {n_jets} ignition kernels initialized")

    def step(self, dt: float):
        """
        Advance flame propagation.

        1. Compute turbulent flame speed
        2. Grow each kernel
        3. Check for kernel merging
        4. Check for wall quenching
        5. Compute burned volume
        """
        # Turbulent flame speed (Zimont correlation)
        s_l = self.config.laminar_flame_speed
        u_prime = s_l * self.config.turbulent_intensity
        da = u_prime * self.config.integral_length_scale / (s_l * self.config.markstein_length)

        # Zimont: S_T = A * u' * Da^0.25 * (integral_scale / flame_thickness)^0.25
        # Simplified: S_T = S_L * (1 + C * (u'/S_L)^n)
        s_t = s_l * (1 + 1.5 * (u_prime / s_l) ** 0.75 * da**0.25)

        # Grow kernels
        r_chamber = self.config.bore / 2
        h_chamber = self.config.stroke  # Approximate cylinder height
        v_total = np.pi * r_chamber**2 * h_chamber

        total_area = 0.0
        v_burned = 0.0

        for kernel in self.kernels:
            if not kernel["active"]:
                continue

            # Grow radius
            r_old = kernel["radius"]
            r_new = r_old + s_t * dt
            kernel["radius"] = r_new

            # Check wall interaction (assume 2D projection for simplicity)
            dist_to_wall = r_chamber - np.linalg.norm(kernel["center"])
            if r_new > dist_to_wall:
                # Flame reached wall - approximate as hemisphere
                r_effective = min(r_new, dist_to_wall + 0.01)
                kernel["radius"] = r_effective

            # Flame area (spherical kernel, 3D)
            area = 4 * np.pi * kernel["radius"] ** 2

            # Burned volume (sphere intersected with cylinder - approximate)
            v_kernel = (4 / 3) * np.pi * kernel["radius"] ** 3
            v_kernel = min(v_kernel, v_total * 0.3)  # Cap at 30% per kernel

            total_area += area
            v_burned += v_kernel

        # Check for kernel merging (simplified)
        for i, k1 in enumerate(self.kernels):
            for k2 in self.kernels[i + 1 :]:
                if not k1["active"] or not k2["active"]:
                    continue
                dist = np.linalg.norm(k1["center"] - k2["center"])
                if dist < k1["radius"] + k2["radius"]:
                    # Merge: deactivate smaller, grow larger
                    if k1["radius"] > k2["radius"]:
                        k1["radius"] = (k1["radius"] ** 3 + k2["radius"] ** 3) ** (1 / 3)
                        k2["active"] = False
                    else:
                        k2["radius"] = (k1["radius"] ** 3 + k2["radius"] ** 3) ** (1 / 3)
                        k1["active"] = False

        # Compute burned fraction
        x_b = min(v_burned / v_total, 1.0)

        # Heat release rate (approximate)
        # HRR ~ rho_u * A_flame * S_T * LHV
        if self.input_data:
            p_cyl = np.mean(self.input_data.boundary_conditions.pressure_gas)
            t_cyl = np.mean(self.input_data.boundary_conditions.temperature_gas)
        else:
            p_cyl = 30e5  # 30 bar
            t_cyl = 800.0  # 800 K

        rho_u = p_cyl / (287.0 * t_cyl)
        hrr = rho_u * total_area * s_t * self.lhv * 0.4  # 40% efficiency for unburned mix

        # Update state
        self.state["burned_fraction"] = x_b
        self.state["flame_area"] = total_area
        self.state["heat_release_rate"] = hrr

        # Record
        self.history["time"].append(self.t)
        self.history["burned_fraction"].append(x_b)
        self.history["flame_area"].append(total_area)
        self.history["hrr"].append(hrr)

        self.t += dt

    def solve_steady_state(self) -> SimulationOutput:
        """Run full combustion and return results."""
        self.setup()

        while self.t < self.config.t_end:
            self.step(self.config.dt)
            if self.state["burned_fraction"] > 0.99:
                break

        # Compute metrics
        burn_10 = next(
            (t for t, xb in zip(self.history["time"], self.history["burned_fraction"]) if xb > 0.1),
            0,
        )
        burn_90 = next(
            (t for t, xb in zip(self.history["time"], self.history["burned_fraction"]) if xb > 0.9),
            self.t,
        )
        burn_duration = burn_90 - burn_10

        peak_hrr = max(self.history["hrr"]) if self.history["hrr"] else 0

        self.results = {
            "final_burned_fraction": self.state["burned_fraction"],
            "burn_duration_10_90": burn_duration,
            "peak_hrr": peak_hrr,
            "history": self.history,
        }

        return SimulationOutput(
            run_id=self.input_data.run_id if self.input_data else "flamefront_test",
            success=True,
            calibration_params={
                "burn_duration_ms": burn_duration * 1000,
                "peak_hrr_mw": peak_hrr / 1e6,
            },
        )

    def post_process(self) -> dict[str, Any]:
        """Return flamefront results."""
        return self.results
