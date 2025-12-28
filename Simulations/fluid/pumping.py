"""Pumping Fluid Dynamics Model.

Implements 1D compressible flow for intake/exhaust port dynamics.
Uses finite volume method with approximate Riemann solver.

Reference: Versteeg & Malalasekera, Computational Fluid Dynamics
"""

from typing import Any

import numpy as np

from Simulations.common.io_schema import SimulationInput, SimulationOutput
from Simulations.common.simulation import BaseSimulation, SimulationConfig


class PumpingConfig(SimulationConfig):
    """Configuration for 1D pumping losses model."""

    # Port geometry
    port_length: float = 0.15  # Port length [m]
    port_diameter: float = 0.04  # Port diameter [m]
    n_cells: int = 20  # Number of FV cells

    # Valve parameters
    valve_diameter: float = 0.035  # Valve head diameter [m]
    max_lift: float = 0.010  # Maximum valve lift [m]
    valve_cd: float = 0.6  # Discharge coefficient

    # Simulation
    dt: float = 1e-6  # Time step [s]
    t_end: float = 0.02  # 20ms (one cycle at 3000 RPM)


class PumpingCFDModel(BaseSimulation):
    """
    1D Finite Volume Model for Intake/Exhaust Pumping.

    Solves compressible Euler equations:
    - dρ/dt + d(ρu)/dx = 0          (mass)
    - d(ρu)/dt + d(ρu² + P)/dx = 0   (momentum)
    - dE/dt + d((E+P)u)/dx = 0       (energy)

    Outputs:
    - Mass flow integral [kg]
    - Pumping work [J]
    - Discharge coefficient (effective)

    Source: Versteeg & Malalasekera (FVM).
    """

    def __init__(self, name: str, config: PumpingConfig):
        super().__init__(name, config)
        self.config: PumpingConfig = config
        self.input_data: SimulationInput | None = None

        # Grid
        self.dx = config.port_length / config.n_cells
        self.x = np.linspace(self.dx / 2, config.port_length - self.dx / 2, config.n_cells)

        # State vectors (conservative variables)
        # U = [rho, rho*u, E]
        self.rho = np.ones(config.n_cells) * 1.2  # Density [kg/m³]
        self.rho_u = np.zeros(config.n_cells)  # Momentum [kg/m²s]
        self.E = np.ones(config.n_cells) * 2.5e5  # Total energy [J/m³]

        # Physical constants
        self.gamma = 1.4
        self.r_gas = 287.0
        self.cv = self.r_gas / (self.gamma - 1)

        # Results accumulator
        self.mass_flow_integral = 0.0
        self.pumping_work = 0.0

    def load_input(self, input_data: SimulationInput):
        """Load simulation input bundle."""
        self.input_data = input_data

    def setup(self):
        """
        Initialize flow field.

        1. Define Port Geometry.
        2. Discretize into Finite Volumes.
        3. Apply Boundary Conditions (Inlet Pressure, Cylinder Pressure).
        """
        if self.input_data:
            p_inlet = self.input_data.operating_point.p_intake
            t_inlet = self.input_data.operating_point.T_intake
        else:
            p_inlet = 1.0e5  # 1 bar
            t_inlet = 300.0  # 300 K

        # Initialize uniform field
        rho_init = p_inlet / (self.r_gas * t_inlet)
        e_int = self.cv * t_inlet  # Internal energy per unit mass

        self.rho[:] = rho_init
        self.rho_u[:] = 0.0
        self.E[:] = rho_init * e_int  # No kinetic energy initially

        self.mass_flow_integral = 0.0
        self.pumping_work = 0.0
        self.t = 0.0

        print(f"[{self.name}] Setup Complete: P={p_inlet / 1e5:.2f} bar, T={t_inlet:.0f} K")

    def _compute_pressure(
        self, rho: np.ndarray, rho_u: np.ndarray, e_total: np.ndarray
    ) -> np.ndarray:
        """Compute pressure from conservative variables using ideal gas EOS."""
        u = rho_u / (rho + 1e-10)
        e_kinetic = 0.5 * rho * u**2
        e_internal = e_total - e_kinetic
        p = (self.gamma - 1) * e_internal
        return np.maximum(p, 1e3)  # Floor at 10 mbar

    def _compute_fluxes(self, rho: np.ndarray, rho_u: np.ndarray, e_total: np.ndarray) -> tuple:
        """
        Compute convective fluxes using Rusanov (local Lax-Friedrichs) scheme.

        Returns fluxes at cell interfaces (n_cells + 1 values).
        """
        n = len(rho)
        p = self._compute_pressure(rho, rho_u, e_total)
        u = rho_u / (rho + 1e-10)

        # Sound speed
        a = np.sqrt(self.gamma * p / (rho + 1e-10))

        # Fluxes at cell centers
        f_rho = rho_u
        f_rhou = rho_u * u + p
        f_e = (e_total + p) * u

        # Interface fluxes (Rusanov scheme)
        flux_rho = np.zeros(n + 1)
        flux_rhou = np.zeros(n + 1)
        flux_e = np.zeros(n + 1)

        for i in range(1, n):
            # Left and right states
            rho_l, rho_r = rho[i - 1], rho[i]
            rhou_l, rhou_r = rho_u[i - 1], rho_u[i]
            e_l, e_r = e_total[i - 1], e_total[i]

            # Max wave speed
            s_max = max(abs(u[i - 1]) + a[i - 1], abs(u[i]) + a[i])

            # Rusanov flux
            flux_rho[i] = 0.5 * (f_rho[i - 1] + f_rho[i]) - 0.5 * s_max * (rho_r - rho_l)
            flux_rhou[i] = 0.5 * (f_rhou[i - 1] + f_rhou[i]) - 0.5 * s_max * (rhou_r - rhou_l)
            flux_e[i] = 0.5 * (f_e[i - 1] + f_e[i]) - 0.5 * s_max * (e_r - e_l)

        return flux_rho, flux_rhou, flux_e

    def step(self, dt: float):
        """
        Advance flow solution by one time step.

        Uses explicit Euler with Rusanov flux.
        """
        n = self.config.n_cells

        # Compute fluxes
        flux_rho, flux_rhou, flux_e = self._compute_fluxes(self.rho, self.rho_u, self.E)

        # Boundary conditions (extrapolate for now)
        flux_rho[0] = flux_rho[1]
        flux_rho[n] = flux_rho[n - 1]
        flux_rhou[0] = flux_rhou[1]
        flux_rhou[n] = flux_rhou[n - 1]
        flux_e[0] = flux_e[1]
        flux_e[n] = flux_e[n - 1]

        # Update conserved variables
        for i in range(n):
            self.rho[i] -= dt / self.dx * (flux_rho[i + 1] - flux_rho[i])
            self.rho_u[i] -= dt / self.dx * (flux_rhou[i + 1] - flux_rhou[i])
            self.E[i] -= dt / self.dx * (flux_e[i + 1] - flux_e[i])

        # Enforce positivity
        self.rho = np.maximum(self.rho, 0.1)
        self.E = np.maximum(self.E, 1e4)

        # Accumulate results
        u_exit = self.rho_u[-1] / (self.rho[-1] + 1e-10)
        a_port = np.pi * (self.config.port_diameter / 2) ** 2
        m_dot = self.rho[-1] * u_exit * a_port
        self.mass_flow_integral += m_dot * dt

        p_exit = self._compute_pressure(self.rho[-1:], self.rho_u[-1:], self.E[-1:])[0]
        self.pumping_work += m_dot * p_exit / self.rho[-1] * dt  # P*dV work

        self.t += dt

    def solve_steady_state(self) -> SimulationOutput:
        """Run full cycle and compute pumping losses."""
        self.setup()

        # Simple steady-state: run until quasi-steady
        n_steps = int(self.config.t_end / self.config.dt)
        for _ in range(min(n_steps, 10000)):  # Cap at 10k steps
            self.step(self.config.dt)

        # Compute effective Cd
        p_inlet = self.input_data.operating_point.p_intake if self.input_data else 1e5
        t_inlet = self.input_data.operating_point.T_intake if self.input_data else 300.0
        rho_inlet = p_inlet / (self.r_gas * t_inlet)

        # Theoretical mass flow (isentropic)
        a_valve = self.config.valve_cd * np.pi * self.config.valve_diameter * self.config.max_lift
        m_dot_ideal = a_valve * rho_inlet * np.sqrt(2 * p_inlet / rho_inlet)

        m_dot_actual = self.mass_flow_integral / self.t if self.t > 0 else 0
        cd_effective = m_dot_actual / (m_dot_ideal + 1e-10)

        self.results = {
            "mass_flow_integral": self.mass_flow_integral,
            "pumping_work": self.pumping_work,
            "cd_effective": cd_effective,
        }

        return SimulationOutput(
            run_id=self.input_data.run_id if self.input_data else "pumping_test",
            success=True,
            calibration_params={
                "mass_flow_kg": self.mass_flow_integral,
                "pumping_work_j": self.pumping_work,
                "cd_effective": float(cd_effective),
            },
        )

    def post_process(self) -> dict[str, Any]:
        """Return pumping analysis results."""
        return self.results
