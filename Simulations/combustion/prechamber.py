"""Prechamber Ignition and Turbulent Jet Model.

Implements 0D time-stepping for prechamber combustion with:
- Mass balance (burned/unburned fractions)
- Pressure rise from heat release
- Nozzle mass flow (choked/unchoked)
- Jet momentum for main chamber turbulence enhancement

Reference: Heywood, "Internal Combustion Engine Fundamentals"
"""

from typing import Any

import numpy as np

from Simulations.common.io_schema import SimulationInput, SimulationOutput
from Simulations.common.simulation import BaseSimulation, SimulationConfig


class PrechamberConfig(SimulationConfig):
    """Configuration for prechamber combustion model."""

    # Geometry
    volume_pc: float = 1e-6  # Prechamber volume [m³]
    nozzle_diameter: float = 1.5e-3  # Nozzle diameter [m]
    n_nozzles: int = 6  # Number of nozzle holes
    cd_nozzle: float = 0.8  # Discharge coefficient

    # Combustion parameters
    phi_pc: float = 1.1  # Prechamber equivalence ratio (rich)
    laminar_flame_speed: float = 0.5  # S_L at stoich [m/s]
    turbulent_intensity: float = 2.0  # u'/S_L ratio

    # Time stepping
    dt: float = 1e-6  # Time step [s]
    t_end: float = 0.002  # Simulation duration [s] (2ms burn)


class PrechamberCombustionModel(BaseSimulation):
    """
    0D Two-Zone Prechamber Combustion Model.

    Zones:
    - Unburned: Fresh mixture (fuel + air)
    - Burned: Combustion products

    Key Outputs:
    - burned_fraction(t): Mass fraction burned
    - P(t): Prechamber pressure trace
    - jet_momentum: Momentum flux through nozzles

    Source: Heywood + Turbulent Jet Ignition literature
    """

    def __init__(self, name: str, config: PrechamberConfig):
        super().__init__(name, config)
        self.config: PrechamberConfig = config
        self.input_data: SimulationInput | None = None

        # State variables
        self.state = {
            "p_pc": 1e5,  # Prechamber pressure [Pa]
            "t_pc": 300.0,  # Prechamber temperature [K]
            "m_unburned": 0.0,  # Unburned mass [kg]
            "m_burned": 0.0,  # Burned mass [kg]
            "burned_fraction": 0.0,
        }

        # History for output
        self.history = {
            "time": [],
            "pressure": [],
            "burned_fraction": [],
            "nozzle_mdot": [],
            "jet_momentum": [],
        }

        # Physical constants
        self.gamma = 1.35  # Ratio of specific heats
        self.r_gas = 287.0  # Gas constant [J/kg-K]
        self.lhv = 44e6  # Lower heating value [J/kg]
        self.cp = 1100.0  # Specific heat [J/kg-K]

    def load_input(self, input_data: SimulationInput):
        """Load simulation input bundle."""
        self.input_data = input_data

    def setup(self):
        """
        Initialize prechamber state.

        1. Define Prechamber Geometry (Volume, Nozzle Dia).
        2. Initialize Species (Fuel, Oxidizer, Inert).
        3. Set Initial State (P, T, Phi_stratified).
        """
        if not self.input_data:
            # Use config defaults if no input
            p_init = 50e5  # 50 bar (near TDC compression)
            t_init = 800.0  # 800 K (compressed)
        else:
            # Use operating point data
            p_init = (
                max(self.input_data.boundary_conditions.pressure_gas)
                if self.input_data.boundary_conditions.pressure_gas
                else 50e5
            )
            t_init = (
                max(self.input_data.boundary_conditions.temperature_gas)
                if self.input_data.boundary_conditions.temperature_gas
                else 800.0
            )

        # Initial mass from ideal gas
        v_pc = self.config.volume_pc
        m_total = p_init * v_pc / (self.r_gas * t_init)

        # Fuel mass based on phi
        # m_fuel = m_air * phi * stoich_ratio
        afr_stoich = 14.7  # Air-fuel ratio
        m_fuel = m_total / (1 + afr_stoich / self.config.phi_pc)

        self.state = {
            "p_pc": p_init,
            "t_pc": t_init,
            "m_unburned": m_total,
            "m_burned": 0.0,
            "burned_fraction": 0.0,
            "m_fuel": m_fuel,
        }

        # Clear history
        for key in self.history:
            self.history[key] = []

        self.t = 0.0
        print(
            f"[{self.name}] Setup Complete: P0={p_init / 1e5:.1f} bar, T0={t_init:.0f} K, m={m_total * 1e6:.2f} mg"
        )

    def step(self, dt: float):
        """
        Advance combustion by one time step.

        Physics:
        1. Calculate Laminar -> Turbulent Flame Speed.
        2. Evolve Burned Mass Fraction (Wiebe or flame area).
        3. Calculate Pressure Rise (dP/dt).
        4. Calculate Mass Flow through Nozzle (Choked/Unchoked).

        Args:
            dt: Time step [s]
        """
        # Current state
        p = self.state["p_pc"]
        t_gas = self.state["t_pc"]
        m_u = self.state["m_unburned"]
        m_b = self.state["m_burned"]
        m_total = m_u + m_b

        if m_total <= 0:
            return

        x_b = m_b / m_total  # Burned fraction

        # 1. Flame speed with turbulence enhancement
        # S_T = S_L * (1 + C * (u'/S_L)^n)
        s_l = self.config.laminar_flame_speed
        turb_factor = 1 + 2.0 * self.config.turbulent_intensity**0.5
        s_t = s_l * turb_factor

        # 2. Burning rate (spherical flame model)
        # dm_burned/dt = rho_u * A_flame * S_T
        rho_u = p / (self.r_gas * t_gas)  # Unburned density

        # Flame radius approximation (spherical flame in volume V)
        v_burned = x_b * self.config.volume_pc
        r_flame = (3 * v_burned / (4 * np.pi)) ** (1 / 3) if v_burned > 0 else 0.001
        a_flame = 4 * np.pi * r_flame**2

        # Limit flame area to prechamber surface area
        r_pc = (3 * self.config.volume_pc / (4 * np.pi)) ** (1 / 3)
        a_max = 4 * np.pi * r_pc**2
        a_flame = min(a_flame, a_max)

        # Burning rate
        dm_burn = rho_u * a_flame * s_t * dt
        dm_burn = min(dm_burn, m_u)  # Can't burn more than available

        # 3. Pressure rise from heat release
        # dP/dt = (gamma - 1) / V * Q_dot
        q_dot = dm_burn * self.lhv * 0.3  # Combustion efficiency ~30% in PC
        dp = (self.gamma - 1) / self.config.volume_pc * q_dot * dt

        # 4. Nozzle mass flow
        # Check for choked flow
        a_nozzle = self.config.n_nozzles * np.pi * (self.config.nozzle_diameter / 2) ** 2
        p_main = 0.8 * p  # Assume main chamber is 80% of PC pressure

        p_ratio = p_main / p
        p_crit = (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1))

        if p_ratio < p_crit:
            # Choked flow
            m_dot = (
                self.config.cd_nozzle
                * a_nozzle
                * p
                * np.sqrt(
                    self.gamma
                    / (self.r_gas * t_gas)
                    * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (self.gamma - 1))
                )
            )
        else:
            # Subsonic flow
            m_dot = (
                self.config.cd_nozzle
                * a_nozzle
                * p
                / np.sqrt(self.r_gas * t_gas)
                * np.sqrt(
                    2
                    * self.gamma
                    / (self.gamma - 1)
                    * (p_ratio ** (2 / self.gamma) - p_ratio ** ((self.gamma + 1) / self.gamma))
                )
            )

        m_dot = max(0, m_dot)
        dm_out = m_dot * dt

        # Jet momentum (for turbulence in main chamber)
        v_jet = np.sqrt(2 * self.cp * t_gas * (1 - p_ratio ** ((self.gamma - 1) / self.gamma)))
        jet_momentum = m_dot * v_jet

        # 5. Temperature rise from adiabatic combustion
        # dT = Q / (m * Cp)
        dt_gas = q_dot * dt / (m_total * self.cp) if m_total > 0 else 0

        # 6. Update state
        self.state["m_unburned"] = max(0, m_u - dm_burn - dm_out * (m_u / m_total))
        self.state["m_burned"] = max(0, m_b + dm_burn - dm_out * (m_b / m_total))
        self.state["p_pc"] = max(
            1e5, p + dp - (self.gamma - 1) * dm_out * self.cp * t_gas / self.config.volume_pc
        )
        self.state["t_pc"] = min(3000, t_gas + dt_gas)  # Cap at 3000K

        m_new = self.state["m_unburned"] + self.state["m_burned"]
        self.state["burned_fraction"] = self.state["m_burned"] / m_new if m_new > 0 else 1.0

        # Record history
        self.history["time"].append(self.t)
        self.history["pressure"].append(self.state["p_pc"])
        self.history["burned_fraction"].append(self.state["burned_fraction"])
        self.history["nozzle_mdot"].append(m_dot)
        self.history["jet_momentum"].append(jet_momentum)

        self.t += dt

    def solve_steady_state(self) -> SimulationOutput:
        """
        Run full combustion event and return summary results.
        """
        self.setup()

        # Run time-stepping
        while self.t < self.config.t_end:
            self.step(self.config.dt)

            # Early exit if fully burned
            if self.state["burned_fraction"] > 0.99:
                break

        # Compute summary metrics
        p_max = max(self.history["pressure"]) if self.history["pressure"] else self.state["p_pc"]
        peak_mdot = max(self.history["nozzle_mdot"]) if self.history["nozzle_mdot"] else 0
        peak_momentum = max(self.history["jet_momentum"]) if self.history["jet_momentum"] else 0

        # Burn duration (10-90%)
        burn_10 = next(
            (t for t, xb in zip(self.history["time"], self.history["burned_fraction"]) if xb > 0.1),
            0,
        )
        burn_90 = next(
            (t for t, xb in zip(self.history["time"], self.history["burned_fraction"]) if xb > 0.9),
            self.t,
        )
        burn_duration = burn_90 - burn_10

        self.results = {
            "p_max_pc": p_max,
            "peak_mdot": peak_mdot,
            "peak_jet_momentum": peak_momentum,
            "burn_duration_10_90": burn_duration,
            "final_burned_fraction": self.state["burned_fraction"],
            "history": self.history,
        }

        return SimulationOutput(
            run_id=self.input_data.run_id if self.input_data else "prechamber_test",
            success=True,
            calibration_params={
                "p_max_pc_bar": p_max / 1e5,
                "burn_duration_ms": burn_duration * 1000,
                "peak_momentum_N": peak_momentum,
            },
        )

    def post_process(self) -> dict[str, Any]:
        """Return combustion results."""
        return self.results
