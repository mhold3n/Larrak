from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from campro.freepiston.core.geom import chamber_volume, piston_area
from campro.freepiston.core.states import MechState
from campro.freepiston.core.thermo import RealGasEOS
from campro.freepiston.core.valves import effective_area_linear
from campro.freepiston.core.xfer import heat_loss_rate, woschni_h
from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class GasComposition:
    """Gas composition tracking for 0D control volume."""

    # Mass fractions
    fresh_air: float = 1.0  # Fresh air mass fraction
    exhaust_gas: float = 0.0  # Exhaust gas mass fraction
    fuel: float = 0.0  # Fuel mass fraction
    burned_gas: float = 0.0  # Burned gas mass fraction

    # Species concentrations (molar fractions)
    O2: float = 0.21  # Oxygen
    N2: float = 0.79  # Nitrogen
    CO2: float = 0.0  # Carbon dioxide
    H2O: float = 0.0  # Water vapor
    CO: float = 0.0  # Carbon monoxide
    H2: float = 0.0  # Hydrogen

    def normalize(self) -> None:
        """Normalize mass fractions to sum to 1.0."""
        total = self.fresh_air + self.exhaust_gas + self.fuel + self.burned_gas
        if total > 0:
            self.fresh_air /= total
            self.exhaust_gas /= total
            self.fuel /= total
            self.burned_gas /= total

    def get_mixture_properties(self, T: float, p: float) -> Dict[str, float]:
        """Get mixture properties based on composition."""
        # Simplified mixture properties calculation
        # In practice, this would use proper mixture rules

        # Air properties (fresh air + N2)
        air_fraction = self.fresh_air + self.N2

        # Exhaust properties (burned gas + CO2 + H2O)
        exhaust_fraction = self.burned_gas + self.CO2 + self.H2O

        # Mixture molecular weight (simplified)
        MW_air = 28.97  # kg/kmol
        MW_exhaust = 30.0  # kg/kmol (approximate)
        MW_mix = air_fraction * MW_air + exhaust_fraction * MW_exhaust

        # Mixture gas constant
        R_mix = 8314.47 / MW_mix  # J/(kg·K)

        # Mixture heat capacity (simplified)
        cp_air = 1005.0  # J/(kg·K)
        cp_exhaust = 1100.0  # J/(kg·K)
        cp_mix = air_fraction * cp_air + exhaust_fraction * cp_exhaust

        # Mixture heat capacity ratio
        gamma_mix = cp_mix / (cp_mix - R_mix)

        return {
            "MW": MW_mix,
            "R": R_mix,
            "cp": cp_mix,
            "gamma": gamma_mix,
        }


@dataclass
class ControlVolumeState:
    """Enhanced 0D control volume state with composition tracking."""

    # Basic thermodynamic state
    rho: float  # Density [kg/m^3]
    T: float  # Temperature [K]
    p: float  # Pressure [Pa]
    u: float  # Internal energy [J/kg]
    h: float  # Enthalpy [J/kg]

    # Composition
    composition: GasComposition

    # Mass and energy
    m: float  # Total mass [kg]
    U: float  # Total internal energy [J]
    H: float  # Total enthalpy [J]

    # Volume and flow
    V: float  # Volume [m^3]
    dV_dt: float  # Volume rate [m^3/s]

    def update_from_thermodynamics(self, eos: RealGasEOS) -> None:
        """Update state from thermodynamics using EOS."""
        # Update pressure from EOS
        self.p = eos.peng_robinson_pressure(self.T, 1.0 / self.rho)

        # Update internal energy and enthalpy
        self.u = eos.h_mix(self.T) - self.p / self.rho
        self.h = eos.h_mix(self.T)

        # Update total quantities
        self.m = self.rho * self.V
        self.U = self.m * self.u
        self.H = self.m * self.h


def volume_from_pistons(*, B: float, Vc: float, x_L: float, x_R: float) -> float:
    return chamber_volume(B=B, Vc=Vc, x_L=x_L, x_R=x_R)


def cv_residual(
    mech: MechState, gas: Dict[str, float], params: Dict[str, object],
) -> Dict[str, float]:
    """Enhanced 0D control-volume residuals for mass and total energy.

    This enhanced version includes:
    - Composition tracking
    - Advanced orifice flow with compressible corrections
    - Blow-by mass flow
    - Proper thermodynamics integration
    """
    geom = params.get("geom", {})
    B = float(geom.get("B", 0.1))
    Vc = float(geom.get("Vc", 1e-5))

    V = volume_from_pistons(B=B, Vc=Vc, x_L=mech.x_L, x_R=mech.x_R)
    dV_dt = piston_area(B) * (mech.v_R - mech.v_L)

    # Extract gas state
    rho = float(gas.get("rho", 1.0))
    T = float(gas.get("T", 1000.0))
    p = float(gas.get("p", 1.0e5))

    # Create composition (default to fresh air)
    composition = GasComposition()
    if "composition" in gas:
        comp_data = gas["composition"]
        composition.fresh_air = comp_data.get("fresh_air", 1.0)
        composition.exhaust_gas = comp_data.get("exhaust_gas", 0.0)
        composition.fuel = comp_data.get("fuel", 0.0)
        composition.burned_gas = comp_data.get("burned_gas", 0.0)
        composition.normalize()

    # Get mixture properties
    mix_props = composition.get_mixture_properties(T, p)
    gamma = mix_props["gamma"]
    R = mix_props["R"]
    cp = mix_props["cp"]

    # Calculate internal energy and enthalpy
    u = cp * T - p / rho  # Internal energy [J/kg]
    h = cp * T  # Enthalpy [J/kg]

    m = rho * V
    U = m * u

    # Advanced orifice flow calculations
    flows = params.get("flows", {})
    Ain_max = float(params.get("valves", {}).get("Ain_max", 0.0))
    Aex_max = float(params.get("valves", {}).get("Aex_max", 0.0))

    # Common defaults for enthalpy references
    T_in = float(flows.get("T_in", T))
    T_ex = float(flows.get("T_ex", T))

    # If explicit mdot provided, honor it and skip orifice model
    if "mdot_in" in flows or "mdot_ex" in flows:
        mdot_in = float(flows.get("mdot_in", 0.0))
        mdot_ex = float(flows.get("mdot_ex", 0.0))
        # Unless blow-by explicitly configured, set to zero for mass conservation tests
        if "blow_by" in params:
            blow_by_params = params.get("blow_by", {})
            gap = float(blow_by_params.get("gap", 1e-6))
            p_crank = float(blow_by_params.get("p_crank", p * 0.8))
            L_gap = float(blow_by_params.get("L_gap", 0.1))
            mu = float(flows.get("mu", 1.8e-5))
            mdot_blow_by = blow_by_mdot(
                gap=gap,
                p_cyl=p,
                p_crank=p_crank,
                T_cyl=T,
                rho_cyl=rho,
                mu=mu,
                L_gap=L_gap,
            )
        else:
            mdot_blow_by = 0.0
    else:
        lift_in = float(flows.get("lift_in", 0.0))
        lift_ex = float(flows.get("lift_ex", 0.0))
        Ain = effective_area_linear(lift=lift_in, A_max=Ain_max)
        Aex = effective_area_linear(lift=lift_ex, A_max=Aex_max)

        # Upstream and downstream conditions
        p_in = float(flows.get("p_in", p))
        p_ex = float(flows.get("p_ex", p))
        rho_in = float(flows.get("rho_in", rho))
        rho_ex = float(flows.get("rho_ex", rho))

        # Calculate Reynolds numbers for discharge coefficients
        mu = float(flows.get("mu", 1.8e-5))  # Dynamic viscosity [Pa·s]
        D_orifice = float(flows.get("D_orifice", 0.01))  # Orifice diameter [m]

        # Inlet flow
        if Ain > 0 and p_in > p:
            Re_in = rho_in * math.sqrt(2.0 * (p_in - p) / rho_in) * D_orifice / mu
            Cd_in = discharge_coefficient(Re=Re_in)
            mdot_in = advanced_orifice_mdot(
                A=Ain,
                Cd=Cd_in,
                rho_up=rho_in,
                p_up=p_in,
                p_down=p,
                T_up=T_in,
                gamma=gamma,
                R=R,
            )
        else:
            mdot_in = 0.0

        # Exhaust flow
        if Aex > 0 and p > p_ex:
            Re_ex = rho * math.sqrt(2.0 * (p - p_ex) / rho) * D_orifice / mu
            Cd_ex = discharge_coefficient(Re=Re_ex)
            mdot_ex = advanced_orifice_mdot(
                A=Aex,
                Cd=Cd_ex,
                rho_up=rho,
                p_up=p,
                p_down=p_ex,
                T_up=T,
                gamma=gamma,
                R=R,
            )
        else:
            mdot_ex = 0.0

        # Blow-by mass flow (disabled unless explicitly configured)
        if "blow_by" in params:
            blow_by_params = params.get("blow_by", {})
            gap = float(blow_by_params.get("gap", 1e-6))  # Ring gap [m]
            p_crank = float(
                blow_by_params.get("p_crank", p * 0.8),
            )  # Crankcase pressure [Pa]
            L_gap = float(blow_by_params.get("L_gap", 0.1))  # Gap length [m]
            mdot_blow_by = blow_by_mdot(
                gap=gap,
                p_cyl=p,
                p_crank=p_crank,
                T_cyl=T,
                rho_cyl=rho,
                mu=mu,
                L_gap=L_gap,
            )
        else:
            mdot_blow_by = 0.0

    # Mass balance
    dm_dt = mdot_in - mdot_ex - mdot_blow_by

    # Energy balance
    # dU/dt = -p dV/dt + h_in mdot_in - h_ex mdot_ex - h_blow_by mdot_blow_by - q_wall
    h_in = cp * T_in  # Inlet enthalpy
    h_ex = cp * T  # Exhaust enthalpy (assume same as cylinder)
    h_blow_by = cp * T  # Blow-by enthalpy (assume same as cylinder)

    # Heat transfer
    xfer = params.get("xfer", {})
    A_p = piston_area(B)
    area = 2.0 * A_p
    Tw = float(xfer.get("Tw", 450.0))
    w_char = float(xfer.get("w_char", 10.0))
    if "h" in xfer:
        h_conv = float(xfer.get("h", 0.0))
    else:
        h_conv = woschni_h(p=p, T=T, B=B, w=w_char)
    q_wall = heat_loss_rate(h=h_conv, area=area, T=T, Tw=Tw)

    # Combustion heat release (if applicable)
    chem = params.get("chem", {})
    Q_comb = float(chem.get("Q_comb", 0.0))  # Heat release rate [W]

    # Energy balance
    dU_dt = (
        (-p * dV_dt)
        + h_in * mdot_in
        - h_ex * mdot_ex
        - h_blow_by * mdot_blow_by
        - q_wall
        + Q_comb
    )

    # Composition evolution (simplified)
    # Fresh air fraction change due to intake and exhaust
    if m > 0:
        # Intake adds fresh air
        dY_fresh_dt = (
            mdot_in * 1.0
            - mdot_ex * composition.fresh_air
            - mdot_blow_by * composition.fresh_air
        ) / m

        # Exhaust gas fraction change
        dY_exhaust_dt = (
            mdot_in * 0.0
            - mdot_ex * composition.exhaust_gas
            - mdot_blow_by * composition.exhaust_gas
        ) / m

        # Fuel fraction change (simplified)
        dY_fuel_dt = (
            mdot_in * 0.0 - mdot_ex * composition.fuel - mdot_blow_by * composition.fuel
        ) / m

        # Burned gas fraction change
        dY_burned_dt = (
            mdot_in * 0.0
            - mdot_ex * composition.burned_gas
            - mdot_blow_by * composition.burned_gas
        ) / m
    else:
        dY_fresh_dt = 0.0
        dY_exhaust_dt = 0.0
        dY_fuel_dt = 0.0
        dY_burned_dt = 0.0

    return {
        "dm_dt": dm_dt,
        "dU_dt": dU_dt,
        "dY_fresh_dt": dY_fresh_dt,
        "dY_exhaust_dt": dY_exhaust_dt,
        "dY_fuel_dt": dY_fuel_dt,
        "dY_burned_dt": dY_burned_dt,
        "mdot_in": mdot_in,
        "mdot_ex": mdot_ex,
        "mdot_blow_by": mdot_blow_by,
        "q_wall": q_wall,
        "Q_comb": Q_comb,
    }


def orifice_mdot(*, A: float, Cd: float, rho: float, dp: float) -> float:
    """Quasi-steady orifice mass flow (subsonic small dp).

    m_dot = Cd * A * sqrt(2 * rho * dp)
    Negative dp returns zero (one-way definition here).
    """
    if A <= 0.0 or Cd <= 0.0 or dp <= 0.0 or rho <= 0.0:
        return 0.0
    from math import sqrt

    return Cd * A * sqrt(2.0 * rho * dp)


def advanced_orifice_mdot(
    *,
    A: float,
    Cd: float,
    rho_up: float,
    p_up: float,
    p_down: float,
    T_up: float,
    gamma: float = 1.4,
    R: float = 287.0,
) -> float:
    """
    Advanced orifice mass flow with compressible flow corrections.

    Handles both subsonic and choked flow conditions.

    Args:
        A: Orifice area [m^2]
        Cd: Discharge coefficient [-]
        rho_up: Upstream density [kg/m^3]
        p_up: Upstream pressure [Pa]
        p_down: Downstream pressure [Pa]
        T_up: Upstream temperature [K]
        gamma: Heat capacity ratio [-]
        R: Gas constant [J/(kg·K)]

    Returns:
        Mass flow rate [kg/s]
    """
    if A <= 0.0 or Cd <= 0.0 or rho_up <= 0.0 or p_up <= 0.0 or T_up <= 0.0:
        return 0.0

    # Pressure ratio
    pr = p_down / p_up

    # Critical pressure ratio for choked flow
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    if pr <= pr_crit:
        # Choked flow
        # Critical velocity
        c_crit = math.sqrt(gamma * R * T_up * (2.0 / (gamma + 1.0)))

        # Critical density
        rho_crit = rho_up * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))

        # Choked mass flow
        mdot = Cd * A * rho_crit * c_crit

    else:
        # Subsonic flow
        # Isentropic flow relations
        term1 = 2.0 * gamma / (gamma - 1.0)
        term2 = pr ** (2.0 / gamma) - pr ** ((gamma + 1.0) / gamma)

        if term2 <= 0.0:
            return 0.0

        # Velocity
        c_up = math.sqrt(gamma * R * T_up)
        u = c_up * math.sqrt(term1 * term2)

        # Density at throat (assuming isentropic)
        rho_throat = rho_up * pr ** (1.0 / gamma)

        # Mass flow
        mdot = Cd * A * rho_throat * u

    return max(0.0, mdot)


def discharge_coefficient(
    *,
    Re: float,
    L_D: float = 0.0,
    beta: float = 0.0,
    roughness: float = 0.0,
) -> float:
    """
    Calculate discharge coefficient based on Reynolds number and geometry.

    Args:
        Re: Reynolds number [-]
        L_D: Length to diameter ratio [-]
        beta: Beta ratio (orifice diameter / pipe diameter) [-]
        roughness: Relative roughness [-]

    Returns:
        Discharge coefficient [-]
    """
    # Base discharge coefficient for sharp-edged orifice
    Cd_base = 0.61

    # Reynolds number correction
    if Re > 10000:
        Cd_re = 1.0
    elif Re > 1000:
        Cd_re = 0.95 + 0.05 * (Re - 1000) / 9000
    else:
        Cd_re = 0.95 * (Re / 1000) ** 0.1

    # Length correction (for long orifices)
    Cd_length = 1.0 - 0.1 * L_D

    # Beta ratio correction
    Cd_beta = 1.0 - 0.2 * beta**2

    # Roughness correction
    Cd_rough = 1.0 - 0.05 * roughness

    # Combined correction
    Cd = Cd_base * Cd_re * Cd_length * Cd_beta * Cd_rough

    return max(0.1, min(1.0, Cd))


def blow_by_mdot(
    *,
    gap: float,
    p_cyl: float,
    p_crank: float,
    T_cyl: float,
    rho_cyl: float,
    mu: float,
    L_gap: float,
) -> float:
    """
    Calculate blow-by mass flow rate through piston ring gap.

    Args:
        gap: Ring gap [m]
        p_cyl: Cylinder pressure [Pa]
        p_crank: Crankcase pressure [Pa]
        T_cyl: Cylinder temperature [K]
        rho_cyl: Cylinder density [kg/m^3]
        mu: Dynamic viscosity [Pa·s]
        L_gap: Gap length [m]

    Returns:
        Blow-by mass flow rate [kg/s]
    """
    if gap <= 0.0 or L_gap <= 0.0:
        return 0.0

    # Pressure difference
    dp = p_cyl - p_crank

    if dp <= 0.0:
        return 0.0

    # Gap area
    A_gap = gap * L_gap

    # Reynolds number based on gap
    Re_gap = rho_cyl * math.sqrt(2.0 * dp / rho_cyl) * gap / mu

    # Discharge coefficient for gap flow
    if Re_gap > 1000:
        Cd_gap = 0.6
    else:
        Cd_gap = 0.6 * (Re_gap / 1000.0) ** 0.1

    # Mass flow rate
    mdot = Cd_gap * A_gap * math.sqrt(2.0 * rho_cyl * dp)

    return mdot


def create_control_volume_state(
    *,
    rho: float,
    T: float,
    V: float,
    dV_dt: float = 0.0,
    composition: Optional[GasComposition] = None,
) -> ControlVolumeState:
    """
    Create a control volume state from basic thermodynamic properties.

    Args:
        rho: Density [kg/m^3]
        T: Temperature [K]
        V: Volume [m^3]
        dV_dt: Volume rate [m^3/s]
        composition: Gas composition (defaults to fresh air)

    Returns:
        ControlVolumeState object
    """
    if composition is None:
        composition = GasComposition()

    # Get mixture properties
    mix_props = composition.get_mixture_properties(T, 1.0e5)  # Default pressure
    R = mix_props["R"]
    cp = mix_props["cp"]

    # Calculate pressure (ideal gas law for now)
    p = rho * R * T

    # Calculate internal energy and enthalpy
    u = cp * T - p / rho
    h = cp * T

    # Calculate total quantities
    m = rho * V
    U = m * u
    H = m * h

    return ControlVolumeState(
        rho=rho,
        T=T,
        p=p,
        u=u,
        h=h,
        composition=composition,
        m=m,
        U=U,
        H=H,
        V=V,
        dV_dt=dV_dt,
    )


def update_control_volume_state(
    state: ControlVolumeState,
    dt: float,
    residuals: Dict[str, float],
    eos: Optional[RealGasEOS] = None,
) -> ControlVolumeState:
    """
    Update control volume state using residuals from cv_residual.

    Args:
        state: Current control volume state
        dt: Time step [s]
        residuals: Residuals from cv_residual function
        eos: Equation of state (optional, uses ideal gas if None)

    Returns:
        Updated ControlVolumeState
    """
    # Update mass
    m_new = state.m + dt * residuals["dm_dt"]

    # Update total internal energy
    U_new = state.U + dt * residuals["dU_dt"]

    # Update composition
    comp_new = GasComposition(
        fresh_air=state.composition.fresh_air + dt * residuals["dY_fresh_dt"],
        exhaust_gas=state.composition.exhaust_gas + dt * residuals["dY_exhaust_dt"],
        fuel=state.composition.fuel + dt * residuals["dY_fuel_dt"],
        burned_gas=state.composition.burned_gas + dt * residuals["dY_burned_dt"],
    )
    comp_new.normalize()

    # Update volume
    V_new = state.V + dt * state.dV_dt

    # Calculate new density
    if V_new > 0:
        rho_new = m_new / V_new
    else:
        rho_new = state.rho

    # Calculate new internal energy per unit mass
    if m_new > 0:
        u_new = U_new / m_new
    else:
        u_new = state.u

    # Get mixture properties
    mix_props = comp_new.get_mixture_properties(state.T, state.p)
    cp = mix_props["cp"]
    R = mix_props["R"]

    # Calculate new temperature (assuming constant cp)
    T_new = (u_new + state.p / rho_new) / cp

    # Calculate new pressure
    if eos is not None:
        # Use real gas EOS
        p_new = eos.peng_robinson_pressure(T_new, 1.0 / rho_new)
    else:
        # Use ideal gas law
        p_new = rho_new * R * T_new

    # Calculate new enthalpy
    h_new = cp * T_new

    # Create new state
    new_state = ControlVolumeState(
        rho=rho_new,
        T=T_new,
        p=p_new,
        u=u_new,
        h=h_new,
        composition=comp_new,
        m=m_new,
        U=U_new,
        H=m_new * h_new,
        V=V_new,
        dV_dt=state.dV_dt,
    )

    return new_state


@dataclass
class ScavengingState:
    """Enhanced scavenging state tracking for two-stroke engines."""

    # Mass tracking
    m_fresh_delivered: float = 0.0  # Total fresh charge delivered [kg]
    m_exhaust_removed: float = 0.0  # Total exhaust gas removed [kg]
    m_fresh_trapped: float = 0.0  # Fresh charge trapped in cylinder [kg]
    m_exhaust_trapped: float = 0.0  # Exhaust gas trapped in cylinder [kg]
    m_total_trapped: float = 0.0  # Total mass trapped in cylinder [kg]
    m_short_circuit: float = 0.0  # Fresh charge lost through exhaust [kg]

    # Phase tracking
    phase_intake_start: float = 0.0  # Intake phase start time [s]
    phase_intake_end: float = 0.0  # Intake phase end time [s]
    phase_exhaust_start: float = 0.0  # Exhaust phase start time [s]
    phase_exhaust_end: float = 0.0  # Exhaust phase end time [s]
    phase_overlap_start: float = 0.0  # Overlap phase start time [s]
    phase_overlap_end: float = 0.0  # Overlap phase end time [s]

    # Efficiency metrics
    eta_scavenging: float = 0.0  # Scavenging efficiency [-]
    eta_trapping: float = 0.0  # Trapping efficiency [-]
    eta_blowdown: float = 0.0  # Blowdown efficiency [-]
    eta_short_circuit: float = 0.0  # Short-circuit loss [-]

    # Quality metrics
    scavenging_uniformity: float = 0.0  # Fresh charge distribution uniformity [-]
    mixing_index: float = 0.0  # Mixing quality index [-]

    def update_mass_flows(
        self, mdot_in: float, mdot_ex: float, dt: float, composition: GasComposition,
    ) -> None:
        """Update mass flow tracking."""
        # Fresh charge delivered
        self.m_fresh_delivered += mdot_in * dt

        # Exhaust gas removed
        self.m_exhaust_removed += mdot_ex * dt * composition.exhaust_gas

        # Short-circuit loss (fresh charge lost through exhaust)
        self.m_short_circuit += mdot_ex * dt * composition.fresh_air

    def update_trapped_masses(self, state: ControlVolumeState) -> None:
        """Update trapped mass tracking."""
        self.m_fresh_trapped = state.m * state.composition.fresh_air
        self.m_exhaust_trapped = state.m * state.composition.exhaust_gas
        self.m_total_trapped = state.m

    def calculate_efficiencies(self) -> None:
        """Calculate all efficiency metrics."""
        # Scavenging efficiency (fresh charge / total trapped)
        if self.m_total_trapped > 0:
            self.eta_scavenging = self.m_fresh_trapped / self.m_total_trapped
        else:
            self.eta_scavenging = 0.0

        # Trapping efficiency (trapped mass / delivered mass)
        if self.m_fresh_delivered > 0:
            self.eta_trapping = self.m_total_trapped / self.m_fresh_delivered
        else:
            self.eta_trapping = 0.0

        # Blowdown efficiency (exhaust removed / initial exhaust)
        if self.m_exhaust_trapped > 0:
            self.eta_blowdown = self.m_exhaust_removed / (
                self.m_exhaust_trapped + self.m_exhaust_removed
            )
        else:
            self.eta_blowdown = 0.0

        # Short-circuit loss
        if self.m_fresh_delivered > 0:
            self.eta_short_circuit = self.m_short_circuit / self.m_fresh_delivered
        else:
            self.eta_short_circuit = 0.0


def detect_scavenging_phases(
    A_in: float,
    A_ex: float,
    A_in_max: float,
    A_ex_max: float,
    threshold: float = 0.01,
) -> Dict[str, bool]:
    """
    Detect current scavenging phase based on valve areas.

    Args:
        A_in: Current intake valve area [m^2]
        A_ex: Current exhaust valve area [m^2]
        A_in_max: Maximum intake valve area [m^2]
        A_ex_max: Maximum exhaust valve area [m^2]
        threshold: Area threshold for phase detection [-]

    Returns:
        Dictionary of phase flags
    """
    # Normalized valve areas
    A_in_norm = A_in / (A_in_max + 1e-9)
    A_ex_norm = A_ex / (A_ex_max + 1e-9)

    # Phase detection
    intake_open = A_in_norm > threshold
    exhaust_open = A_ex_norm > threshold
    overlap = intake_open and exhaust_open

    return {
        "intake_open": intake_open,
        "exhaust_open": exhaust_open,
        "overlap": overlap,
        "intake_only": intake_open and not exhaust_open,
        "exhaust_only": exhaust_open and not intake_open,
        "both_closed": not intake_open and not exhaust_open,
    }


def calculate_scavenging_metrics(
    state: ControlVolumeState,
    mdot_in: float,
    mdot_ex: float,
    dt: float,
) -> Dict[str, float]:
    """
    Calculate scavenging efficiency metrics for two-stroke engines.

    Args:
        state: Current control volume state
        mdot_in: Intake mass flow rate [kg/s]
        mdot_ex: Exhaust mass flow rate [kg/s]
        dt: Time step [s]

    Returns:
        Dictionary of scavenging metrics
    """
    # Fresh charge trapped mass
    m_fresh_trapped = state.m * state.composition.fresh_air

    # Total trapped mass
    m_total_trapped = state.m

    # Scavenging efficiency (fresh charge / total trapped)
    if m_total_trapped > 0:
        eta_scav = m_fresh_trapped / m_total_trapped
    else:
        eta_scav = 0.0

    # Trapping efficiency (trapped mass / delivered mass)
    m_delivered = mdot_in * dt
    if m_delivered > 0:
        eta_trap = m_total_trapped / m_delivered
    else:
        eta_trap = 0.0

    # Short-circuit loss (fresh charge lost through exhaust)
    m_fresh_lost = mdot_ex * dt * state.composition.fresh_air
    if mdot_in * dt > 0:
        eta_short_circuit = m_fresh_lost / (mdot_in * dt)
    else:
        eta_short_circuit = 0.0

    return {
        "scavenging_efficiency": eta_scav,
        "trapping_efficiency": eta_trap,
        "short_circuit_loss": eta_short_circuit,
        "fresh_charge_trapped": m_fresh_trapped,
        "total_trapped": m_total_trapped,
        "fresh_charge_lost": m_fresh_lost,
    }


def enhanced_scavenging_tracking(
    state: ControlVolumeState,
    mdot_in: float,
    mdot_ex: float,
    A_in: float,
    A_ex: float,
    A_in_max: float,
    A_ex_max: float,
    dt: float,
    scavenging_state: ScavengingState,
    time: float,
) -> ScavengingState:
    """
    Enhanced scavenging state tracking with phase detection.

    Args:
        state: Current control volume state
        mdot_in: Intake mass flow rate [kg/s]
        mdot_ex: Exhaust mass flow rate [kg/s]
        A_in: Current intake valve area [m^2]
        A_ex: Current exhaust valve area [m^2]
        A_in_max: Maximum intake valve area [m^2]
        A_ex_max: Maximum exhaust valve area [m^2]
        dt: Time step [s]
        scavenging_state: Current scavenging state
        time: Current time [s]

    Returns:
        Updated scavenging state
    """
    # Detect current phase
    phases = detect_scavenging_phases(A_in, A_ex, A_in_max, A_ex_max)

    # Update phase timing
    if phases["intake_open"] and scavenging_state.phase_intake_start == 0.0:
        scavenging_state.phase_intake_start = time
    if not phases["intake_open"] and scavenging_state.phase_intake_start > 0.0:
        scavenging_state.phase_intake_end = time

    if phases["exhaust_open"] and scavenging_state.phase_exhaust_start == 0.0:
        scavenging_state.phase_exhaust_start = time
    if not phases["exhaust_open"] and scavenging_state.phase_exhaust_start > 0.0:
        scavenging_state.phase_exhaust_end = time

    if phases["overlap"] and scavenging_state.phase_overlap_start == 0.0:
        scavenging_state.phase_overlap_start = time
    if not phases["overlap"] and scavenging_state.phase_overlap_start > 0.0:
        scavenging_state.phase_overlap_end = time

    # Update mass flow tracking
    scavenging_state.update_mass_flows(mdot_in, mdot_ex, dt, state.composition)

    # Update trapped masses
    scavenging_state.update_trapped_masses(state)

    # Calculate efficiencies
    scavenging_state.calculate_efficiencies()

    # Calculate quality metrics (simplified for 0D model)
    scavenging_state.scavenging_uniformity = 1.0 - scavenging_state.eta_short_circuit
    scavenging_state.mixing_index = (
        1.0 - abs(scavenging_state.eta_scavenging - 0.5) * 2.0
    )

    return scavenging_state


def calculate_phase_timing_metrics(
    scavenging_state: ScavengingState,
) -> Dict[str, float]:
    """
    Calculate phase timing metrics for scavenging optimization.

    Args:
        scavenging_state: Current scavenging state

    Returns:
        Dictionary of phase timing metrics
    """
    metrics = {}

    # Phase durations
    if (
        scavenging_state.phase_intake_start > 0.0
        and scavenging_state.phase_intake_end > 0.0
    ):
        metrics["intake_duration"] = (
            scavenging_state.phase_intake_end - scavenging_state.phase_intake_start
        )
    else:
        metrics["intake_duration"] = 0.0

    if (
        scavenging_state.phase_exhaust_start > 0.0
        and scavenging_state.phase_exhaust_end > 0.0
    ):
        metrics["exhaust_duration"] = (
            scavenging_state.phase_exhaust_end - scavenging_state.phase_exhaust_start
        )
    else:
        metrics["exhaust_duration"] = 0.0

    if (
        scavenging_state.phase_overlap_start > 0.0
        and scavenging_state.phase_overlap_end > 0.0
    ):
        metrics["overlap_duration"] = (
            scavenging_state.phase_overlap_end - scavenging_state.phase_overlap_start
        )
    else:
        metrics["overlap_duration"] = 0.0

    # Phase timing ratios
    total_cycle_time = max(metrics["intake_duration"], metrics["exhaust_duration"])
    if total_cycle_time > 0:
        metrics["intake_ratio"] = metrics["intake_duration"] / total_cycle_time
        metrics["exhaust_ratio"] = metrics["exhaust_duration"] / total_cycle_time
        metrics["overlap_ratio"] = metrics["overlap_duration"] / total_cycle_time
    else:
        metrics["intake_ratio"] = 0.0
        metrics["exhaust_ratio"] = 0.0
        metrics["overlap_ratio"] = 0.0

    return metrics
