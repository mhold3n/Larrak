"""0D Thermodynamic ODE Kernel for Phase 1 optimization.

This module implements the system of ODEs: dy/dtheta = f(theta, y, u)
for a multi-zone (or single-zone aggregate) engine model.
"""

from __future__ import annotations

from typing import Protocol

import casadi as ca
import numpy as np

from thermo.geometry import GeometryInterface


class CombustionSurrogate(Protocol):
    """Interface for combustion heat release generation."""

    def get_heat_release_rate(
        self,
        theta: ca.SX,
        controls: dict[str, ca.SX],
        state_dict: dict[str, ca.SX],
    ) -> ca.SX:
        """Return dQ_comb/dtheta [J/rad]."""
        ...


class WiebeSurrogate:
    """Standard Wiebe function surrogate."""

    def __init__(self, Q_total: float):
        self.Q_total = Q_total

    def get_heat_release_rate(
        self,
        theta: ca.SX,
        controls: dict[str, ca.SX],
        state_dict: dict[str, ca.SX],
    ) -> ca.SX:
        # Extract controls or use defaults
        # controls expected: 'theta_start', 'theta_duration', 'wiebe_m', 'wiebe_a'
        theta_start = controls.get("theta_start", 0.0)  # rad (0 is TDC combustion)
        theta_dur = controls.get("theta_duration", np.radians(60.0))
        m = controls.get("wiebe_m", 2.0)
        a = controls.get("wiebe_a", 5.0)

        # Normalize theta
        # Handling periodicity relative to theta_start is tricky.
        # Assuming theta is continuous cycle angle here.

        # theta_norm = (theta - theta_start) / theta_dur
        # Using smooth approximation or just the formula
        # dQ = Q * a * (m+1)/dur * x^m * exp(-a * x^(m+1))

        tr = (theta - theta_start) / theta_dur

        # Activation: 0 if tr < 0 or tr > 1
        # Use soft activation for gradient flow?
        # CasADi if_else is better here

        active = ca.if_else(ca.logic_and(tr >= 0, tr <= 1), 1.0, 0.0)

        val = (
            self.Q_total
            * a
            * (m + 1)
            / theta_dur
            * ca.power(tr, m)
            * ca.exp(-a * ca.power(tr, m + 1))
        )
        return val * active


class PrechamberSurrogate:
    """
    Two-stage combustion surrogate for Prechamber (PC) + Main chamber.
    Modeled as two superimposed Wiebe functions.
    """

    def __init__(self, Q_total: float):
        self.Q_total = Q_total

    def get_heat_release_rate(
        self,
        theta: ca.SX,
        controls: dict[str, ca.SX],
        state_dict: dict[str, ca.SX],
    ) -> ca.SX:
        # Controls: 'theta_inj_main', 'theta_inj_pc', 'f_pc', 'phi_main'
        # f_pc: Fraction of fuel energy in prechamber (0.0 to 0.1 typical)

        f_pc = controls.get("f_pc", 0.05)
        theta_main = controls.get("theta_start", 0.0)
        # PC usually fires slightly before or triggers main
        # Let's assume PC timing is coupled or separate
        theta_pc = controls.get("theta_pc", theta_main - np.radians(5.0))

        # PC burn is fast/sharp
        dur_pc = np.radians(10.0)
        # Main burn is standard
        dur_main = controls.get("theta_duration", np.radians(50.0))

        # Wiebe 1 (PC)
        tr_pc = (theta - theta_pc) / dur_pc
        active_pc = ca.if_else(ca.logic_and(tr_pc >= 0, tr_pc <= 1), 1.0, 0.0)
        dQ_pc = (
            (f_pc * self.Q_total)
            * 5.0
            * 3.0
            / dur_pc
            * ca.power(tr_pc, 2)
            * ca.exp(-5.0 * ca.power(tr_pc, 3))
            * active_pc
        )

        # Wiebe 2 (Main)
        tr_m = (theta - theta_main) / dur_main
        active_m = ca.if_else(ca.logic_and(tr_m >= 0, tr_m <= 1), 1.0, 0.0)
        dQ_m = (
            ((1.0 - f_pc) * self.Q_total)
            * 5.0
            * 3.0
            / dur_main
            * ca.power(tr_m, 2)
            * ca.exp(-5.0 * ca.power(tr_m, 3))
            * active_m
        )

        return dQ_pc + dQ_m


class ThermoODE:
    """0D Thermodynamic ODE System."""

    def __init__(
        self,
        geometry: GeometryInterface,
        combustion: CombustionSurrogate,
        num_species: int = 1,  # 1 for just 'fuel fraction' or 'burned fraction' logic
    ):
        self.geo = geometry
        self.comb = combustion
        self.ns = num_species

        # Constants
        self.R_gas = 287.0  # J/kgK (Air)
        self.cv = 718.0  # J/kgK
        self.cp = 1005.0  # J/kgK

    def get_state_names(self) -> list[str]:
        # y = [m_c, T_c, Y..., p_int, T_int, p_exh, T_exh]
        # For Phase 1 simplification, we might treat manifolds as infinite boundaries (Parameters)
        # or dynamic states. Plan said "Manifold states".
        # Let's start with Cylinder states first, Manifolds usually parameters in simple genset opt.
        # But user spec said: "Manifolds: m_int, p_int..."
        # Let's implement full vector.

        names = ["m_cyl", "T_cyl"] + [f"Y_cyl_{i}" for i in range(self.ns)]
        names += ["m_int", "p_int", "T_int"]  # Intake manifold state? Or just p/T?
        # If intake is a fixed supply, we don't need ODEs.
        # User spec: "Manifold/port volumes: p_int...". This implies dynamics.
        names += ["m_exh", "p_exh", "T_exh"]
        return names

    def dynamics(
        self,
        theta: ca.SX,
        y: ca.SX,
        u: dict[str, ca.SX],
        p: dict[str, ca.SX],
        omega: ca.SX,  # dTheta/dt (rad/s) needed for time-based rates
    ) -> ca.SX:
        """Compute dy/dtheta.

        Args:
            theta: Current angle
            y: State vector
            u: Controls (combustion, etc.)
            p: Parameters (geometry constants, etc.)
            omega: Angular velocity [rad/s]
        """
        # Unpack states
        # Assuming simplified ordering for prototype:
        # y[0] = m_c
        # y[1] = T_c
        # y[2] = Y_fuel

        m_c = y[0]
        T_c = y[1]
        Y_f = y[2]

        # Geometry
        V_c = self.geo.Volume(theta)
        dV_dth = self.geo.dV_dtheta(theta)
        A_wall = self.geo.Area_wall(theta)
        A_int = self.geo.Area_intake(theta)
        A_exh = self.geo.Area_exhaust(theta)

        # Derived
        P_c = m_c * self.R_gas * T_c / V_c

        # Flows (Quasi-1D)
        # P_int, T_int from parameters or states. Let's assume params for boundary conditions for now
        # to keep this prototype simple, unless user strictly wants manifold dynamics NOW.
        # "Control volumes: Intake manifold... Mass flows across interface".
        # Let's use parameters for manifold conditions P_plenum_in, T_plenum_in for now.
        P_in = 2.0e5  # 2 bar boost
        T_in = 300.0
        P_out = 1.0e5

        # Mass flow in (Intake)
        # dm_in = Cd * A * P / sqrt(RT) * func(pr)
        # Simplified:
        dm_in_dt = 0.6 * A_int * ca.sqrt(2 * (P_in - P_c) * 1.2)  # Incompressible-ish placeholder
        # Directionality check needed for backflow
        dm_in_dt = ca.if_else(
            P_in > P_c,
            0.6 * A_int * ca.sqrt(2 * 1.2 * (P_in - P_c)),
            -0.6 * A_int * ca.sqrt(2 * 1.2 * (P_c - P_in)),  # Backflow
        )

        # Mass flow out (Exhaust)
        dm_out_dt = ca.if_else(
            P_c > P_out,
            0.6 * A_exh * ca.sqrt(2 * 1.2 * (P_c - P_out)),
            0.0,  # No inflow from exhaust usually
        )

        # Combustion
        # dQ/dtheta
        dQ_comb_dth = self.comb.get_heat_release_rate(theta, u, {"phi": 1.0})
        dQ_comb_dt = dQ_comb_dth * omega

        # Heat Loss
        # Q_wall = h * A * (T - Tw)
        h_coeff = (
            0.01 * ca.power(P_c, 0.8) * ca.power(T_c, -0.55) * ca.power(3.0 * omega, 0.8)
        )  # Hohenberg-ish
        dQ_wall_dt = h_coeff * A_wall * (T_c - 400.0)

        # Energy Balance (dT/dt)
        # m cv dT/dt = sum(mh)in - sum(mh)out + Q_comb - Q_wall - P dV/dt - u dm/dt
        # T_c_dot = (1/m_c c_v) * [ ... ]

        h_in = self.cp * T_in
        h_out = self.cp * T_c  # flow out carries cylindere temp
        h_c = self.cv * T_c + self.R_gas * T_c  # h = u + Pv = u + RT
        u_c = self.cv * T_c

        dV_dt = dV_dth * omega

        # Energy terms
        E_flux = dm_in_dt * h_in - dm_out_dt * h_out
        Work = P_c * dV_dt

        du_dt_total = dQ_comb_dt - dQ_wall_dt - Work + E_flux

        # d(mu)/dt = m du/dt + u dm/dt = du_total/dt
        # du/dt = (du_total/dt - u_c * dm/dt) / m_c
        # dT/dt = (1/cv) * du/dt

        dm_dt = dm_in_dt - dm_out_dt

        dT_dt = (du_dt_total - u_c * dm_dt) / (m_c * self.cv)

        # Species
        # dY/dt
        # Simple burnout: dYf = - dQ_comb / LHV
        LHV = 44e6
        dYf_dt = -dQ_comb_dt / LHV / m_c  # approx

        # Convert time derivatives to theta derivatives
        # dy/dtheta = (dy/dt) / omega

        dy_dth = ca.vertcat(dm_dt / omega, dT_dt / omega, dYf_dt / omega)

        # Pad with zeros if we defined more states
        return dy_dth
