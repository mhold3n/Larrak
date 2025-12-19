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
        params: dict[str, ca.SX] | None = None,
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
        params: dict[str, ca.SX] | None = None,
    ) -> ca.SX:
        # Extract Q_total from params if available, else usage self.Q_total default
        # This supports both legacy (fixed) and new (parameterized) modes
        Q_tot = self.Q_total
        if params and "Q_total" in params:
            Q_tot = params["Q_total"]
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

        # Combustion Efficiency (eta_comb)
        # 1. Base efficiency (98%)
        # 2. Rich penalty (1/phi) for phi > 1
        phi = controls.get("phi", 1.0)
        # Protect against phi=0
        phi_safe = ca.fmax(0.01, phi)
        eta_comb = 0.98 * ca.if_else(phi_safe > 1.0, 1.0 / phi_safe, 1.0)
        
        # Scale Q_total by efficiency
        Q_eff = Q_tot * eta_comb

        val = (
            Q_eff
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
        params: dict[str, ca.SX] | None = None,
    ) -> ca.SX:
        # Controls: 'theta_inj_main', 'theta_inj_pc', 'f_pc', 'phi_main'
        # f_pc: Fraction of fuel energy in prechamber (0.0 to 0.1 typical)

        # Extract Q_total from params if available, else usage self.Q_total default
        Q_tot = self.Q_total
        if params and "Q_total" in params:
            Q_tot = params["Q_total"]

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
            (f_pc * Q_tot)
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
            ((1.0 - f_pc) * Q_tot)
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
        # Tuned for Hot Exhaust Gas (Gamma ~ 1.32)
        # R = 287.0
        # cv = R / (gamma - 1) = 287 / 0.32 = 896.8
        # cp = cv + R = 1183.8
        self.R_gas = 287.0
        self.cv = 896.8
        self.cp = 1183.8

    def calculate_fmep(self, rpm: ca.SX, p_max_pa: ca.SX) -> ca.SX:
        """Calculate Friction Mean Effective Pressure (FMEP) using Chen-Flynn model.
        
        Equation: FMEP [bar] = A + B * Pmax [bar] + C * S_p + ...
        Here simplified to RPM dependence.
        
        Args:
           rpm: Engine speed [RPM]
           p_max_pa: Peak cylinder pressure [Pa]
           
        Returns:
           FMEP [Pa]
        """

        # Coefficients (Chen-Flynn / Heywood generic Diesel)
        # CALIBRATION: Phase 3c (Final?) - Strong Friction for Realistic Eff
        A = 2.0  # bar (Constant)
        B = 0.005  # Scaling with peak pressure
        # Speed term approximated as linear/quadratic with RPM for this 0D model
        # 1000 RPM ~ 0.1 bar friction increase?
        freq = rpm / 1000.0
        
        fmep_bar = A + B * (p_max_pa / 1e5) + 0.09 * freq + 0.0009 * freq**2
        return fmep_bar * 1e5

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
        kinematics: dict[str, ca.SX] | None = None,
    ) -> ca.SX:
        """Compute dy/dtheta.

        Args:
            theta: Current angle
            y: State vector
            u: Controls (combustion, etc.)
            p: Parameters (geometry constants, etc.)
            omega: Angular velocity [rad/s]
            kinematics: Optional override for geometric values {V, dV_dt, A_wall}
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
        # Geometry
        # Check for valve timing overrides in parameters
        # Default behavior: None (geometry uses internal defaults)
        
        # Generic Control Override (Phase 4b)
        if "intake_alpha" in u and u["intake_alpha"] is not None:
             # Scaling Factor: Matches the max peak of the fixed geometry (~0.005 m2)
             A_int = u["intake_alpha"] * 0.005
        else:
            # Intake Parametric
            int_open = p.get("intake_open") # Expecting Rad if present
            int_dur = p.get("intake_dur")
            A_int = self.geo.Area_intake(theta, open_rad=int_open, duration_rad=int_dur)
        
        if "exhaust_alpha" in u and u["exhaust_alpha"] is not None:
             A_exh = u["exhaust_alpha"] * 0.006 # Exhaust usually slightly larger
        else:
            # Exhaust Parametric
            exh_open = p.get("exhaust_open")
            exh_dur = p.get("exhaust_dur")
            A_exh = self.geo.Area_exhaust(theta, open_rad=exh_open, duration_rad=exh_dur)

        if kinematics is not None:
            V_c = kinematics["V"]
            A_wall = kinematics["A_wall"]
            # Kinematics dictionary provides time-derivative dV/dt directly
            dV_dt_val = kinematics["dV_dt"]
        else:
            V_c = self.geo.Volume(theta)
            dV_dth = self.geo.dV_dtheta(theta)
            A_wall = self.geo.Area_wall(theta)
            dV_dt_val = dV_dth * omega

        # Derived
        P_c = m_c * self.R_gas * T_c / V_c

        # Flows (Quasi-1D)
        # P_int, T_int from parameters or states. Let's assume params for boundary conditions for now
        # to keep this prototype simple, unless user strictly wants manifold dynamics NOW.
        # "Control volumes: Intake manifold... Mass flows across interface".
        # Let's use parameters for manifold conditions P_plenum_in, T_plenum_in for now.
        P_in = p.get("p_int", 2.0e5)  # Default 2 bar if not provided
        T_in = p.get("T_int", 300.0)
        P_out = 1.0e5

        # Mass flow in (Intake)
        # Mass flow in (Intake)
        # Smooth Flow Approximation (Essential for MA86 on Windows, and stability for MA57)
        def smooth_flow(delta_p, eps=1e-2): 
             return delta_p * ca.power(delta_p**2 + eps**2, -0.25)

        # Note: Fixed rho is standard for simple 0D control models
        rho_est = 1.2
        dm_in_dt = 0.6 * A_int * ca.sqrt(2 * rho_est) * smooth_flow(P_in - P_c)

        # Exhaust Flow
        dm_out_dt = 0.6 * A_exh * ca.sqrt(2 * rho_est) * smooth_flow(P_c - P_out)

        # Combustion
        # dQ/dtheta
        dQ_comb_dth = self.comb.get_heat_release_rate(theta, u, {"phi": 1.0}, params=p)
        dQ_comb_dt = dQ_comb_dth * omega

        # Heat Loss (Woschni Correlation)
        # h [W/m2K] = 3.26 * B^-0.2 * P[kPa]^0.8 * T[K]^-0.55 * w[m/s]^0.8
        
        B_m = 0.1 # Default Bore 0.1m if self.geo.B not accessible easily or symbolic
        # Actually self.geo should have it.
        # But self.geo might be a symbolic interface.
        # Let's try to access self.geo.B
        if hasattr(self.geo, "B"):
             B_m = self.geo.B
        elif hasattr(self.geo, "bore"):
             B_m = self.geo.bore
             
        # Mean Piston Speed (Sp) = 2 * Stroke * RPM / 60 = Stroke * RPM / 30
        # omega [rad/s] -> RPM = omega * 30 / pi
        # Sp = S * (omega * 30 / pi) / 30 = S * omega / pi
        S_m = 0.2 # Default Stroke
        if hasattr(self.geo, "S"):
            S_m = self.geo.S
             
        Sp = S_m * omega / np.pi
        
        # Woschni Velocity w
        # Without P-Pmot term: w = C1 * Sp
        # C1 = 2.28 (Combustion/Expansion typical avg) + Swirl boost?
        # Let's use 6.18 for gas exchange, 2.28 for compression.
        # Averaging to ~3.0 to account for swirl + turbulence.
        w_gas = 3.0 * Sp
        
        # P in kPa
        P_kpa = P_c / 1000.0
        
        # Woschni Coeff
        # 3.26 * B^-0.2 ...
        # CALIBRATION: Phase 3c - Strong Heat Loss
        C_woschni = 12.0
        
        h_coeff = (
            C_woschni 
            * ca.power(B_m, -0.2) 
            * ca.power(ca.fmax(1e-1, P_kpa), 0.8) 
            * ca.power(ca.fmax(100.0, T_c), -0.55) 
            * ca.power(ca.fmax(0.1, w_gas), 0.8)
        )
        
        dQ_wall_dt = h_coeff * A_wall * (T_c - 450.0) # Wall temp 450K

        # Energy Balance (dT/dt)
        # m cv dT/dt = sum(mh)in - sum(mh)out + Q_comb - Q_wall - P dV/dt - u dm/dt
        # T_c_dot = (1/m_c c_v) * [ ... ]

        h_in = self.cp * T_in
        h_out = self.cp * T_c  # flow out carries cylindere temp
        h_c = self.cv * T_c + self.R_gas * T_c  # h = u + Pv = u + RT
        u_c = self.cv * T_c

        dV_dt = dV_dt_val

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
