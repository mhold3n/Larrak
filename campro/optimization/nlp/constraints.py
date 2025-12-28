"""
Thermodynamic Constraints for the NLP.
Provides CasADi expressions for Feasibility Gates (Thermal, Mechanical).
These surrogates are calibrated by Phase 4 simulations.
"""

from typing import Any, Optional

import casadi as ca


class ThermalConstraints:
    """
    CasADi-compatible thermal surrogate models.
    Calibrated by Simulations/thermal modules.
    """

    def __init__(self, calibration: Optional[dict[str, float]] = None):
        # Default Calibration Coefficients (To be updated by Phase 4 Loop)
        # Model: T_crown = T_coolant + C1 * (P_max * RPM)^C2
        # Or more physics based: T_crown = T_gas_eff - (T_gas_eff - T_oil) / (1 + R_rat)
        self.coeffs = {
            "h_gas_scale": 1.0,  # Scaling for Woschni
            "R_cond_eff": 0.005,  # Effective conduction resistance [K/W]
        }
        if calibration:
            self.coeffs.update(calibration)

    def get_max_crown_temp(self, p_max_sym, mean_temp_sym, rpm_sym, T_oil=360.0):
        """
        Symbolic expression for Max Piston Crown Temperature.
        Inputs are CasADi symbols or expressions.

        Physics Model:
        - Woschni-like HTC scaling: h_gas ~ P^0.8 * RPM^0.8
        - Resistance divider: T_crown = (h_gas*T_gas + U_cool*T_oil) / (h_gas + U_cool)
        """
        # 1. Estimate Heat Flux Driver
        # Q_flux ~ h_gas * (T_gas_effective - T_wall)
        # T_gas_eff is roughly mean cycle temp + Correction for phasing
        T_gas_eff = mean_temp_sym * 1.1

        # 2. Estimate Gas Side HTC
        # h_gas ~ P^0.8 * v^0.8 ~ P_max^0.8 * RPM^0.8
        h_gas = (
            0.5 * (p_max_sym / 1e5) ** 0.8 * (rpm_sym / 1000.0) ** 0.8 * self.coeffs["h_gas_scale"]
        )

        # 3. Simple Resistance Divider
        # R_gas = 1 / h_gas (Ignoring Area scaling for proportional model)
        # R_total = R_gas + R_cond + R_oil
        # But let's simplify: Delta_T_gas_layer = Q * R_gas
        # T_crown = T_gas_eff - Delta_T_gas_layer

        # We need a stable surrogate.
        # Let's use the flux balance: h_gas(Tg - Tc) = U_cool(Tc - Toil)
        # h_gas*Tg + U_cool*Toil = Tc * (h_gas + U_cool)
        # Tc = (h_gas*Tg + U_cool*Toil) / (h_gas + U_cool)

        U_cool = 1.0 / self.coeffs["R_cond_eff"]  # Combined Conduction + Oil Conv

        T_crown = (h_gas * T_gas_eff + U_cool * T_oil) / (h_gas + U_cool)

        return T_crown


def add_thermal_constraints(
    builder: Any,
    p_max_sym: Any,
    T_mean_sym: Any,
    rpm_val: float,
    T_limit: float = 600.0,
    T_oil: float = 360.0,
    calibration: Optional[dict[str, float]] = None,
) -> Any:
    """
    Add T_crown_max as a HARD constraint to the NLP.

    Constraint: g_thermal = T_crown_max(x) - T_limit <= 0

    Args:
        builder: CollocationBuilder with .g, .lbg, .ubg lists
        p_max_sym: CasADi expression for peak cylinder pressure [Pa]
        T_mean_sym: CasADi expression for mean cycle temperature [K]
        rpm_val: Engine speed [RPM] (fixed per solve)
        T_limit: Maximum allowable crown temperature [K] (default 600K for aluminum)
        T_oil: Oil temperature for cooling [K]
        calibration: Optional calibration coefficients for thermal model

    Returns:
        T_crown_sym: CasADi expression for crown temperature (for diagnostics)
    """
    # Build thermal surrogate
    thermal = ThermalConstraints(calibration=calibration)

    # Compute symbolic crown temperature
    T_crown_sym = thermal.get_max_crown_temp(p_max_sym, T_mean_sym, rpm_val, T_oil)

    # Add hard constraint: T_crown - T_limit <= 0
    # Normalized for better conditioning: (T_crown - T_limit) / T_limit <= 0
    g_thermal = (T_crown_sym - T_limit) / T_limit

    builder.g.append(g_thermal)
    builder.lbg.append(-ca.inf)  # No lower bound
    builder.ubg.append(0.0)  # Upper bound: must be <= 0

    return T_crown_sym


def add_pressure_constraints(
    builder: Any,
    p_max_sym: Any,
    P_limit: float = 250e5,
) -> None:
    """
    Add peak cylinder pressure as a HARD constraint to the NLP.

    Constraint: g_pressure = P_max - P_limit <= 0

    Args:
        builder: CollocationBuilder with .g, .lbg, .ubg lists
        p_max_sym: CasADi expression for peak cylinder pressure [Pa]
        P_limit: Maximum allowable pressure [Pa] (default 250 bar)
    """
    # Add hard constraint: P_max - P_limit <= 0
    # Normalized for conditioning
    g_pressure = (p_max_sym - P_limit) / P_limit

    builder.g.append(g_pressure)
    builder.lbg.append(-ca.inf)
    builder.ubg.append(0.0)
