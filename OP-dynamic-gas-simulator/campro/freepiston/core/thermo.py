from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from campro.logging import get_logger

log = get_logger(__name__)

R_UNIVERSAL = 8.314462618  # J/(mol K)


@dataclass
class JANAFCoeffs:
    """JANAF polynomial coefficients for temperature-dependent properties.

    Coefficients for cp(T) = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    Valid over temperature range [T_low, T_high] in Kelvin.
    """

    a1: float  # J/(mol K)
    a2: float  # J/(mol K^2)
    a3: float  # J/(mol K^3)
    a4: float  # J/(mol K^4)
    a5: float  # J/(mol K^5)
    T_low: float  # K
    T_high: float  # K
    h_formation: float  # J/mol at 298.15 K
    s_formation: float  # J/(mol K) at 298.15 K, 1 bar


@dataclass
class RealGasEOS:
    """Real gas equation of state with temperature-dependent properties.

    Supports Peng-Robinson EOS and JANAF polynomial fits for transport properties.
    """

    # Component data
    components: Dict[
        str, Dict[str, float],
    ]  # species -> {W, Tc, Pc, omega, janaf_coeffs}
    mole_fractions: Dict[str, float]

    def __post_init__(self):
        """Validate and compute mixture properties."""
        if abs(sum(self.mole_fractions.values()) - 1.0) > 1e-6:
            raise ValueError("Mole fractions must sum to 1.0")

        # Compute mixture molecular weight
        self.W_mix = sum(
            frac * self.components[species]["W"]
            for species, frac in self.mole_fractions.items()
        )

        # Compute mixture critical properties using Kay's rule
        self.Tc_mix = sum(
            frac * self.components[species]["Tc"]
            for species, frac in self.mole_fractions.items()
        )
        self.Pc_mix = sum(
            frac * self.components[species]["Pc"]
            for species, frac in self.mole_fractions.items()
        )
        self.omega_mix = sum(
            frac * self.components[species]["omega"]
            for species, frac in self.mole_fractions.items()
        )

    def gas_constant(self) -> float:
        """Mixture gas constant R = R_universal / W_mix."""
        return R_UNIVERSAL / self.W_mix

    def cp_mix(self, T: float) -> float:
        """Mixture heat capacity at constant pressure [J/(kg K)].

        Uses JANAF polynomial fits for each component.
        """
        cp_total = 0.0
        for species, frac in self.mole_fractions.items():
            if "janaf_coeffs" in self.components[species]:
                coeffs = self.components[species]["janaf_coeffs"]
                # JANAF polynomial: cp = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
                cp_species = (
                    coeffs.a1
                    + coeffs.a2 * T
                    + coeffs.a3 * T**2
                    + coeffs.a4 * T**3
                    + coeffs.a5 * T**4
                )
                cp_total += frac * cp_species
            else:
                # Fallback to constant cp for species without JANAF data
                gamma = 1.4  # Default
                R_species = R_UNIVERSAL / self.components[species]["W"]
                cp_species = gamma * R_species / (gamma - 1.0)
                cp_total += frac * cp_species

        # Convert from J/(mol K) to J/(kg K)
        return cp_total / self.W_mix

    def cv_mix(self, T: float) -> float:
        """Mixture heat capacity at constant volume [J/(kg K)]."""
        cp = self.cp_mix(T)
        R = self.gas_constant()
        return cp - R

    def gamma_mix(self, T: float) -> float:
        """Mixture heat capacity ratio."""
        cp = self.cp_mix(T)
        cv = self.cv_mix(T)
        return cp / cv

    def h_mix(self, T: float) -> float:
        """Mixture enthalpy [J/kg] relative to 298.15 K.

        Integrates cp(T) from 298.15 K to T.
        """
        T_ref = 298.15
        if abs(T - T_ref) < 1e-6:
            return 0.0

        # Simple trapezoidal integration (could be improved with adaptive quadrature)
        n_points = max(10, int(abs(T - T_ref) / 50.0))  # ~50K intervals
        T_points = [T_ref + i * (T - T_ref) / n_points for i in range(n_points + 1)]

        h_total = 0.0
        for i in range(n_points):
            T_mid = 0.5 * (T_points[i] + T_points[i + 1])
            cp_mid = self.cp_mix(T_mid)
            dT = T_points[i + 1] - T_points[i]
            h_total += cp_mid * dT

        return h_total

    def s_mix(self, T: float, p: float) -> float:
        """Mixture entropy [J/(kg K)] relative to 298.15 K, 1 bar.

        s(T,p) = s(T,1bar) - R*ln(p/1bar)
        """
        p_ref = 1e5  # 1 bar in Pa
        R = self.gas_constant()

        # Entropy change due to temperature at constant pressure
        s_T = 0.0
        if abs(T - 298.15) > 1e-6:
            # Integrate cp/T from 298.15 K to T
            n_points = max(10, int(abs(T - 298.15) / 50.0))
            T_points = [
                298.15 + i * (T - 298.15) / n_points for i in range(n_points + 1)
            ]

            for i in range(n_points):
                T_mid = 0.5 * (T_points[i] + T_points[i + 1])
                cp_mid = self.cp_mix(T_mid)
                dT = T_points[i + 1] - T_points[i]
                s_T += (cp_mid / T_mid) * dT

        # Entropy change due to pressure at constant temperature
        s_p = -R * math.log(p / p_ref)

        return s_T + s_p

    def peng_robinson_pressure(self, T: float, v: float) -> float:
        """Peng-Robinson equation of state: p = RT/(v-b) - a(T)/(v^2+2bv-b^2).

        Parameters
        ----------
        T : float
            Temperature [K]
        v : float
            Specific volume [m^3/kg]

        Returns
        -------
        p : float
            Pressure [Pa]
        """
        R = self.gas_constant()

        # Peng-Robinson parameters
        kappa = 0.37464 + 1.54226 * self.omega_mix - 0.26992 * self.omega_mix**2
        alpha = (1.0 + kappa * (1.0 - math.sqrt(T / self.Tc_mix))) ** 2

        a = 0.45724 * (R * self.Tc_mix) ** 2 / self.Pc_mix * alpha
        b = 0.07780 * R * self.Tc_mix / self.Pc_mix

        # Convert specific volume to molar volume
        v_molar = v * self.W_mix

        # Peng-Robinson EOS
        p = R * T / (v_molar - b) - a / (v_molar**2 + 2 * b * v_molar - b**2)

        return p

    def density_from_pressure(self, T: float, p: float) -> float:
        """Compute density from pressure using Peng-Robinson EOS.

        Uses Newton-Raphson iteration to solve p(T,v) = p_target.
        """
        R = self.gas_constant()

        # Initial guess from ideal gas
        rho_guess = p / (R * T)
        v_guess = 1.0 / rho_guess

        # Newton-Raphson iteration
        for _ in range(20):  # Max iterations
            p_calc = self.peng_robinson_pressure(T, v_guess)
            if abs(p_calc - p) < 1e-6:
                break

            # Numerical derivative for dp/dv
            dv = 1e-6 * v_guess
            p_plus = self.peng_robinson_pressure(T, v_guess + dv)
            dp_dv = (p_plus - p_calc) / dv

            v_guess = v_guess - (p_calc - p) / dp_dv
            v_guess = max(v_guess, 1e-6)  # Prevent negative volume

        return 1.0 / v_guess

    def transport_properties(self, T: float) -> Tuple[float, float, float]:
        """Compute temperature-dependent transport properties.

        Returns
        -------
        mu : float
            Dynamic viscosity [Pa s]
        k : float
            Thermal conductivity [W/(m K)]
        Pr : float
            Prandtl number
        """
        # Sutherland's law for viscosity (simplified)
        T_ref = 273.15  # K
        mu_ref = 1.716e-5  # Pa s for air at 273.15 K
        S = 110.4  # Sutherland constant for air

        mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

        # Thermal conductivity from Eucken's relation
        cp = self.cp_mix(T)
        cv = self.cv_mix(T)
        gamma = cp / cv
        k = mu * cp * (4 * gamma) / (9 * gamma - 5)

        # Prandtl number
        Pr = mu * cp / k

        return mu, k, Pr


@dataclass
class IdealMix:
    """Calorically simple ideal mixture with constant gamma.

    Parameters
    ----------
    gamma_ref : float
        Heat capacity ratio.
    W_mix : float
        Mixture molecular weight [kg/mol].
    """

    gamma_ref: float
    W_mix: float

    def gas_constants(self) -> Tuple[float, float]:
        R = R_UNIVERSAL / self.W_mix
        cp = self.gamma_ref * R / (self.gamma_ref - 1.0)
        return R, cp

    def h_T(self, T: float) -> float:
        """Sensible enthalpy relative to 0 K with constant cp."""
        _, cp = self.gas_constants()
        return cp * T

    def s_Tp(self, T: float, p: float) -> float:
        """Ideal-mix entropy up to a constant reference.

        s(T,p) = cp ln(T) - R ln(p)
        """
        R, cp = self.gas_constants()

        if T <= 0.0 or p <= 0.0:
            raise ValueError("T and p must be positive")
        return cp * math.log(T) - R * math.log(p)
