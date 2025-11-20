from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from campro.constants import CASADI_PHYSICS_EPSILON
from campro.logging import get_logger

log = get_logger(__name__)


class MassFlowFunction(Protocol):
    def __call__(
        self,
        *,
        ca: Any,
        p_up: Any,
        T_up: Any,
        rho_up: Any,
        p_down: Any,
        T_down: Any,
        A_eff: Any,
        gamma: float,
        R: float,
    ) -> Any:  # pragma: no cover - protocol
        ...


class HeatTransferFunction(Protocol):
    def __call__(
        self,
        *,
        ca: Any,
        p_gas: Any,
        T_gas: Any,
        T_wall: Any,
        B: float,
        x_L: Any,
        x_R: Any,
    ) -> Any:  # pragma: no cover - protocol
        ...


@dataclass
class GasModel:
    mode: Literal["0d", "1d"]
    mdot_in: MassFlowFunction
    mdot_out: MassFlowFunction
    qdot_wall: HeatTransferFunction


def _mdot_orifice() -> MassFlowFunction:
    def fn(
        *,
        ca: Any,
        p_up: Any,
        T_up: Any,
        rho_up: Any,
        p_down: Any,
        T_down: Any,
        A_eff: Any,
        gamma: float,
        R: float,
    ) -> Any:
        # Compressible orifice with choked flow handling (symbolic-friendly)
        # Protect pressures to prevent extreme pressure ratios
        p_min = 1e3  # Minimum pressure [Pa] to prevent extreme ratios
        p_up_safe = ca.fmax(p_up, p_min)
        p_down_safe = ca.fmax(p_down, p_min)
        
        eps = 1e-12
        pr = (p_down_safe + eps) / (p_up_safe + eps)
        
        # Clamp pressure ratio to reasonable range to prevent Inf from fractional powers
        # pr_min = 1e-6 prevents extremely small ratios, pr_max = 1e6 prevents extremely large ratios
        pr_min = 1e-6
        pr_max = 1e6
        pr = ca.fmax(ca.fmin(pr, pr_max), pr_min)
        
        pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        c_up = ca.sqrt(ca.fmax(gamma * R * T_up, CASADI_PHYSICS_EPSILON))
        # Choked branch
        c_crit = ca.sqrt(
            ca.fmax(gamma * R * T_up * (2.0 / (gamma + 1.0)), CASADI_PHYSICS_EPSILON),
        )
        rho_crit = rho_up * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0))
        mdot_choked = A_eff * rho_crit * c_crit
        # Subsonic branch (isentropic)
        term1 = 2.0 * gamma / (gamma - 1.0)
        term2 = ca.fmax(
            0.0,
            ca.fmax(pr, CASADI_PHYSICS_EPSILON) ** (2.0 / gamma)
            - ca.fmax(pr, CASADI_PHYSICS_EPSILON) ** ((gamma + 1.0) / gamma),
        )
        u = c_up * ca.sqrt(ca.fmax(term1 * term2, CASADI_PHYSICS_EPSILON))
        rho_throat = rho_up * ca.fmax(pr, CASADI_PHYSICS_EPSILON) ** (1.0 / gamma)
        mdot_sub = A_eff * rho_throat * u
        # Select based on critical pressure ratio
        use_choked = ca.if_else(pr <= pr_crit, 1.0, 0.0)
        mdot_result = use_choked * mdot_choked + (1.0 - use_choked) * mdot_sub
        
        # Final protection: ensure result is never NaN/Inf before returning
        # Clamp to reasonable physical range [0, mdot_max]
        mdot_max = 1e3  # Reasonable maximum mass flow rate [kg/s]
        # Use conditional to handle Inf: if fabs > mdot_max, clamp to mdot_max with correct sign
        mdot_safe = ca.if_else(
            ca.fabs(mdot_result) > mdot_max,
            ca.sign(mdot_result) * mdot_max,  # Preserve sign but clamp magnitude
            ca.fmax(mdot_result, 0.0)  # Ensure non-negative for normal values
        )
        return ca.fmin(mdot_safe, mdot_max)  # Final clamp to [0, mdot_max]

    return fn


def _q_woschni_like() -> HeatTransferFunction:
    def fn(
        *, ca: Any, p_gas: Any, T_gas: Any, T_wall: Any, B: float, x_L: Any, x_R: Any,
    ) -> Any:
        # Simple Woschni-like h and area model
        h = (
            130.0
            * ca.fmax(p_gas / 1.0e5, CASADI_PHYSICS_EPSILON) ** 0.8
            * ca.fmax(T_gas / 1000.0, CASADI_PHYSICS_EPSILON) ** 0.55
        )
        A_wall = (
            ca.pi * B * ca.fmax(0.0, x_R - x_L)
            + 2.0 * ca.pi * ca.fmax(B / 2.0, CASADI_PHYSICS_EPSILON) ** 2
        )
        return h * A_wall * (T_gas - T_wall)

    return fn


def build_gas_model(P: dict[str, Any]) -> GasModel:
    """Factory: unified gas model interface for NLP assembly.

    Returns small symbolic-friendly closures for mass flow and wall heat transfer,
    configured by `flow.mode`.
    """
    flow = P.get("flow", {})
    mode: Literal["0d", "1d"] = "0d"
    if isinstance(flow.get("mode"), str) and flow["mode"].lower() == "1d":
        mode = "1d"

    # For now, both modes use symbolic-friendly orifice/heat-transfer closures.
    # In future, the "1d" mode can swap to higher-fidelity closures calibrated
    # against the FV solver or CasADi external callbacks.
    mdot = _mdot_orifice()
    qwall = _q_woschni_like()

    return GasModel(mode=mode, mdot_in=mdot, mdot_out=mdot, qdot_wall=qwall)
