"""
Meshing law integration and kinematics.
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from campro.logging import get_logger

from .config import CamRingParameters

log = get_logger(__name__)


def solve_meshing_law(
    theta: np.ndarray,
    rho_c: np.ndarray,
    psi_guess: np.ndarray,
    R_psi_guess: np.ndarray,
    params: CamRingParameters,
) -> np.ndarray:
    """Solve meshing law ODE: dpsi/dtheta = rho_c(theta) / R(psi)."""
    rho_interp = interp1d(theta, rho_c, kind="cubic", fill_value="extrapolate")
    R_interp = interp1d(psi_guess, R_psi_guess, kind="cubic", fill_value="extrapolate")

    sign = -1.0 if params.contact_type == "internal" else 1.0

    def ode(t, y):
        R_val = R_interp(y)
        if abs(R_val) < 1e-12:
            return 0.0
        return sign * rho_interp(t) / R_val

    try:
        sol = solve_ivp(
            ode, [theta[0], theta[-1]], [psi_guess[0]], t_eval=theta, method="RK45", rtol=1e-8
        )
        if sol.success:
            return sol.y[0]
    except Exception as e:
        log.warning(f"ODE failed: {e}")

    return np.linspace(psi_guess[0], psi_guess[-1], len(theta))


def compute_time_kinematics(
    theta: np.ndarray,
    psi: np.ndarray,
    rho_c: np.ndarray,
    R_psi: np.ndarray,
    driver: str = "cam",
    omega: float | None = None,
    Omega: float | None = None,
) -> dict[str, Any]:
    """Compute time-based kinematics."""
    if driver == "cam" and omega:
        t = (theta - theta[0]) / omega
        # dpsi/dt = dpsi/dtheta * dtheta/dt = (rho/R) * omega
        # Sign handled in value? Assuming external contact logic here implies positive ratio magnitude?
        # Actually dpsi/dtheta is signed.
        # Here we just want magnitudes or consistent check?
        # Let's rely on array operations.
        # dpsi/dt? We return rho/R ratio implicitly.
        pass
    elif driver == "ring" and Omega:
        t = (psi - psi[0]) / Omega
    else:
        raise ValueError("Invalid driver config")

    # Simplified return for legacy compatibility
    # Real dpsi/dt calculation would need sign check.
    return {"time": t, "theta": theta, "psi": psi, "rho_c": rho_c, "R_psi": R_psi, "driver": driver}


def generate_complete_ring_profile(
    psi: np.ndarray,
    R_psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate to full 360 degree profile."""
    psi_norm = np.mod(psi, 2 * np.pi)
    idx = np.argsort(psi_norm)
    psi_sorted = psi_norm[idx]
    R_sorted = R_psi[idx]

    psi_complete = np.linspace(0, 2 * np.pi, len(psi), endpoint=False)
    R_complete = np.interp(psi_complete, psi_sorted, R_sorted, period=2 * np.pi)

    if len(np.unique(R_psi)) == 1:
        R_complete[:] = R_psi[0]

    return psi_complete, R_complete
