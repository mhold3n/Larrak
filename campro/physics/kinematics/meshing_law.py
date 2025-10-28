"""
Meshing law component for cam-ring systems.

This module provides a modular component for solving the meshing law
that relates cam rotation to ring follower rotation.
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from campro.logging import get_logger

from ..base import BaseComponent, ComponentResult, ComponentStatus

log = get_logger(__name__)


class MeshingLawComponent(BaseComponent):
    """
    Component for solving the meshing law between cam and ring follower.

    The meshing law relates cam rotation θ to ring follower rotation ψ through:
    ρ_c(θ)dθ = R(ψ)dψ

    where ρ_c(θ) is the cam osculating radius and R(ψ) is the ring radius.
    """

    def _validate_parameters(self) -> None:
        """Validate component parameters."""
        # No specific parameters required for meshing law computation

    def _validate_meshing_inputs(self, inputs: dict[str, Any]) -> bool:
        """Validate meshing law inputs with custom logic."""
        required = ["theta", "rho_c", "psi_initial", "R_psi"]

        for input_name in required:
            if input_name not in inputs:
                log.error(f"Missing required input: {input_name}")
                return False

        # Validate array inputs
        if not isinstance(inputs["theta"], np.ndarray):
            log.error("Input theta must be numpy array")
            return False
        if not isinstance(inputs["rho_c"], np.ndarray):
            log.error("Input rho_c must be numpy array")
            return False

        # Validate scalar inputs
        try:
            float(inputs["psi_initial"])
        except (ValueError, TypeError):
            log.error("Input psi_initial must be a number")
            return False

        # R_psi can be scalar or array
        if not isinstance(inputs["R_psi"], (int, float, np.ndarray)):
            log.error("Input R_psi must be a number or numpy array")
            return False

        return True

    def compute(self, inputs: dict[str, Any]) -> ComponentResult:
        """
        Solve the meshing law to relate cam and ring angles.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Input data containing:
            - 'theta': Cam angles (radians) - numpy array
            - 'rho_c': Cam osculating radius ρ_c(θ) - numpy array
            - 'psi_initial': Initial ring angle (radians) - scalar
            - 'R_psi': Ring radius function R(ψ) - scalar or array

        Returns
        -------
        ComponentResult
            Result containing ring angles ψ(θ)
        """
        try:
            # Validate inputs with custom validation
            if not self._validate_meshing_inputs(inputs):
                return ComponentResult(
                    status=ComponentStatus.FAILED,
                    outputs={},
                    metadata={},
                    error_message="Invalid inputs",
                )

            theta = inputs["theta"]
            rho_c = inputs["rho_c"]
            psi_initial = float(inputs["psi_initial"])  # Convert to scalar
            R_psi = inputs["R_psi"]

            log.info(f"Solving meshing law for {len(theta)} points")

            # Create interpolation function for R(ψ)
            # For now, assume R(ψ) is constant or can be interpolated
            if isinstance(R_psi, (int, float)):
                # Constant ring radius
                R_func = lambda psi: R_psi
            else:
                # Interpolate ring radius function
                psi_points = np.linspace(0, 2 * np.pi, len(R_psi))
                R_func = interp1d(
                    psi_points,
                    R_psi,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

            # Solve meshing law: dψ/dθ = ρ_c(θ) / R(ψ)
            psi = self._solve_meshing_law(theta, rho_c, psi_initial, R_func)

            # Prepare outputs
            outputs = {
                "psi": psi,
                "theta": theta,
                "rho_c": rho_c,
            }

            # Prepare metadata
            metadata = {
                "num_points": len(theta),
                "psi_initial": float(psi_initial),
                "psi_final": float(psi[-1]),
                "psi_range": float(psi[-1] - psi[0]),
            }

            log.info(
                f"Meshing law solved successfully: psi range {metadata['psi_range']:.3f} rad",
            )

            return ComponentResult(
                status=ComponentStatus.COMPLETED,
                outputs=outputs,
                metadata=metadata,
            )

        except Exception as e:
            log.error(f"Error solving meshing law: {e}")
            return ComponentResult(
                status=ComponentStatus.FAILED,
                outputs={},
                metadata={},
                error_message=str(e),
            )

    def _solve_meshing_law(
        self, theta: np.ndarray, rho_c: np.ndarray, psi_initial: float, R_func: callable,
    ) -> np.ndarray:
        """
        Solve the meshing law differential equation.

        Parameters
        ----------
        theta : np.ndarray
            Cam angles
        rho_c : np.ndarray
            Cam osculating radius
        psi_initial : float
            Initial ring angle
        R_func : callable
            Ring radius function R(ψ)

        Returns
        -------
        np.ndarray
            Ring angles ψ(θ)
        """
        # Create interpolation function for ρ_c(θ)
        rho_c_func = interp1d(
            theta, rho_c, kind="linear", bounds_error=False, fill_value="extrapolate",
        )

        # Define the differential equation: dψ/dθ = ρ_c(θ) / R(ψ)
        def dpsi_dtheta(theta_val, psi_val):
            rho_c_val = rho_c_func(theta_val)
            R_val = R_func(psi_val)

            # Avoid division by zero
            if abs(R_val) < 1e-12:
                return 0.0

            return rho_c_val / R_val

        # Solve the ODE
        try:
            sol = solve_ivp(
                dpsi_dtheta,
                [theta[0], theta[-1]],
                [psi_initial],
                t_eval=theta,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
            )

            if sol.success:
                return sol.y[0]
            log.warning("ODE solver failed, using simplified integration")
            return self._simplified_integration(theta, rho_c, psi_initial, R_func)

        except Exception as e:
            log.warning(f"ODE solver error: {e}, using simplified integration")
            return self._simplified_integration(theta, rho_c, psi_initial, R_func)

    def _simplified_integration(
        self, theta: np.ndarray, rho_c: np.ndarray, psi_initial: float, R_func: callable,
    ) -> np.ndarray:
        """
        Simplified integration method as fallback.

        Parameters
        ----------
        theta : np.ndarray
            Cam angles
        rho_c : np.ndarray
            Cam osculating radius
        psi_initial : float
            Initial ring angle
        R_func : callable
            Ring radius function

        Returns
        -------
        np.ndarray
            Ring angles
        """
        psi = np.zeros_like(theta)
        psi[0] = psi_initial

        for i in range(1, len(theta)):
            dtheta = theta[i] - theta[i - 1]
            R_val = R_func(psi[i - 1])

            if abs(R_val) > 1e-12:
                dpsi = rho_c[i - 1] / R_val * dtheta
            else:
                dpsi = 0.0

            psi[i] = psi[i - 1] + dpsi

        return psi

    def get_required_inputs(self) -> list[str]:
        """Get list of required input names."""
        return ["theta", "rho_c", "psi_initial", "R_psi"]

    def get_outputs(self) -> list[str]:
        """Get list of output names."""
        return ["psi", "theta", "rho_c"]
