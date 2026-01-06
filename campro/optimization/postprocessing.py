"""
Optimization Post-Processing Module.

This module handles calculation of derived results after the optimization loop,
including Conjugate Profile generation and Mechanical (Phase 3) calculations.
It is designed to be fault-tolerant: failures in these "artifact generation"
steps should not fail the overall optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from campro.logging import get_logger
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)


class OptimizationPostProcessor:
    """Handles post-processing of optimization results."""

    def __init__(self, reporter: StructuredReporter | None = None):
        """
        Initialize the post-processor.

        Args:
            reporter: Optional reporter for structured logging.
        """
        self.reporter = reporter or StructuredReporter(
            context="POSTPROC", logger=log, stream_out=None, stream_err=None
        )

    def generate_conjugate_profiles(
        self, x_opt_unscaled: np.ndarray, meta: dict[str, Any], optimization_result: dict[str, Any]
    ) -> None:
        """
        Generate Conjugate Gear Profiles.

        Updates optimization_result["profiles"] in-place if successful.
        Failures are logged but not raised.

        Args:
            x_opt_unscaled: Unscaled optimization variables.
            meta: Metadata from NLP build.
            optimization_result: Result dictionary to populate.
        """
        if x_opt_unscaled is None or "get_profiles" not in meta or "variables_detailed" not in meta:
            return

        try:
            self.reporter.info("Generating Conjugate Gear Profiles...")

            indices_map = meta["variables_detailed"]

            # Extract PSI (High Res State)
            if "psi" in indices_map:
                psi_idx = indices_map["psi"]
                psi_vals = x_opt_unscaled[psi_idx]
            else:
                raise KeyError("psi state not found")

            # Extract RADIUS CONTROLS (Low Res)
            if "r_planet" in indices_map and "R_ring" in indices_map:
                r_idx = indices_map["r_planet"]
                R_idx = indices_map["R_ring"]
                r_vals_coarse = x_opt_unscaled[r_idx]
                R_vals_coarse = x_opt_unscaled[R_idx]
            else:
                if "i_ratio" in indices_map:
                    raise KeyError(
                        "Legacy i_ratio found but get_profiles expects radii. Model mismatch."
                    )
                raise KeyError("Radius controls (r_planet, R_ring) not found")

            # Extract TIME (High Res)
            if "t_cycle" in indices_map:
                t_idx = indices_map["t_cycle"]
                t_vals = x_opt_unscaled[t_idx]
            else:
                # Fallback to linear grid
                # Use T_cycle if found, else guess
                cycle_time_est = optimization_result.get("T_cycle", 0.02)
                t_vals = np.linspace(0, cycle_time_est, len(psi_vals))

            # Upsample Radii to match psi length
            # Reconstruct time grid for intervals
            K_est = len(r_vals_coarse)
            t_grid_coarse = np.linspace(t_vals[0], t_vals[-1], K_est + 1)

            # Use 'previous' interpolation: val[k] holds for [t_k, t_k+1)
            f_r = interp1d(
                t_grid_coarse[:-1],
                r_vals_coarse,
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )
            f_R = interp1d(
                t_grid_coarse[:-1],
                R_vals_coarse,
                kind="previous",
                bounds_error=False,
                fill_value="extrapolate",
            )

            r_vals_fine = f_r(t_vals)
            R_vals_fine = f_R(t_vals)

            # Computed Ratio for plotting
            i_vals_fine = R_vals_fine / (r_vals_fine + 1e-9)

            # Evaluate CasADi function
            f_prof = meta["get_profiles"]
            # Signature: [r, R, t, psi]
            res = f_prof(r_vals_fine, R_vals_fine, t_vals, psi_vals)

            # Unpack and store coordinates
            profiles = {
                "Px": np.array(res[0]).flatten().tolist(),
                "Py": np.array(res[1]).flatten().tolist(),
                "Rx": np.array(res[2]).flatten().tolist(),
                "Ry": np.array(res[3]).flatten().tolist(),
                # Store vectors for plotting
                "t": t_vals.tolist(),
                "i": i_vals_fine.tolist(),
                "r": r_vals_fine.tolist(),
                "R": R_vals_fine.tolist(),
                "psi": psi_vals.tolist(),
            }

            optimization_result["profiles"] = profiles
            self.reporter.info(f"Generated conjugate profiles (N={len(t_vals)})")

        except Exception as e:
            self.reporter.warning(f"Failed to generate conjugate profiles: {e}")
            # Add metadata about partial failure
            optimization_result.setdefault("warnings", []).append(f"Profile generation failed: {e}")

    def process_mechanical_results(
        self,
        x_opt_unscaled: np.ndarray,
        params: dict[str, Any],
        meta: dict[str, Any],
        optimization_result: dict[str, Any],
    ) -> None:
        """
        Process Phase 3 Mechanical Results.

        Updates optimization_result["mechanical"] in-place if successful.
        Failures are logged but not raised.

        Args:
            x_opt_unscaled: Unscaled optimization variables.
            params: Problem parameters.
            meta: Metadata from NLP build.
            optimization_result: Result dictionary to populate.
        """
        is_phase3_mechanical = "load_profile" in params
        if not is_phase3_mechanical or x_opt_unscaled is None or "variable_groups" not in meta:
            return

        try:
            self.reporter.info("Processing Phase 3 Mechanical Results...")
            var_groups = meta["variable_groups"]

            # Helper to extract variable by name
            def get_var(name: str) -> np.ndarray | None:
                idx = var_groups.get(name)
                if idx is None:
                    return None
                if isinstance(idx, slice):
                    return x_opt_unscaled[idx]
                return x_opt_unscaled[idx]

            psi_vals = get_var("psi")
            r_vals = get_var("r_planet")
            R_vals = get_var("R_ring")

            if psi_vals is not None and r_vals is not None and R_vals is not None:
                # Grid (Angle Domain)
                phi_grid = meta.get("time_grid")
                if phi_grid is None:
                    K_val = len(r_vals)
                    phi_grid = np.linspace(0, 2 * np.pi, K_val + 1)

                # Handle Collocation Points in psi_vals
                # psi_vals contains [X0, Xc_0_1...Xc_0_C, X1, ...]
                # We only want the grid points X0, X1, ...
                C_val = meta.get("C", 3)
                psi_grid_points = psi_vals[:: C_val + 1]

                # Ensure shapes align
                # Controls r, R are K points
                # psi_grid_points should be K+1 points

                K_intervals = len(r_vals)
                phi_eval = phi_grid[:-1]  # Start of intervals

                # Truncate psi to K points (start of intervals)
                if len(psi_grid_points) > K_intervals:
                    psi_eval = psi_grid_points[:K_intervals]
                else:
                    # Fallback if logic mismatch
                    psi_eval = psi_grid_points

                # Interpolate Load Profile
                F_gas_eval = np.zeros(K_intervals)
                load_prof = params.get("load_profile")
                if load_prof:
                    # Linear interp: assumes load_prof['angle'] is sorted 0..2pi
                    F_gas_eval = np.interp(phi_eval, load_prof["angle"], load_prof["F_gas"])

                # Kinematics
                # xL = (R - r) * cos(psi)
                x_L_eval = (R_vals - r_vals) * np.cos(psi_eval)

                # dx/dphi approx = - (R - r) * sin(psi) * (R/r - 1)
                i_ratio = R_vals / (r_vals + 1e-9)
                dx_dphi_eval = -(R_vals - r_vals) * np.sin(psi_eval) * (i_ratio - 1.0)

                # Torque Output (Ideal)
                T_out_ideal = F_gas_eval * dx_dphi_eval

                # Friction
                alpha = 20.0 * (np.pi / 180.0)
                mu = 0.05
                N_c = np.abs(F_gas_eval) / np.cos(alpha)
                T_loss = mu * N_c * r_vals

                # Net Torque
                # efficiency = np.ones_like(T_out_ideal)
                mask_power = np.abs(T_out_ideal) > 1e-6

                # Manual efficiency calculation to avoid divide-by-zero
                efficiency = np.ones_like(T_out_ideal)
                valid_indices = np.where(mask_power)
                if len(valid_indices[0]) > 0:
                    efficiency[valid_indices] = 1.0 - np.abs(
                        T_loss[valid_indices] / T_out_ideal[valid_indices]
                    )

                # Stress Proxy: Conformity (1/r + 1/R)
                curvature_sum = 1.0 / r_vals + 1.0 / R_vals

                mech_results = {
                    "phi": phi_eval.tolist(),
                    "psi": psi_eval.tolist(),
                    "r": r_vals.tolist(),
                    "R": R_vals.tolist(),
                    "i_ratio": i_ratio.tolist(),
                    "x_L": x_L_eval.tolist(),
                    "F_gas": F_gas_eval.tolist(),
                    "T_out_ideal": T_out_ideal.tolist(),
                    "T_loss": T_loss.tolist(),
                    "efficiency": efficiency.tolist(),
                    "curvature_sum": curvature_sum.tolist(),
                    "mean_efficiency": float(np.mean(efficiency[efficiency > 0]))
                    if np.any(efficiency > 0)
                    else 0.0,
                    "mean_torque": float(np.mean(T_out_ideal)),
                }

                optimization_result["mechanical"] = mech_results
                self.reporter.info(
                    f"Mechanical Summary: Mean Eff={mech_results['mean_efficiency']:.2%}, "
                    f"Mean Torque={mech_results['mean_torque']:.2f} Nm"
                )

        except Exception as e:
            self.reporter.warning(f"Failed to process mechanical results: {e}")
            optimization_result.setdefault("warnings", []).append(
                f"Mechanical analysis failed: {e}"
            )
