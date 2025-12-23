"""NLP Diagnostic Suite for Phase 1 Optimization.

Three-layer analysis:
1. NLP Health - feasibility, scaling, Jacobian conditioning
2. Dynamics Stability - re-integration, divergence detection
3. Solution Uniqueness - sensitivity slices, Hessian eigenvalues
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

# Optional scipy for advanced linear algebra
try:
    from scipy import linalg as sp_linalg

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class DiagnosticResult:
    """Container for diagnostic results."""

    name: str
    status: str  # "PASS", "WARNING", "CRITICAL"
    message: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "data": self.data,
        }


class NLPDiagnostics:
    """Comprehensive NLP diagnostic suite."""

    def __init__(
        self,
        nlp_dict: dict,
        meta: dict,
        name: str = "NLP Diagnostic",
    ):
        """
        Args:
            nlp_dict: CasADi NLP dict with 'x', 'f', 'g' keys
            meta: Metadata dict with 'w0', 'lbw', 'ubw', 'lbg', 'ubg', etc.
            name: Identifier for this diagnostic run
        """
        self.nlp = nlp_dict
        self.meta = meta
        self.name = name

        self.x = nlp_dict["x"]
        self.f = nlp_dict["f"]
        self.g = nlp_dict["g"]
        self.w0 = np.array(meta.get("w0", []))
        self.lbw = np.array(meta.get("lbw", []))
        self.ubw = np.array(meta.get("ubw", []))
        self.lbg = np.array(meta.get("lbg", []))
        self.ubg = np.array(meta.get("ubg", []))

        self.results: list[DiagnosticResult] = []
        self.solution: dict | None = None

    # =========================================================================
    # LAYER 1: NLP Health
    # =========================================================================

    def run_feasibility_check(self, max_iter: int = 500) -> DiagnosticResult:
        """Test if constraints are satisfiable with zero objective."""

        # Feasibility NLP: minimize 0
        nlp_feas = {"x": self.x, "f": 0, "g": self.g}

        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "linear_solver": "mumps",
                "print_level": 0,
                "max_iter": max_iter,
                "tol": 1e-4,
            },
        }

        try:
            solver = ca.nlpsol("feasibility", "ipopt", nlp_feas, opts)
            res = solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
            stats = solver.stats()

            success = stats.get("success", False)
            status_str = stats.get("return_status", "Unknown")

            # Check constraint violation at solution
            g_fn = ca.Function("g_eval", [self.x], [self.g])
            g_sol = g_fn(res["x"]).full().flatten()

            # Compute violation
            violation = 0.0
            for i, gv in enumerate(g_sol):
                lb, ub = self.lbg[i], self.ubg[i]
                if gv < lb:
                    violation = max(violation, lb - gv)
                elif gv > ub:
                    violation = max(violation, gv - ub)

            if success and violation < 1e-4:
                result = DiagnosticResult(
                    name="Feasibility Check",
                    status="PASS",
                    message=f"Feasible solution found in {stats.get('iter_count', 0)} iterations",
                    data={
                        "solver_status": status_str,
                        "iterations": stats.get("iter_count", 0),
                        "max_violation": float(violation),
                    },
                )
            else:
                result = DiagnosticResult(
                    name="Feasibility Check",
                    status="CRITICAL",
                    message=f"Infeasible: {status_str}, max violation = {violation:.2e}",
                    data={
                        "solver_status": status_str,
                        "iterations": stats.get("iter_count", 0),
                        "max_violation": float(violation),
                    },
                )
        except Exception as e:
            result = DiagnosticResult(
                name="Feasibility Check",
                status="CRITICAL",
                message=f"Solver error: {e}",
                data={"error": str(e)},
            )

        self.results.append(result)
        return result

    def run_scaling_analysis(self) -> DiagnosticResult:
        """Analyze variable and constraint scaling (Jacobian Rows)."""

        # Variable scaling (Magnitude of initial guess)
        w0_abs = np.abs(self.w0)
        w0_abs[w0_abs < 1e-6] = 1e-6  # Clamp zeros to 1e-6 (Range < 7 orders)

        var_min = float(w0_abs.min())
        var_max = float(w0_abs.max())
        var_log_range = np.log10(var_max / var_min)

        # Identify poorly scaled variables
        very_small = np.sum(w0_abs < 1e-6)
        very_large = np.sum(w0_abs > 1e6)

        # Constraint Scaling: Measure Jacobian Row Norms (Gradient Magnitude)
        # This is the true measure of equation scaling seen by the solver.
        jac_g = ca.jacobian(self.g, self.x)
        jac_fn = ca.Function("jac_eval", [self.x], [jac_g])
        J = jac_fn(self.w0).full()

        # Row norms (Infinity norm: max absolute element in each row)
        row_norms = np.max(np.abs(J), axis=1)
        row_norms[row_norms < 1e-10] = 1e-10  # Clamp zero rows (e.g. redundant constraints)

        con_min = float(row_norms.min())
        con_max = float(row_norms.max())
        con_log_range = np.log10(con_max / con_min)

        # Status based on log range
        max_range = max(var_log_range, con_log_range)

        if max_range < 7:
            status = "PASS"
            msg = f"Good scaling (range: {max_range:.1f} orders)"
        elif max_range < 9:
            status = "WARNING"
            msg = f"Moderate scaling issues (range: {max_range:.1f} orders)"
        else:
            status = "CRITICAL"
            msg = f"Severe scaling issues (range: {max_range:.1f} orders)"

        result = DiagnosticResult(
            name="Scaling Analysis",
            status=status,
            message=msg,
            data={
                "variable_range": [var_min, var_max],
                "constraint_grad_range": [con_min, con_max],
                "variable_log_range": float(var_log_range),
                "constraint_log_range": float(con_log_range),
                "variables_below_1e-6": int(very_small),
                "variables_above_1e6": int(very_large),
            },
        )

        self.results.append(result)
        return result

    def run_jacobian_analysis(self, check_nan: bool = True) -> DiagnosticResult:
        """Analyze constraint Jacobian at w0."""

        # Compute Jacobian
        jac_g = ca.jacobian(self.g, self.x)
        jac_fn = ca.Function("jacobian_eval", [self.x], [jac_g])

        J = jac_fn(self.w0).full()

        # Check for NaN
        nan_count = int(np.sum(np.isnan(J)))
        inf_count = int(np.sum(np.isinf(J)))

        if nan_count > 0 or inf_count > 0:
            result = DiagnosticResult(
                name="Jacobian Analysis",
                status="CRITICAL",
                message=f"Jacobian contains {nan_count} NaN, {inf_count} Inf values",
                data={
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "shape": list(J.shape),
                },
            )
            self.results.append(result)
            return result

        # SVD analysis
        if HAS_SCIPY:
            try:
                U, s, Vt = sp_linalg.svd(J, full_matrices=False)

                # Condition number
                cond = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")

                # Numerical rank (singular values > 1e-10 * max)
                rank_tol = 1e-10 * s[0]
                rank = int(np.sum(s > rank_tol))
                full_rank = min(J.shape)
                rank_deficiency = full_rank - rank

                # Near-zero singular values
                near_zero = int(np.sum(s < 1e-6))

                if cond < 1e4 and rank_deficiency == 0:
                    status = "PASS"
                    msg = f"Well-conditioned (κ = {cond:.2e})"
                elif cond < 1e8 and rank_deficiency < 5:
                    status = "WARNING"
                    msg = f"Moderate conditioning (κ = {cond:.2e}, rank deficiency = {rank_deficiency})"
                else:
                    status = "CRITICAL"
                    msg = f"Ill-conditioned (κ = {cond:.2e}, rank deficiency = {rank_deficiency})"

                result = DiagnosticResult(
                    name="Jacobian Analysis",
                    status=status,
                    message=msg,
                    data={
                        "condition_number": cond,
                        "rank": rank,
                        "full_rank": full_rank,
                        "rank_deficiency": rank_deficiency,
                        "near_zero_sv": near_zero,
                        "singular_values_top5": s[:5].tolist(),
                        "singular_values_bottom5": s[-5:].tolist() if len(s) >= 5 else s.tolist(),
                    },
                )
            except Exception as e:
                result = DiagnosticResult(
                    name="Jacobian Analysis",
                    status="WARNING",
                    message=f"SVD failed: {e}",
                    data={"error": str(e)},
                )
        else:
            # Fallback without scipy
            result = DiagnosticResult(
                name="Jacobian Analysis",
                status="WARNING",
                message="Scipy not available for SVD analysis",
                data={"shape": list(J.shape)},
            )

        self.results.append(result)
        return result

    # =========================================================================
    # LAYER 2: Dynamics Stability
    # =========================================================================

    def run_dynamics_stability(
        self,
        solution: np.ndarray | None = None,
        ode_func: callable | None = None,
    ) -> DiagnosticResult:
        """Check if dynamics are stable at solution."""

        if solution is None:
            solution = self.w0

        # Extract state trajectories from solution
        # This is problem-specific - we'll do a generic check

        # Check for divergent values
        sol = np.array(solution)
        max_val = float(np.max(np.abs(sol)))
        any_inf = bool(np.any(np.isinf(sol)))
        any_nan = bool(np.any(np.isnan(sol)))

        # Check for negative values where they shouldn't be
        # (mass, temperature, pressure should be positive)
        min_val = float(np.min(sol))

        if any_nan or any_inf:
            status = "CRITICAL"
            msg = "Solution contains NaN/Inf values"
        elif max_val > 1e10:
            status = "CRITICAL"
            msg = f"Solution diverged (max = {max_val:.2e})"
        elif min_val < -1e3:
            status = "WARNING"
            msg = f"Large negative values in solution (min = {min_val:.2e})"
        else:
            status = "PASS"
            msg = f"Solution values in reasonable range [{min_val:.2e}, {max_val:.2e}]"

        result = DiagnosticResult(
            name="Dynamics Stability",
            status=status,
            message=msg,
            data={
                "max_value": max_val,
                "min_value": min_val,
                "any_nan": any_nan,
                "any_inf": any_inf,
            },
        )

        self.results.append(result)
        return result

    def run_dynamics_reintegration(
        self,
        ode_func: callable | None = None,
        state_indices: list[int] | None = None,
        theta_span: tuple = (0, 2 * np.pi),
        n_points: int = 100,
    ) -> DiagnosticResult:
        """Re-integrate dynamics outside NLP to check for divergence.

        Args:
            ode_func: Function f(theta, y) -> dy/dtheta. If None, skips re-integration.
            state_indices: Indices in w0 corresponding to initial state y0.
            theta_span: Integration span (theta_start, theta_end).
            n_points: Number of evaluation points.
        """
        if ode_func is None:
            result = DiagnosticResult(
                name="Dynamics Re-integration",
                status="WARNING",
                message="ODE function not provided, skipping re-integration",
                data={},
            )
            self.results.append(result)
            return result

        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            result = DiagnosticResult(
                name="Dynamics Re-integration",
                status="WARNING",
                message="scipy.integrate not available",
                data={},
            )
            self.results.append(result)
            return result

        # Extract initial state
        if state_indices is None:
            # Default: first 6 values are states
            state_indices = list(range(min(6, len(self.w0))))

        y0 = self.w0[state_indices]
        theta_eval = np.linspace(theta_span[0], theta_span[1], n_points)

        try:
            sol = solve_ivp(
                ode_func,
                theta_span,
                y0,
                t_eval=theta_eval,
                method="RK45",
                max_step=0.1,
            )

            if not sol.success:
                result = DiagnosticResult(
                    name="Dynamics Re-integration",
                    status="CRITICAL",
                    message=f"Integration failed: {sol.message}",
                    data={"message": sol.message},
                )
            else:
                # Check for divergence
                y_final = sol.y[:, -1]
                max_val = float(np.max(np.abs(sol.y)))
                any_nan = bool(np.any(np.isnan(sol.y)))
                any_inf = bool(np.any(np.isinf(sol.y)))

                # Check for stiffness (rapid oscillations)
                # Use variance of derivatives as proxy
                dy = np.diff(sol.y, axis=1)
                stiffness_score = float(np.max(np.std(dy, axis=1)))

                if any_nan or any_inf:
                    status = "CRITICAL"
                    msg = "Re-integrated trajectory contains NaN/Inf"
                elif max_val > 1e6:
                    status = "CRITICAL"
                    msg = f"Re-integrated trajectory diverged (max = {max_val:.2e})"
                elif stiffness_score > 100:
                    status = "WARNING"
                    msg = f"Stiff dynamics detected (score = {stiffness_score:.1f})"
                else:
                    status = "PASS"
                    msg = f"Re-integration stable (max = {max_val:.2e})"

                result = DiagnosticResult(
                    name="Dynamics Re-integration",
                    status=status,
                    message=msg,
                    data={
                        "y0": y0.tolist(),
                        "y_final": y_final.tolist(),
                        "max_value": max_val,
                        "stiffness_score": stiffness_score,
                        "n_points": n_points,
                    },
                )
        except Exception as e:
            result = DiagnosticResult(
                name="Dynamics Re-integration",
                status="CRITICAL",
                message=f"Integration error: {e}",
                data={"error": str(e)},
            )

        self.results.append(result)
        return result

    # =========================================================================
    # LAYER 3: Solution Uniqueness
    # =========================================================================

    def run_sensitivity_slices(
        self,
        control_indices: list[int] | None = None,
        n_points: int = 11,
        variation: float = 0.2,
    ) -> DiagnosticResult:
        """Compute 1D slices of objective around current point."""

        if control_indices is None:
            # Auto-detect control indices from meta
            control_indices = self.meta.get("control_indices", list(range(min(3, len(self.w0)))))

        # Objective function
        f_fn = ca.Function("f_eval", [self.x], [self.f])
        g_fn = ca.Function("g_eval", [self.x], [self.g])

        slices = {}
        flatness_score = 0.0

        for idx in control_indices:
            if idx >= len(self.w0):
                continue

            center = self.w0[idx]
            lb = self.lbw[idx] if idx < len(self.lbw) else center - abs(center) * variation
            ub = self.ubw[idx] if idx < len(self.ubw) else center + abs(center) * variation

            # Compute range
            delta = max(abs(center) * variation, (ub - lb) * 0.1)
            values = np.linspace(max(lb, center - delta), min(ub, center + delta), n_points)

            objectives = []
            violations = []

            for v in values:
                w_test = self.w0.copy()
                w_test[idx] = v

                try:
                    obj = float(f_fn(w_test))
                    g_vals = g_fn(w_test).full().flatten()

                    # Max violation
                    viol = 0.0
                    for i, gv in enumerate(g_vals):
                        if i < len(self.lbg) and gv < self.lbg[i]:
                            viol = max(viol, self.lbg[i] - gv)
                        if i < len(self.ubg) and gv > self.ubg[i]:
                            viol = max(viol, gv - self.ubg[i])

                    objectives.append(obj)
                    violations.append(viol)
                except Exception:
                    objectives.append(float("nan"))
                    violations.append(float("nan"))

            # Compute flatness (std / range)
            obj_arr = np.array([o for o in objectives if not np.isnan(o)])
            if len(obj_arr) > 2:
                obj_range = obj_arr.max() - obj_arr.min()
                if abs(obj_arr.mean()) > 1e-10:
                    rel_flatness = obj_range / abs(obj_arr.mean())
                else:
                    rel_flatness = obj_range
                flatness_score = max(flatness_score, 1.0 / (rel_flatness + 0.01))

            slices[str(idx)] = {
                "values": values.tolist(),
                "objectives": objectives,
                "violations": violations,
            }

        if flatness_score > 10:
            status = "WARNING"
            msg = f"Objective appears flat (flatness score = {flatness_score:.1f})"
        else:
            status = "PASS"
            msg = f"Objective has reasonable curvature (flatness score = {flatness_score:.1f})"

        result = DiagnosticResult(
            name="Sensitivity Slices",
            status=status,
            message=msg,
            data={
                "slices": slices,
                "flatness_score": flatness_score,
                "n_controls_analyzed": len(slices),
            },
        )

        self.results.append(result)
        return result

    def run_hessian_analysis(
        self,
        lambda_g: np.ndarray | None = None,
    ) -> DiagnosticResult:
        """Analyze Hessian of Lagrangian for solution uniqueness.

        Near-zero eigenvalues indicate flat directions (non-unique optima)
        or over-parameterized controls.

        Args:
            lambda_g: Lagrange multipliers for constraints. If None, uses zeros.
        """
        if not HAS_SCIPY:
            result = DiagnosticResult(
                name="Hessian Analysis",
                status="WARNING",
                message="scipy not available for eigenvalue analysis",
                data={},
            )
            self.results.append(result)
            return result

        try:
            # Hessian of objective
            H_f = ca.hessian(self.f, self.x)[0]

            # Hessian of Lagrangian: H_L = H_f + sum(lambda_i * H_g_i)
            if lambda_g is None:
                # Use zero multipliers (approximation)
                lambda_g = np.zeros(self.g.shape[0])

            # For simplicity, compute Hessian of objective only
            # (full Lagrangian Hessian is expensive for large problems)
            H_fn = ca.Function("hessian_eval", [self.x], [H_f])
            H = H_fn(self.w0).full()

            # Check for NaN/Inf
            if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                result = DiagnosticResult(
                    name="Hessian Analysis",
                    status="CRITICAL",
                    message="Hessian contains NaN/Inf values",
                    data={"nan_count": int(np.sum(np.isnan(H)))},
                )
                self.results.append(result)
                return result

            # Eigenvalue analysis
            try:
                eigenvalues = sp_linalg.eigvalsh(H)  # Symmetric, real eigenvalues
            except Exception:
                # Fallback for non-symmetric
                eigenvalues = np.real(sp_linalg.eigvals(H))

            eigenvalues = np.sort(eigenvalues)

            # Count near-zero eigenvalues (flat directions)
            near_zero_threshold = 1e-6
            near_zero_count = int(np.sum(np.abs(eigenvalues) < near_zero_threshold))

            # Count negative eigenvalues (non-convex)
            negative_count = int(np.sum(eigenvalues < -near_zero_threshold))

            # Condition number of positive part
            pos_eigs = eigenvalues[eigenvalues > near_zero_threshold]
            if len(pos_eigs) >= 2:
                hess_cond = float(pos_eigs[-1] / pos_eigs[0])
            else:
                hess_cond = float("inf")

            # Status determination
            if near_zero_count > len(eigenvalues) * 0.1:
                status = "WARNING"
                msg = f"Many flat directions ({near_zero_count} near-zero eigenvalues)"
            elif negative_count > len(eigenvalues) * 0.1:
                status = "WARNING"
                msg = f"Non-convex objective ({negative_count} negative eigenvalues)"
            elif near_zero_count > 0:
                status = "PASS"
                msg = f"Some flat directions ({near_zero_count} near-zero eigenvalues)"
            else:
                status = "PASS"
                msg = f"Well-curved objective (no flat directions)"

            result = DiagnosticResult(
                name="Hessian Analysis",
                status=status,
                message=msg,
                data={
                    "n_eigenvalues": len(eigenvalues),
                    "near_zero_count": near_zero_count,
                    "negative_count": negative_count,
                    "hessian_condition": hess_cond,
                    "eigenvalues_smallest5": eigenvalues[:5].tolist(),
                    "eigenvalues_largest5": eigenvalues[-5:].tolist(),
                },
            )
        except Exception as e:
            result = DiagnosticResult(
                name="Hessian Analysis",
                status="WARNING",
                message=f"Hessian analysis failed: {e}",
                data={"error": str(e)},
            )

        self.results.append(result)
        return result

    # =========================================================================
    # Report Generation
    # =========================================================================

    def run_all(self, include_hessian: bool = True) -> list[DiagnosticResult]:
        """Run all diagnostics.

        Args:
            include_hessian: Whether to run Hessian analysis (can be slow for large NLPs).
        """
        self.results = []

        # Layer 1: NLP Health
        self.run_feasibility_check()
        self.run_scaling_analysis()
        self.run_jacobian_analysis()

        # Layer 2: Dynamics Stability
        self.run_dynamics_stability()

        # Layer 3: Solution Uniqueness
        self.run_sensitivity_slices()
        if include_hessian:
            self.run_hessian_analysis()

        return self.results

    def get_summary(self) -> dict:
        """Get overall summary of diagnostics."""
        status_counts = {"PASS": 0, "WARNING": 0, "CRITICAL": 0}
        for r in self.results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        if status_counts["CRITICAL"] > 0:
            overall = "CRITICAL"
        elif status_counts["WARNING"] > 0:
            overall = "WARNING"
        else:
            overall = "PASS"

        return {
            "overall_status": overall,
            "status_counts": status_counts,
            "n_diagnostics": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

    def generate_html_report(self, output_path: str) -> str:
        """Generate HTML diagnostic report."""

        summary = self.get_summary()

        status_colors = {
            "PASS": "#28a745",
            "WARNING": "#ffc107",
            "CRITICAL": "#dc3545",
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NLP Diagnostic Report - {self.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; 
                        color: white; font-weight: bold; }}
        .status-PASS {{ background: {status_colors["PASS"]}; }}
        .status-WARNING {{ background: {status_colors["WARNING"]}; color: #333; }}
        .status-CRITICAL {{ background: {status_colors["CRITICAL"]}; }}
        .diagnostic {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .diagnostic-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .diagnostic-data {{ background: #f8f9fa; padding: 15px; border-radius: 4px; 
                           margin-top: 15px; font-family: monospace; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
        .timestamp {{ color: #888; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 NLP Diagnostic Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Overall Status</h3>
            <span class="status-badge status-{summary["overall_status"]}">{summary["overall_status"]}</span>
        </div>
        <div class="summary-card">
            <h3>Diagnostics Run</h3>
            <p><strong>{summary["n_diagnostics"]}</strong> checks completed</p>
        </div>
        <div class="summary-card">
            <h3>Results</h3>
            <p>✅ {summary["status_counts"].get("PASS", 0)} Pass | 
               ⚠️ {summary["status_counts"].get("WARNING", 0)} Warning | 
               ❌ {summary["status_counts"].get("CRITICAL", 0)} Critical</p>
        </div>
    </div>
    
    <h2>Diagnostic Details</h2>
"""

        for r in self.results:
            data_html = ""
            if r.data:
                data_html = (
                    "<div class='diagnostic-data'><pre>"
                    + json.dumps(r.data, indent=2, default=str)
                    + "</pre></div>"
                )

            html += f"""
    <div class="diagnostic">
        <div class="diagnostic-header">
            <h3>{r.name}</h3>
            <span class="status-badge status-{r.status}">{r.status}</span>
        </div>
        <p>{r.message}</p>
        {data_html}
    </div>
"""

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            html += """
    <h2>📋 Recommendations</h2>
    <div class="diagnostic">
        <ul>
"""
            for rec in recommendations:
                html += f"            <li>{rec}</li>\n"
            html += """        </ul>
    </div>
"""

        html += """
</div>
</body>
</html>
"""

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on results."""
        recs = []

        for r in self.results:
            if r.status == "CRITICAL":
                if "Feasibility" in r.name:
                    recs.append(
                        "🔴 <strong>Feasibility:</strong> Check if constraints are consistent. Try relaxing bounds or removing conflicting constraints."
                    )
                elif "Jacobian" in r.name and "NaN" in r.message:
                    recs.append(
                        "🔴 <strong>NaN in Jacobian:</strong> Protect sqrt/power/log from invalid arguments. Check for division by zero."
                    )
                elif "Jacobian" in r.name:
                    recs.append(
                        "🔴 <strong>Ill-conditioned Jacobian:</strong> Normalize variables/constraints to O(1). Add regularization."
                    )
                elif "Scaling" in r.name:
                    recs.append(
                        "🔴 <strong>Scaling:</strong> Apply variable/constraint scaling. Nondimensionalize physical quantities."
                    )
            elif r.status == "WARNING":
                if "flat" in r.message.lower():
                    recs.append(
                        "🟡 <strong>Flat objective:</strong> Add regularization term (e.g., ||u||² or ||du/dt||²)."
                    )
                elif "Scaling" in r.name:
                    recs.append(
                        "🟡 <strong>Scaling:</strong> Consider normalizing variables to improve conditioning."
                    )

        return recs


def diagnose_nlp(nlp_dict: dict, meta: dict, output_dir: str, name: str = "nlp") -> str:
    """Convenience function to run diagnostics and generate report.

    Args:
        nlp_dict: CasADi NLP dict
        meta: Metadata with w0, bounds, etc.
        output_dir: Directory for output report
        name: Diagnostic run name

    Returns:
        Path to generated HTML report
    """
    diag = NLPDiagnostics(nlp_dict, meta, name)
    diag.run_all()

    report_path = os.path.join(
        output_dir, f"nlp_diagnostic_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    diag.generate_html_report(report_path)

    return report_path


def diagnose_doe_results(csv_path: str, output_dir: str) -> str:
    """Analyze DOE results CSV for convergence patterns and failure modes.

    Args:
        csv_path: Path to DOE results CSV file
        output_dir: Directory for output report

    Returns:
        Path to generated HTML report
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Convergence analysis
    if "status" in df.columns:
        status_col = "status"
    elif "solver_status" in df.columns:
        status_col = "solver_status"
    else:
        status_col = None

    results = []

    # Overall convergence rate
    if status_col:
        total = len(df)
        succeeded = len(df[df[status_col].str.contains("Succeed", case=False, na=False)])
        failed = total - succeeded

        # Categorize failures
        iter_exceeded = len(df[df[status_col].str.contains("Iteration", case=False, na=False)])
        restoration = len(df[df[status_col].str.contains("Restoration", case=False, na=False)])
        infeas = len(df[df[status_col].str.contains("Infeasible", case=False, na=False)])
        skipped = len(df[df[status_col].str.contains("Skip", case=False, na=False)])

        conv_rate = succeeded / total if total > 0 else 0

        if conv_rate > 0.9:
            status = "PASS"
            msg = f"High convergence rate ({conv_rate * 100:.1f}%)"
        elif conv_rate > 0.5:
            status = "WARNING"
            msg = f"Moderate convergence rate ({conv_rate * 100:.1f}%)"
        else:
            status = "CRITICAL"
            msg = f"Low convergence rate ({conv_rate * 100:.1f}%)"

        results.append(
            DiagnosticResult(
                name="Convergence Rate",
                status=status,
                message=msg,
                data={
                    "total_cases": total,
                    "succeeded": succeeded,
                    "failed": failed,
                    "convergence_rate": conv_rate,
                    "iter_exceeded": iter_exceeded,
                    "restoration_failed": restoration,
                    "infeasible": infeas,
                    "skipped": skipped,
                },
            )
        )

    # Parameter range analysis
    param_cols = [c for c in df.columns if c in ["rpm", "q_total", "phi", "p_boost_actual_bar"]]
    for col in param_cols:
        if col in df.columns and status_col:
            # Check if failures correlate with parameter range
            if df[col].dtype in [np.float64, np.int64]:
                failed_df = df[~df[status_col].str.contains("Succeed", case=False, na=False)]
                success_df = df[df[status_col].str.contains("Succeed", case=False, na=False)]

                if len(failed_df) > 0 and len(success_df) > 0:
                    failed_mean = failed_df[col].mean()
                    success_mean = success_df[col].mean()
                    bias = (failed_mean - success_mean) / (success_mean + 1e-10)

                    if abs(bias) > 0.3:
                        results.append(
                            DiagnosticResult(
                                name=f"Parameter Bias: {col}",
                                status="WARNING",
                                message=f"Failures biased toward {'higher' if bias > 0 else 'lower'} {col}",
                                data={
                                    "failed_mean": float(failed_mean),
                                    "success_mean": float(success_mean),
                                    "bias_ratio": float(bias),
                                },
                            )
                        )

    # Iteration count analysis
    if "iter_count" in df.columns or "iterations" in df.columns:
        iter_col = "iter_count" if "iter_count" in df.columns else "iterations"
        avg_iter = df[iter_col].mean()
        max_iter = df[iter_col].max()

        results.append(
            DiagnosticResult(
                name="Iteration Statistics",
                status="PASS" if avg_iter < 500 else "WARNING",
                message=f"Average iterations: {avg_iter:.0f}, max: {max_iter:.0f}",
                data={
                    "average_iterations": float(avg_iter),
                    "max_iterations": float(max_iter),
                    "std_iterations": float(df[iter_col].std()),
                },
            )
        )

    # Generate HTML report
    summary = {
        "overall_status": "CRITICAL"
        if any(r.status == "CRITICAL" for r in results)
        else "WARNING"
        if any(r.status == "WARNING" for r in results)
        else "PASS",
        "n_diagnostics": len(results),
        "status_counts": {
            "PASS": sum(1 for r in results if r.status == "PASS"),
            "WARNING": sum(1 for r in results if r.status == "WARNING"),
            "CRITICAL": sum(1 for r in results if r.status == "CRITICAL"),
        },
    }

    status_colors = {"PASS": "#28a745", "WARNING": "#ffc107", "CRITICAL": "#dc3545"}

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DOE Diagnostic Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; 
                        color: white; font-weight: bold; }}
        .status-PASS {{ background: {status_colors["PASS"]}; }}
        .status-WARNING {{ background: {status_colors["WARNING"]}; color: #333; }}
        .status-CRITICAL {{ background: {status_colors["CRITICAL"]}; }}
        .diagnostic {{ background: white; margin: 15px 0; padding: 20px; border-radius: 8px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .diagnostic-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .diagnostic-data {{ background: #f8f9fa; padding: 15px; border-radius: 4px; 
                           margin-top: 15px; font-family: monospace; font-size: 13px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 DOE Diagnostic Report</h1>
    <p>Source: {os.path.basename(csv_path)}</p>
    
    <div class="summary">
        <div class="summary-card">
            <h3>Overall Status</h3>
            <span class="status-badge status-{summary["overall_status"]}">{summary["overall_status"]}</span>
        </div>
        <div class="summary-card">
            <h3>Diagnostics</h3>
            <p>✅ {summary["status_counts"]["PASS"]} | ⚠️ {summary["status_counts"]["WARNING"]} | ❌ {summary["status_counts"]["CRITICAL"]}</p>
        </div>
    </div>
    
    <h2>Details</h2>
"""

    for r in results:
        data_html = (
            f"<div class='diagnostic-data'><pre>{json.dumps(r.data, indent=2, default=str)}</pre></div>"
            if r.data
            else ""
        )
        html += f"""
    <div class="diagnostic">
        <div class="diagnostic-header">
            <h3>{r.name}</h3>
            <span class="status-badge status-{r.status}">{r.status}</span>
        </div>
        <p>{r.message}</p>
        {data_html}
    </div>
"""

    html += """
</div>
</body>
</html>
"""

    # Write report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(
        output_dir, f"doe_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    with open(report_path, "w") as f:
        f.write(html)

    return report_path


if __name__ == "__main__":
    # Example usage
    from thermo.nlp import build_thermo_nlp

    print("Building NLP...")
    nlp_tuple = build_thermo_nlp(n_coll=20, Q_total=5000.0, omega_val=300.0, p_int=1e5)
    nlp_dict, meta = nlp_tuple

    print("Running diagnostics...")
    report = diagnose_nlp(nlp_dict, meta, "out/diagnostics", "phase1_test")
    print(f"Report generated: {report}")
