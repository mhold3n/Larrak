"""
Optimization Builder Module.

This module handles the construction of the Nonlinear Programming (NLP) problem,
including bounds setup and scaling. It effectively decouples the "Build" phase
from the "Solve" phase of the optimization pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import casadi as ca
import numpy as np

from campro.logging import get_logger
from campro.optimization.initialization.setup import setup_optimization_bounds
from campro.optimization.nlp import build_collocation_nlp
from campro.optimization.nlp.scaling import compute_unified_data_driven_scaling
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)


class OptimizationBuilder:
    """Handles NLP construction, bounds setup, and scaling."""

    def __init__(self, params: dict[str, Any], reporter: StructuredReporter | None = None):
        """
        Initialize the builder.

        Args:
            params: Optimization parameters dictionary.
            reporter: Optional reporter for structured logging.
        """
        self.params = params
        self.reporter = reporter or StructuredReporter(
            context="BUILDER", logger=log, stream_out=None, stream_err=None
        )

    def build_nlp(
        self, initial_trajectory: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Build the Collocation NLP.

        Args:
            initial_trajectory: Optional initial trajectory guess.

        Returns:
            Tuple of (nlp, meta) dictionary.
        """
        num_intervals = int(self.params.get("num", {}).get("K", 10))
        poly_degree = int(self.params.get("num", {}).get("C", 3))
        use_combustion = bool(self.params.get("combustion", {}).get("use_integrated_model", False))

        self.reporter.info(
            f"Building NLP: K={num_intervals}, C={poly_degree}, combustion={use_combustion}"
        )

        nlp, meta = build_collocation_nlp(self.params, initial_trajectory=initial_trajectory)

        n_vars = meta.get("n_vars", 0)
        n_constraints = meta.get("n_constraints", 0)
        self.reporter.info(f"NLP Built: n_vars={n_vars}, n_constraints={n_constraints}")

        return nlp, meta

    def setup_bounds(
        self, nlp: dict[str, Any], meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Set up optimization bounds and initial guess.

        Args:
            nlp: The NLP dictionary.
            meta: The metadata dictionary.

        Returns:
            Tuple of (x0, lbx, ubx, lbg, ubg, p).
        """
        n_vars = meta.get("n_vars", 0)
        n_constraints = meta.get("n_constraints", 0)

        # Verify dimensions if meta is missing or zero (fallback logic from driver)
        if n_vars == 0 or n_constraints == 0:
            if "x" in nlp:
                n_vars = nlp["x"].shape[0] if hasattr(nlp["x"], "shape") else nlp["x"].size1()
            if "g" in nlp:
                n_constraints = (
                    nlp["g"].shape[0] if hasattr(nlp["g"], "shape") else nlp["g"].size1()
                )

        self.reporter.info("Setting up optimization bounds...")
        warm_start = self.params.get("warm_start", {})

        x0, lbx, ubx, lbg, ubg, p = setup_optimization_bounds(
            n_vars,
            n_constraints,
            self.params,
            builder=None,  # Legacy argument, can be explicit if needed
            warm_start=warm_start,
            meta=meta,
        )

        if x0 is None or lbx is None or ubx is None:
            raise ValueError("Failed to set up optimization bounds and initial guess")

        return x0, lbx, ubx, lbg, ubg, p

    def compute_scaling(
        self,
        nlp: dict[str, Any],
        meta: dict[str, Any],
        x0: np.ndarray,
        lbx: np.ndarray,
        ubx: np.ndarray,
        lbg: np.ndarray,
        ubg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute variable, constraint, and objective scaling factors.

        Args:
            nlp: NLP dictionary.
            meta: Metadata dictionary.
            x0: Initial guess.
            lbx: Lower variable bounds.
            ubx: Upper variable bounds.
            lbg: Lower constraint bounds.
            ubg: Upper constraint bounds.

        Returns:
            Tuple of (scale, scale_g, scale_f).
        """
        variable_groups = meta.get("variable_groups", {})

        self.reporter.info("Computing unified data-driven scaling...")

        scale, scale_g, _, scaling_quality = compute_unified_data_driven_scaling(
            nlp,
            x0,
            lbx,
            ubx,
            lbg,
            ubg,
            variable_groups=variable_groups,
            meta=meta,
            reporter=self.reporter,
        )

        # Log quality
        cond = scaling_quality.get("condition_number", np.inf)
        score = scaling_quality.get("quality_score", 0.0)
        self.reporter.info(f"Scaling Quality: condition_number={cond:.2e}, score={score:.2f}")

        # Compute Objective Scaling (Global Gradient Based)
        scale_f = self._compute_objective_scaling(nlp, x0, scale)

        return scale, scale_g, scale_f

    def _compute_objective_scaling(
        self, nlp: dict[str, Any], x0: np.ndarray, scale: np.ndarray
    ) -> float:
        """
        Compute global objective scaling factor based on gradient magnitude.
        """
        try:
            if "f" in nlp and "x" in nlp:
                f_expr = nlp["f"]
                x_sym = nlp["x"]
                grad_f_expr = ca.gradient(f_expr, x_sym)
                grad_f_func = ca.Function("grad_f_func", [x_sym], [grad_f_expr])

                # Gradient at unscaled x0
                grad_f0 = np.array(grad_f_func(x0)).flatten()

                # Scaled gradient approximation: g_tilde = grad_f * scale
                g_tilde = grad_f0 * scale
                g_max = np.max(np.abs(g_tilde))

                target_grad = 10.0
                g_min_threshold = 1e-2

                raw_w0 = target_grad / max(g_max, g_min_threshold)
                scale_f = float(np.clip(raw_w0, 1e-3, 1e3))

                self.reporter.info(f"Objective Scaling: g_max={g_max:.2e}, scale_f={scale_f:.2e}")
                return scale_f
        except Exception as e:
            self.reporter.warning(f"Failed to compute objective scaling: {e}")

        return 1.0
