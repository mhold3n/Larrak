"""Driver for Phase 1 Thermodynamic Optimization.

This module orchestrates the solving of the Thermo NLP.
"""

from __future__ import annotations

import time
from typing import Any

import casadi as ca
import numpy as np

from campro.logging import get_logger
from thermo.nlp import build_thermo_nlp

log = get_logger(__name__)


def solve_thermo_cycle(
    n_coll: int = 50,
    max_iter: int = 200,
    **kwargs,
) -> dict[str, Any]:
    """Solve the Phase 1 Thermodynamic Cycle.

    Args:
        n_coll: Number of collocation intervals
        max_iter: Maximum IPOPT iterations
        **kwargs: Hardware parameters

    Returns:
        Solution dictionary
    """
    log.info("Building Thermo NLP...")

    start_time = time.time()
    # build_thermo_nlp now returns (nlp, meta) via export_nlp()
    nlp_data, meta = build_thermo_nlp(n_coll=n_coll, **kwargs)
    build_time = time.time() - start_time
    log.info(f"NLP built in {build_time:.3f}s")

    # Solver Options
    opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.print_level": 5,
        "ipopt.tol": 1e-4,
        "ipopt.mu_strategy": "adaptive",
        # "ipopt.linear_solver": "ma57", # Try enabling if safe
    }

    # Create Solver
    # export_nlp returns 'nlp' dict expected by ca.nlpsol
    solver = ca.nlpsol("solver", "ipopt", nlp_data, opts)

    # Solve
    log.info("Solving NLP...")
    solve_start = time.time()
    # meta has keys: w0, lbw, ubw, lbg, ubg
    res = solver(
        x0=meta["w0"],
        lbx=meta["lbw"],
        ubx=meta["ubw"],
        lbg=meta["lbg"],
        ubg=meta["ubg"],
    )
    solve_time = time.time() - solve_start
    log.info(f"Solve complete in {solve_time:.3f}s")

    success = solver.stats()["success"]
    log.info(f"Solver success: {success}, Objective: {res['f']}")

    return {
        "success": success,
        "objective": float(res["f"]),
        "x_opt": np.array(res["x"]).flatten(),
        "stats": solver.stats(),
        # TODO: Add full trajectory unpacking helper
    }
