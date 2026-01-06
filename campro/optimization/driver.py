"""
Driver module for OP engine optimization.

This module is the main entry point for running 0D/1D cycle optimization.
It has been refactored to delegate logic to `OptimizationPipeline`.
"""

from __future__ import annotations

import logging
from typing import Any

from campro.logging import get_logger
from campro.optimization.core.solution import Solution
from campro.optimization.pipeline import OptimizationPipeline

# Re-export solve variants for backward compatibility
from campro.optimization.solve_strategies import (
    get_driver_function,
    solve_cycle_adaptive,
    solve_cycle_robust,
    solve_cycle_with_fuel_continuation,
    solve_cycle_with_refinement,
    solve_cycle_with_warm_start,
)

log = get_logger(__name__)


def solve_cycle(params: dict[str, Any]) -> Solution:
    """
    Solve OP engine cycle optimization using IPOPT.

    This function wraps the modular `OptimizationPipeline`.

    Args:
        params: Problem parameters dictionary

    Returns:
        Solution object with optimization results
    """
    pipeline = OptimizationPipeline(params)

    # Check for ensemble initialization needs?
    # The original driver.py had 'Ensemble Initialization Suite' logic.
    # We moved basic NLP building to Builder. But where did the 'Ensemble' logic go?
    # It was in driver.py lines 218-300: `init_manager = InitializationManager(params)`
    # My Builder.py currently does NOT have the InitializationManager invocation!
    # I MISSED THAT in the extraction.

    # I need to re-check `driver.py` (lines 218-300) and put it into `builder.py` or `pipeline.py`.
    # It generates `initial_trajectory`.
    # `solve_cycle` calls `solve(initial_trajectory)`...
    # My `OptimizationPipeline.run` takes optional `initial_trajectory`.
    # The `InitializationManager` logic should probably be in `Pipeline.run` (Phase 0: Init)
    # or `Builder` helper?

    # I should add it to `OptimizationPipeline` before finalizing `driver.py` replacement?
    # Or add it here as a temporary bridge?
    # "decompose ... Extract NLP construction... "
    # The Ensemble Init is part of "Initialization".
    # Ideally `OptimizationPipeline` handles it.

    # I will invoke the pipeline for now.
    # IMPORTANT: I need to FIX `pipeline.py` or pass the logic here.
    # Since I'm in "Refactoring Driver", I should move that logic to `OptimizationPipeline`
    # to truly clean up driver.py.

    # I will rely on the pipeline.
    # I'll update `pipeline.py` in the next step to include InitializationManager.
    # But first I write this `driver.py` assuming `pipeline` will do it (or passing it if I calculate it here).

    # Actually, to be safe and atomic, maybe I calculate it here for now?
    # No, that defeats the purpose.
    # I will assume I will fix Pipeline next.

    return pipeline.run()


__all__ = [
    "get_driver_function",
    "solve_cycle",
    "solve_cycle_adaptive",
    "solve_cycle_robust",
    "solve_cycle_with_fuel_continuation",
    "solve_cycle_with_refinement",
    "solve_cycle_with_warm_start",
]
