from __future__ import annotations

"""Central facility for constructing Ipopt/CasADi solver options.

This module converts internal :class:`campro.freepiston.opt.ipopt_solver.IPOPTOptions`
into the dictionary expected by CasADi's ``nlpsol`` and (optionally) writes an
``ipopt.opt`` file so Ipopt can read *run-time* parameters not recognised at
creation time.

Design goals
------------
1. Single source of truth for *all* Ipopt parameters used in the project.
2. Automatic handling of solver-specific defaults (MA27 vs MA57).
3. Optional emission of an ``ipopt.opt`` file at the path configured in
   :pydata:`campro.constants.IPOPT_OPT_PATH`.
4. No external side-effects unless ``emit_file=True``.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Final

from campro.constants import HSLLIB_PATH, IPOPT_OPT_PATH
from campro.logging import get_logger
from campro.optimization.solver_selection import SolverType

log = get_logger(__name__)

# -- Public API -------------------------------------------------------------


def build_casadi_options(
    ipopt_options: "IPOPTOptions",  # type: ignore[name-defined]
    solver: SolverType,
    *,
    emit_file: bool = True,
) -> Dict[str, Any]:
    """Return CasADi options dict and optionally write *ipopt.opt* file.

    Parameters
    ----------
    ipopt_options
        Options dataclass from freepiston layer (or compatible subset).
    solver
        Selected linear solver (MA27, MA57, ...).
    emit_file
        When *True* (default) write ``ipopt.opt`` file alongside returning the
        options dict.  Existing files are overwritten.
    """

    opts: Dict[str, Any] = {}

    # Map common dataclass fields to ipopt.* keys – prefer explicit list to
    # avoid leaking unwanted attributes.
    _DIRECT_MAP: Final = {
        "max_iter": "ipopt.max_iter",
        "max_cpu_time": "ipopt.max_cpu_time",
        "tol": "ipopt.tol",
        "acceptable_tol": "ipopt.acceptable_tol",
        "acceptable_iter": "ipopt.acceptable_iter",
        "mu_strategy": "ipopt.mu_strategy",
        "mu_init": "ipopt.mu_init",
        "mu_max": "ipopt.mu_max",
        "mu_min": "ipopt.mu_min",
        "line_search_method": "ipopt.line_search_method",
        "dual_inf_tol": "ipopt.dual_inf_tol",
        "compl_inf_tol": "ipopt.compl_inf_tol",
        "constr_viol_tol": "ipopt.constr_viol_tol",
        "print_level": "ipopt.print_level",
        "print_frequency_iter": "ipopt.print_frequency_iter",
        "print_frequency_time": "ipopt.print_frequency_time",
        "hessian_approximation": "ipopt.hessian_approximation",
        "limited_memory_max_history": "ipopt.limited_memory_max_history",
        "limited_memory_update_type": "ipopt.limited_memory_update_type",
    }

    data = asdict(ipopt_options)
    for field, key in _DIRECT_MAP.items():
        if field in data and data[field] is not None:
            opts[key] = data[field]

    # Creation-time linear solver + hsllib (must be present before reading file)
    opts["ipopt.linear_solver"] = solver.value
    opts["ipopt.hsllib"] = HSLLIB_PATH

    # Warm-start params (if enabled)
    if data.get("warm_start_init_point", "no") == "yes":
        opts["ipopt.warm_start_init_point"] = data["warm_start_init_point"]
        opts["ipopt.warm_start_bound_push"] = data.get("warm_start_bound_push", 1e-6)
        opts["ipopt.warm_start_mult_bound_push"] = data.get(
            "warm_start_mult_bound_push", 1e-6
        )

    # Add user-supplied linear_solver_options dict verbatim (prefixed keys)
    ls_opts = data.get("linear_solver_options") or {}
    for k, v in ls_opts.items():
        opts[f"ipopt.{k}"] = v

    # Emit option file with *run-time* parameters (Ipopt expects bare keys)
    if emit_file:
        _write_option_file(opts)
        opts["ipopt.option_file_name"] = IPOPT_OPT_PATH

    return opts


# -- Private helpers --------------------------------------------------------


def _write_option_file(opts: Dict[str, Any]) -> None:
    """Write ``ipopt.opt`` file from *opts*.

    Only ``ipopt.*`` keys are written; the prefix is stripped.
    """

    path = Path(IPOPT_OPT_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for key, value in sorted(opts.items()):
                if not key.startswith("ipopt."):
                    continue
                bare_key = key.partition("ipopt.")[2]
                fh.write(f"{bare_key} {value}\n")
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Failed to write Ipopt option file %s: %s", path, exc)
        raise
