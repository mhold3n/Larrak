import textwrap

from campro.optimization.ipopt_log_parser import parse_ipopt_log_text


def _sample_log(
    exit_status: str = "Optimal Solution Found", ls_ratio: float = 0.42,
) -> str:
    """Return minimal synthetic Ipopt log covering key metrics."""
    return textwrap.dedent(
        f"""
        EXIT: {exit_status}
        Number of Iterations....: 35

        Total CPU secs in IPOPT (w/o function evaluations)   =      1.234
        Time for linear solve                                  =      0.519
        Linear solve time ratio = {ls_ratio}

        Primal infeasibility      =  1.0e-08
        Dual infeasibility        =  3.0e-07
        Complementarity           =  2.0e-09

        WARNING: Small pivot in MA27! (my synthetic warning): 1
        Restoration phase start
        Restoration phase start

        Number of Iterative Refinements: 4
        """,
    )


def test_parse_ipopt_log_text_basic():
    text = _sample_log()
    stats = parse_ipopt_log_text(text)

    assert stats.status == "Optimal Solution Found"
    assert stats.n_iterations == 35
    assert abs(stats.cpu_time - 1.234) < 1e-6
    assert abs(stats.ls_time - 0.519) < 1e-6
    assert abs(stats.ls_time_ratio - 0.42) < 1e-6

    # Derived counts
    assert stats.restoration_count == 2
    assert stats.small_pivot_warnings == 1
    assert stats.refactorizations == 4

    # Infeasibilities
    assert abs(stats.primal_inf - 1.0e-08) < 1e-12
    assert abs(stats.dual_inf - 3.0e-07) < 1e-12
    assert abs(stats.compl_inf - 2.0e-09) < 1e-12
