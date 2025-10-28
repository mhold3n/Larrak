import types

from campro.api.adapters import (
    motion_result_to_solve_report,
    unified_data_to_solve_report,
)


def test_motion_result_to_solve_report_smoke(tmp_path):
    # Minimal fake OptimizationResult
    res = types.SimpleNamespace(
        status=types.SimpleNamespace(value="converged"),
        convergence_info={"primal_inf": 0.0, "dual_inf": 0.0},
        iterations=42,
    )
    report = motion_result_to_solve_report(res)
    assert report.status in {"Solve_Success", "Failed"}
    assert isinstance(report.artifacts, dict)


def test_unified_data_to_solve_report_smoke():
    data = types.SimpleNamespace(
        convergence_info={
            "primary": {"iterations": 10, "status": "converged"},
            "secondary": {"iterations": 5, "status": "converged"},
        },
        primary_ipopt_analysis=None,
        secondary_ipopt_analysis=None,
        tertiary_ipopt_analysis=None,
    )
    report = unified_data_to_solve_report(data)  # type: ignore[arg-type]
    assert report.n_iter >= 0
    assert isinstance(report.residuals, dict)
