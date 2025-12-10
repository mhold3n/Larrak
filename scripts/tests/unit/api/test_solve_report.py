from campro.api import SolveReport


def test_solve_report_defaults():
    r = SolveReport(run_id="123", status="Solve_Success")
    assert r.run_id == "123"
    assert r.status == "Solve_Success"
    assert isinstance(r.kkt, dict)
    assert isinstance(r.residuals, dict)
    assert isinstance(r.scaling_stats, dict)
    assert isinstance(r.artifacts, dict)
