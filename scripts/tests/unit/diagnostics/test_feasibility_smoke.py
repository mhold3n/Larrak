import pytest


def _has_casadi():
    try:
        import casadi as _  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skip(reason="Segfaults on macOS due to CasADi instability")
def test_feasibility_easy_spec_near_zero():
    from campro.diagnostics.feasibility import check_feasibility_nlp

    constraints = {
        "stroke": 10.0,
        "cycle_time": 1.0,
        "upstroke_percent": 50.0,
    }
    bounds = {
        "max_velocity": 1000.0,
        "max_acceleration": 1000.0,
        "max_jerk": 1000.0,
    }

    rep = check_feasibility_nlp(constraints, bounds)

    # Allow modest numerical residual due to FD and finite slacks
    assert rep.max_violation <= 1e-3


@pytest.mark.skip(reason="Segfaults on macOS due to CasADi instability")
def test_ipopt_progress_smoke_under_100_iters(tmp_path, monkeypatch):
    # Force runs dir to temporary path if environment variable is used; otherwise rely on default
    from pathlib import Path

    from campro.diagnostics.feasibility import check_feasibility_nlp
    from campro.logging import get_logger
    from campro.optimization.solvers.ipopt_log_parser import parse_ipopt_log_file

    log = get_logger(__name__)

    # Use a moderately sized, feasible problem
    constraints = {
        "stroke": 20.0,
        "cycle_time": 1.0,
        "upstroke_percent": 60.0,
    }
    bounds = {
        "max_velocity": 200.0,
        "max_acceleration": 800.0,
        "max_jerk": 2000.0,
    }

    rep = check_feasibility_nlp(constraints, bounds)
    assert rep.max_violation <= 1e-3

    # Pick the newest Ipopt log in runs/
    runs_dir = Path("runs")
    logs = sorted(runs_dir.glob("*-ipopt.log"), key=lambda p: p.stat().st_mtime)
    assert logs, "Ipopt log was not created"
    log_path = logs[-1]
    stats_obj = parse_ipopt_log_file(log_path)
    stats = stats_obj.as_dict()
    # Parse the iter table to estimate progress
    text = log_path.read_text(errors="ignore").splitlines()
    rows = []
    for line in text:
        s = line.strip()
        if not s or not s[0].isdigit():
            continue
        parts = s.split()
        # iter, objective, inf_pr, inf_du are the first 4 columns in Ipopt default table
        if len(parts) >= 4:
            try:
                it = int(parts[0])
                inf_pr = float(parts[2])
                inf_du = float(parts[3])
                rows.append((it, inf_pr, inf_du))
            except Exception as e:
                log.debug(f"Skipping line due to parsing error: {e}")
                continue

    # Expect a reasonable number of iterations and strong reduction in dual infeasibility
    assert rows, "No iteration table rows parsed from Ipopt log"
    first_it, _, first_du = rows[0]
    last_it, _, last_du = rows[-1]
    assert last_it <= 120
    if last_du > 0:
        assert (first_du / last_du) >= 10.0
    else:
        # Reached machine zero — treat as huge reduction
        assert True
