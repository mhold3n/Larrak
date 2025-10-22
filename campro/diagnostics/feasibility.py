from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FeasibilityReport:
    feasible: bool
    max_violation: float
    violations: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


def _pair_ok(bounds: Dict[str, Tuple[float, float]], key: str) -> float:
    try:
        lb, ub = bounds[key]
        return 0.0 if float(lb) < float(ub) else abs(float(lb) - float(ub)) + 1.0
    except Exception:
        return 0.0


def check_feasibility(constraints: Dict, bounds: Dict) -> FeasibilityReport:
    """Phase-0 feasibility check using fast heuristics.

    The goal is to detect obvious inconsistencies early, before building the NLP.
    This is intentionally lightweight and conservative.
    """
    violations: Dict[str, float] = {}
    recs: List[str] = []

    stroke = float(constraints.get("stroke", 0.0) or 0.0)
    cycle_time = float(constraints.get("cycle_time", 0.0) or 0.0)
    up_pct = float(constraints.get("upstroke_percent", 60.0) or 0.0)
    zero_pct = constraints.get("zero_accel_percent")
    zero_pct = float(zero_pct) if zero_pct is not None else None

    # Basic parameter sanity
    if stroke <= 0:
        violations["stroke_positive"] = 1.0
        recs.append("Stroke must be positive")
    if cycle_time <= 0:
        violations["cycle_time_positive"] = 1.0
        recs.append("Cycle time must be positive")
    if not (0.0 <= up_pct <= 100.0):
        violations["upstroke_percent_range"] = abs(up_pct)
        recs.append("Upstroke percent must be in [0, 100]")
    if zero_pct is not None and not (0.0 <= zero_pct <= 100.0):
        violations["zero_accel_percent_range"] = abs(zero_pct)
        recs.append("Zero-accel percent must be in [0, 100]")

    # Velocity / acceleration / jerk requirements (rough heuristics)
    try:
        up_time = cycle_time * up_pct / 100.0
        if up_time > 0 and stroke > 0:
            # Required average velocity for upstroke
            v_req = stroke / up_time
            v_max = bounds.get("max_velocity")
            if isinstance(v_max, (int, float)):
                if v_req > float(v_max):
                    violations["velocity_requirement"] = v_req - float(v_max)
                    recs.append("Increase max_velocity or upstroke duration/cycle time")

            # Simple bound for acceleration: triangular profile
            a_req = 4.0 * stroke / (up_time ** 2)
            a_max = bounds.get("max_acceleration")
            if isinstance(a_max, (int, float)):
                if a_req > float(a_max):
                    violations["acceleration_requirement"] = a_req - float(a_max)
                    recs.append("Increase max_acceleration or upstroke duration/cycle time")

            # Rough jerk estimate for bang-bang acceleration
            j_req = 8.0 * stroke / (up_time ** 3)
            j_max = bounds.get("max_jerk")
            if isinstance(j_max, (int, float)):
                if j_req > float(j_max):
                    violations["jerk_requirement"] = j_req - float(j_max)
                    recs.append("Increase max_jerk or upstroke duration/cycle time")
    except Exception:
        # Ignore heuristic errors
        pass

    # Generic bound ordering checks if provided as pairs under 'pairs'
    pairs = constraints.get("pairs") or {}
    if isinstance(pairs, dict):
        for key, lr in pairs.items():
            try:
                lb, ub = lr
                if float(lb) >= float(ub):
                    violations[f"pair_order:{key}"] = abs(float(lb) - float(ub)) + 1.0
                    recs.append(f"Ensure {key} lower < upper bound")
            except Exception:
                continue

    max_violation = max(violations.values()) if violations else 0.0
    feasible = max_violation == 0.0

    return FeasibilityReport(
        feasible=feasible,
        max_violation=max_violation,
        violations=violations,
        recommendations=recs,
    )


def check_feasibility_nlp(constraints: Dict, bounds: Dict) -> FeasibilityReport:
    """Phase-0 feasibility via slack-minimization NLP using CasADi/Ipopt.

    - Variables: position samples x[i] on a uniform grid over θ∈[0, 2π], plus
      nonnegative slack variables for equalities and inequalities.
    - Objective: minimize sum of squared slacks.
    - Constraints: encode equalities with two-sided slacks and inequalities with
      per-sample nonnegative slack dominating constraint residuals.
    """
    try:
        import casadi as ca  # type: ignore
    except Exception as exc:  # pragma: no cover
        # Fall back to heuristic if CasADi not available
        return check_feasibility(constraints, bounds)

    stroke = float(constraints.get("stroke", 0.0) or 0.0)
    cycle_time = float(constraints.get("cycle_time", 0.0) or 0.0)
    up_pct = float(constraints.get("upstroke_percent", 60.0) or 0.0)

    # Bounds (optional)
    v_max = bounds.get("max_velocity")
    a_max = bounds.get("max_acceleration")
    j_max = bounds.get("max_jerk")
    v_max = float(v_max) if isinstance(v_max, (int, float)) else None
    a_max = float(a_max) if isinstance(a_max, (int, float)) else None
    j_max = float(j_max) if isinstance(j_max, (int, float)) else None

    # Problem size
    N = 72
    theta = [2.0 * ca.pi * i / (N - 1) for i in range(N)]
    dtheta = float(2.0 * 3.141592653589793 / (N - 1))
    idx = list(range(N))

    # Decision variables: x (position samples)
    x = ca.SX.sym("x", N)

    # Equalities: x(0)=0, x(2π)=0, x(θ_up)=stroke
    idx_up = int(round(up_pct / 100.0 * (N - 1)))
    idx_up = max(0, min(N - 1, idx_up))

    # Slack variables (nonnegative)
    s_eq = ca.SX.sym("s_eq", 3)  # for three equalities

    # Inequality slacks per-sample
    s_v = ca.SX.sym("s_v", N) if v_max is not None else ca.SX([])
    s_a = ca.SX.sym("s_a", N) if a_max is not None else ca.SX([])
    s_j = ca.SX.sym("s_j", N) if j_max is not None else ca.SX([])

    # Stack decision vector
    z = ca.vertcat(x, s_eq, s_v, s_a, s_j)

    # Helper for periodic indices
    def ip(i):
        return (i + 1) % N

    def im(i):
        return (i - 1) % N

    def ipp(i):
        return (i + 2) % N

    def imm(i):
        return (i - 2) % N

    # Finite differences (periodic)
    v = ca.SX.zeros(N, 1)
    a = ca.SX.zeros(N, 1)
    j = ca.SX.zeros(N, 1) if j_max is not None else None
    for i in idx:
        v[i] = (x[ip(i)] - x[im(i)]) / (2.0 * dtheta)
        a[i] = (x[ip(i)] - 2.0 * x[i] + x[im(i)]) / (dtheta * dtheta)
        if j is not None:
            # Third derivative central difference (approximate)
            j[i] = (x[ipp(i)] - 2.0 * x[ip(i)] + 2.0 * x[im(i)] - x[imm(i)]) / (
                2.0 * (dtheta ** 3)
            )

    g_list: List[ca.SX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    # Equality constraints with two-sided slack: -s <= e(x) <= s
    e_vals = [x[0] - 0.0, x[idx_up] - stroke, x[N - 1] - 0.0]
    for k, e in enumerate(e_vals):
        # e - s_eq[k] <= 0
        g_list.append(e - s_eq[k])
        lbg.append(-ca.inf)
        ubg.append(0.0)
        # -e - s_eq[k] <= 0
        g_list.append(-e - s_eq[k])
        lbg.append(-ca.inf)
        ubg.append(0.0)

    # Inequalities per-sample with slack dominance
    if v_max is not None:
        for i in idx:
            g_list.append(v[i] - v_max - s_v[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)
            g_list.append(-v[i] - v_max - s_v[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)
    if a_max is not None:
        for i in idx:
            g_list.append(a[i] - a_max - s_a[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)
            g_list.append(-a[i] - a_max - s_a[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)
    if j_max is not None and j is not None:
        for i in idx:
            g_list.append(j[i] - j_max - s_j[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)
            g_list.append(-j[i] - j_max - s_j[i])
            lbg.append(-ca.inf)
            ubg.append(0.0)

    g = ca.vertcat(*g_list) if g_list else ca.SX([])

    # Objective: minimize sum of squared slacks + tiny regularization on x
    obj = ca.sumsqr(s_eq)
    if s_v.numel() > 0:
        obj = obj + ca.sumsqr(s_v)
    if s_a.numel() > 0:
        obj = obj + ca.sumsqr(s_a)
    if s_j.numel() > 0:
        obj = obj + ca.sumsqr(s_j)
    obj = obj + 1e-10 * ca.sumsqr(x)

    nlp = {"x": z, "f": obj, "g": g}

    # Bounds: slacks >= 0, x free
    import numpy as np

    n_x = N
    n_eq = 3
    n_sv = N if v_max is not None else 0
    n_sa = N if a_max is not None else 0
    n_sj = N if j_max is not None else 0

    total = n_x + n_eq + n_sv + n_sa + n_sj
    lbx = -np.inf * np.ones(total)
    ubx = np.inf * np.ones(total)

    # s_eq >= 0
    lbx[n_x : n_x + n_eq] = 0.0
    # s_v >= 0
    start = n_x + n_eq
    if n_sv:
        lbx[start : start + n_sv] = 0.0
    # s_a >= 0
    start += n_sv
    if n_sa:
        lbx[start : start + n_sa] = 0.0
    # s_j >= 0
    start += n_sa
    if n_sj:
        lbx[start : start + n_sj] = 0.0

    # Initial guess: S-curve profile
    theta_np = np.linspace(0.0, 2.0 * np.pi, N)
    x0 = stroke * 0.5 * (1.0 - np.cos(theta_np))
    z0 = np.zeros(total)
    z0[:n_x] = x0

    # Set up solver
    from campro.optimization.ipopt_factory import create_ipopt_solver

    opts = {
        "ipopt.max_iter": 400,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.print_level": 0,
        # Enable warm-start if available
        "ipopt.warm_start_init_point": "yes",
        "ipopt.warm_start_bound_push": 1e-6,
        "ipopt.warm_start_mult_bound_push": 1e-6,
    }
    solver = create_ipopt_solver("feas_phase0", nlp, opts, linear_solver="ma27")

    try:
        # Try to load warm-start from previous run if shapes match
        try:
            from campro.diagnostics.warmstart import load_warmstart, save_warmstart
            warm_args = load_warmstart(total, len(lbg))
        except Exception:
            warm_args = {}

        res = solver(x0=z0, lbx=lbx, ubx=ubx, lbg=np.array(lbg), ubg=np.array(ubg), **warm_args)
        z_opt = np.array(res["x"]).reshape((-1,))
        # Extract slacks
        ptr = n_x
        s_eq_val = z_opt[ptr : ptr + n_eq]
        ptr += n_eq
        s_v_val = z_opt[ptr : ptr + n_sv] if n_sv else np.array([])
        ptr += n_sv
        s_a_val = z_opt[ptr : ptr + n_sa] if n_sa else np.array([])
        ptr += n_sa
        s_j_val = z_opt[ptr : ptr + n_sj] if n_sj else np.array([])

        # Summarize violations
        violations: Dict[str, float] = {}
        eq_names = ["eq_start", "eq_upstroke", "eq_end"]
        for name, v in zip(eq_names, s_eq_val):
            violations[name] = float(abs(v))
        if n_sv:
            violations["v_max"] = float(np.max(s_v_val))
        if n_sa:
            violations["a_max"] = float(np.max(s_a_val))
        if n_sj:
            violations["j_max"] = float(np.max(s_j_val))

        max_violation = max(violations.values()) if violations else 0.0
        feasible = max_violation <= 1e-6

        recs: List[str] = []
        # Basic recommendation based on dominant violation
        if not feasible:
            dominant = max(violations, key=violations.get)
            if dominant in ("v_max",):
                recs.append("Increase max_velocity or upstroke duration/cycle time")
            elif dominant in ("a_max",):
                recs.append("Increase max_acceleration or upstroke duration/cycle time")
            elif dominant in ("j_max",):
                recs.append("Increase max_jerk or upstroke duration/cycle time")
            else:
                recs.append("Relax boundary/ stroke conditions or adjust phases")

        report = FeasibilityReport(
            feasible=feasible,
            max_violation=max_violation,
            violations=violations,
            recommendations=recs,
        )

        # Persist warm-start arrays for future runs
        try:
            lam_g = None
            if "lam_g" in res:
                lam_g = np.array(res["lam_g"]).reshape((-1,))
            save_warmstart(z_opt, lam_g=lam_g, tag="feas")
        except Exception:
            pass

        return report
    except Exception:
        # Fallback to heuristic in case of solver error
        return check_feasibility(constraints, bounds)
