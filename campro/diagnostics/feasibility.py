from __future__ import annotations

from dataclasses import dataclass, field

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class FeasibilityReport:
    feasible: bool
    max_violation: float
    violations: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


def _pair_ok(bounds: dict[str, tuple[float, float]], key: str) -> float:
    try:
        lb, ub = bounds[key]
        return 0.0 if float(lb) < float(ub) else abs(float(lb) - float(ub)) + 1.0
    except Exception:
        return 0.0


def check_feasibility(constraints: dict, bounds: dict) -> FeasibilityReport:
    """Phase-0 feasibility check using fast heuristics.

    The goal is to detect obvious inconsistencies early, before building the NLP.
    This is intentionally lightweight and conservative.
    """
    violations: dict[str, float] = {}
    recs: list[str] = []

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
            a_req = 4.0 * stroke / (up_time**2)
            a_max = bounds.get("max_acceleration")
            if isinstance(a_max, (int, float)):
                if a_req > float(a_max):
                    violations["acceleration_requirement"] = a_req - float(a_max)
                    recs.append(
                        "Increase max_acceleration or upstroke duration/cycle time",
                    )

            # Rough jerk estimate for bang-bang acceleration
            j_req = 8.0 * stroke / (up_time**3)
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
            except Exception as e:
                log.debug(f"Skipping constraint pair {key} due to error: {e}")
                continue

    max_violation = max(violations.values()) if violations else 0.0
    feasible = max_violation == 0.0

    return FeasibilityReport(
        feasible=feasible,
        max_violation=max_violation,
        violations=violations,
        recommendations=recs,
    )


def check_feasibility_nlp(constraints: dict, bounds: dict) -> FeasibilityReport:
    """Phase-0 feasibility via slack-minimization NLP using CasADi/Ipopt.

    All computations use per-degree units:
    - Variables: position samples x[i] on a uniform grid over θ∈[0, duration_angle_deg], plus
      nonnegative slack variables for equalities and inequalities.
    - Objective: minimize sum of squared slacks.
    - Constraints: encode equalities with two-sided slacks and inequalities with
      per-sample nonnegative slack dominating constraint residuals.
    - Velocity, acceleration, jerk are computed in per-degree units (m/deg, m/deg², m/deg³).
    """
    try:
        import casadi as ca  # type: ignore
    except Exception:  # pragma: no cover
        # Fall back to heuristic if CasADi not available
        return check_feasibility(constraints, bounds)

    stroke = float(constraints.get("stroke", 0.0) or 0.0)
    # Use duration_angle_deg if provided, otherwise default to 360° (not 2π radians)
    duration_angle_deg = constraints.get("duration_angle_deg")
    if duration_angle_deg is None:
        # Default to 360° for backward compatibility, but prefer explicit value
        duration_angle_deg = 360.0
    duration_angle_deg = float(duration_angle_deg)
    if duration_angle_deg <= 0:
        # Invalid duration, fall back to heuristic
        return check_feasibility(constraints, bounds)

    up_pct = float(constraints.get("upstroke_percent", 60.0) or 0.0)

    # Bounds (optional) - these are already in per-degree units
    v_max = bounds.get("max_velocity")
    a_max = bounds.get("max_acceleration")
    j_max = bounds.get("max_jerk")
    v_max = float(v_max) if isinstance(v_max, (int, float)) else None
    a_max = float(a_max) if isinstance(a_max, (int, float)) else None
    j_max = float(j_max) if isinstance(j_max, (int, float)) else None

    # Problem size
    N = 72
    # Use degrees instead of radians
    theta_deg = [duration_angle_deg * i / (N - 1) for i in range(N)]
    dtheta_deg = duration_angle_deg / (N - 1)  # degrees per step
    idx = list(range(N))

    # Decision variables: x (position samples)
    x = ca.SX.sym("x", N)

    # Equalities: x(0)=0, x(θ_up)=stroke, x(duration_angle_deg)=0
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
    def ip(i: int) -> int:
        return (i + 1) % N

    def im(i: int) -> int:
        return (i - 1) % N

    def ipp(i: int) -> int:
        return (i + 2) % N

    def imm(i: int) -> int:
        return (i - 2) % N

    # Finite differences in per-degree units (periodic)
    # v = dx/dθ (m/deg), a = d²x/dθ² (m/deg²), j = d³x/dθ³ (m/deg³)
    v = ca.SX.zeros(N, 1)
    a = ca.SX.zeros(N, 1)
    j = ca.SX.zeros(N, 1) if j_max is not None else None
    for i in idx:
        # Central difference for velocity: v[i] = (x[i+1] - x[i-1]) / (2 * dtheta_deg)
        v[i] = (x[ip(i)] - x[im(i)]) / (2.0 * dtheta_deg)
        # Second derivative for acceleration: a[i] = (x[i+1] - 2*x[i] + x[i-1]) / (dtheta_deg²)
        a[i] = (x[ip(i)] - 2.0 * x[i] + x[im(i)]) / (dtheta_deg * dtheta_deg)
        if j is not None:
            # Third derivative central difference (approximate)
            j[i] = (x[ipp(i)] - 2.0 * x[ip(i)] + 2.0 * x[im(i)] - x[imm(i)]) / (
                2.0 * (dtheta_deg**3)
            )

    g_list: list[ca.SX] = []
    lbg: list[float] = []
    ubg: list[float] = []

    # Equality constraints with two-sided slack: -s <= e(x) <= s
    # x(0)=0, x(θ_up)=stroke, x(duration_angle_deg)=0
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
    # Bounds are already in per-degree units, so direct comparison
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

    # Initial guess: S-curve profile over degrees that respects all boundary conditions
    # x[0] = 0, x[idx_up] = stroke, x[N-1] = 0
    theta_np = np.linspace(0.0, duration_angle_deg, N)
    x0 = np.zeros(N)

    # Use 5th-order smoothstep S-curve (same as actual optimizer)
    # Smoothstep: 10*t³ - 15*t⁴ + 6*t⁵
    def smoothstep(t: np.ndarray) -> np.ndarray:
        """5th-order smoothstep function (0≤t≤1)."""
        t = np.clip(t, 0.0, 1.0)
        return 10 * t**3 - 15 * t**4 + 6 * t**5

    # Normal case: upstroke then downstroke
    # Boundary conditions: x[0] = 0, x[idx_up] = stroke, x[N-1] = 0
    if idx_up > 0 and idx_up < N - 1:
        # Upstroke phase: 0 to stroke (from index 0 to idx_up)
        upstroke_indices = np.arange(idx_up + 1)
        up_phase = upstroke_indices / idx_up
        x0[: idx_up + 1] = stroke * smoothstep(up_phase)

        # Downstroke phase: stroke to 0 (from index idx_up to N-1)
        downstroke_indices = np.arange(idx_up, N)
        down_phase = (downstroke_indices - idx_up) / (N - 1 - idx_up)
        x0[idx_up:] = stroke * (1.0 - smoothstep(down_phase))
    elif idx_up == 0:
        # Edge case: upstroke at start (invalid for full cycle, but handle gracefully)
        # Start at stroke, then downstroke to 0
        if N > 1:
            downstroke_indices = np.arange(N)
            down_phase = downstroke_indices / (N - 1)
            x0[:] = stroke * (1.0 - smoothstep(down_phase))
        x0[0] = stroke  # Upstroke point
        # Note: x[0] = 0 requirement will be violated, but that's expected for this edge case
    elif idx_up == N - 1:
        # Edge case: upstroke at end (invalid for full cycle, but handle gracefully)
        # Upstroke to stroke at end
        upstroke_indices = np.arange(N)
        up_phase = upstroke_indices / max(idx_up, 1)
        x0[:] = stroke * smoothstep(up_phase)
        x0[N - 1] = stroke  # Upstroke point
        # Note: x[N-1] = 0 requirement will be violated, but that's expected for this edge case

    # Ensure boundary conditions
    x0[0] = 0.0  # Start at BDC
    if 0 < idx_up < N - 1:
        x0[idx_up] = stroke  # TDC at upstroke point (only if valid)
    x0[N - 1] = 0.0  # End at BDC (full cycle)

    z0 = np.zeros(total)
    z0[:n_x] = x0

    # Set up solver
    from campro.optimization.solvers.ipopt_factory import create_ipopt_solver

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

        res = solver(
            x0=z0,
            lbx=lbx,
            ubx=ubx,
            lbg=np.array(lbg),
            ubg=np.array(ubg),
            **warm_args,
        )
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
        violations: dict[str, float] = {}
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

        recs: list[str] = []
        # Basic recommendation based on dominant violation
        if not feasible:
            dominant = max(violations, key=lambda k: violations[k])
            if dominant in ("v_max",):
                recs.append("Increase max_velocity or upstroke duration/duration_angle_deg")
            elif dominant in ("a_max",):
                recs.append("Increase max_acceleration or upstroke duration/duration_angle_deg")
            elif dominant in ("j_max",):
                recs.append("Increase max_jerk or upstroke duration/duration_angle_deg")
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
