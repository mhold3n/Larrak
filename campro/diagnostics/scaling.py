from __future__ import annotations

from typing import Any, Union

import numpy as np

Number = Union[int, float]
ArrayLike = Union[Number, np.ndarray]


def compute_scaling_vector(bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Compute variable scaling so typical magnitudes are O(1).

    For each variable, uses the max(|lb|, |ub|) to compute a scale factor s
    such that the scaled variable z = s * x has magnitude ~O(1).
    If the magnitude is too small, returns 1.0 to avoid blow-ups.
    """
    scales: dict[str, float] = {}
    for var, lr in bounds.items():
        if not isinstance(lr, (tuple, list)) or len(lr) != 2:
            continue
        lb, ub = float(lr[0]), float(lr[1])
        magnitude = max(abs(lb), abs(ub))
        scales[var] = (1.0 / magnitude) if magnitude > 1e-6 else 1.0
    return scales


# ---- Numeric helpers (NumPy) -------------------------------------------------


def scale_value(value: ArrayLike, scale: ArrayLike) -> ArrayLike:
    """Return scaled value z = s * x (NumPy path).

    Works with scalars or arrays; broadcasting rules apply.
    """
    return np.asarray(value) * np.asarray(scale)


def unscale_value(value: ArrayLike, scale: ArrayLike) -> ArrayLike:
    """Return unscaled value x = z / s (NumPy path)."""
    return np.asarray(value) / np.asarray(scale)


def scale_bounds(
    bounds: tuple[ArrayLike, ArrayLike], scale: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale lower/upper bounds by factor s (z = s*x).

    Returns new (lb_scaled, ub_scaled).
    """
    lb, ub = bounds
    return scale_value(lb, scale), scale_value(ub, scale)


def scale_dict(
    values: dict[str, ArrayLike], scales: dict[str, ArrayLike],
) -> dict[str, np.ndarray]:
    """Elementwise scale a dict of arrays by matching keys in scales."""
    out: dict[str, np.ndarray] = {}
    for k, v in values.items():
        s = scales.get(k, 1.0)
        out[k] = scale_value(v, s)
    return out


def unscale_dict(
    values: dict[str, ArrayLike], scales: dict[str, ArrayLike],
) -> dict[str, np.ndarray]:
    """Elementwise unscale a dict of arrays by matching keys in scales."""
    out: dict[str, np.ndarray] = {}
    for k, v in values.items():
        s = scales.get(k, 1.0)
        out[k] = unscale_value(v, s)
    return out


# ---- CasADi helpers (stubs – safe to import lazily) -------------------------


def _import_casadi():  # pragma: no cover - lightweight lazy import
    try:
        import casadi as ca  # type: ignore

        return ca
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("CasADi is required for symbol scaling helpers") from exc


def make_scaled_symbol(
    name: str, shape: tuple[int, ...] | int, scale: ArrayLike, *, kind: str = "MX",
):
    """Create a scaled decision variable for CasADi NLPs.

    Defines z as the decision variable and returns (z, x_expr) where
    x_expr = z / s is the physical variable to use in constraints/objective.

    Parameters
    ----------
    name: base symbol name (e.g., "x")
    shape: int for vector or (n, m) for matrix
    scale: scalar or array-like scale s applied as z = s * x
    kind: 'MX' or 'SX'

    Returns
    -------
    (z, x_expr): tuple of CasADi symbols/expressions
    """
    ca = _import_casadi()

    if isinstance(shape, int):
        n, m = int(shape), 1
    elif isinstance(shape, tuple) and len(shape) == 2:
        n, m = int(shape[0]), int(shape[1])
    else:
        raise ValueError("shape must be int or (n, m)")

    Sym = getattr(ca, kind)
    z = Sym.sym(name, n, m)

    # Broadcast scale to match shape
    if np.isscalar(scale):
        s = float(scale)
        s_expr = ca.DM.full(n, m, s)
    else:
        arr = np.asarray(scale, dtype=float)
        if arr.shape == (n, m):
            s_expr = ca.DM(arr)
        elif arr.shape == (n,) and m == 1:
            s_expr = ca.DM(arr).reshape((n, 1))
        else:
            raise ValueError(
                f"scale shape {arr.shape} incompatible with variable shape ({n}, {m})",
            )

    # Physical variable expression
    x_expr = z / s_expr
    return z, x_expr


def scale_expr(expr: Any, scale: ArrayLike) -> Any:
    """Scale a CasADi expression: z = s * x."""
    ca = _import_casadi()
    if np.isscalar(scale):
        return expr * float(scale)
    arr = ca.DM(np.asarray(scale, dtype=float))
    return expr * arr


def unscale_expr(expr: Any, scale: ArrayLike) -> Any:
    """Unscale a CasADi expression: x = z / s."""
    ca = _import_casadi()
    if np.isscalar(scale):
        return expr / float(scale)
    arr = ca.DM(np.asarray(scale, dtype=float))
    return expr / arr


def build_scaled_nlp(
    nlp: dict[str, Any], scale: ArrayLike, *, kind: str | None = None,
) -> dict[str, Any]:
    """Build a scaled CasADi NLP from an unscaled one.

    Given an unscaled NLP dict {'x': x, 'f': f, 'g': g}, constructs a new
    decision variable z so that x = z / s, and returns a new NLP dict
    {'x': z, 'f': f(z/s), 'g': g(z/s)}. Bounds and initial guesses must be
    scaled separately by the caller (lbz = s*lbx, ubz = s*ubx, z0 = s*x0).

    Parameters
    ----------
    nlp: The original NLP dict using physical variables.
    scale: Scalar or array-like scale factor s.
    kind: Optional override for symbol type ('SX' or 'MX'); inferred from x.
    """
    ca = _import_casadi()
    if "x" not in nlp or "f" not in nlp:
        raise ValueError("nlp must contain keys 'x' and 'f'")
    x = nlp["x"]
    f = nlp["f"]
    g = nlp.get("g", ca.DM([]))

    # Infer symbol type and shape
    sym_kind = kind
    if sym_kind is None:
        sym_kind = "SX" if isinstance(x, ca.SX) else "MX"
    shape = getattr(x, "shape", None)
    if not shape:
        # Fallback for older CasADi builds
        n = int(getattr(x, "size1", lambda: 0)())
        m = int(getattr(x, "size2", lambda: 1)())
        shape = (n, m)

    # Create scaled variable and physical expression
    z, x_expr = make_scaled_symbol("z", shape, scale, kind=sym_kind)

    # Substitute x -> (z/s) in f and g
    f_scaled = ca.substitute(f, x, x_expr)
    g_scaled = ca.substitute(g, x, x_expr) if g is not None else g

    return {"x": z, "f": f_scaled, "g": g_scaled}


def solve_scaled_nlpsol(
    name: str,
    nlp: dict[str, Any],
    scale: ArrayLike,
    *,
    x0: ArrayLike | None = None,
    lbx: ArrayLike | None = None,
    ubx: ArrayLike | None = None,
    lbg: ArrayLike | None = None,
    ubg: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
    linear_solver: str | None = None,
):
    """Create a scaled Ipopt solver and scaled argument dict ready to call.

    Returns (solver, kwargs) where solver is a CasADi nlpsol and kwargs is a
    dict containing scaled x0/lbx/ubx/lbg/ubg suitable for passing to solver().
    """
    ca = _import_casadi()
    from campro.optimization.ipopt_factory import create_ipopt_solver

    nlp_scaled = build_scaled_nlp(nlp, scale)
    solver = create_ipopt_solver(
        name, nlp_scaled, options or {}, linear_solver=linear_solver,
    )

    kwargs: dict[str, Any] = {}
    if x0 is not None:
        kwargs["x0"] = scale_value(x0, scale)
    if lbx is not None and ubx is not None:
        lbz, ubz = scale_bounds((lbx, ubx), scale)
        kwargs["lbx"] = lbz
        kwargs["ubx"] = ubz
    elif lbx is not None:
        kwargs["lbx"] = scale_value(lbx, scale)
    elif ubx is not None:
        kwargs["ubx"] = scale_value(ubx, scale)
    if lbg is not None:
        kwargs["lbg"] = lbg
    if ubg is not None:
        kwargs["ubg"] = ubg

    return solver, kwargs
