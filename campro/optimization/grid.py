"""
Universal collocation grid and mapping utilities.

Provides a canonical angular grid over [0, 2π) and helpers to map series
between grids with periodic interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
try:  # Optional SciPy for higher-order mapping options
    from scipy.interpolate import PchipInterpolator, BarycentricInterpolator  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_SCIPY = False


@dataclass(frozen=True)
class GridSpec:
    method: str
    degree: int
    n_points: int
    periodic: bool = True
    # hp-mesh metadata (future-proof):
    family: str = "uniform"  # e.g., uniform, LGL, Radau, Chebyshev
    segments: int = 1         # number of segments if multi-interval


@dataclass
class UniversalGrid:
    n_points: int = 360
    periodic: bool = True

    def __post_init__(self) -> None:
        # Uniform angular grid in radians on [1°, 360°] (π/180 to 2π)
        # This avoids wraparound issues at 0°/360° boundary
        self.theta = np.linspace(np.pi / 180.0, 2.0 * np.pi, int(self.n_points), endpoint=True)

    @property
    def theta_rad(self) -> np.ndarray:
        return self.theta

    @property
    def theta_deg(self) -> np.ndarray:
        return np.degrees(self.theta)


class GridMapper:
    @staticmethod
    def trapz_weights(theta: np.ndarray) -> np.ndarray:
        """Periodic trapezoidal weights on possibly nonuniform theta (radians)."""
        n = len(theta)
        if n == 0:
            return np.array([])
        two_pi = 2.0 * np.pi
        th = np.mod(theta, two_pi)
        ord_ = np.argsort(th)
        th = th[ord_]
        # Append wrap to compute segment lengths
        th_ext = np.concatenate([th, th[:1] + two_pi])
        seg = np.diff(th_ext)
        # Weight at node i = 0.5*(seg_left + seg_right)
        w = 0.5 * (np.roll(seg, 1) + seg)
        # Undo ordering to original layout
        inv = np.empty_like(ord_)
        inv[ord_] = np.arange(n)
        return w[inv]

    @staticmethod
    def fft_derivative(theta: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Spectral derivative for uniform periodic grids (best-effort check)."""
        n = len(theta)
        if n == 0:
            return np.array([])
        # Check uniformity
        d = np.diff(theta)
        if not np.allclose(d, d.mean(), rtol=1e-3, atol=1e-6):
            # Fallback to finite differences
            return np.gradient(values, theta)
        L = 2.0 * np.pi
        k = np.fft.fftfreq(n, d=(L / n) / (2.0 * np.pi))  # cycles per 2π
        k = 1j * k
        vhat = np.fft.fft(values)
        dhat = k * vhat
        return np.real(np.fft.ifft(dhat))
    @staticmethod
    def periodic_linear_resample(
        from_theta: np.ndarray, from_values: np.ndarray, to_theta: np.ndarray,
    ) -> np.ndarray:
        """Resample periodic data defined on from_theta to to_theta using linear wrap-around.

        Assumes from_theta and to_theta are in radians on [0, 2π).
        """
        if len(from_theta) == 0:
            return np.zeros_like(to_theta)

        # Normalize and ensure monotonic increasing
        two_pi = 2.0 * np.pi
        ft = np.mod(from_theta, two_pi)
        order = np.argsort(ft)
        ft = ft[order]
        fv = np.asarray(from_values)[order]

        # Append wrap-around point
        ft_ext = np.concatenate([ft, ft[:1] + two_pi])
        fv_ext = np.concatenate([fv, fv[:1]])

        tt = np.mod(to_theta, two_pi)
        return np.interp(tt, ft_ext, fv_ext)

    @staticmethod
    def periodic_pchip_resample(
        from_theta: np.ndarray, from_values: np.ndarray, to_theta: np.ndarray,
    ) -> np.ndarray:
        """Periodic shape-preserving cubic interpolation.

        Requires SciPy. Falls back to linear if unavailable.
        """
        if not _HAVE_SCIPY:
            return GridMapper.periodic_linear_resample(from_theta, from_values, to_theta)
        two_pi = 2.0 * np.pi
        ft = np.mod(from_theta, two_pi)
        ord_ = np.argsort(ft)
        ft = ft[ord_]
        fv = np.asarray(from_values)[ord_]
        # Extend for periodicity
        ft_ext = np.concatenate([ft[:1] - two_pi, ft, ft[-1:] + two_pi])
        fv_ext = np.concatenate([fv[-1:], fv, fv[:1]])
        pchip = PchipInterpolator(ft_ext, fv_ext, extrapolate=True)
        tt = np.mod(to_theta, two_pi)
        return pchip(tt)

    @staticmethod
    def barycentric_resample(
        from_theta: np.ndarray, from_values: np.ndarray, to_theta: np.ndarray,
    ) -> np.ndarray:
        """Barycentric Lagrange interpolation for nonuniform nodes (non-periodic basis).

        Best used for stable evaluation between collocation-style nodes.
        Requires SciPy; falls back to linear if unavailable.
        """
        if not _HAVE_SCIPY:
            return GridMapper.periodic_linear_resample(from_theta, from_values, to_theta)
        interp = BarycentricInterpolator(from_theta, from_values)
        return interp(to_theta)

    @staticmethod
    def l2_project(
        from_theta: np.ndarray,
        from_values: np.ndarray,
        to_theta: np.ndarray,
        weights_from: np.ndarray | None = None,
    ) -> np.ndarray:
        """Conservative L2 projection from source nodes to target nodes.

        Constructs target Lagrange basis (via barycentric) evaluated at source nodes,
        solves (Phi^T W Phi) c = Phi^T W f, returning c as values on to_theta.
        """
        if not _HAVE_SCIPY:
            # Fallback: linear resample
            return GridMapper.periodic_linear_resample(from_theta, from_values, to_theta)
        # Build basis evaluation matrix Phi_{i,j} = L_j(from_theta_i)
        n_to = len(to_theta)
        Phi = np.zeros((len(from_theta), n_to), dtype=float)
        # Precompute barycentric weights for target nodes
        bary = BarycentricInterpolator(to_theta)
        # There is no direct access to basis; approximate by evaluating impulses
        # Build identity at target nodes and evaluate interpolant at from nodes
        I = np.eye(n_to, dtype=float)
        for j in range(n_to):
            bary.set_yi(I[:, j])  # type: ignore[attr-defined]
            Phi[:, j] = bary(from_theta)
        W = np.diag(weights_from) if weights_from is not None else np.eye(len(from_theta))
        A = Phi.T @ W @ Phi
        b = Phi.T @ W @ np.asarray(from_values)
        # Regularize lightly for safety
        A += 1e-12 * np.eye(n_to)
        coeffs = np.linalg.solve(A, b)
        return coeffs

    @staticmethod
    def map_state_triplet(
        from_theta: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        to_theta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map (p, v, a) independently via periodic linear interpolation.

        For robust physics conservation, callers should re-derive v/a from p if
        needed. Here we keep it simple to avoid introducing heavy dependencies.
        """
        p_m = GridMapper.periodic_linear_resample(from_theta, position, to_theta)
        v_m = GridMapper.periodic_linear_resample(from_theta, velocity, to_theta)
        a_m = GridMapper.periodic_linear_resample(from_theta, acceleration, to_theta)
        return p_m, v_m, a_m

    @staticmethod
    def operators(from_theta: np.ndarray, to_theta: np.ndarray, method: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
        """Return (P_U→Gi, P_Gi→U) interpolation/projection operators.

        Currently builds dense matrices by sampling basis; suitable for small N.
        """
        n_from = len(from_theta)
        n_to = len(to_theta)
        if method == "projection" and _HAVE_SCIPY:
            # Build Phi(from, to) and compute projectors approximately
            Phi = np.zeros((n_from, n_to), dtype=float)
            I = np.eye(n_to)
            bary = BarycentricInterpolator(to_theta)
            for j in range(n_to):
                bary.set_yi(I[:, j])  # type: ignore[attr-defined]
                Phi[:, j] = bary(from_theta)
            W = np.diag(GridMapper.trapz_weights(from_theta))
            A = Phi.T @ W @ Phi + 1e-12 * np.eye(n_to)
            P_u2g = np.linalg.solve(A, Phi.T @ W)  # maps f(from) to coeffs(on to)
            # For back map use interpolation on to→from (Phi^T) or a simple linear interp
            P_g2u = Phi  # evaluates to-node coefficients at from-nodes
            return P_u2g, P_g2u
        # Linear: build by sampling Kronecker basis at target
        # Build P_g2u: evaluate target basis at from nodes using barycentric if available
        if _HAVE_SCIPY:
            Phi = np.zeros((n_from, n_to), dtype=float)
            I = np.eye(n_to)
            bary = BarycentricInterpolator(to_theta)
            for j in range(n_to):
                bary.set_yi(I[:, j])  # type: ignore[attr-defined]
                Phi[:, j] = bary(from_theta)
            # P_u2g: least-squares fit using Phi
            P_u2g = np.linalg.pinv(Phi)
            P_g2u = Phi
            return P_u2g, P_g2u
        # Fallback coarse operators via linear interp matrices (dense sampling)
        # Construct P_u2g by solving n_to independent linear systems built by impulses at to nodes
        P_g2u = np.zeros((n_from, n_to))
        eye = np.eye(n_to)
        for j in range(n_to):
            col = GridMapper.periodic_linear_resample(to_theta, eye[:, j], from_theta)
            P_g2u[:, j] = col
        P_u2g = np.linalg.pinv(P_g2u)
        return P_u2g, P_g2u

    @staticmethod
    def pullback_gradient(grad_gi: np.ndarray, P_gi_to_u: np.ndarray) -> np.ndarray:
        """Pull back a gradient defined on Gi to the universal grid U.

        grad_U = P_Gi→U^T · grad_Gi
        """
        return P_gi_to_u.T @ grad_gi

    @staticmethod
    def pushforward_jacobian(
        J_gi: np.ndarray, P_u_to_gi: np.ndarray, P_gi_to_u: np.ndarray,
    ) -> np.ndarray:
        """Push Jacobian from Gi to U using linearization of maps.

        ∂r_U/∂x_U = P_Gi→U^T · (∂r_Gi/∂x_Gi) · P_U→Gi
        """
        return P_gi_to_u.T @ J_gi @ P_u_to_gi

    # Diagnostics
    @staticmethod
    def integral_conservation_error(theta_u: np.ndarray, f_u: np.ndarray, theta_g: np.ndarray, f_g: np.ndarray) -> float:
        """Absolute difference between integrals on U and Gi using trapz weights on each grid."""
        Iu = float(np.sum(f_u * GridMapper.trapz_weights(theta_u)))
        Ig = float(np.sum(f_g * GridMapper.trapz_weights(theta_g)))
        return abs(Iu - Ig)

    @staticmethod
    def harmonic_probe_error(theta_u: np.ndarray, P_u2g: np.ndarray, P_g2u: np.ndarray, k: int = 5) -> float:
        """Map sin/cos(kθ) U→Gi→U and report max amplitude error over the round-trip."""
        s = np.sin(k * theta_u)
        c = np.cos(k * theta_u)
        s_rt = P_g2u @ (P_u2g @ s)
        c_rt = P_g2u @ (P_u2g @ c)
        return max(float(np.max(np.abs(s - s_rt))), float(np.max(np.abs(c - c_rt))))

    @staticmethod
    def derivative_drift(theta_u: np.ndarray, theta_g: np.ndarray, P_u2g: np.ndarray, P_g2u: np.ndarray, p_u: np.ndarray) -> float:
        """|| D_U p_U − P_Gi→U D_Gi (P_U→Gi p_U) ||_inf using gradient approximations."""
        d_u = np.gradient(p_u, theta_u)
        p_g = P_u2g @ p_u
        d_g = np.gradient(p_g, theta_g)
        d_u_rt = P_g2u @ d_g
        return float(np.max(np.abs(d_u - d_u_rt)))


