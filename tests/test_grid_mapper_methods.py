from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from campro.optimization.grid import GridMapper


def test_linear_mapping_roundtrip():
    th_u = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    f_u = np.sin(3 * th_u) + 0.5 * np.cos(5 * th_u)
    th_g = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    f_g = GridMapper.periodic_linear_resample(th_u, f_u, th_g)
    # Back to U using interpolation matrix
    P_u2g, P_g2u = GridMapper.operators(th_u, th_g, method="linear")
    f_u_rt = P_g2u @ (P_u2g @ f_u)
    assert np.max(np.abs(f_u - f_u_rt)) < 5e-2


def test_projection_conserves_integral():
    th_u = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    f_u = 1.0 + 0.2 * np.cos(th_u)
    th_g = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    w_u = GridMapper.trapz_weights(th_u)
    f_g = GridMapper.l2_project(th_u, f_u, th_g, weights_from=w_u)
    w_g = GridMapper.trapz_weights(th_g)
    Iu = float(np.sum(f_u * w_u))
    Ig = float(np.sum(f_g * w_g))
    assert abs(Iu - Ig) < 1e-6


def test_harmonic_probe_error_small():
    th_u = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    th_g = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    P_u2g, P_g2u = GridMapper.operators(th_u, th_g, method="linear")
    err = GridMapper.harmonic_probe_error(th_u, P_u2g, P_g2u, k=7)
    assert err < 0.2


