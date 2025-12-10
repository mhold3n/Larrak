import numpy as np

from campro.diagnostics.scaling import (
    compute_scaling_vector,
    scale_value,
    unscale_value,
)


def test_compute_scaling_vector_basic():
    scales = compute_scaling_vector({"x": (-10.0, 20.0), "y": (-1e-9, 1e-9)})
    assert abs(scales["x"] - 1.0 / 20.0) < 1e-12
    assert scales["y"] == 1.0  # too small magnitude → 1.0


def test_scale_unscale_roundtrip():
    x = np.array([1.0, 2.0, -3.0])
    s = 0.5
    z = scale_value(x, s)
    x2 = unscale_value(z, s)
    assert np.allclose(x, x2)
