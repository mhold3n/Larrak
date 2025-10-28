from __future__ import annotations

import pytest

from campro.freepiston.opt.colloc import make_grid


def test_radau_iia_s1_coefficients():
    grid = make_grid(5, 1, kind="radau")
    # Radau IIA with 1 stage reduces to implicit Euler
    assert grid.nodes == [1.0]
    assert grid.weights == [1.0]
    assert grid.a == [[1.0]]


def test_radau_iia_unsupported_stages():
    with pytest.raises(NotImplementedError):
        make_grid(5, 2, kind="radau")


def test_radau_iia_s3_coefficients_shape_and_sanity():
    grid = make_grid(4, 3, kind="radau")
    assert len(grid.nodes) == 3
    assert len(grid.weights) == 3
    assert len(grid.a) == 3 and len(grid.a[0]) == 3
    # basic sanity: weights sum to 1 on [0,1]
    assert abs(sum(grid.weights) - 1.0) < 1e-12
