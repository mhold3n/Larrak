import numpy as np
import pytest

from campro.tribology.ehl import hamrock_dowson_hmin, lambda_map


def test_hamrock_dowson_hmin_stub_raises():
    with pytest.raises(NotImplementedError):
        hamrock_dowson_hmin(1.0, 1.0, 1.0, 1.0)


def test_lambda_map_stub_raises():
    with pytest.raises(NotImplementedError):
        lambda_map(np.array([0.0]))
