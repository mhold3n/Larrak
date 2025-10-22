import pytest

from campro.tca.rigid_tca import rigid_tca
import numpy as np


def test_rigid_tca_stub_raises():
    with pytest.raises(NotImplementedError):
        rigid_tca({}, {}, 0.0, np.array([0.0]))

