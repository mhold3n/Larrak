import pytest

from campro.profiles.noncircular_internal import synthesize_internal_pair
import numpy as np


def test_noncircular_internal_stub_raises():
    with pytest.raises(NotImplementedError):
        synthesize_internal_pair(np.array([0.0]), np.array([0.0]), {})

