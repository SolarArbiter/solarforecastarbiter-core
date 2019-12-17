import pytest
import numpy as np
from solarforecastarbiter.metrics import summary


@pytest.mark.parametrize("ts", [
    [1, 2, 3],
    np.random.rand(10),
    np.random.rand(1000),
])
def test_scalar(ts):
    for metric in summary._MAP:
        f = summary._MAP[metric][0]
        assert np.isscalar(f(ts))
