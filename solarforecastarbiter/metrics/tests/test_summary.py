import pytest
import numpy as np
from solarforecastarbiter.metrics import summary


@pytest.mark.parametrize("ts,value", [
    ([1, 2, 3], 2),
    ([2, np.nan, 4], 3),
])
def test_mean(ts, value):
    assert summary.mean(ts) == value
