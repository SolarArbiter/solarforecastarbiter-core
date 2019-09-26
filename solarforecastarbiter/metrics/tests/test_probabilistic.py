import pytest
import numpy as np
from solarforecastarbiter.metrics import probabilistic as prob


@pytest.mark.parametrize("f,o,value", [
    (1.00, 1, 0.00),  # forecast 100%, event occurred
    (1.00, 0, 1.00),  # forecast 100%, event did not occur
    (0.70, 1, (0.70 - 1) ** 2),  # forecast 70%, event occurred
    (0.30, 1, (0.30 - 1) ** 2),  # forecast 30%, event occurred
    (0.50, 1, 0.25),  # forecast 50%, event occurred
    (0.50, 0, 0.25),  # forecast 50%, event did not occurr
])
def test_brier_score(f, o, value):
    assert prob.brier_score(f, o) == value
