import pytest
import numpy as np
from solarforecastarbiter.metrics import valuation

@pytest.mark.parametrize("fx,obs,cost,value", [
    # 10 USD per MW
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 10, 0.0),
    (np.array([1, 2, 3]), np.array([2, 3, 4]), 10, 3 * 10),
])
def test_fixed_cost(fx, obs, cost, value):
    return valuation.fixed_error_cost(fx, obs, cost) == value
