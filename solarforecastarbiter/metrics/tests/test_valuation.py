import pytest
import numpy as np
from solarforecastarbiter.metrics import valuation


@pytest.mark.parametrize("fx,obs,cost,value", [
    # cost: 10 USD per kW
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 10, 0.0),
    (np.array([1, 2, 3]), np.array([2, 3, 4]), 10, 3 * 10),

    # cost: 1 USD per W/m^2
    (np.array([500, 600, 650]), np.array([500, 580, 630]), 1, 20 + 20),
])
def test_fixed_cost(fx, obs, cost, value):
    return valuation.fixed_error_cost(obs, fx, cost) == value
