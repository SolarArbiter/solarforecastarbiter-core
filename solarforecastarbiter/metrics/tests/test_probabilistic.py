import pytest
import numpy as np
import pandas as pd
from solarforecastarbiter.metrics import probabilistic as prob


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    # forecast 10 MW with 100% probability
    (10, 100, 8, 0.0),   # actual: 8 MW
    (10, 100, 10, 0.0),  # actual: 10 MW
    (10, 100, 17, 1.0),  # actual: 17 MW

    # forecast 8 MW with 50% probability
    (8, 50, 2, 0.25),   # actual: 2 MW
    (8, 50, 8, 0.25),   # actual: 8 MW
    (8, 50, 13, 0.25),  # actual: 13 MW

    # forecast 350 W/m^2 with 30% probability
    (350, 30, 270, (0.3 - 1) ** 2),  # actual: 270 W/m^2
    (350, 30, 350, (0.3 - 1) ** 2),  # actual: 350 W/m^2
    (350, 30, 521, (0.3 - 0) ** 2),  # actual: 521 W/m^2

    # forecast 2 kWh with 70% probability
    (2, 70, 1, (0.7 - 1) ** 2),  # actual: 1 kWh
    (2, 70, 2, (0.7 - 1) ** 2),  # actual: 2 kWh
    (2, 70, 4, (0.7 - 0) ** 2),  # actual: 4 kWh

    # vector inputs
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([8, 8]), 0.0),
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([10, 10]), 0.0),
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([17, 17]), 1.0),
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([2, 14]), 0.5),
])
def test_brier_score(fx, fx_prob, obs, value):
    assert prob.brier_score(fx, fx_prob, obs) == value


@pytest.mark.parametrize("fx,fx_prob,ref,ref_prob,obs,value", [
    (7, 100, 5, 100, 8, 1.0 - 1.0 / 1.0),
    (10, 50, 5, 100, 8, 1.0 - 0.25 / 1.0),
])
def test_brier_skill_score(fx, fx_prob, ref, ref_prob, obs, value):
    assert prob.brier_skill_score(fx, fx_prob, ref, ref_prob, obs) == value


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([8, 8]), 0.0),
    (
        np.asarray([10, 10]),
        np.asarray([100, 50]),
        np.asarray([8, 8]),
        (1 * 0.0 + 1 * 0.25) / 2,
    ),
])
def test_reliability(fx, fx_prob, obs, value):
    assert prob.reliability(fx, fx_prob, obs) == value


@pytest.mark.parametrize("fx,obs,value", [
    (10, 8, 1.0 * (1 - 1.0)),
    (5, 8, 0.0 * (1 - 0.0)),
    (np.asarray([5, 10]), np.asarray([8, 8]), 0.5 * (1 - 0.5)),
])
def test_unc(fx, obs, value):
    assert prob.uncertainty(fx, obs) == value


@pytest.mark.parametrize("fx_lower,fx_upper,value", [
    # scalar
    (0.0, 1.0, 1.0),
    (0.5, 0.6, 0.1),

    # vector
    (np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([0.2, 0.7]), np.array([0.3, 0.8]), 0.1),
    (np.array([0.0, 0.1]), np.array([0.5, 0.6]), 0.5),
])
def test_sharpness_scalar(fx_lower, fx_upper, value):
    sh = prob.sharpness(fx_lower, fx_upper)
    assert pytest.approx(sh) == value


#@pytest.mark.parametrize("F,O,q,value", [
#    (
#        np.array([[0.0, 1.0], [0.0, 1.0]]),  # predicted CDF [-]
#        np.array([[0, 0], [0, 0]]),          # observed CDF [-] (binary)
#        np.array([0, 1]),                    # actuals [MW] = grid
#        (0.5 + 0.5) / 2,
#    ),
#    (
#        np.array([[0.0, 1.0], [0.0, 1.0]]),
#        np.array([[0, 1], [0, 1]]),
#        np.array([0, 1]),
#        (0.0 + 0.0) / 2,
#    ),
#    (
#        np.array([[0.0, 0.2], [0.0, 0.2]]),
#        np.array([[0, 1], [0, 1]]),
#        np.array([0, 1]),
#        (0.4 + 0.4) / 2,
#    ),
#    (
#        np.array([[0.0, 1.0], [0.0, 0.4]]),
#        np.array([[0, 1], [0, 1]]),
#        np.array([0, 1]),
#        (0.0 + 0.3) / 2,
#    ),
#    (
#        np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
#        np.array([[0, 1, 1], [0, 1, 1]]),
#        np.array([0, 1, 2]),
#        (0.5 + 0.5) / 2,
#    ),
#    (
#        np.array([[0.0, 0.2, 0.4], [0.0, 0.5, 1.0]]),
#        np.array([[0, 1, 1], [0, 1, 1]]),
#        np.array([0, 1, 2]),
#        ((0.4 + 0.7) + 0.5) / 2,
#    ),
#])
#def test_crps(F, O, q, value):
#    assert prob.crps(F, O, q) == value
