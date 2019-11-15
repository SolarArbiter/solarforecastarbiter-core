import pytest
import numpy as np
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


@pytest.mark.parametrize("f,value", [
    ([0.1234], [0.1]),
    ([0.1689], [0.2]),
    (np.ones(999) * 0.1234, np.ones(999) * 0.1),
    (np.ones(1000) * 0.1234, np.ones(1000) * 0.12),
    (np.ones(1000) * 0.1580, np.ones(1000) * 0.16),
    (np.ones(1001) * 0.1580, np.ones(1001) * 0.16),
])
def test_unique_forecasts(f, value):
    np.testing.assert_array_equal(prob._unique_forecasts(f), value)


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    (
        np.asarray([10, 10]),
        np.asarray([100, 100]),
        np.asarray([8, 8]),
        0.0,
    ),
    (
        np.asarray([3, 4]),
        np.asarray([100, 100]),
        np.asarray([10, 11]),
        1.0,
    ),
    (
        np.asarray([10, 5]),
        np.asarray([100, 50]),
        np.asarray([8, 8]),
        0.125,
    ),
    (
        np.asarray([10, 10]),
        np.asarray([100, 50]),
        np.asarray([8, 8]),
        0.125,
    ),
])
def test_reliability(fx, fx_prob, obs, value):
    assert prob.reliability(fx, fx_prob, obs) == value


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    (
        np.asarray([10, 10]),
        np.asarray([100, 100]),
        np.asarray([8, 8]),
        0.0,
    ),
    (
        np.asarray([8, 8]),
        np.asarray([100, 100]),
        np.asarray([10, 10]),
        0.0,
    ),
    (
        np.asarray([10, 5]),
        np.asarray([100, 50]),
        np.asarray([8, 8]),
        0.25,
    ),
])
def test_resolution(fx, fx_prob, obs, value):
    assert prob.resolution(fx, fx_prob, obs) == value


@pytest.mark.parametrize("fx,obs,value", [
    # scalar inputs
    (10, 8, 0.0),
    (5, 8, 0.0),

    # vector inputs
    (np.asarray([10, 5]), np.asarray([8, 8]), 0.25),
    (np.asarray([8, 8]), np.asarray([10, 10]), 0.0),
])
def test_unc(fx, obs, value):
    assert prob.uncertainty(fx, obs) == value


@pytest.mark.parametrize("fx,fx_prob,obs", [
    # scalar inputs
    (np.asarray([10]), np.asarray([100]), np.asarray([8])),
    (np.asarray([10]), np.asarray([100]), np.asarray([10])),
    (np.asarray([10]), np.asarray([100]), np.asarray([15])),

    # vector inputs
    (np.asarray([10, 10]), np.asarray([100, 100]), np.asarray([8, 8])),
    (np.asarray([10, 5]), np.asarray([100, 50]), np.asarray([8, 8])),
    (np.asarray([8, 8]), np.asarray([100, 100]), np.asarray([10, 10])),
])
def test_brier_decomposition(fx, fx_prob, obs):
    bs = prob.brier_score(fx, fx_prob, obs)
    rel = prob.reliability(fx, fx_prob, obs)
    res = prob.resolution(fx, fx_prob, obs)
    unc = prob.uncertainty(fx, obs)
    assert bs == rel - res + unc


@pytest.mark.parametrize("fx_lower,fx_upper,value", [
    # scalar inputs
    (0.0, 1.0, 1.0),
    (0.5, 0.6, 0.1),

    # vector inputs
    (np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([0.2, 0.7]), np.array([0.3, 0.8]), 0.1),
    (np.array([0.0, 0.1]), np.array([0.5, 0.6]), 0.5),
])
def test_sharpness(fx_lower, fx_upper, value):
    sh = prob.sharpness(fx_lower, fx_upper)
    assert pytest.approx(sh) == value
