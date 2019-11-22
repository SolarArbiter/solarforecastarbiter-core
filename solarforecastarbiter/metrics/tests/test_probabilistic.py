import pytest
import numpy as np
from solarforecastarbiter.metrics import probabilistic as prob


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    # forecast <= 10 MW with 100% probability
    (10, 100, 8, 0.0),   # actual: 8 MW
    (10, 100, 10, 0.0),  # actual: 10 MW
    (10, 100, 17, 1.0),  # actual: 17 MW

    # forecast <= 8 MW with 50% probability
    (8, 50, 2, 0.25),   # actual: 2 MW
    (8, 50, 8, 0.25),   # actual: 8 MW
    (8, 50, 13, 0.25),  # actual: 13 MW

    # forecast <= 350 W/m^2 with 30% probability
    (350, 30, 270, (0.3 - 1) ** 2),  # actual: 270 W/m^2
    (350, 30, 350, (0.3 - 1) ** 2),  # actual: 350 W/m^2
    (350, 30, 521, (0.3 - 0) ** 2),  # actual: 521 W/m^2

    # forecast <= 2 kWh with 70% probability
    (2, 70, 1, (0.7 - 1) ** 2),  # actual: 1 kWh
    (2, 70, 2, (0.7 - 1) ** 2),  # actual: 2 kWh
    (2, 70, 4, (0.7 - 0) ** 2),  # actual: 4 kWh

    # vector inputs
    (np.array([10, 10]), np.array([100, 100]), np.array([8, 8]), 0.0),
    (np.array([10, 10]), np.array([100, 100]), np.array([10, 10]), 0.0),
    (np.array([10, 10]), np.array([100, 100]), np.array([17, 17]), 1.0),
    (np.array([10, 10]), np.array([100, 100]), np.array([2, 14]), 0.5),
])
def test_brier_score(fx, fx_prob, obs, value):
    assert prob.brier_score(obs, fx, fx_prob) == value


@pytest.mark.parametrize("fx,fx_prob,ref,ref_prob,obs,value", [
    (7, 100, 5, 100, 8, 1.0 - 1.0 / 1.0),
    (10, 50, 5, 100, 8, 1.0 - 0.25 / 1.0),
])
def test_brier_skill_score(fx, fx_prob, ref, ref_prob, obs, value):
    assert prob.brier_skill_score(obs, fx, fx_prob, ref, ref_prob) == value


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
        np.array([10, 10]),
        np.array([100, 100]),
        np.array([8, 8]),
        0.0,
    ),
    (
        np.array([3, 4]),
        np.array([100, 100]),
        np.array([10, 11]),
        1.0,
    ),
    (
        np.array([10, 5]),
        np.array([100, 50]),
        np.array([8, 8]),
        0.125,
    ),
    (
        np.array([10, 10]),
        np.array([100, 50]),
        np.array([8, 8]),
        0.125,
    ),

    # effects of determining unique forecasts
    (
        np.ones(999) * 2,
        np.ones(999) * 51,
        np.ones(999) * 1,
        (0.5 - 1.0) ** 2
    ),
    (
        np.ones(1000) * 10,
        np.ones(1000) * 51,
        np.ones(1000) * 8,
        (0.51 - 1.0) ** 2
    ),
])
def test_reliability(fx, fx_prob, obs, value):
    assert prob.reliability(obs, fx, fx_prob) == pytest.approx(value)


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    (
        np.array([10, 10]),
        np.array([100, 100]),
        np.array([8, 8]),
        0.0,
    ),
    (
        np.array([8, 8]),
        np.array([100, 100]),
        np.array([10, 10]),
        0.0,
    ),
    (
        np.array([10, 5]),
        np.array([100, 50]),
        np.array([8, 8]),
        0.25,
    ),

    # effects of determining unique forecasts
    (
        np.array([10, 5]),
        np.array([95, 100]),
        np.array([8, 8]),
        (2 * (0.5 - 0.5) ** 2) / 2,
    ),
    (
        np.concatenate([np.ones(20) * 10, np.ones(10) * 5]),
        np.concatenate([np.ones(20) * 95, np.ones(10) * 100]),
        np.concatenate([np.ones(20) * 8, np.ones(10) * 8]),
        (30 * (0.5 - 0.5) ** 2) / 30,
    ),
    (
        np.concatenate([np.ones(1000) * 10, np.ones(500) * 5]),
        np.concatenate([np.ones(1000) * 95, np.ones(500) * 100]),
        np.concatenate([np.ones(1000) * 8, np.ones(500) * 8]),
        (1000 * (1 - 2/3) ** 2 + 500 * (0 - 2/3) ** 2) / 1500,
    ),
])
def test_resolution(fx, fx_prob, obs, value):
    assert prob.resolution(obs, fx, fx_prob) == value


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    # scalar inputs
    (np.array([10]), np.array([100]), np.array([8]), 0.0),
    (np.array([10]), np.array([35]), np.array([8]), 0.0),
    (np.array([5]), np.array([100]), np.array([8]), 0.0),
    (np.array([5]), np.array([35]), np.array([8]), 0.0),

    # vector inputs
    (np.array([10, 5]), np.array([100, 100]), np.array([8, 8]), 0.25),
    (np.array([8, 8]), np.array([100, 100]), np.array([10, 10]), 0.0),
    (np.array([10, 5]), np.array([15, 25]), np.array([8, 8]), 0.25),
    (np.array([8, 8]), np.array([33, 31]), np.array([10, 10]), 0.0),
])
def test_unc(fx, fx_prob, obs, value):
    assert prob.uncertainty(obs, fx, fx_prob) == value


@pytest.mark.parametrize("fx,fx_prob,obs", [
    # scalar inputs
    (np.array([10]), np.array([100]), np.array([8])),
    (np.array([10]), np.array([100]), np.array([10])),
    (np.array([10]), np.array([100]), np.array([15])),

    # vector inputs
    (np.array([10, 10]), np.array([100, 100]), np.array([8, 8])),
    (np.array([10, 5]), np.array([100, 50]), np.array([8, 8])),
    (np.array([8, 8]), np.array([100, 100]), np.array([10, 10])),
])
def test_brier_decomposition(fx, fx_prob, obs):
    bs = prob.brier_score(obs, fx, fx_prob)
    rel, res, unc = prob.brier_decomposition(obs, fx, fx_prob)
    assert pytest.approx(bs) == rel - res + unc


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


@pytest.mark.parametrize("fx,fx_prob,obs,value", [
    # 1 sample, 1 CDF interval
    (
        np.array([[10]]),
        np.array([[100]]),
        np.array([8]),
        0.0,
    ),

    # 2 samples, 2 CDF intervals
    (
        np.array([[10, 20], [10, 20]]),
        np.array([[100, 100], [100, 100]]),
        np.array([8, 8]),
        0.0,
    ),
    (
        np.array([[10, 20], [10, 20]]),
        np.array([[0, 100], [0, 100]]),
        np.array([8, 8]),
        (1.0 ** 2) * 10,
    ),
    (
        np.array([[10, 20], [10, 20]]),
        np.array([[50, 100], [50, 100]]),
        np.array([8, 8]),
        (0.5 ** 2) * 10,
    ),
    (
        np.array([[10, 20], [5, 10]]),
        np.array([[50, 100], [50, 100]]),
        np.array([8, 8]),
        ((0.5 ** 2) * 10 + (0.5 ** 2) * 5) / 2,
    ),
    (
        np.array([[10, 20], [10, 20]]),
        np.array([[50, 100], [30, 100]]),
        np.array([8, 8]),
        ((0.5 ** 2) * 10 + (0.7 ** 2) * 10) / 2,
    ),

    # 2 samples, 3 CDF intervals
    (
        np.array([[10, 20, 30], [10, 20, 30]]),
        np.array([[100, 100, 100], [100, 100, 100]]),
        np.array([8, 8]),
        0.0,
    ),
    (
        np.array([[10, 20, 30], [10, 20, 30]]),
        np.array([[0, 100, 100], [0, 100, 100]]),
        np.array([8, 8]),
        (1.0 ** 2) * 10,
    ),
    (
        np.array([[10, 20, 30], [10, 20, 30]]),
        np.array([[0, 0, 100], [0, 0, 100]]),
        np.array([8, 8]),
        ((1.0 ** 2) * 10 + (1.0 ** 2) * 10),
    ),
    (
        np.array([[10, 20, 30], [10, 20, 30]]),
        np.array([[0, 50, 100], [0, 50, 100]]),
        np.array([8, 8]),
        (1.0 ** 2) * 10 + (0.5 ** 2) * 10,
    ),
])
def test_crps(fx, fx_prob, obs, value):
    crps = prob.continuous_ranked_probability_score(obs, fx, fx_prob)
    assert crps == value
