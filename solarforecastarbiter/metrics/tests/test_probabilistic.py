import pytest
import numpy as np
import pandas as pd
from solarforecastarbiter.metrics import probabilistic as prob


@pytest.mark.parametrize("y_pred,y_prob,y_true,value", [
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
def test_brier_score(y_pred, y_prob, y_true, value):
    assert prob.brier_score(y_pred, y_prob, y_true) == value


#@pytest.mark.parametrize("fx,obs,ref,value", [
#    (1.00, 0, 1.00, 1.0 - 1.0 / 1.0),
#    (1.00, 0, 0.50, 1.0 - 1.0 / 0.25),
#])
#def test_brier_skill_score_scalar(fx, obs, ref, value):
#    assert prob.brier_skill_score(fx, obs, ref) == value
#
#
#@pytest.mark.parametrize("fx,obs,ref,value", [
#    (
#        np.array([1.0, 1.0]),
#        np.array([0, 0]),
#        np.array([1.0, 1.0]),
#        1.0 - 1.0 / 1.0
#    ),
#    (
#        np.array([1.0, 1.0]),
#        np.array([0, 0]),
#        np.array([0.5, 0.5]),
#        1.0 - 1.0 / 0.25
#    ),
#])
#def test_brier_skill_score_vector(fx, obs, ref, value):
#    assert prob.brier_skill_score(fx, obs, ref) == value
#
#
#@pytest.mark.parametrize("fx,obs,ref,value", [
#    (
#        pd.Series([1.0, 1.0]),
#        pd.Series([0, 0]),
#        pd.Series([1.0, 1.0]),
#        1.0 - 1.0 / 1.0
#    ),
#    (
#        pd.Series([1.0, 1.0]),
#        pd.Series([0, 0]),
#        pd.Series([0.5, 0.5]),
#        1.0 - 1.0 / 0.25
#    ),
#])
#def test_brier_skill_score_series(fx, obs, ref, value):
#    assert prob.brier_skill_score(fx, obs, ref) == value
#
#
#@pytest.mark.parametrize("fx_lower,fx_upper,value", [
#    (0.0, 1.0, 1.0),
#    (0.5, 0.6, 0.1),
#])
#def test_sharpness_scalar(fx_lower, fx_upper, value):
#    sh = prob.sharpness(fx_lower, fx_upper)
#    assert pytest.approx(sh) == value
#
#
#@pytest.mark.parametrize("fx_lower,fx_upper,value", [
#    (np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0),
#    (np.array([0.2, 0.7]), np.array([0.3, 0.8]), 0.1),
#])
#def test_sharpness_vector(fx_lower, fx_upper, value):
#    sh = prob.sharpness(fx_lower, fx_upper)
#    assert pytest.approx(sh) == value
#
#
#@pytest.mark.parametrize("fx_lower,fx_upper,value", [
#    (pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]), 1.0),
#    (pd.Series([0.2, 0.7]), pd.Series([0.3, 0.8]), 0.1),
#])
#def test_sharpness_series(fx_lower, fx_upper, value):
#    sh = prob.sharpness(fx_lower, fx_upper)
#    assert pytest.approx(sh) == value
#
#
#@pytest.mark.parametrize("obs,value", [
#    (0, 0.0),
#    (1, 0.0),
#    ([0, 0], 0.0),
#    ([1, 1], 0.0),
#    ([0, 1], 0.25),
#    (pd.Series([0, 0]), 0.0),
#    (pd.Series([1, 1]), 0.0),
#    (pd.Series([0, 1]), 0.25),
#])
#def test_unc(obs, value):
#    assert prob.uncertainty(obs) == value
#
#
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
