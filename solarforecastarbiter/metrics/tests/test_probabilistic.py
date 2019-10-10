import pytest
import numpy as np
import pandas as pd
from solarforecastarbiter.metrics import probabilistic as prob


@pytest.mark.parametrize("fx,obs,value", [
    (1.00, 1, 0.00),             # forecast 100%, event occurred
    (1.00, 0, 1.00),             # forecast 100%, event did not occur
    (0.70, 1, (0.70 - 1) ** 2),  # forecast 70%, event occurred
    (0.30, 1, (0.30 - 1) ** 2),  # forecast 30%, event occurred
    (0.50, 1, 0.25),             # forecast 50%, event occurred
    (0.50, 0, 0.25),             # forecast 50%, event did not occurr
])
def test_brier_score_scalar(fx, obs, value):
    assert prob.brier_score(fx, obs) == value


@pytest.mark.parametrize("fx,obs,value", [
    (np.array([1.0, 1.0]), np.array([1, 1]), 0.0),
    (np.array([1.0, 1.0]), np.array([0, 0]), 1.0),
])
def test_brier_score_vector(fx, obs, value):
    assert prob.brier_score(fx, obs) == value


@pytest.mark.parametrize("fx,obs,value", [
    (pd.Series([1.0, 1.0]), pd.Series([1, 1]), 0.0),
    (pd.Series([1.0, 1.0]), pd.Series([0, 0]), 1.0),
])
def test_brier_score_series(fx, obs, value):
    assert prob.brier_score(fx, obs) == value


@pytest.mark.parametrize("fx,obs,ref,value", [
    (1.00, 0, 1.00, 1.0 - 1.0 / 1.0),
    (1.00, 0, 0.50, 1.0 - 1.0 / 0.25),
])
def test_brier_skill_score_scalar(fx, obs, ref, value):
    assert prob.brier_skill_score(fx, obs, ref) == value


@pytest.mark.parametrize("fx,obs,ref,value", [
    (
        np.array([1.0, 1.0]),
        np.array([0, 0]),
        np.array([1.0, 1.0]),
        1.0 - 1.0 / 1.0
    ),
    (
        np.array([1.0, 1.0]),
        np.array([0, 0]),
        np.array([0.5, 0.5]),
        1.0 - 1.0 / 0.25
    ),
])
def test_brier_skill_score_vector(fx, obs, ref, value):
    assert prob.brier_skill_score(fx, obs, ref) == value


@pytest.mark.parametrize("fx,obs,ref,value", [
    (
        pd.Series([1.0, 1.0]),
        pd.Series([0, 0]),
        pd.Series([1.0, 1.0]),
        1.0 - 1.0 / 1.0
    ),
    (
        pd.Series([1.0, 1.0]),
        pd.Series([0, 0]),
        pd.Series([0.5, 0.5]),
        1.0 - 1.0 / 0.25
    ),
])
def test_brier_skill_score_series(fx, obs, ref, value):
    assert prob.brier_skill_score(fx, obs, ref) == value


@pytest.mark.parametrize("fx_lower,fx_upper,value", [
    (0.0, 1.0, 1.0),
    (0.5, 0.6, 0.1),
])
def test_sharpness_scalar(fx_lower, fx_upper, value):
    sh = prob.sharpness(fx_lower, fx_upper)
    assert pytest.approx(sh) == value


@pytest.mark.parametrize("fx_lower,fx_upper,value", [
    (np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([0.2, 0.7]), np.array([0.3, 0.8]), 0.1),
])
def test_sharpness_vector(fx_lower, fx_upper, value):
    sh = prob.sharpness(fx_lower, fx_upper)
    assert pytest.approx(sh) == value


@pytest.mark.parametrize("fx_lower,fx_upper,value", [
    (pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0]), 1.0),
    (pd.Series([0.2, 0.7]), pd.Series([0.3, 0.8]), 0.1),
])
def test_sharpness_series(fx_lower, fx_upper, value):
    sh = prob.sharpness(fx_lower, fx_upper)
    assert pytest.approx(sh) == value


@pytest.mark.parametrize("obs,value", [
    (0, 0.0),
    (1, 0.0),
    ([0, 0], 0.0),
    ([1, 1], 0.0),
    ([0, 1], 0.25),
    (pd.Series([0, 0]), 0.0),
    (pd.Series([1, 1]), 0.0),
    (pd.Series([0, 1]), 0.25),
])
def test_unc(obs, value):
    assert prob.uncertainty(obs) == value
