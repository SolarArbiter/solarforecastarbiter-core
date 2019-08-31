import pytest
import numpy as np
from solarforecastarbiter.metrics import deterministic


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 1]), 1 / 3),
])
def test_mae(y_true, y_pred, value):
    assert deterministic.mean_absolute(y_true, y_pred) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([1, 0, 2]), 0.0),
])
def test_mbe(y_true, y_pred, value):
    assert deterministic.mean_bias(y_true, y_pred) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1]), np.array([1, 2]), 1.0),
])
def test_rmse(y_true, y_pred, value):
    assert deterministic.root_mean_square(y_true, y_pred) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([1, 1]), np.array([2, 2]), 100.0),
    (np.array([2, 2]), np.array([3, 3]), 50.0),
])
def test_mape(y_true, y_pred, value):
    assert deterministic.mean_absolute_percentage(y_true, y_pred) == value
