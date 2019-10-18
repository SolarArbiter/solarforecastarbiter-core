import pytest
import numpy as np
from solarforecastarbiter.metrics import deterministic


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 1]), 1 / 3),
])
def test_mae(y_true, y_pred, value):
    mae = deterministic.mean_absolute(y_true, y_pred)
    assert mae == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([1, 0, 2]), 0.0),
])
def test_mbe(y_true, y_pred, value):
    mbe = deterministic.mean_bias(y_true, y_pred)
    assert mbe == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
    (np.array([0, 1]), np.array([1, 2]), 1.0),
])
def test_rmse(y_true, y_pred, value):
    rmse = deterministic.root_mean_square(y_true, y_pred)
    assert rmse == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([1, 1]), np.array([2, 2]), 100.0),
    (np.array([2, 2]), np.array([3, 3]), 50.0),
])
def test_mape(y_true, y_pred, value):
    mape = deterministic.mean_absolute_percentage(y_true, y_pred)
    assert mape == value


@pytest.mark.parametrize("y_true,y_pred,y_norm,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 1.0, 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 55.0, 0.0),
    (np.array([0, 1]), np.array([1, 2]), 1.0, 100.0),
    (np.array([0, 1]), np.array([1, 2]), 100.0, 1.0),
])
def test_nrmse(y_true, y_pred, y_norm, value):
    nrmse = deterministic.normalized_root_mean_square(y_true, y_pred, y_norm)
    assert nrmse == value


@pytest.mark.parametrize("y_true,y_pred,y_ref,value", [
    (np.array([0, 1]), np.array([0, 2]), np.array([0, 2]), 1.0 - 1.0 / 1.0),
    (np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), 1.0 - 1.0 / 2.0),
])
def test_skill(y_true, y_pred, y_ref, value):
    s = deterministic.forecast_skill(y_true, y_pred, y_ref)
    assert s == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1]), np.array([0, 1]), 1.0),
    (np.array([1, 2]), np.array([-1, -2]), -1.0),
])
def test_r(y_true, y_pred, value):
    r = deterministic.pearson_correlation_coeff(y_true, y_pred)
    assert pytest.approx(r) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1]), np.array([0, 1]), 1.0),
])
def test_r2(y_true, y_pred, value):
    r2 = deterministic.coeff_determination(y_true, y_pred)
    assert pytest.approx(r2) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
])
def test_crmse(y_true, y_pred, value):
    crmse = deterministic.centered_root_mean_square(y_true, y_pred)
    assert crmse == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
    ([0, 1], [0, 2], 0.5),
    ([0, 1, 2], [0, 0, 2], 1.0 / 6.0),
])
def test_ksi(y_true, y_pred, value):
    ksi = deterministic.kolmogorov_smirnov_integral(y_true, y_pred)
    assert pytest.approx(ksi) == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
])
def test_ksi_norm(y_true, y_pred, value):
    ksi = deterministic.kolmogorov_smirnov_integral(
        y_true, y_pred, normed=True
    )
    assert ksi == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
])
def test_over(y_true, y_pred, value):
    ov = deterministic.over(y_true, y_pred)
    assert ov == value


@pytest.mark.parametrize("y_true,y_pred,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
    (np.array([1, 2]), np.array([1, 2]), 0.0),
])
def test_cpi(y_true, y_pred, value):
    cpi = deterministic.combined_performance_index(y_true, y_pred)
    assert cpi == value
