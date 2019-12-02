import pytest
import numpy as np
from solarforecastarbiter.metrics import deterministic


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 1]), 1 / 3),
    (np.array([0, 1, 2]), np.array([0, 1, 3]), 1 / 3),
])
def test_mae(obs, fx, value):
    mae = deterministic.mean_absolute(obs, fx)
    assert mae == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([1, 0, 2]), 0.0),
    (np.array([0, 1, 2]), np.array([1, 2, 3]), 1.0),
    (np.array([0, 1, 2]), np.array([1, 3, 4]), (1 + 2 + 2) / 3),
    (np.array([5, 5, 5]), np.array([4, 4, 4]), -1.0),
    (np.array([5, 5, 5]), np.array([4, 3, 3]), -(1 + 2 + 2) / 3),
])
def test_mbe(obs, fx, value):
    mbe = deterministic.mean_bias(obs, fx)
    assert mbe == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
    (np.array([0, 1]), np.array([1, 2]), 1.0),
    (np.array([1, 2]), np.array([0, 1]), 1.0),
])
def test_rmse(obs, fx, value):
    rmse = deterministic.root_mean_square(obs, fx)
    assert rmse == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([1, 1]), np.array([2, 2]), 100.0),
    (np.array([2, 2]), np.array([3, 3]), 50.0),
    (np.array([1, 2]), np.array([1, 2]), 0.0),
])
def test_mape(obs, fx, value):
    mape = deterministic.mean_absolute_percentage(obs, fx)
    assert mape == value


@pytest.mark.parametrize("obs,fx,norm,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 55, 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 1]), 20, 1 / 3 / 20 * 100),
])
def test_nmae(obs, fx, norm, value):
    nmae = deterministic.normalized_mean_absolute(obs, fx, norm)
    assert nmae == value


@pytest.mark.parametrize("obs,fx,norm,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 55, 0.0),
    (np.array([0, 1, 2]), np.array([1, 0, 2]), 20, 0.0),
    (np.array([0, 1, 2]), np.array([1, 3, 4]), 7, (1 + 2 + 2) / 3 / 7 * 100),
    (np.array([5, 5, 5]), np.array([4, 4, 4]), 2, -1.0 / 2 * 100),
    (np.array([5, 5, 5]), np.array([4, 3, 3]), 2, -(1 + 2 + 2) / 3 / 2 * 100),
])
def test_nmbe(obs, fx, norm, value):
    nmbe = deterministic.normalized_mean_bias(obs, fx, norm)
    assert nmbe == value


@pytest.mark.parametrize("obs,fx,norm,value", [
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 1.0, 0.0),
    (np.array([0, 1, 2]), np.array([0, 1, 2]), 55.0, 0.0),
    (np.array([0, 1]), np.array([1, 2]), 1.0, 100.0),
    (np.array([0, 1]), np.array([1, 2]), 100.0, 1.0),
])
def test_nrmse(obs, fx, norm, value):
    nrmse = deterministic.normalized_root_mean_square(obs, fx, norm)
    assert nrmse == value


@pytest.mark.parametrize("obs,fx,ref,value", [
    (np.array([0, 1]), np.array([0, 2]), np.array([0, 2]), 0.0),
    (np.array([0, 1]), np.array([0, 2]), np.array([0, 3]), 0.5),
])
def test_skill(obs, fx, ref, value):
    s = deterministic.forecast_skill(obs, fx, ref)
    assert s == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1]), np.array([0, 1]), 1.0),
    (np.array([1, 2]), np.array([-1, -2]), -1.0),
])
def test_r(obs, fx, value):
    r = deterministic.pearson_correlation_coeff(obs, fx)
    assert pytest.approx(r) == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1]), np.array([0, 1]), 1.0),
    (np.array([1, 2, 3]), np.array([2, 2, 2]), 0.0),
])
def test_r2(obs, fx, value):
    r2 = deterministic.coeff_determination(obs, fx)
    assert pytest.approx(r2) == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
    (np.array([0, 2]), np.array([0, 4]), 1.0),
    (np.array([0, 2]), np.array([0, 6]), 2.0),
])
def test_crmse(obs, fx, value):
    crmse = deterministic.centered_root_mean_square(obs, fx)
    assert crmse == value


@pytest.mark.parametrize("obs,fx,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
    ([0, 1], [0, 2], 0.5),
    ([0, 1, 2], [0, 0, 2], 1.0 / 3.0),
])
def test_ksi(obs, fx, value):
    ksi = deterministic.kolmogorov_smirnov_integral(obs, fx)
    assert pytest.approx(ksi) == value


@pytest.mark.parametrize("obs,fx,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
    ([0, 1, 2], [0, 0, 2], 1 / 3 / (1.63 / np.sqrt(3) * 2) * 100),
])
def test_ksi_norm(obs, fx, value):
    ksi = deterministic.kolmogorov_smirnov_integral(
        obs, fx, normed=True
    )
    assert pytest.approx(ksi) == value


@pytest.mark.parametrize("obs,fx,value", [
    ([0, 1], [0, 1], 0.0),
    ([1, 2], [1, 2], 0.0),
    ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], 0.8 - 1.63 / np.sqrt(5)),
])
def test_over(obs, fx, value):
    ov = deterministic.over(obs, fx)
    assert ov == value


@pytest.mark.parametrize("obs,fx,value", [
    (np.array([0, 1]), np.array([0, 1]), 0.0),
    (np.array([1, 2]), np.array([1, 2]), 0.0),
    (
        np.array([0, 1, 2]),
        np.array([0, 0, 2]),
        1/4 * (1/3 + 0 + 2 * np.sqrt(1/3))
    ),
])
def test_cpi(obs, fx, value):
    cpi = deterministic.combined_performance_index(obs, fx)
    assert pytest.approx(cpi) == value
