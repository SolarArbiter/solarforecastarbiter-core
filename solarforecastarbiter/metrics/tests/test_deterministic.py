import datetime as dt
from functools import partial


import pandas as pd
from pandas.testing import assert_series_equal
import pytest
import numpy as np
from numpy.testing import assert_allclose


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import deterministic


@pytest.fixture
def error_func_deadband():
    return partial(deterministic.error_deadband, deadband=0.05)


@pytest.mark.parametrize('deadband,expected', [
    (0., [1, 0, 0, 0, 0]),
    (0.1, [1, 0, 0, 0, 1]),
    (1., [1, 0, 1, 0, 1]),
])
def test_deadband_mask(deadband, expected):
    obs = np.array([0, 0, 1, 0, 1.])
    fx = np.array([0, 1, 0, 1.05, 0.95])
    expected = np.array(expected, dtype=bool)
    out = deterministic.deadband_mask(obs, fx, deadband)
    assert_allclose(out, expected)


def test_error():
    obs = np.array([2, 1, 0.])
    fx = np.array([1, 2, 0.])
    expected = np.array([-1, 1, 0.])
    out = deterministic.error(obs, fx)
    assert_allclose(out, expected)


@pytest.mark.parametrize('deadband,expected', [
    (0., [0, 1, -1, 1.05, -0.05]),
    (0.1, [0, 1, -1, 1.05, 0]),
    (1., [0, 1, 0, 1.05, 0]),
])
def test_error_deadband(deadband, expected):
    obs = np.array([0, 0, 1, 0, 1.])
    fx = np.array([0, 1, 0, 1.05, 0.95])
    expected = np.array(expected)
    out = deterministic.error_deadband(obs, fx, deadband)
    assert_allclose(out, expected)


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
    (np.array([0, 1]), np.array([0, 2]), np.array([0, 1]), np.NINF),
    (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), 0.0),
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
    assert r == value


@pytest.mark.parametrize("obs,fx", [
    # len(obs) < 2 or len(fx) < 2
    (np.array([0]), np.array([1])),

    # len(obs) != len(fx)
    (np.array([0, 1, 2]), np.array([0, 1, 2, 3])),
    (np.array([2, 3, 4]), np.array([2, 3, 5, 6])),
])
def test_r_nan(obs, fx):
    r = deterministic.pearson_correlation_coeff(obs, fx)
    assert np.isnan(r)


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


@pytest.fixture
def deadband_obs_fx():
    obs = np.array([1, 2, 3, 4])
    # 2.1 and 3.8 are outside the 5% deadband on some platforms due to
    # floating point arithmetic errors
    fx = np.array([2, 2.09, 2, 3.81])
    return obs, fx


@pytest.mark.parametrize('func,expect,expect_deadband,args', [
    (deterministic.mean_absolute, 0.57, 0.5, []),
    (deterministic.mean_bias, -0.025, 0., []),
    (deterministic.root_mean_square,
     0.7148776119029046, 0.7071067811865476, []),
    (deterministic.mean_absolute_percentage,
     35.64583333333333, 33.33333333333333, []),
    (deterministic.normalized_mean_absolute, 5.7, 5.0, [10.]),
    (deterministic.normalized_mean_bias, -0.25, 0., [10.]),
    (deterministic.normalized_root_mean_square,
     7.148776119029046, 7.071067811865476, [10.]),
]
)
def test_deadband(func, error_func_deadband, deadband_obs_fx, expect,
                  expect_deadband, args):
    obs, fx = deadband_obs_fx
    out = func(obs, fx, *args)
    out_deadband = func(obs, fx, *args, error_fnc=error_func_deadband)
    assert_allclose(out, expect)
    assert_allclose(out_deadband, expect_deadband)


@pytest.fixture()
def default_cost_err():
    # sum: 8.6, netsum: 16.6
    return pd.Series([0, 0, -1, .1, 2.5, 2.1, 1.9, -3, 6],
                     index=pd.date_range(
                         start='2020-05-01T00:00Z', freq='1h', periods=9))


@pytest.mark.parametrize('params,exp', [
    (datamodel.ConstantCost(
        cost=1.0, aggregation='sum', net=True), 8.6),
    (datamodel.ConstantCost(
        cost=1.0, aggregation='sum', net=False), 16.6),
    (datamodel.ConstantCost(
        cost=0.0, aggregation='sum', net=False), 0),
    (datamodel.ConstantCost(
        cost=1.0, aggregation='mean', net=False), 16.6/9),
    (datamodel.ConstantCost(
        cost=1.0, aggregation='mean', net=True), 8.6/9),
    (datamodel.ConstantCost(
        cost=2.0, aggregation='sum', net=True), 17.2),
])
def test_constant_cost(default_cost_err, params, exp):
    fx = default_cost_err
    obs = pd.Series(0, index=fx.index)
    cost = deterministic._constant_cost(obs, fx, params,
                                        deterministic.error)
    assert cost == exp


@pytest.mark.parametrize('times,costs,index,tz,fill,exp', [
    ([dt.time(0), dt.time(2)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'UTC', 'ffill',
     pd.Series([1.0, 1.0, 2.0])),
    ([dt.time(0), dt.time(4)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'UTC', 'ffill',
     pd.Series([1.0, 1.0, 1.0])),
    ([dt.time(1), dt.time(2)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'UTC', 'ffill',
     pd.Series([2.0, 1.0, 2.0])),
    ([dt.time(1), dt.time(1, 30)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:01Z', freq='1h', periods=3),
     'UTC', 'bfill',
     pd.Series([1.0, 2.0, 1.0])),
    ([dt.time(17), dt.time(19)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'MST', 'ffill',
     pd.Series([1.0, 1.0, 2.0])),
    # missing first time
    ([dt.time(15), dt.time(17)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'Etc/GMT+10', 'ffill',
     pd.Series([2.0, 1.0, 1.0])),
    # missing last time
    ([dt.time(9), dt.time(11)], [1.0, 2.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=3),
     'Etc/GMT-10', 'bfill',
     pd.Series([2.0, 2.0, 1.0])),
    ([dt.time(9)], [1.0],
     pd.date_range('2020-01-01T00:00Z', freq='1h', periods=2),
     'Etc/GMT-1', 'bfill',
     pd.Series([1.0, 1.0])),
    ([dt.time(0, 5), dt.time(0, 10)], [1.0, -99.9],
     pd.date_range('2020-01-01T00:00Z', freq='5min', periods=3),
     'UTC', 'ffill',
     pd.Series([-99.9, 1.0, -99.9])),
])
def test_make_time_of_day_cost_ser(times, costs, index, tz, fill, exp):
    ser = deterministic._make_time_of_day_cost_ser(times, costs, index, tz,
                                                   fill)
    exp.index = index
    exp.name = 'cost'
    assert_series_equal(ser, exp)


def test_band_masks(banded_cost_params, default_cost_err):
    masks = deterministic._band_masks(banded_cost_params.parameters.bands,
                                      default_cost_err)
    expmasks = [
        pd.Series([1, 1, 1, 1, 0, 0, 1, 0, 0], dtype=bool,
                  index=default_cost_err.index),
        pd.Series([0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=bool,
                  index=default_cost_err.index),
        pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool,
                  index=default_cost_err.index),
    ]
    assert sum(masks).all()
    for i, m in enumerate(masks):
        assert_series_equal(m, expmasks[i])


@pytest.fixture(params=['constant', 'timeofday', 'datetime'])
def costs(request):
    if request.param == 'constant':
        return datamodel.ConstantCost(
            cost=1.0,
            aggregation='mean',
            net=False
        )
    elif request.param == 'timeofday':
        return datamodel.TimeOfDayCost(
            times=[dt.time(0), dt.time(12)],
            costs=[1.0, -1.0],
            aggregation='sum',
            net=True,
            fill='forward'
        )
    elif request.param == 'datetime':
        return datamodel.DatetimeCost(
            datetimes=[pd.Timestamp('2020-05-01T00:00Z'),
                       pd.Timestamp('2020-05-02T12:00Z')],
            costs=[1.0, 3.0],
            aggregation='sum',
            net=False,
            fill='forward'
        )


def test_error_band_cost(banded_cost_params):
    ind = pd.date_range(start='2020-05-01T00:00Z', freq='1h', periods=8)
    obs = pd.Series([1, 6, 1, 2, 2.00, 2.0, 0, 4], index=ind)
    fx = pd.Series([1., 2, 3, 0, -0.1, 1.9, 3, 5], index=ind)
    err = deterministic._error_band_cost(
        obs, fx, banded_cost_params.parameters, deterministic.error)
    exp = ((0 + 0 + 2 + -2 + 0.0 + -0.1 + 0 + 1) +
           (0 + 0 + 0 + 0. + 0.0 + 0.00 + 3 * 0.9 + 0) +
           (0 + -4 * -0.2 + 0 + 0. + -2.1 * -0.2 + 0.00 + 0 + 0))
    assert err == exp
