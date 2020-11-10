from contextlib import nullcontext as does_not_raise
from functools import partial

import pandas as pd
from pandas.testing import assert_series_equal

from solarforecastarbiter import datamodel
from solarforecastarbiter.reference_forecasts import persistence
from solarforecastarbiter.conftest import default_observation

import pytest


def load_data_base(data, observation, data_start, data_end):
    # slice doesn't care about closed or interval label
    # so here we manually adjust start and end times
    if 'instant' in observation.interval_label:
        pass
    elif observation.interval_label == 'ending':
        data_start += pd.Timedelta('1s')
    elif observation.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    return data[data_start:data_end]


@pytest.fixture
def powerplant_metadata():
    """1:1 AC:DC"""
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=200, dc_capacity=200, temperature_coefficient=-0.3,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    metadata = datamodel.SolarPowerPlant(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver',
        modeling_parameters=modeling_params)
    return metadata


@pytest.mark.parametrize('interval_label,closed,end', [
    ('beginning', 'left', '20190404 1400'),
    ('ending', 'right', '20190404 1400'),
    ('instant', None, '20190404 1359')
])
def test_persistence_scalar(site_metadata, interval_label, closed, end):
    # interval beginning obs
    observation = default_observation(
        site_metadata, interval_length='5min', interval_label=interval_label)
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    data = pd.Series(100., index=data_index)
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp(end, tz=tz)
    interval_length = pd.Timedelta('5min')

    load_data = partial(load_data_base, data)

    fx = persistence.persistence_scalar(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)

    expected_index = pd.date_range(
        start='20190404 1300', end=end, freq='5min', tz=tz,
        closed=closed)
    expected = pd.Series(100., index=expected_index)
    assert_series_equal(fx, expected)


@pytest.mark.parametrize('obs_interval_label', ('beginning', 'ending',
                                                'instant'))
@pytest.mark.parametrize('interval_label,closed,end', [
    ('beginning', 'left', '20190406 0000'),
    ('ending', 'right', '20190406 0000'),
    ('instant', None, '20190405 2359')
])
def test_persistence_interval(site_metadata, obs_interval_label,
                              interval_label, closed, end):
    # interval beginning obs
    observation = default_observation(
        site_metadata, interval_length='5min',
        interval_label=obs_interval_label)
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    # each element of data is equal to the hour value of its label
    data = pd.Series(data_index.hour, index=data_index, dtype=float)
    if obs_interval_label == 'ending':
        # e.g. timestamp 12:00:00 should be equal to 11
        data = data.shift(1).fillna(0)
    data_start = pd.Timestamp('20190404 0000', tz=tz)
    data_end = pd.Timestamp(end, tz=tz) - pd.Timedelta('1d')
    forecast_start = pd.Timestamp('20190405 0000', tz=tz)
    interval_length = pd.Timedelta('60min')

    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(
        start='20190405 0000', end=end, freq='60min', tz=tz, closed=closed)
    expected_vals = list(range(0, 24))
    expected = pd.Series(expected_vals, index=expected_index, dtype=float)

    # handle permutations of parameters that should fail
    if data_end.minute == 59 and obs_interval_label != 'instant':
        expectation = pytest.raises(ValueError)
    elif data_end.minute == 0 and obs_interval_label == 'instant':
        expectation = pytest.raises(ValueError)
    else:
        expectation = does_not_raise()

    with expectation:
        fx = persistence.persistence_interval(
            observation, data_start, data_end, forecast_start,
            interval_length, interval_label, load_data)
        assert_series_equal(fx, expected)


def test_persistence_interval_missing_data(site_metadata):
    # interval beginning obs
    observation = default_observation(
        site_metadata, interval_length='5min',
        interval_label='ending')
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404T1200', end='20190406', freq='5min', tz=tz)
    # each element of data is equal to the hour value of its label
    end = '20190406 0000'
    data = pd.Series(data_index.hour, index=data_index, dtype=float)
    data = data.shift(1)
    data_start = pd.Timestamp('20190404 0000', tz=tz)
    data_end = pd.Timestamp(end, tz=tz) - pd.Timedelta('1d')
    forecast_start = pd.Timestamp('20190405 0000', tz=tz)
    interval_length = pd.Timedelta('60min')

    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(
        start='20190405 0000', end=end, freq='60min', tz=tz, closed='right')
    expected_vals = [None] * 12 + list(range(12, 24))
    expected = pd.Series(expected_vals, index=expected_index, dtype=float)
    fx = persistence.persistence_interval(
        observation, data_start, data_end, forecast_start,
        interval_length, 'ending', load_data)
    assert_series_equal(fx, expected)


@pytest.fixture
def uniform_data():
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    data = pd.Series(100., index=data_index)
    return data


@pytest.mark.parametrize(
    'interval_label,expected_index,expected_ghi,expected_ac,obsscale', (
        ('beginning',
         ['20190404 1300', '20190404 1330'],
         [96.41150694741889, 91.6991546408236],
         [99.28349914087346, 98.28165269708589],
         1),
        ('ending',
         ['20190404 1330', '20190404 1400'],
         [96.2818141290749, 91.5132934827808],
         [99.25690632023922, 98.2405479197069],
         1),
        # test clipped at 2x clearsky
        ('beginning',
         ['20190404 1300', '20190404 1330'],
         [1926.5828549018618, 1832.4163238767312],
         [395.9216674046528, 391.9265149579709],
         50)
    )
)
def test_persistence_scalar_index(
        powerplant_metadata, uniform_data, interval_label,
        expected_index, expected_ghi, expected_ac, obsscale):
    # ac_capacity is 200 from above
    observation = default_observation(
        powerplant_metadata, interval_length='5min',
        interval_label='beginning')
    observation_ac = default_observation(
        powerplant_metadata, interval_length='5min',
        interval_label='beginning', variable='ac_power')

    data = uniform_data * obsscale
    tz = data.index.tzinfo
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    interval_length = pd.Timedelta('30min')

    load_data = partial(load_data_base, data)
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    expected_index = pd.DatetimeIndex(expected_index, tz=tz)
    expected = pd.Series(expected_ghi, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    expected = pd.Series(expected_ac, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)


def test_persistence_scalar_index_instant_obs_fx(
        site_metadata, powerplant_metadata, uniform_data):
    # instantaneous obs and fx
    interval_length = pd.Timedelta('30min')
    interval_label = 'instant'
    observation = default_observation(
        site_metadata, interval_length='5min', interval_label=interval_label)
    observation_ac = default_observation(
        powerplant_metadata, interval_length='5min',
        interval_label=interval_label, variable='ac_power')
    data = uniform_data
    tz = data.index.tzinfo
    load_data = partial(load_data_base, data)
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1259', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1359', tz=tz)
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    expected_index = pd.DatetimeIndex(
        ['20190404 1300', '20190404 1330'], tz=tz)
    expected_values = [96.59022431746838, 91.99405501672328]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    expected_values = [99.32046515783028, 98.34762206379594]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    # instant obs and fx, but with offset added to starts instead of ends
    data_start = pd.Timestamp('20190404 1201', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1301', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    expected_index = pd.DatetimeIndex(
        ['20190404 1300', '20190404 1330'], tz=tz)
    expected_values = [96.55340033645147, 91.89662922267517]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)


def test_persistence_scalar_index_invalid_times_instant(site_metadata):
    data = pd.Series(100., index=[0])
    load_data = partial(load_data_base, data)
    tz = 'America/Phoenix'
    interval_label = 'instant'
    observation = default_observation(
        site_metadata, interval_length='5min', interval_label=interval_label)
    # instant obs that cover the whole interval - not allowed!
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    interval_length = pd.Timedelta('30min')
    with pytest.raises(ValueError):
        persistence.persistence_scalar_index(
            observation, data_start, data_end, forecast_start, forecast_end,
            interval_length, interval_label, load_data)


@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
@pytest.mark.parametrize('data_start,data_end,forecast_start,forecast_end', (
    ('20190404 1201', '20190404 1300', '20190404 1300', '20190404 1400'),
    ('20190404 1200', '20190404 1259', '20190404 1300', '20190404 1400'),
    ('20190404 1200', '20190404 1300', '20190404 1301', '20190404 1400'),
    ('20190404 1200', '20190404 1300', '20190404 1300', '20190404 1359'),
))
def test_persistence_scalar_index_invalid_times_interval(
        site_metadata, interval_label, data_start, data_end, forecast_start,
        forecast_end):
    data = pd.Series(100., index=[0])
    load_data = partial(load_data_base, data)
    tz = 'America/Phoenix'
    interval_length = pd.Timedelta('30min')

    # base times to mess with
    data_start = pd.Timestamp(data_start, tz=tz)
    data_end = pd.Timestamp(data_end, tz=tz)
    forecast_start = pd.Timestamp(forecast_start, tz=tz)
    forecast_end = pd.Timestamp(forecast_end, tz=tz)

    # interval average obs with invalid starts/ends
    observation = default_observation(
        site_metadata, interval_length='5min', interval_label=interval_label)
    errtext = "with interval_label beginning or ending"
    with pytest.raises(ValueError) as excinfo:
        persistence.persistence_scalar_index(
            observation, data_start, data_end, forecast_start, forecast_end,
            interval_length, interval_label, load_data)
    assert errtext in str(excinfo.value)


def test_persistence_scalar_index_invalid_times_invalid_label(site_metadata):
    data = pd.Series(100., index=[0])
    load_data = partial(load_data_base, data)
    tz = 'America/Phoenix'
    interval_length = pd.Timedelta('30min')

    interval_label = 'invalid'
    observation = default_observation(
        site_metadata, interval_length='5min')
    object.__setattr__(observation, 'interval_label', interval_label)
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    with pytest.raises(ValueError) as excinfo:
        persistence.persistence_scalar_index(
            observation, data_start, data_end, forecast_start, forecast_end,
            interval_length, interval_label, load_data)
    assert "invalid interval_label" in str(excinfo.value)


def test_persistence_scalar_index_low_solar_elevation(
        site_metadata, powerplant_metadata):

    interval_label = 'beginning'
    observation = default_observation(
        site_metadata, interval_length='5min', interval_label=interval_label)
    observation_ac = default_observation(
        powerplant_metadata, interval_length='5min',
        interval_label=interval_label, variable='ac_power')

    # at ABQ Baseline, solar apparent zenith for these points is
    # 2019-05-13 12:00:00+00:00     91.62
    # 2019-05-13 12:05:00+00:00     90.09
    # 2019-05-13 12:10:00+00:00     89.29
    # 2019-05-13 12:15:00+00:00     88.45
    # 2019-05-13 12:20:00+00:00     87.57
    # 2019-05-13 12:25:00+00:00     86.66

    tz = 'UTC'
    data_start = pd.Timestamp('20190513 1200', tz=tz)
    data_end = pd.Timestamp('20190513 1230', tz=tz)
    index = pd.date_range(start=data_start, end=data_end,
                          freq='5min', closed='left')

    # clear sky 5 min avg (from 1 min avg) GHI is
    # [0., 0.10932908, 1.29732454, 4.67585122, 10.86548521, 19.83487399]
    # create data series that could produce obs / clear of
    # 0/0, 1/0.1, -1/1.3, 5/5, 10/10, 20/20
    # average without limits is (10 - 1 + 1 + 1 + 1) / 5 = 2.4
    # average with element limits of [0, 2] = (2 + 0 + 1 + 1 + 1) / 5 = 1

    data = pd.Series([0, 1, -1, 5, 10, 20.], index=index)
    forecast_start = pd.Timestamp('20190513 1230', tz=tz)
    forecast_end = pd.Timestamp('20190513 1300', tz=tz)
    interval_length = pd.Timedelta('5min')
    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(
        start=forecast_start, end=forecast_end, freq='5min', closed='left')

    # clear sky 5 min avg GHI is
    # [31.2, 44.5, 59.4, 75.4, 92.4, 110.1]
    expected_vals = [31.2, 44.5, 59.4, 75.4, 92.4, 110.1]
    expected = pd.Series(expected_vals, index=expected_index)

    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    assert_series_equal(fx, expected, check_less_precise=1, check_names=False)

    expected = pd.Series([0.2, 0.7, 1.2, 1.6, 2., 8.9], index=expected_index)
    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data)
    assert_series_equal(fx, expected, check_less_precise=1, check_names=False)


@pytest.mark.parametrize("interval_label", [
    'beginning', 'ending'
])
@pytest.mark.parametrize("obs_values,axis,constant_values,expected_values", [
    # constant_values = variable values
    # forecasts = percentiles [%]
    ([0, 0, 0, 20, 20, 20], 'x', [10, 20], [50, 100]),

    # constant_values = percentiles [%]
    # forecasts = variable values
    ([0, 0, 0, 4, 4, 4], 'y', [50], [2]),

    # invalid axis
    pytest.param([0, 0, 0, 4, 4, 4], 'percentile', [-1], None,
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
])
def test_persistence_probabilistic(site_metadata, interval_label, obs_values,
                                   axis, constant_values, expected_values):

    tz = 'UTC'
    interval_length = '5min'
    observation = default_observation(
        site_metadata,
        interval_length=interval_length,
        interval_label=interval_label
    )

    data_start = pd.Timestamp('20190513 1200', tz=tz)
    data_end = pd.Timestamp('20190513 1230', tz=tz)
    closed = datamodel.CLOSED_MAPPING[interval_label]
    index = pd.date_range(start=data_start, end=data_end, freq='5min',
                          closed=closed)

    data = pd.Series(obs_values, index=index, dtype=float)
    forecast_start = pd.Timestamp('20190513 1230', tz=tz)
    forecast_end = pd.Timestamp('20190513 1300', tz=tz)
    interval_length = pd.Timedelta('5min')
    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    forecasts = persistence.persistence_probabilistic(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, constant_values
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(expected_values[i], index=expected_index, dtype=float)
        )


@pytest.mark.parametrize("obs_values,axis,constant_values,expected_values", [
    # constant_values = variable values
    # forecasts = percentiles [%]
    ([0] * 11 + [20] * 11, 'x', [10, 20], [50, 100]),
    ([0] * 11 + [20] * 11, 'x', [10, 20], [50, 100]),

    # constant_values = percentiles [%]
    # forecasts = variable values
    ([0] * 11 + [4] * 11, 'y', [50], [2]),

    # invalid axis
    pytest.param([0] * 11 + [4] * 11, 'percentile', [-1], None,
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),

    # insufficient observation data
    pytest.param([5.3, 7.3, 1.4] * 4, 'x', [50], None,
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    pytest.param([], 'x', [50], None,
                 marks=pytest.mark.xfail(raises=ValueError, strict=True))
])
def test_persistence_probabilistic_timeofday(site_metadata, obs_values, axis,
                                             constant_values, expected_values):

    tz = 'UTC'
    interval_label = "beginning"
    interval_length = '1h'
    observation = default_observation(
        site_metadata,
        interval_length=interval_length,
        interval_label=interval_label
    )

    # all observations at 9am each day
    data_end = pd.Timestamp('20190513T0900', tz=tz)
    data_start = data_end - pd.Timedelta("{}D".format(len(obs_values)))
    closed = datamodel.CLOSED_MAPPING[interval_label]
    index = pd.date_range(start=data_start, end=data_end, freq='1D',
                          closed=closed)
    data = pd.Series(obs_values, index=index, dtype=float)

    # forecast 9am
    forecast_start = pd.Timestamp('20190514T0900', tz=tz)
    forecast_end = pd.Timestamp('20190514T1000', tz=tz)
    interval_length = pd.Timedelta('1h')

    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    forecasts = persistence.persistence_probabilistic_timeofday(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, constant_values
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(expected_values[i], index=expected_index, dtype=float)
        )


@pytest.mark.parametrize("data_end,forecast_start", [
    # no timezone
    (pd.Timestamp("20190513T0900"), pd.Timestamp("20190514T0900")),

    # same timezone
    (
        pd.Timestamp("20190513T0900", tz="UTC"),
        pd.Timestamp("20190514T0900", tz="UTC")
    ),

    # different timezone
    (
        pd.Timestamp("20190513T0200", tz="US/Pacific"),
        pd.Timestamp("20190514T0900", tz="UTC")
    ),

    # obs timezone, but no fx timezone
    (
        pd.Timestamp("20190513T0900", tz="UTC"),
        pd.Timestamp("20190514T0900")
    ),

    # no obs timezone, but fx timezone
    (
        pd.Timestamp("20190513T0900"),
        pd.Timestamp("20190514T0900", tz="UTC")
    ),
])
def test_persistence_probabilistic_timeofday_timezone(site_metadata, data_end,
                                                      forecast_start):

    obs_values = [0] * 11 + [20] * 11
    axis, constant_values, expected_values = 'x', [10, 20], [50, 100]

    interval_label = "beginning"
    interval_length = '1h'
    observation = default_observation(
        site_metadata,
        interval_length=interval_length,
        interval_label=interval_label
    )

    # all observations at 9am each day
    data_start = data_end - pd.Timedelta("{}D".format(len(obs_values)))
    closed = datamodel.CLOSED_MAPPING[interval_label]
    index = pd.date_range(start=data_start, end=data_end, freq='1D',
                          closed=closed)
    data = pd.Series(obs_values, index=index, dtype=float)

    # forecast 9am
    forecast_end = forecast_start + pd.Timedelta("1h")
    interval_length = pd.Timedelta('1h')

    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    # if forecast without timezone, then use obs timezone
    if data.index.tzinfo is not None and forecast_start.tzinfo is None:
        expected_index = expected_index.tz_localize(data.index.tzinfo)

    forecasts = persistence.persistence_probabilistic_timeofday(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, constant_values
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(expected_values[i], index=expected_index, dtype=float)
        )


@pytest.mark.parametrize("interval_label", [
    'beginning', 'ending'
])
@pytest.mark.parametrize("obs_values,axis,constant_values,expected_values", [
    # constant_values = variable values
    # forecasts = percentiles [%]
    ([0] * 15 + [20] * 15, 'x', [10, 20], [50, 100]),

    # constant_values = percentiles [%]
    # forecasts = variable values
    ([0] * 15 + [4] * 15, 'y', [50], [2]),

    ([None] * 30, 'y', [50], [None])
])
def test_persistence_probabilistic_resampling(
    site_metadata,
    interval_label,
    obs_values, axis,
    constant_values,
    expected_values
):

    tz = 'UTC'
    interval_length = '1min'
    observation = default_observation(
        site_metadata,
        interval_length=interval_length,
        interval_label=interval_label
    )

    data_start = pd.Timestamp('20190513 1200', tz=tz)
    data_end = pd.Timestamp('20190513 1230', tz=tz)
    closed = datamodel.CLOSED_MAPPING[interval_label]
    index = pd.date_range(start=data_start, end=data_end, freq='1min',
                          closed=closed)

    data = pd.Series(obs_values, index=index, dtype=float)
    forecast_start = pd.Timestamp('20190513 1230', tz=tz)
    forecast_end = pd.Timestamp('20190513 1300', tz=tz)
    interval_length = pd.Timedelta('5min')
    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    forecasts = persistence.persistence_probabilistic(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, constant_values
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(expected_values[i], index=expected_index, dtype=float)
        )


@pytest.mark.parametrize("obs_values,axis,constant_values,expected_values", [
    # constant_values = variable values
    # forecasts = percentiles [%]
    ([0] * 22 + [20] * 22, 'x', [10, 20], [50, 100]),

    # constant_values = percentiles [%]
    # forecasts = variable values
    ([0] * 22 + [4] * 22, 'y', [50], [2]),
])
def test_persistence_probabilistic_timeofday_resample(
    site_metadata,
    obs_values,
    axis,
    constant_values,
    expected_values
):

    tz = 'UTC'
    interval_label = "beginning"
    observation = default_observation(
        site_metadata,
        interval_length='30min',
        interval_label=interval_label
    )

    # all observations at 9am each day
    data_end = pd.Timestamp('20190513T0900', tz=tz)
    data_start = data_end - pd.Timedelta("{}D".format(len(obs_values) / 2))
    closed = datamodel.CLOSED_MAPPING[interval_label]
    index = pd.date_range(start=data_start, end=data_end, freq='30min',
                          closed=closed)
    index = index[index.hour == 9]

    data = pd.Series(obs_values, index=index, dtype=float)

    # forecast 9am
    forecast_start = pd.Timestamp('20190514T0900', tz=tz)
    forecast_end = pd.Timestamp('20190514T1000', tz=tz)
    interval_length = pd.Timedelta('1h')

    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    forecasts = persistence.persistence_probabilistic_timeofday(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, constant_values
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(expected_values[i], index=expected_index, dtype=float)
        )


@pytest.mark.parametrize("interval_label", [
    'beginning', 'ending'
])
@pytest.mark.parametrize('axis', ['x', 'y'])
def test_persistence_probabilistic_no_data(
        site_metadata, interval_label, axis):

    tz = 'UTC'
    interval_length = '5min'
    observation = default_observation(
        site_metadata,
        interval_length=interval_length,
        interval_label=interval_label
    )

    data_start = pd.Timestamp('20190513 1200', tz=tz)
    data_end = pd.Timestamp('20190513 1230', tz=tz)
    closed = datamodel.CLOSED_MAPPING[interval_label]

    data = pd.Series([], index=pd.DatetimeIndex([], tz=tz), dtype=float)
    forecast_start = pd.Timestamp('20190513 1230', tz=tz)
    forecast_end = pd.Timestamp('20190513 1300', tz=tz)
    interval_length = pd.Timedelta('5min')
    load_data = partial(load_data_base, data)

    expected_index = pd.date_range(start=forecast_start, end=forecast_end,
                                   freq=interval_length, closed=closed)

    forecasts = persistence.persistence_probabilistic(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data, axis, [0.0, 25.0, 50.0]
    )
    assert isinstance(forecasts, list)
    for i, fx in enumerate(forecasts):
        pd.testing.assert_index_equal(fx.index, expected_index,
                                      check_categorical=False)

        pd.testing.assert_series_equal(
            fx,
            pd.Series(None, index=expected_index, dtype=float)
        )
