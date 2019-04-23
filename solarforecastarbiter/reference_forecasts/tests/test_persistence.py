from functools import partial

import pandas as pd
from pandas.util.testing import assert_series_equal

from solarforecastarbiter import datamodel
from solarforecastarbiter.reference_forecasts import persistence

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


def _observation(site_metadata, interval_length, interval_label):
    name = 'Albuquerque Baseline ghi'
    variable = 'ghi'
    value_type = 'mean'
    uncertainty = 1
    obs = datamodel.Observation(
        name=name, variable=variable,
        value_type=value_type, site=site_metadata, uncertainty=uncertainty,
        interval_length=pd.Timedelta(interval_length),
        interval_label=interval_label)
    return obs


@pytest.fixture
def powerplant_metadata():
    modeling_params = datamodel.FixedTiltModelingParameters(
        ac_capacity=200, dc_capacity=200, temperature_coefficient=-0.003,
        dc_loss_factor=3, ac_loss_factor=0,
        surface_tilt=30, surface_azimuth=180)
    metadata = datamodel.SolarPowerPlant(
        name='Albuquerque Baseline', latitude=35.05, longitude=-106.54,
        elevation=1657.0, timezone='America/Denver', network='Sandia',
        modeling_parameters=modeling_params)
    return metadata


def _observation_ac(site_metadata, interval_length, interval_label):
    name = 'Test AC Power'
    variable = 'ac_power'
    value_type = 'mean'
    uncertainty = 1
    obs = datamodel.Observation(
        name=name, variable=variable,
        value_type=value_type, site=site_metadata, uncertainty=uncertainty,
        interval_length=pd.Timedelta(interval_length),
        interval_label=interval_label)
    return obs


def test_persistence_scalar(site_metadata):
    # interval beginning obs
    observation = _observation(site_metadata, '5min', 'beginning')
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    data = pd.Series(100., index=data_index)
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    interval_length = pd.Timedelta('5min')

    # interval beginning fx
    interval_label = 'beginning'
    load_data = partial(load_data_base, data)
    fx = persistence.persistence_scalar(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190404 1300', end='20190404 1400', freq='5min', tz=tz,
        closed='left')
    expected = pd.Series(100., index=expected_index)
    assert_series_equal(fx, expected)

    # interval ending fx
    interval_label = 'ending'
    fx = persistence.persistence_scalar(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190404 1300', end='20190404 1400', freq='5min', tz=tz,
        closed='right')
    expected = pd.Series(100., index=expected_index)
    assert_series_equal(fx, expected)

    # instantaneous obs and fx
    observation = _observation(site_metadata, '5min', 'instant')
    interval_label = 'instant'
    forecast_end = pd.Timestamp('20190404 1359', tz=tz)
    fx = persistence.persistence_scalar(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190404 1300', end='20190404 1359', freq='5min', tz=tz)
    expected = pd.Series(100., index=expected_index)
    assert_series_equal(fx, expected)


def test_persistence_interval(site_metadata):
    # interval beginning obs
    observation = _observation(site_metadata, '5min', 'beginning')
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    # each element of data is equal to the hour value of its label
    data = pd.Series(data_index.hour, index=data_index)
    data_start = pd.Timestamp('20190404 0000', tz=tz)
    data_end = pd.Timestamp('20190405 0000', tz=tz)
    forecast_start = pd.Timestamp('20190405 0000', tz=tz)
    interval_length = pd.Timedelta('60min')

    # interval beginning fx
    interval_label = 'beginning'
    load_data = partial(load_data_base, data)
    fx = persistence.persistence_interval(
        observation, data_start, data_end, forecast_start,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190405 0000', end='20190406 0000', freq='60min', tz=tz,
        closed='left')
    expected_vals = list(range(0, 24))
    expected = pd.Series(expected_vals, index=expected_index)
    assert_series_equal(fx, expected)

    # interval ending fx
    interval_label = 'ending'
    fx = persistence.persistence_interval(
        observation, data_start, data_end, forecast_start,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190405 0000', end='20190406 0000', freq='60min', tz=tz,
        closed='right')
    expected = pd.Series(expected_vals, index=expected_index)
    assert_series_equal(fx, expected)
    assert len(fx) == 24

    # instantaneous obs and fx
    observation = _observation(site_metadata, '5min', 'instant')
    data_end = pd.Timestamp('20190404 2359', tz=tz)
    interval_label = 'instant'
    fx = persistence.persistence_interval(
        observation, data_start, data_end, forecast_start,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.date_range(
        start='20190405 0000', end='20190405 2359', freq='60min', tz=tz)
    expected = pd.Series(expected_vals, index=expected_index)
    assert_series_equal(fx, expected)


def test_persistence_scalar_index(site_metadata, powerplant_metadata):
    # interval beginning obs
    observation = _observation(site_metadata, '5min', 'beginning')
    observation_ac = _observation_ac(powerplant_metadata, '5min', 'beginning')
    tz = 'America/Phoenix'
    data_index = pd.date_range(
        start='20190404', end='20190406', freq='5min', tz=tz)
    data = pd.Series(100., index=data_index)
    data_start = pd.Timestamp('20190404 1200', tz=tz)
    data_end = pd.Timestamp('20190404 1300', tz=tz)
    forecast_start = pd.Timestamp('20190404 1300', tz=tz)
    forecast_end = pd.Timestamp('20190404 1400', tz=tz)
    interval_length = pd.Timedelta('30min')

    # interval beginning fx
    interval_label = 'beginning'
    load_data = partial(load_data_base, data)
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.DatetimeIndex(
        ['20190404 1300', '20190404 1330'], tz=tz)
    expected_values = [96.41150694741889, 91.6991546408236]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_values = [99.28349914087346, 98.28165269708589]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    # interval ending fx
    interval_label = 'ending'
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.DatetimeIndex(
        ['20190404 1330', '20190404 1400'], tz=tz)
    expected_values = [96.2818141290749, 91.5132934827808]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_values = [99.25690632023922, 98.2405479197069]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    # instantaneous obs and fx
    observation = _observation(site_metadata, '5min', 'instant')
    observation_ac = _observation_ac(powerplant_metadata, '5min', 'instant')
    interval_label = 'instant'
    data_end = pd.Timestamp('20190404 1259', tz=tz)
    forecast_end = pd.Timestamp('20190404 1359', tz=tz)
    fx = persistence.persistence_scalar_index(
        observation, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_index = pd.DatetimeIndex(
        ['20190404 1300', '20190404 1330'], tz=tz)
    expected_values = [96.59022431746838, 91.99405501672328]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)

    fx = persistence.persistence_scalar_index(
        observation_ac, data_start, data_end, forecast_start, forecast_end,
        interval_length, interval_label, load_data=load_data)
    expected_values = [99.32046515783028, 98.34762206379594]
    expected = pd.Series(expected_values, index=expected_index)
    assert_series_equal(fx, expected, check_names=False)
