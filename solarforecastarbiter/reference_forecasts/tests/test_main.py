import datetime as dt
from functools import partial
from pathlib import Path
import re

import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.io import api, nwp, utils
from solarforecastarbiter.reference_forecasts import main, models
from solarforecastarbiter.conftest import default_forecast, default_observation

init_time = pd.Timestamp('20190515T0000Z')
start = pd.Timestamp('20190515T0100Z')
end = pd.Timestamp('20190515T0600Z')

index_exp = pd.date_range(start=start, end=end, freq='1h')
ghi_exp = pd.Series([0, 10.]*3, index=index_exp)
dni_exp = pd.Series([0, 15.]*3, index=index_exp)
dhi_exp = pd.Series([0, 9.]*3, index=index_exp)
temp_air_exp = pd.Series([10, 11.]*3, index=index_exp)
wind_speed_exp = pd.Series([0, 1.]*3, index=index_exp)
cloud_cover_exp = pd.Series([100., 0.]*3, index=index_exp)
load_forecast_return_value_3 = (cloud_cover_exp, temp_air_exp, wind_speed_exp)
load_forecast_return_value_5 = (
    ghi_exp, dni_exp, dhi_exp, temp_air_exp, wind_speed_exp)
out_forecast_exp = (ghi_exp, dni_exp, dhi_exp, temp_air_exp, wind_speed_exp)


def check_out(out, expected, site_type):
    assert len(out) == 6
    for o, e in zip(out[0:5], expected):
        assert isinstance(o, pd.Series)
        # assert_series_equal(o, e)
    if site_type == 'powerplant':
        assert isinstance(out[5], pd.Series)
    elif site_type == 'site':
        assert out[5] is None


@pytest.mark.parametrize('model,load_forecast_return_value', [
    (models.gfs_quarter_deg_hourly_to_hourly_mean,
     load_forecast_return_value_3),
    (models.gfs_quarter_deg_to_hourly_mean, load_forecast_return_value_3),
    (models.hrrr_subhourly_to_hourly_mean, load_forecast_return_value_5),
    (models.hrrr_subhourly_to_subhourly_instantaneous,
        load_forecast_return_value_5),
    pytest.param(models.nam_12km_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
    (models.nam_12km_hourly_to_hourly_instantaneous,
     load_forecast_return_value_3),
    pytest.param(models.rap_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
])
def test_run_nwp(model, load_forecast_return_value, site_powerplant_site_type,
             mocker):
    BASE_PATH = Path(nwp.__file__).resolve().parents[0] / 'tests/data'
    load_forecast = partial(nwp.load_forecast, base_path=BASE_PATH)
    model = partial(model, load_forecast=load_forecast)
    site, site_type = site_powerplant_site_type
    out = main.run_nwp(site, model, init_time, start, end)
    check_out(out, out_forecast_exp, site_type)


@pytest.fixture
def obs_5min_begin(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')
    return observation


@pytest.fixture
def observation_values_text():
    """JSON text representation of test data"""
    tz = 'UTC'
    data_index = pd.date_range(
        start='20190101', end='20190102', freq='5min', tz=tz, closed='left')
    # each element of data is equal to the hour value of its label
    data = pd.DataFrame({'value': data_index.hour, 'quality_flag': 0},
                        index=data_index)
    text = utils.observation_df_to_json_payload(data)
    return text.encode()


@pytest.fixture
def session(requests_mock, observation_values_text):
    session = api.APISession('')
    matcher = re.compile(f'{session.base_url}/observations/.*/values')
    requests_mock.register_uri('GET', matcher, content=observation_values_text)
    return session


def test_run_persistence_scalar(session, site_metadata, obs_5min_begin,
                                mocker):
    run_time = pd.Timestamp('20190101T1945Z')
    # intraday, index=False
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190101T2300Z')
    mocker.spy(main.persistence, 'persistence_scalar')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time)
    assert isinstance(out, pd.Series)
    assert main.persistence.persistence_scalar.call_count == 1


def test_run_persistence_scalar_index(session, site_metadata, obs_5min_begin,
                                      mocker):
    run_time = pd.Timestamp('20190101T1945Z')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190101T2300Z')
    # intraday, index=True
    mocker.spy(main.persistence, 'persistence_scalar_index')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time, index=True)
    assert isinstance(out, pd.Series)
    assert main.persistence.persistence_scalar_index.call_count == 1


def test_run_persistence_interval(session, site_metadata, obs_5min_begin,
                                  mocker):
    run_time = pd.Timestamp('20190102T1945Z')
    # day ahead, index = False
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190102T2300Z')
    mocker.spy(main.persistence, 'persistence_interval')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time)
    assert isinstance(out, pd.Series)
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_interval_index(session, site_metadata,
                                        obs_5min_begin):
    # index=True not supported for day ahead
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time, index=True)
    assert 'index=True not supported' in str(excinfo.value)


def test_run_persistence_interval_too_long(session, site_metadata,
                                           obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('48h'),  # too long
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time)
    assert 'midnight to midnight' in str(excinfo.value)


def test_run_persistence_interval_not_midnight_to_midnight(session,
                                                           site_metadata,
                                                           obs_5min_begin):
    # not midnight to midnight
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=22),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2200Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time)
    assert 'midnight to midnight' in str(excinfo.value)
