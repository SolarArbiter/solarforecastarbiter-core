import datetime as dt
from functools import partial
import inspect
from pathlib import Path
import re


import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api, nwp, utils
from solarforecastarbiter.reference_forecasts import main, models
from solarforecastarbiter.conftest import default_forecast, default_observation


BASE_PATH = Path(nwp.__file__).resolve().parents[0] / 'tests/data'


@pytest.mark.parametrize('model', [
    models.gfs_quarter_deg_hourly_to_hourly_mean,
    models.gfs_quarter_deg_to_hourly_mean,
    models.hrrr_subhourly_to_hourly_mean,
    models.hrrr_subhourly_to_subhourly_instantaneous,
    models.nam_12km_cloud_cover_to_hourly_mean,
    models.nam_12km_hourly_to_hourly_instantaneous,
    models.rap_cloud_cover_to_hourly_mean,
])
def test_run_nwp(model, site_powerplant_site_type, mocker):
    """ to later patch the return value of load forecast, do something like
    def load(*args, **kwargs):
        return load_forecast_return_value
    mocker.patch.object(inspect.unwrap(model), '__defaults__',
        (partial(load),))
    """
    mocker.patch.object(inspect.unwrap(model), '__defaults__',
                        (partial(nwp.load_forecast, base_path=BASE_PATH),))
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.utils.get_init_time',
        return_value=pd.Timestamp('20190515T0000Z'))
    site, site_type = site_powerplant_site_type
    fx = datamodel.Forecast('Test', dt.time(5), pd.Timedelta('1h'),
                            pd.Timedelta('1h'), pd.Timedelta('6h'),
                            'beginning', 'interval_mean', 'ghi', site)
    run_time = pd.Timestamp('20190515T1100Z')
    issue_time = pd.Timestamp('20190515T1100Z')
    out = main.run_nwp(fx, model, run_time, issue_time)

    for var in ('ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed',
                'ac_power'):
        if site_type == 'site' and var == 'ac_power':
            assert out.ac_power is None
        else:
            ser = getattr(out, var)
            assert len(ser) >= 6
            assert isinstance(ser, pd.Series)
            assert ser.index[0] == pd.Timestamp('20190515T1200Z')
            assert ser.index[-1] < pd.Timestamp('20190515T1800Z')


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
