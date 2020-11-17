from dataclasses import replace
import datetime as dt
from functools import partial
import inspect
from pathlib import Path
import re
import types
import uuid


import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
import pytz


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
    models.gefs_half_deg_to_hourly_mean
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
            assert isinstance(ser, (pd.Series, pd.DataFrame))
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
        start='20190101', end='20190112', freq='5min', tz=tz, closed='left')
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


@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
def test_run_persistence_scalar(session, site_metadata, obs_5min_begin,
                                interval_label, mocker):
    run_time = pd.Timestamp('20190101T1945Z')
    # intraday, index=False
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label=interval_label)
    issue_time = pd.Timestamp('20190101T2300Z')
    mocker.spy(main.persistence, 'persistence_scalar')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time)
    assert isinstance(out, pd.Series)
    assert len(out) == 1
    assert main.persistence.persistence_scalar.call_count == 1


@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
def test_run_persistence_scalar_index(session, site_metadata, obs_5min_begin,
                                      interval_label, mocker):
    run_time = pd.Timestamp('20190101T1945Z')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label=interval_label)
    issue_time = pd.Timestamp('20190101T2300Z')
    # intraday, index=True
    mocker.spy(main.persistence, 'persistence_scalar_index')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time, index=True)
    assert isinstance(out, pd.Series)
    assert len(out) == 1
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
    assert len(out) == 24
    assert main.persistence.persistence_interval.call_count == 1


@pytest.mark.parametrize('il', ['ending', 'beginning'])
def test_run_persistence_interval_assert_data(
        session, site_metadata, obs_5min_begin, mocker, il):
    run_time = pd.Timestamp('20190102T1945Z')
    # day ahead, index = False
    obs = obs_5min_begin.replace(interval_label=il)
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190102T2300Z')
    mocker.spy(main.persistence, 'persistence_interval')
    data_start = pd.Timestamp('20190101T0000Z')
    data_end = pd.Timestamp('20190102T0000Z')
    index = pd.date_range(start=data_start, end=data_end, freq='1h')
    data = pd.Series([0, 1, 2] + [0] * 22, index=index)
    data = utils.adjust_timeseries_for_interval_label(
        data, il, data_start, data_end)
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, obs, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (data_start, data_end)
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    if il == 'ending':
        assert out.loc['20190103T0000Z'] == 1
        assert out.loc['20190103T0100Z'] == 2
        assert out.loc['20190103T0200Z'] == 0
    else:
        assert out.loc['20190103T0000Z'] == 0
        assert out.loc['20190103T0100Z'] == 1
        assert out.loc['20190103T0200Z'] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_interval_two_day(
        session, site_metadata, obs_5min_begin, mocker):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=0),
        lead_time_to_start=pd.Timedelta('2d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    run_time = pd.Timestamp('20190102T1945Z')
    issue_time = pd.Timestamp('20190103T0000Z')
    mocker.spy(main.persistence, 'persistence_interval')
    index = pd.date_range('20190101T0000Z', periods=24, freq='1h')
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190101T0000Z'),
                                          pd.Timestamp('20190102T0000Z'))
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    assert out.loc['20190105T0000Z'] == 0
    assert out.loc['20190105T0100Z'] == 1
    assert out.loc['20190105T0200Z'] == 2
    assert main.persistence.persistence_interval.call_count == 1


@pytest.mark.parametrize('fxil', ['ending', 'beginning'])
@pytest.mark.parametrize('obsil', ['ending', 'beginning'])
def test_run_persistence_interval_tz(session, site_metadata, obs_5min_begin,
                                     mocker, fxil, obsil):
    run_time = pd.Timestamp('20190102T2245-05:00')
    site = site_metadata.replace(timezone='Etc/GMT+5')
    # day ahead, index = False
    obs = obs_5min_begin.replace(site=site, interval_label=obsil)
    forecast = default_forecast(
        site,
        issue_time_of_day=pytz.timezone('Etc/GMT+5').localize(
            dt.time(hour=23)),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label=fxil)
    issue_time = pd.Timestamp('20190102T2300-05:00')
    mocker.spy(main.persistence, 'persistence_interval')
    data_start = pd.Timestamp('20190101T0500Z')
    data_end = pd.Timestamp('20190102T0500Z')
    index = pd.date_range(start=data_start, end=data_end, freq='1h')
    data = pd.Series([0, 1, 2] + [0] * 22, index=index)
    data = utils.adjust_timeseries_for_interval_label(
        data, obsil, data_start, data_end)
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, obs, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (data_start, data_end)
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    if fxil == 'ending' and obsil == 'ending':
        i = 6
    elif fxil == 'beginning' and obsil == 'beginning':
        i = 6
    elif fxil == 'ending' and obsil == 'beginning':
        # obs: [0, 1) = 0, [1, 2) = 1, [2, 3) = 2, [3, 4) = 0
        # fx: (0, 1] = 0, (1, 2] = 1, (2, 3] = 2, (3, 4] = 0
        i = 7
    elif fxil == 'beginning' and obsil == 'ending':
        # obs: (0, 1] = 1, (1, 2] = 2, (2, 3] = 0, (3, 4] = 0
        # fx: [0, 1) = 1, [1, 2) = 2, [2, 3) = 0, [3, 4) = 0
        # persistence interval moves endpoint into wrong interval
        # but alternative would require broader data start/end
        i = 5
    assert out.loc[f'20190103T0{i}00Z'] == 1
    assert out.loc[f'20190103T0{i+1}00Z'] == 2
    assert out.loc[f'20190103T0{i+2}00Z'] == 0
    assert out.iloc[-1] == 0
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_interval_dst(session, site_metadata, obs_5min_begin,
                                      mocker):
    run_time = pd.Timestamp('20200308T0000', tz='America/New_York')
    site = site_metadata.replace(timezone='America/New_York')
    # day ahead, index = False
    obs = obs_5min_begin.replace(site=site)
    forecast = default_forecast(
        site,
        issue_time_of_day=dt.time(
            hour=2, tzinfo=run_time.tzinfo),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='ending')
    issue_time = pd.Timestamp('20200308T0300-04:00')
    mocker.spy(main.persistence, 'persistence_interval')
    index = pd.date_range('20200306T0300-05:00', periods=24, freq='1h')
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, obs, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20200306T0300-05:00'),
                                          pd.Timestamp('20200307T0300-05:00'))
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    assert out.loc['20200308T0500-04:00'] == 0
    assert out.loc['20200308T0600-04:00'] == 1
    assert out.loc['20200308T0700-04:00'] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead(session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190111T1945Z')
    index = pd.date_range('20190105T0000Z', freq='1h', periods=24)
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T2300Z')
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190105T0000Z'),
                                          pd.Timestamp('20190106T0000Z'))
    assert isinstance(out, pd.Series)
    assert out.loc[pd.Timestamp('20190112T0100Z')] == 1
    assert out.loc[pd.Timestamp('20190112T0200Z')] == 2
    assert len(out) == 24
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead_day_lead(session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190111T2345Z')
    index = pd.date_range('20190106T0000Z', freq='1h', periods=24)
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190112T0000Z')
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190106T0000Z'),
                                          pd.Timestamp('20190107T0000Z'))
    assert isinstance(out, pd.Series)
    assert out.loc[pd.Timestamp('20190113T0100Z')] == 1
    assert out.loc[pd.Timestamp('20190113T0200Z')] == 2
    assert len(out) == 24
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead_alttz(session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190111T0445Z')
    index = pd.date_range('20190104T0100-05:00', freq='1h', periods=24)
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    site = site_metadata.replace(timezone='Etc/GMT+5')
    forecast = default_forecast(
        site, variable=variable,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T0500Z')
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190104T0100-05:00'),
                                          pd.Timestamp('20190105T0100-05:00'))
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    assert out.loc[pd.Timestamp('20190111T0200-05:00')] == 1
    assert out.loc[pd.Timestamp('20190111T0300-05:00')] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead_olderrun(session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190110T0445Z')
    # next issue time would be on 1/10, but try 1/11 issue. should work because
    # data lookback so long
    index = pd.date_range('20190104T0100-05:00', freq='1h', periods=24)
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    site = site_metadata.replace(timezone='Etc/GMT+5')
    forecast = default_forecast(
        site, variable=variable,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T0500Z')
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190104T0100-05:00'),
                                          pd.Timestamp('20190105T0100-05:00'))
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    assert out.loc[pd.Timestamp('20190111T0200-05:00')] == 1
    assert out.loc[pd.Timestamp('20190111T0300-05:00')] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead_not_midnight(
        session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    site = site_metadata.replace(timezone='Etc/GMT+5')
    forecast = default_forecast(
        site, variable=variable,
        issue_time_of_day=dt.time(hour=16),  # start at noon
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T1600Z')
    run_time = pd.Timestamp('20190111T1545Z')
    index = pd.date_range('20190104T1200-05:00', freq='1h', periods=24)
    data = pd.Series([0, 1, 2] + [0] * 21, index=index)
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190104T1200-05:00'),
                                          pd.Timestamp('20190105T1200-05:00'))
    assert isinstance(out, pd.Series)
    assert len(out) == 24
    assert out.loc[pd.Timestamp('20190111T1300-05:00')] == 1
    assert out.loc[pd.Timestamp('20190111T1400-05:00')] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_weekahead_early_runtime(
        session, site_metadata, mocker):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    site = site_metadata.replace(timezone='Etc/GMT+5')
    forecast = default_forecast(
        site, variable=variable,
        issue_time_of_day=dt.time(hour=9),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T0900Z')
    # data end = 2019-01-05T10:00
    run_time = pd.Timestamp('2019-01-05T09:59Z')
    with pytest.raises(ValueError):
        main.run_persistence(session, observation, forecast, run_time,
                             issue_time)


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


def test_run_persistence_interval_long(session, site_metadata,
                                       mocker, obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('48h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190423T1945Z')
    index = pd.date_range('20190421T0000Z', freq='1h', periods=48)
    data = pd.Series([0, 1, 2] + [0] * 45, index=index)
    mocker.spy(main.persistence, 'persistence_interval')
    load_data = mocker.MagicMock(return_value=data)
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time, load_data=load_data)
    assert load_data.call_args[0][1:] == (pd.Timestamp('20190421T0000Z'),
                                          pd.Timestamp('20190423T0000Z'))
    assert isinstance(out, pd.Series)
    assert len(out) == 48
    assert out.loc[pd.Timestamp('20190424T0100Z')] == 1
    assert out.loc[pd.Timestamp('20190424T0200Z')] == 2
    assert main.persistence.persistence_interval.call_count == 1


def test_run_persistence_incompatible_issue(session, site_metadata,
                                            obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2330Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time)
    assert 'incompatible' in str(excinfo.value).lower()


def test_run_persistence_fx_too_short(session, site_metadata,
                                      obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1min'),
        run_length=pd.Timedelta('3min'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time)
    assert 'requires observation.interval_length' in str(excinfo.value)


def test_run_persistence_incompatible_instant_fx(session, site_metadata,
                                                 obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='instant')
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs_5min_begin, forecast, run_time,
                             issue_time)
    assert 'instantaneous forecast' in str(excinfo.value).lower()


def test_run_persistence_incompatible_instant_interval(session, site_metadata,
                                                       obs_5min_begin):
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='instant')
    obs = obs_5min_begin.replace(interval_label='instant',
                                 interval_length=pd.Timedelta('10min'))
    issue_time = pd.Timestamp('20190423T2300Z')
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(session, obs, forecast, run_time,
                             issue_time)
    assert 'identical interval length' in str(excinfo.value)


def test_verify_nwp_forecasts_compatible(ac_power_forecast_metadata):
    fx0 = ac_power_forecast_metadata
    fx1 = replace(fx0, run_length=pd.Timedelta('10h'), interval_label='ending')
    df = pd.DataFrame({'forecast': [fx0, fx1], 'model': ['a', 'b']})
    errs = main._verify_nwp_forecasts_compatible(df)
    assert set(errs) == {'model', 'run_length', 'interval_label'}


@pytest.mark.parametrize('string,expected', [
    ('{"is_reference_forecast": true}', True),
    ('{"is_reference_persistence_forecast": true}', False),
    ('{"is_reference_forecast": "True"}', True),
    ('{"is_reference_forecast":"True"}', True),
    ('is_reference_forecast" : "True"}', True),
    ('{"is_reference_forecast" : true, "otherkey": badjson, 9}', True),
    ('reference_forecast": true', False),
    ('{"is_reference_forecast": false}', False),
    ("is_reference_forecast", False)
])
def test_is_reference_forecast(string, expected):
    assert main._is_reference_forecast(string) == expected


def test_find_reference_nwp_forecasts_json_err(ac_power_forecast_metadata,
                                               mocker):
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    extra_params = '{"model": "themodel", "is_reference_forecast": true}'
    fxs = [replace(ac_power_forecast_metadata, extra_parameters=extra_params),
           replace(ac_power_forecast_metadata,
                   extra_parameters='{"model": "yes"}'),
           replace(ac_power_forecast_metadata, extra_parameters='{"is_reference_forecast": true'),  # NOQA
           replace(ac_power_forecast_metadata, extra_parameters='')]
    out = main.find_reference_nwp_forecasts(fxs)
    assert logger.warning.called
    assert len(out) == 1


def test_find_reference_nwp_forecasts_no_model(ac_power_forecast_metadata,
                                               mocker):
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    fxs = [replace(ac_power_forecast_metadata, extra_parameters='{}',
                   forecast_id='0'),
           replace(ac_power_forecast_metadata,
                   extra_parameters='{"piggyback_on": "0", "is_reference_forecast": true}',  # NOQA
                   forecast_id='1')]
    out = main.find_reference_nwp_forecasts(fxs)
    assert len(out) == 0
    assert logger.debug.called
    assert logger.error.called


def test_find_reference_nwp_forecasts_no_init(ac_power_forecast_metadata):
    fxs = [replace(ac_power_forecast_metadata,
                   extra_parameters='{"model": "am", "is_reference_forecast": true}',  # NOQA
                   forecast_id='0'),
           replace(ac_power_forecast_metadata,
                   extra_parameters='{"piggyback_on": "0", "model": "am", "is_reference_forecast": true}',  # NOQA
                   forecast_id='1')]
    out = main.find_reference_nwp_forecasts(fxs)
    assert len(out) == 2
    assert out.next_issue_time.unique() == [None]
    assert out.piggyback_on.unique() == ['0']


def test_find_reference_nwp_forecasts(ac_power_forecast_metadata):
    fxs = [replace(ac_power_forecast_metadata,
                   extra_parameters='{"model": "am", "is_reference_forecast": true}',  # NOQA
                   forecast_id='0'),
           replace(ac_power_forecast_metadata,
                   extra_parameters='{"piggyback_on": "0", "model": "am", "is_reference_forecast": true}',  # NOQA
                   forecast_id='1')]
    out = main.find_reference_nwp_forecasts(
        fxs, pd.Timestamp('20190501T0000Z'))
    assert len(out) == 2
    assert out.next_issue_time.unique()[0] == pd.Timestamp('20190501T0500Z')
    assert out.piggyback_on.unique() == ['0']


@pytest.fixture()
def forecast_list(ac_power_forecast_metadata):
    model = 'nam_12km_cloud_cover_to_hourly_mean'
    prob_dict = ac_power_forecast_metadata.to_dict()
    prob_dict['constant_values'] = (0, 50, 100)
    prob_dict['axis'] = 'y'
    prob_dict['extra_parameters'] = '{"model": "gefs_half_deg_to_hourly_mean", "is_reference_forecast": true}'  # NOQA
    return [replace(ac_power_forecast_metadata,
                    extra_parameters=(
                        '{"model": "%s", "is_reference_forecast": true}'
                        % model),
                    forecast_id='0'),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"model": "gfs_quarter_deg_hourly_to_hourly_mean", "is_reference_forecast": true}',  # NOQA
                    forecast_id='1'),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"piggyback_on": "0", "model": "%s", "is_reference_forecast": true}' % model,  # NOQA
                    forecast_id='2',
                    variable='ghi'),
            datamodel.ProbabilisticForecast.from_dict(prob_dict),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"piggyback_on": "0", "model": "%s", "is_reference_forecast": true}' % model,  # NOQA
                    forecast_id='3',
                    variable='dni',
                    provider='Organization 2'
                    ),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"piggyback_on": "0", "model": "badmodel", "is_reference_forecast": true}',  # NOQA
                    forecast_id='4'),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"piggyback_on": "6", "model": "%s", "is_reference_forecast": true}' % model,  # NOQA
                    forecast_id='5',
                    variable='ghi'),
            replace(ac_power_forecast_metadata,
                    extra_parameters='{"piggyback_on": "0", "model": "%s", "is_reference_forecast": false}' % model,  # NOQA
                    forecast_id='7',
                    variable='ghi'),
            ]


def test_process_nwp_forecast_groups(mocker, forecast_list):
    api = mocker.MagicMock()
    run_nwp = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_nwp')
    post_vals = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main._post_forecast_values')

    class res:
        ac_power = [0]
        ghi = [0]

    run_nwp.return_value = res
    fxs = main.find_reference_nwp_forecasts(forecast_list[:-4])
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    main.process_nwp_forecast_groups(api, pd.Timestamp('20190501T0000Z'), fxs)
    assert not logger.error.called
    assert not logger.warning.called
    assert post_vals.call_count == 4


@pytest.mark.parametrize('run_time', [None, pd.Timestamp('20190501T0000Z')])
def test_process_nwp_forecast_groups_issue_time(mocker, forecast_list,
                                                run_time):
    api = mocker.MagicMock()
    run_nwp = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_nwp')
    post_vals = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main._post_forecast_values')

    class res:
        ac_power = [0]
        ghi = [0]

    run_nwp.return_value = res
    fxs = main.find_reference_nwp_forecasts(forecast_list[:-4], run_time)
    main.process_nwp_forecast_groups(api, pd.Timestamp('20190501T0000Z'), fxs)
    assert post_vals.call_count == 4
    run_nwp.assert_called_with(mocker.ANY, mocker.ANY, mocker.ANY,
                               pd.Timestamp('20190501T0500Z'))


def test_process_nwp_forecast_groups_missing_var(mocker, forecast_list):
    api = mocker.MagicMock()
    run_nwp = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_nwp')
    post_vals = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main._post_forecast_values')

    class res:
        ac_power = [0]
        ghi = [0]
        dni = None

    run_nwp.return_value = res
    fxs = main.find_reference_nwp_forecasts(forecast_list[:-3])
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    main.process_nwp_forecast_groups(api, pd.Timestamp('20190501T0000Z'), fxs)
    assert not logger.error.called
    assert logger.warning.called
    assert post_vals.call_count == 4


def test_process_nwp_forecast_groups_bad_model(mocker, forecast_list):
    api = mocker.MagicMock()
    run_nwp = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_nwp')
    post_vals = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main._post_forecast_values')

    class res:
        ac_power = [0]
        ghi = [0]
        dni = None

    run_nwp.return_value = res
    fxs = main.find_reference_nwp_forecasts(forecast_list[4:-1])
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    main.process_nwp_forecast_groups(api, pd.Timestamp('20190501T0000Z'), fxs)
    assert logger.error.called
    assert not logger.warning.called
    assert post_vals.call_count == 0


def test_process_nwp_forecast_groups_missing_runfor(mocker, forecast_list):
    api = mocker.MagicMock()
    run_nwp = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_nwp')

    class res:
        ac_power = [0]
        ghi = [0]
        dni = None

    run_nwp.return_value = res
    fxs = main.find_reference_nwp_forecasts(forecast_list[-2:])
    logger = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.logger')
    main.process_nwp_forecast_groups(api, pd.Timestamp('20190501T0000Z'), fxs)
    assert logger.error.called
    assert not logger.warning.called
    assert api.post_forecast_values.call_count == 0


@pytest.mark.parametrize('ind', [0, 1, 2])
def test__post_forecast_values_regular(mocker, forecast_list, ind):
    api = mocker.MagicMock()
    fx = forecast_list[ind]
    main._post_forecast_values(api, fx, [0], 'whatever')
    assert api.post_forecast_values.call_count == 1


def test__post_forecast_values_cdf(mocker, forecast_list):
    api = mocker.MagicMock()
    fx = forecast_list[3]

    ser = pd.Series([0, 1])
    vals = pd.DataFrame({i: ser for i in range(21)})
    main._post_forecast_values(api, fx, vals, 'gefs')
    assert api.post_probabilistic_forecast_constant_value_values.call_count == 3  # NOQA


def test__post_forecast_values_cdf_not_gefs(mocker, forecast_list):
    api = mocker.MagicMock()
    fx = forecast_list[3]

    ser = pd.Series([0, 1])
    vals = pd.DataFrame({i: ser for i in range(21)})
    with pytest.raises(ValueError):
        main._post_forecast_values(api, fx, vals, 'gfs')


def test__post_forecast_values_cdf_less_cols(mocker, forecast_list):
    api = mocker.MagicMock()
    fx = forecast_list[3]

    ser = pd.Series([0, 1])
    vals = pd.DataFrame({i: ser for i in range(10)})
    with pytest.raises(TypeError):
        main._post_forecast_values(api, fx, vals, 'gefs')


def test__post_forecast_values_cdf_not_df(mocker, forecast_list):
    api = mocker.MagicMock()
    fx = forecast_list[3]

    ser = pd.Series([0, 1])
    with pytest.raises(TypeError):
        main._post_forecast_values(api, fx, ser, 'gefs')


def test__post_forecast_values_cdf_no_cv_match(mocker, forecast_list):
    api = mocker.MagicMock()
    fx = replace(forecast_list[3], constant_values=(
        replace(forecast_list[3].constant_values[0], constant_value=3.0
                ),))

    ser = pd.Series([0, 1])
    vals = pd.DataFrame({i: ser for i in range(21)})
    with pytest.raises(KeyError):
        main._post_forecast_values(api, fx, vals, 'gefs')


@pytest.mark.parametrize('issue_buffer,empty', [
    (pd.Timedelta('10h'), False),
    (pd.Timedelta('1h'), True),
    (pd.Timedelta('5h'), False)
])
def test_make_latest_nwp_forecasts(forecast_list, mocker, issue_buffer, empty):
    session = mocker.patch('solarforecastarbiter.io.api.APISession')
    session.return_value.get_user_info.return_value = {'organization': ''}
    session.return_value.list_forecasts.return_value = forecast_list[:-3]
    session.return_value.list_probabilistic_forecasts.return_value = []
    run_time = pd.Timestamp('20190501T0000Z')
    # last fx has different org
    fxdf = main.find_reference_nwp_forecasts(forecast_list[:-4], run_time)
    process = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.process_nwp_forecast_groups')  # NOQA
    main.make_latest_nwp_forecasts('', run_time, issue_buffer)
    if empty:
        process.assert_not_called()
    else:
        assert_frame_equal(process.call_args[0][-1], fxdf)


@pytest.mark.parametrize('string,expected', [
    ('{"is_reference_forecast": true}', False),
    ('{"is_reference_persistence_forecast": true}', True),
    ('{"is_reference_persistence_forecast": "True"}', True),
    ('{"is_reference_persistence_forecast":"True"}', True),
    ('is_reference_persistence_forecast" : "True"}', True),
    ('{"is_reference_persistence_forecast" : true, "otherkey": badjson, 9}',
     True),
    ('reference_persistence_forecast": true', False),
    ('{"is_reference_persistence_forecast": false}', False),
    ("is_reference_persistence_forecast", False)
])
def test_is_reference_persistence_forecast(string, expected):
    assert main._is_reference_persistence_forecast(string) == expected


@pytest.fixture
def perst_fx_obs(mocker, ac_power_observation_metadata,
                 ac_power_forecast_metadata):
    observations = [
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        ),
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        ),
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        )
    ]

    def make_extra(obs):
        extra = (
            '{"is_reference_persistence_forecast": true,'
            f'"observation_id": "{obs.observation_id}"'
            '}'
        )
        return extra

    forecasts = [
        ac_power_forecast_metadata.replace(
            name='FX0',
            extra_parameters=make_extra(observations[0]),
            run_length=pd.Timedelta('1h'),
            forecast_id=str(uuid.uuid1())
        ),
        ac_power_forecast_metadata.replace(
            name='FX no persist',
            run_length=pd.Timedelta('1h'),
            forecast_id=str(uuid.uuid1())
        ),
        ac_power_forecast_metadata.replace(
            name='FX bad js',
            extra_parameters='is_reference_persistence_forecast": true other',
            run_length=pd.Timedelta('1h'),
            forecast_id=str(uuid.uuid1())
        ),
    ]
    return forecasts, observations


def test_generate_reference_persistence_forecast_parameters(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    # one hour ahead forecast, so 14Z was made at 13Z
    # enough data to do 14Z and 15Z issue times but not 16Z
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T13:00Z')
    assert param_list[0].issue_times == (
        pd.Timestamp('2020-05-20T14:00Z'),
        pd.Timestamp('2020-05-20T15:00Z')
    )


def test_generate_reference_persistence_forecast_parameters_no_forecast_yet(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.NaT, pd.NaT)
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T14:00Z')
    assert param_list[0].issue_times == (pd.Timestamp('2020-05-20T15:00Z'),)


def test_generate_reference_persistence_forecast_parameters_no_data(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.NaT, pd.NaT)
    session.get_forecast_time_range.return_value = (
        pd.NaT, pd.NaT)
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_parameters_diff_org(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': 'a new one'}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_parameters_not_reference_fx(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [fx.replace(extra_parameters='') for fx in forecasts]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_parameters_no_obs_id(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts[0] = forecasts[0].replace(
        extra_parameters='{"is_reference_persistence_forecast": true}')
    forecasts[1] = forecasts[1].replace(
        extra_parameters='{"is_reference_persistence_forecast": true, "observation_id": "idnotinobs"}')  # NOQA
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_parameters_ending_label(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [fx.replace(
        interval_label='ending', lead_time_to_start=pd.Timedelta('0h'))
                 for fx in forecasts]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T16:00Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].issue_times == (
            pd.Timestamp('2020-05-20T14:00Z'),
            pd.Timestamp('2020-05-20T15:00Z'),
            pd.Timestamp('2020-05-20T16:00Z'),
    )


def test_generate_reference_persistence_forecast_parameters_no_lead(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [fx.replace(
        lead_time_to_start=pd.Timedelta('0h'))
                 for fx in forecasts]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T16:00Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].issue_times == (
            pd.Timestamp('2020-05-20T15:00Z'),
            pd.Timestamp('2020-05-20T16:00Z'),
    )


def test_generate_reference_persistence_forecast_parameters_off_time(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:10Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    # one hour ahead forecast, so 14Z was made at 13Z
    # enough data to do 14Z and 15Z issue times but not 16Z
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].issue_times == (
            pd.Timestamp('2020-05-20T14:00Z'),
            pd.Timestamp('2020-05-20T15:00Z'),
    )


def test_generate_reference_persistence_forecast_parameters_multiple(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts[0] = forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}')
    )
    forecasts[1] = forecasts[1].replace(
        extra_parameters=(
            '{"is_reference_persistence_forecast": true, "observation_id": "' +
            observations[1].observation_id + '"}'))
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 2
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is True
    assert param_list[0].issue_times == (pd.Timestamp('2020-05-20T14:00Z'),)
    assert param_list[1].forecast == forecasts[1]
    assert param_list[1].observation == observations[1]
    assert param_list[1].index is False
    assert param_list[1].issue_times == (pd.Timestamp('2020-05-20T14:00Z'),)


def test_generate_reference_persistence_forecast_parameters_up_to_date(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T13:59Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    # next would be at 14 and use data incl 13:59:59
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_make_latest_persistence_forecasts(mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        return_value=pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    main.make_latest_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 4
    assert session.get_observation_values.call_count == 2
    assert session.post_forecast_values.call_count == 2
    assert [ll[1]['index'] for ll in run_pers.call_args_list] == [
        False, False, True, True]


def test_make_latest_persistence_forecasts_up_to_date(mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T16:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        return_value=pd.Series(dtype=float))
    main.make_latest_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 0
    assert session.get_observation_values.call_count == 0
    assert session.post_forecast_values.call_count == 0


def test_make_latest_persistence_forecasts_some_errors(mocker, perst_fx_obs):
    # test that some persistence forecast parameters are invalid for the
    # observation and that no peristence values are posted
    # and that the failure doesn't interrupt other posts
    forecasts, observations = perst_fx_obs
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []

    def sometimes_fail(*args, **kwargs):
        i.append(1)
        if len(i) > 3:
            raise ValueError('Failed')
        else:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    logger = mocker.spy(main, 'logger')
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        side_effect=sometimes_fail, autospec=True)
    main.make_latest_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 6
    assert session.get_observation_values.call_count == 3
    assert session.post_forecast_values.call_count == 2
    assert logger.error.call_count == 3
    assert len(i) == 6
    assert [li[1]['index'] for li in run_pers.call_args_list] == [
        False, False, True, True, True, True]


def test_make_latest_persistence_forecasts_multi_issue_err(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T18:33Z'))
    session.get_forecast_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T19:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []

    def sometimes_fail(*args, **kwargs):
        i.append(1)
        li = len(i)
        if li in (2, 3):
            raise ValueError('Failed')
        else:
            return pd.Series(
                [0.0], name='value',
                index=[pd.Timestamp('2020-05-20T15:00Z') +
                       pd.Timedelta('1h') * li])

    logger = mocker.spy(main, 'logger')
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        side_effect=sometimes_fail, autospec=True)
    main.make_latest_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 5
    assert session.get_observation_values.call_count == 1
    assert logger.error.call_count == 2
    assert len(i) == 5
    expected_sers = [
        pd.Series([0.0], index=[pd.Timestamp('2020-05-20T16:00Z')],
                  name='value'),
        pd.Series([0.0, 0.0], index=[pd.Timestamp('2020-05-20T19:00Z'),
                                     pd.Timestamp('2020-05-20T20:00Z')],
                  name='value'),
    ]
    for i, cl in enumerate(session.post_forecast_values.call_args_list):
        assert_series_equal(cl[0][1], expected_sers[i])
    # one post per valid range after invalid period dropped
    assert session.post_forecast_values.call_count == 2


@pytest.fixture
def perst_prob_fx_obs(mocker, ac_power_observation_metadata,
                      ac_power_forecast_metadata, prob_forecasts_y):
    observations = [
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        ),
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        ),
        ac_power_observation_metadata.replace(
            observation_id=str(uuid.uuid1())
        )
    ]

    def make_extra(obs):
        extra = (
            '{"is_reference_persistence_forecast": true,'
            f'"observation_id": "{obs.observation_id}"'
            '}'
        )
        return extra

    forecasts = [
        prob_forecasts_y.replace(
            variable='ac_power',
            run_length=pd.Timedelta('1h'),
            issue_time_of_day=dt.time(0),
            interval_length=pd.Timedelta('1h'),
            lead_time_to_start=pd.Timedelta('1h'),
            extra_parameters=make_extra(observations[1]),
            provider=ac_power_forecast_metadata.provider,
            constant_values=[
                prob_forecasts_y.constant_values[0],
                prob_forecasts_y.constant_values[0].replace(
                    constant_value=0.0,
                    forecast_id=str(uuid.uuid1())
                ),
                prob_forecasts_y.constant_values[0].replace(
                    constant_value=100.0,
                    forecast_id=str(uuid.uuid1())
                ),
            ]
        ),
        prob_forecasts_y.replace(
            forecast_id=str(uuid.uuid1()),
            variable='ac_power',
            interval_label='ending',
            run_length=pd.Timedelta('1h'),
            interval_length=pd.Timedelta('1h'),
            issue_time_of_day=dt.time(0),
            lead_time_to_start=pd.Timedelta('1h'),
            extra_parameters='',
            provider=ac_power_forecast_metadata.provider,
            constant_values=[
                prob_forecasts_y.constant_values[0],
                prob_forecasts_y.constant_values[0].replace(
                    constant_value=0.0,
                    forecast_id=str(uuid.uuid1())
                ),
                prob_forecasts_y.constant_values[0].replace(
                    constant_value=100.0,
                    forecast_id=str(uuid.uuid1())
                ),
            ]
        )
    ]
    return forecasts, observations


def test_generate_reference_persistence_forecast_parameters_prob_fx(
        mocker, perst_prob_fx_obs):
    forecasts, observations = perst_prob_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_probabilistic_forecast_constant_value_time_range.return_value = (  # NOQA
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    # one hour ahead forecast, so 14Z was made at 13Z
    # enough data to do 14Z and 15Z issue times but not 16Z
    param_gen = main.generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[1]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T13:00Z')
    assert param_list[0].issue_times == (
        pd.Timestamp('2020-05-20T14:00Z'),
        pd.Timestamp('2020-05-20T15:00Z')
    )


def test_make_latest_probabilistic_persistence_forecasts(
        mocker, perst_prob_fx_obs):
    forecasts, observations = perst_prob_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_probabilistic_forecast_constant_value_time_range.return_value = (  # NOQA
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    # can do 13, 14, 15, init times
    session.list_probabilistic_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)
    cvs = len(forecasts[-1].constant_values)
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        return_value=[pd.Series(dtype=float, index=pd.DatetimeIndex([]))] *
        cvs)
    main.make_latest_probabilistic_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 2
    assert session.get_observation_values.call_count == 1
    assert session.post_probabilistic_forecast_constant_value_values.call_count == cvs  # NOQA


def test_make_latest_probabilistic_persistence_forecasts_err(
        mocker, perst_prob_fx_obs):
    forecasts, observations = perst_prob_fx_obs
    session = mocker.MagicMock()
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T15:33Z'))
    session.get_probabilistic_forecast_constant_value_time_range.return_value = (  # NOQA
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:00Z'))
    # can do 13, 14, 15, init times
    session.list_probabilistic_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    max_run_time = pd.Timestamp('2020-05-20T16:00Z')
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        side_effect=ValueError)
    main.make_latest_probabilistic_persistence_forecasts('', max_run_time)
    assert run_pers.call_count == 2
    assert session.get_observation_values.call_count == 1
    assert session.post_probabilistic_forecast_constant_value_values.call_count == 0  # NOQA


@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
def test_run_persistence_probabilistic(
        session, perst_prob_fx_obs, obs_5min_begin,
        interval_label, mocker):
    run_time = pd.Timestamp('20190101T1945Z')
    # intraday, index=False
    forecast = perst_prob_fx_obs[0][0]
    issue_time = pd.Timestamp('20190101T2300Z')
    prob = mocker.spy(main.persistence, 'persistence_probabilistic')
    out = main.run_persistence(session, obs_5min_begin, forecast, run_time,
                               issue_time)
    assert isinstance(out, list)
    assert len(out) == 3
    assert isinstance(out[0], pd.Series)
    assert prob.call_count == 1


@pytest.mark.parametrize('intervallabel', ['beginning', 'ending'])
@pytest.mark.parametrize('start,end', [
    (pd.Timestamp('20190101T0000Z'), pd.Timestamp('20190112T0000Z')),
    (pd.Timestamp('20190101T0000Z'), pd.Timestamp('20190114T0000Z')),
    (pd.Timestamp('20190103T0000Z'), pd.Timestamp('20190109T0000Z')),
    (pd.Timestamp('20190201T0000Z'), pd.Timestamp('20190202T0000Z')),
    (pd.Timestamp('20190201T0000Z'), pd.Timestamp('20190102T0000Z')),
])
def test_data_loading(session, obs_5min_begin, start, end, intervallabel):
    obs = obs_5min_begin.replace(interval_label=intervallabel)
    full_start = pd.Timestamp('20190101T0000Z')
    full_end = pd.Timestamp('20190112T0000Z')
    preload_data = main._preload_load_data(
        session, obs, full_start, full_end)(
            obs, start, end)
    default_load = main._default_load_data(session)(obs, start, end)
    assert_series_equal(preload_data, default_load)


@pytest.mark.parametrize('runlength,leadtime,il,itd,nextt,exp', [
    ('1h', '1h', 'ending', '00:00', '2020-01-10T16:00Z',
     [pd.Timestamp('2020-01-10T11:00Z'), pd.Timestamp('2020-01-10T12:00Z'),
      pd.Timestamp('2020-01-10T13:00Z')]),
    ('1h', '1h', 'beginning', '00:00', '2020-01-10T16:00Z',
     [pd.Timestamp('2020-01-10T12:00Z'), pd.Timestamp('2020-01-10T13:00Z'),
      pd.Timestamp('2020-01-10T14:00Z')]),
    ('24h', '6h', 'ending', '06:00', '2020-01-13T13:00Z',
     [pd.Timestamp('2020-01-10T06:00Z'), pd.Timestamp('2020-01-11T06:00Z'),
      pd.Timestamp('2020-01-12T06:00Z')]),
])
def test__nwp_issue_time_generator(
        single_forecast, runlength, leadtime, il, itd, nextt, exp):
    last = pd.Timestamp('2020-01-10T12:00Z')
    fx = single_forecast.replace(
        run_length=pd.Timedelta(runlength), interval_label=il,
        lead_time_to_start=pd.Timedelta(leadtime),
        interval_length=pd.Timedelta('1h'),
        issue_time_of_day=dt.datetime.strptime(itd, '%H:%M').time())
    out = list(main._nwp_issue_time_generator(fx, last, pd.Timestamp(nextt)))
    assert out == exp


def test_fill_nwp_forecast_gaps(forecast_list, mocker):
    session = mocker.patch('solarforecastarbiter.io.api.APISession')
    session.return_value.get_user_info.return_value = {'organization': ''}
    session.return_value.list_forecasts.return_value = forecast_list[:-3]
    session.return_value.list_probabilistic_forecasts.return_value = []
    # last fx has different org
    run_nwp = mocker.patch.object(
        main, 'run_nwp',
        side_effect=FileNotFoundError)
    session.return_value.get_value_gaps.return_value = [
        (pd.Timestamp('2020-01-10T18:00Z'), pd.Timestamp('2020-01-12T06:00Z'))
    ]
    main.fill_nwp_forecast_gaps('token', pd.Timestamp('2020-01-01T00:00Z'),
                                pd.Timestamp('2020-01-19T00:00Z'))
    # run_nwp should be called for 3 issue times and 3 forecasts
    assert run_nwp.call_count == 9
    # forecast 2 piggy bakcs on 0
    for fx in [forecast_list[0], forecast_list[1], forecast_list[3]]:
        for i in range(3):
            rt = pd.Timestamp('2020-01-10T17:00Z') + pd.Timedelta(hours=i * 12)
            run_nwp.assert_any_call(fx, mocker.ANY, rt, rt)


def test_generate_reference_persistence_forecast_gaps_parameters(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T13:00Z')
    assert param_list[0].data_end == pd.Timestamp('2020-05-20T13:59:59Z')
    assert param_list[0].issue_times == (
        pd.Timestamp('2020-05-20T14:00Z'),
    )


def test_generate_reference_persistence_forecast_gaps_parameters_multiple_gaps(  # NOQA
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z')),
        # need forecast with data from 15
        (pd.Timestamp('2020-05-20T16:00Z'), pd.Timestamp('2020-05-20T17:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T13:00Z')
    assert param_list[0].data_end == pd.Timestamp('2020-05-20T15:59:59Z')
    assert param_list[0].issue_times == (
        pd.Timestamp('2020-05-20T14:00Z'),
        pd.Timestamp('2020-05-20T16:00Z'),
    )


def test_generate_reference_persistence_forecast_gaps_parameters_up_to_date(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = []
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_gaps_parameters_no_data(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        (pd.Timestamp('2020-05-20T20:00Z'), pd.Timestamp('2020-05-20T22:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp(None), pd.Timestamp(None))
    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_gaps_parameters_diff_org(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': 'notright'}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_gaps_parameters_not_reference(  # NOQA
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [fx.replace(extra_parameters='') for fx in forecasts]
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_gaps_parameters_no_obs(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts[0] = forecasts[0].replace(
        extra_parameters='{"is_reference_persistence_forecast": true}')
    forecasts[1] = forecasts[1].replace(
        extra_parameters='{"is_reference_persistence_forecast": true, "observation_id": "idnotinobs"}')  # NOQA
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 0


def test_generate_reference_persistence_forecast_gaps_parameters_ending(
       mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts = [fx.replace(
        interval_label='ending', lead_time_to_start=pd.Timedelta('0h'))
                 for fx in forecasts]
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))

    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 1
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is False
    assert param_list[0].data_start == pd.Timestamp('2020-05-20T13:00:01Z')
    assert param_list[0].data_end == pd.Timestamp('2020-05-20T14:00:00Z')
    assert param_list[0].issue_times == (
        pd.Timestamp('2020-05-20T14:00Z'),
    )


def test_generate_reference_persistence_forecast_gaps_parameters_multiple(
        mocker, perst_fx_obs, perst_prob_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts[0] = forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}')
    )
    forecasts[1] = forecasts[1].replace(
        extra_parameters=(
            '{"is_reference_persistence_forecast": true, "observation_id": "' +
            observations[1].observation_id + '"}'))
    probfx = perst_prob_fx_obs[0][0].replace(
        extra_parameters=(
            '{"is_reference_persistence_forecast": true, "observation_id": "' +
            observations[1].observation_id + '"}'))
    fxs = [forecasts[0], forecasts[1], probfx]
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-20T14:33Z'))
    param_gen = main.generate_reference_persistence_forecast_gaps_parameters(
        session, fxs, observations, start, end
    )
    assert isinstance(param_gen, types.GeneratorType)
    param_list = list(param_gen)
    assert len(param_list) == 3
    assert param_list[0].forecast == forecasts[0]
    assert param_list[0].observation == observations[0]
    assert param_list[0].index is True
    assert param_list[0].issue_times == (pd.Timestamp('2020-05-20T14:00Z'),)
    assert param_list[1].forecast == forecasts[1]
    assert param_list[1].observation == observations[1]
    assert param_list[1].index is False
    assert param_list[1].issue_times == (pd.Timestamp('2020-05-20T14:00Z'),)
    assert param_list[2].forecast == probfx
    assert param_list[2].observation == observations[1]
    assert param_list[2].index is False
    assert param_list[2].issue_times == (pd.Timestamp('2020-05-20T14:00Z'),)


def test_fill_persistence_forecasts_gaps(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z')),
        # need forecast with data from 15
        (pd.Timestamp('2020-05-20T16:00Z'), pd.Timestamp('2020-05-20T17:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []

    def yield_ser(*args, **kwargs):
        if not len(i):
            i.append(0)
            return pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T14:00Z')]))
        else:
            return pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T16:00Z')]))

    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        new=yield_ser)
    main.fill_persistence_forecasts_gaps('', start, end)
    assert i == [0]
    assert session.get_observation_values.call_count == 1
    assert session.post_forecast_values.call_count == 2
    # gap from 15 to 16, means post must be called twice
    assert session.post_forecast_values.call_args_list[0][0][1].index == \
        pd.DatetimeIndex([pd.Timestamp('2020-05-20T14:00Z')])
    assert session.post_forecast_values.call_args_list[1][0][1].index == \
        pd.DatetimeIndex([pd.Timestamp('2020-05-20T16:00Z')])


def test_fill_persistence_forecasts_continuous(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z')),
        # need forecast with data from 14
        (pd.Timestamp('2020-05-20T15:00Z'), pd.Timestamp('2020-05-20T17:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []

    def get_ser(*args, **kwargs):
        if len(i) == 0:
            i.append(0)
            return pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T14:00Z')]))
        elif len(i) == 1:
            i.append(0)
            return pd.Series([2.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T15:00Z')]))
        else:
            return pd.Series([-1.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T16:00Z')]))

    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        new=get_ser)
    main.fill_persistence_forecasts_gaps('', start, end)
    assert i == [0, 0]
    assert session.get_observation_values.call_count == 1
    assert session.post_forecast_values.call_count == 1
    assert_series_equal(
        session.post_forecast_values.call_args[0][1],
        pd.Series([0., 2.0, -1.0], dtype=float, index=pd.DatetimeIndex([
            pd.Timestamp('2020-05-20T14:00Z'),
            pd.Timestamp('2020-05-20T15:00Z'),
            pd.Timestamp('2020-05-20T16:00Z')])))


def test_fill_persistence_forecasts_no_gaps(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = []
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
    )
    main.fill_persistence_forecasts_gaps('', start, end)
    assert run_pers.call_count == 0
    assert session.get_observation_values.call_count == 0
    assert session.post_forecast_values.call_count == 0


def test_fill_persistence_forecasts_some_errs(
        mocker, perst_fx_obs):
    forecasts, observations = perst_fx_obs
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    forecasts += [forecasts[0].replace(
        extra_parameters=(forecasts[0].extra_parameters[:-1] +
                          ', "index_persistence": true}'))]
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z')),
        # need forecast with data from 15
        (pd.Timestamp('2020-05-20T16:00Z'), pd.Timestamp('2020-05-20T17:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))
    session.list_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []

    def sometimes_fail(*args, **kwargs):
        i.append(1)
        if len(i) == 1:
            return pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T14:00Z')]))
        elif len(i) == 2:
            return pd.Series([1.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T16:00Z')]))
        elif len(i) == 3:
            return pd.Series([2.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T14:00Z')]))
        else:
            raise ValueError('Failed')

    logger = mocker.spy(main, 'logger')
    run_pers = mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        side_effect=sometimes_fail, autospec=True)
    main.fill_persistence_forecasts_gaps('', start, end)
    assert run_pers.call_count == 6
    assert logger.error.call_count == 3
    assert session.get_observation_values.call_count == 3
    assert session.post_forecast_values.call_count == 3
    assert_series_equal(
        session.post_forecast_values.call_args_list[0][0][1],
        pd.Series([0.], index=pd.DatetimeIndex([
            pd.Timestamp('2020-05-20T14:00Z')]))
    )
    assert_series_equal(
        session.post_forecast_values.call_args_list[1][0][1],
        pd.Series([1.], index=pd.DatetimeIndex([
            pd.Timestamp('2020-05-20T16:00Z')]))
    )
    assert_series_equal(
        session.post_forecast_values.call_args_list[2][0][1],
        pd.Series([2.], index=pd.DatetimeIndex([
            pd.Timestamp('2020-05-20T14:00Z')]))
    )


def test_fill_probabilistic_persistence_forecasts_gaps(
        mocker, perst_prob_fx_obs):
    forecasts, observations = perst_prob_fx_obs
    session = mocker.MagicMock()
    start = pd.Timestamp('2020-05-20T00:00Z')
    end = pd.Timestamp('2020-05-20T22:00Z')
    session.get_value_gaps.return_value = [
        # need forecast with data from 13
        (pd.Timestamp('2020-05-20T14:00Z'), pd.Timestamp('2020-05-20T15:00Z')),
        # need forecast with data from 15
        (pd.Timestamp('2020-05-20T16:00Z'), pd.Timestamp('2020-05-20T17:00Z'))
    ]
    session.get_user_info.return_value = {'organization': ''}
    session.get_observation_time_range.return_value = (
        pd.Timestamp('2019-01-01T12:00Z'), pd.Timestamp('2020-05-21T00:00Z'))
    session.list_probabilistic_forecasts.return_value = forecasts
    session.list_observations.return_value = observations
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.api.APISession',
        return_value=session)

    i = []
    cvs = len(forecasts[-1].constant_values)

    def yield_ser(*args, **kwargs):
        if not len(i):
            i.append(0)
            return [pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T14:00Z')]))] * cvs
        else:
            return [pd.Series([0.], dtype=float, index=pd.DatetimeIndex([
                pd.Timestamp('2020-05-20T16:00Z')]))] * cvs

    mocker.patch(
        'solarforecastarbiter.reference_forecasts.main.run_persistence',
        new=yield_ser)
    main.fill_probabilistic_persistence_forecasts_gaps('', start, end)
    assert i == [0]
    assert session.get_observation_values.call_count == 1
    assert session.post_probabilistic_forecast_constant_value_values.call_count == 2 * cvs # NOQA
