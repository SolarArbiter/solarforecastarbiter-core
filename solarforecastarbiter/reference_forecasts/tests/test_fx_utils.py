from dataclasses import replace
import datetime as dt


import pandas as pd
import pytest


from solarforecastarbiter.reference_forecasts import utils
from solarforecastarbiter.conftest import default_forecast, default_observation


@pytest.mark.parametrize('issuetime,rl,lt,expected', [
    ('06:00', '1h', '1h', [dt.time(i) for i in range(6, 24)] + [dt.time(0)]),
    ('00:00', '4h', '1h', [dt.time(0), dt.time(4), dt.time(8), dt.time(12),
                           dt.time(16), dt.time(20), dt.time(0)]),
    ('16:00', '8h', '3h', [dt.time(16), dt.time(0)]),
    ('00:30', '4h', '120h', [dt.time(0, 30), dt.time(4, 30), dt.time(8, 30),
                             dt.time(12, 30), dt.time(16, 30),
                             dt.time(20, 30), dt.time(0, 30)])
])
def test_issue_times(single_forecast, issuetime, rl, lt, expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.datetime.strptime(issuetime, '%H:%M').time(),
        run_length=pd.Timedelta(rl),
        lead_time_to_start=pd.Timedelta(lt))
    out = utils.get_issue_times(fx)
    assert out == expected


@pytest.mark.parametrize('issuetime,rl,lt,start,expected', [
    ('05:00', '6h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0500Z'), pd.Timestamp('20190101T1100Z'),
      pd.Timestamp('20190101T1700Z'), pd.Timestamp('20190101T2300Z'),
      pd.Timestamp('20190102T0500Z')]),
    ('11:00', '12h', '3h', pd.Timestamp('20190101T2300Z'),
     [pd.Timestamp('20190101T1100Z'), pd.Timestamp('20190101T2300Z'),
      pd.Timestamp('20190102T1100Z')])
])
def test_issue_times_start(single_forecast, issuetime, rl, lt, start,
                           expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.datetime.strptime(issuetime, '%H:%M').time(),
        run_length=pd.Timedelta(rl),
        lead_time_to_start=pd.Timedelta(lt))
    out = utils.get_issue_times(fx, start)
    assert out == expected


@pytest.mark.parametrize('runtime,expected', [
    (pd.Timestamp('20190501T1100Z'), pd.Timestamp('20190501T1100Z')),
    (pd.Timestamp('20190501T1030Z'), pd.Timestamp('20190501T1100Z')),
    (pd.Timestamp('20190501T0030Z'), pd.Timestamp('20190501T0500Z')),
    (pd.Timestamp('20190501T2359Z'), pd.Timestamp('20190502T0500Z')),
    (pd.Timestamp('20190501T2200Z'), pd.Timestamp('20190501T2300Z'))
])
def test_get_next_issue_time(single_forecast, runtime, expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.time(5, 0),
        run_length=pd.Timedelta('6h'),
        lead_time_to_start=pd.Timedelta('1h'))
    out = utils.get_next_issue_time(fx, runtime)
    assert out == expected


def test_get_init_time():
    run_time = pd.Timestamp('20190501T1200Z')
    fetch_metadata = {'delay_to_first_forecast': '1h',
                      'avg_max_run_length': '5h',
                      'update_freq': '6h'}
    assert utils.get_init_time(run_time, fetch_metadata) == pd.Timestamp(
        '20190501T0600Z')


def test_get_data_start_end_labels_1h_window_limit(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')
    forecast = default_forecast(
        site_metadata,
        run_length=pd.Timedelta('12h'),  # test 1 hr limit on window
        interval_label='beginning')
    # ensure data no later than run time
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190422T1845Z')
    assert data_end == pd.Timestamp('20190422T1945Z')


def test_get_data_start_end_labels_subhourly_window_limit(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='beginning')
    forecast = default_forecast(
        site_metadata,
        run_length=pd.Timedelta('5min'),  # test subhourly limit on window
        interval_label='beginning')
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190422T1940Z')
    assert data_end == pd.Timestamp('20190422T1945Z')


def test_get_data_start_end_labels_obs_longer_than_fx_run(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('15min'))
    forecast = default_forecast(
        site_metadata,
        run_length=pd.Timedelta('5min'))
    run_time = pd.Timestamp('20190422T1945Z')
    # obs interval cannot be longer than forecast interval
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time)
    assert ("observation.interval_length <= forecast.run_length" in
            str(excinfo.value))


def test_get_data_start_end_labels_obs_longer_than_1h(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('2h'))
    forecast = default_forecast(
        site_metadata,
        run_length=pd.Timedelta('5min'))
    run_time = pd.Timestamp('20190422T1945Z')
    # obs interval cannot be longer than 1 hr
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time)
    assert 'observation.interval_length <= 1h' in str(excinfo.value)


def test_get_data_start_end_labels_obs_longer_than_1h_day_ahead(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('2h'), interval_label='beginning')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),  # day ahead
        interval_label='beginning')
    run_time = pd.Timestamp('20190422T1945Z')
    # day ahead doesn't care about obs interval length
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190422T0000Z')


def test_get_data_start_end_labels_obs_fx_instant_mismatch(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),  # interval_length must be equal
        run_length=pd.Timedelta('1d'),
        interval_label='instant')            # if interval_label also instant
    run_time = pd.Timestamp('20190422T1945Z')
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time)
    assert 'with identical interval length' in str(excinfo.value)


def test_get_data_start_end_labels_obs_fx_instant(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),  # interval_length must be equal
        run_length=pd.Timedelta('1d'),
        interval_label='instant')              # if interval_label also instant
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')


def test_get_data_start_end_labels_obs_instant_fx_avg(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')


def test_get_data_start_end_labels_obs_instant_fx_avg_ending(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='ending')
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190421T000001Z')
    assert data_end == pd.Timestamp('20190422T0000Z')


def test_get_data_start_end_labels_obs_instant_fx_avg_intraday(site_metadata):
    run_time = pd.Timestamp('20190422T1945Z')
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('15min'),
        interval_label='ending')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time)
    assert data_start == pd.Timestamp('20190422T193001Z')
    assert data_end == pd.Timestamp('20190422T1945Z')


def test_get_data_start_end_labels_obs_avg_fx_instant(site_metadata):
    run_time = pd.Timestamp('20190422T1945Z')
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='ending')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='instant')
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time)
    assert 'made from interval average obs' in str(excinfo.value)


@pytest.mark.parametrize("variable,expected_start,expected_end", [
    # generate forecast on Wednesday 4/10 that applies to Thursday 4/11 using
    # data from the previous Thursday (4/4)
    ("net_load", "20190404T0000Z", "20190405T0000Z"),

    # generate forecast on Wednesday 4/10 that applies to Thursday 4/11 using
    # data from the previous day (Tuesday 4/9)
    ("ghi", "20190409T0000Z", "20190410T0000Z"),
])
def test_get_forecast_start_end_time_weekahead(site_metadata, variable,
                                               expected_start, expected_end):
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning'
    )

    run_time = pd.Timestamp('20190410T0630Z')
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=10),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, run_time)
    assert data_start == pd.Timestamp(expected_start)
    assert data_end == pd.Timestamp(expected_end)


@pytest.fixture
def forecast_hr_begin(site_metadata):
    return default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')


@pytest.mark.parametrize('adjust_for_interval_label', [True, False])
def test_get_forecast_start_end_time_same_as_issue_time(
        forecast_hr_begin, adjust_for_interval_label):
    # time is same as issue time of day
    issue_time = pd.Timestamp('20190422T0500')
    fx_start, fx_end = utils.get_forecast_start_end(
        forecast_hr_begin, issue_time, adjust_for_interval_label)
    assert fx_start == pd.Timestamp('20190422T0600')
    fx_end_exp = pd.Timestamp('20190422T0700')
    if adjust_for_interval_label:
        fx_end_exp -= pd.Timedelta('1n')
    assert fx_end == fx_end_exp


@pytest.mark.parametrize('adjust_for_interval_label', [True, False])
def test_get_forecast_start_end_time_same_as_issue_time_n_x_run(
        forecast_hr_begin, adjust_for_interval_label):
    # time is same as issue time of day + n * run_length
    issue_time = pd.Timestamp('20190422T1200')
    fx_start, fx_end = utils.get_forecast_start_end(
        forecast_hr_begin, issue_time, adjust_for_interval_label)
    assert fx_start == pd.Timestamp('20190422T1300')
    fx_end_exp = pd.Timestamp('20190422T1400')
    if adjust_for_interval_label:
        fx_end_exp -= pd.Timedelta('1n')
    assert fx_end == fx_end_exp


def test_get_forecast_start_end_time_before_issue_time(forecast_hr_begin):
    # time is before issue time of day but otherwise valid
    issue_time = pd.Timestamp('20190422T0400')
    with pytest.raises(ValueError):
        fx_start, fx_end = utils.get_forecast_start_end(
            forecast_hr_begin, issue_time)


def test_get_forecast_start_end_time_invalid(forecast_hr_begin):
    # time is invalid
    issue_time = pd.Timestamp('20190423T0505')
    with pytest.raises(ValueError):
        fx_start, fx_end = utils.get_forecast_start_end(
            forecast_hr_begin, issue_time)


@pytest.mark.parametrize('adjust_for_interval_label', [True, False])
def test_get_forecast_start_end_time_instant(
        site_metadata, adjust_for_interval_label):
    # instant
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1h'),
        interval_label='instant')
    issue_time = pd.Timestamp('20190422T0500')
    fx_start, fx_end = utils.get_forecast_start_end(
        forecast, issue_time, adjust_for_interval_label)
    assert fx_start == pd.Timestamp('20190422T0600')
    fx_end_exp = pd.Timestamp('20190422T0700')
    if adjust_for_interval_label:
        fx_end_exp -= pd.Timedelta('1n')
    assert fx_end == fx_end_exp


@pytest.mark.parametrize('fxargs,last_time,expected', [
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('0h'),
          run_length='1h',
          interval_label='beginning'),
     pd.Timestamp('2020-05-20T10:00Z'),
     pd.Timestamp('2020-05-20T11:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('0h'),
          run_length='1h',
          interval_label='beginning'),
     pd.Timestamp('2020-05-20T10:30Z'),  # inconsistent w/ fx
     pd.Timestamp('2020-05-20T11:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('0h'),
          run_length='1h',
          interval_label='ending'),
     pd.Timestamp('2020-05-20T10:00Z'),
     pd.Timestamp('2020-05-20T10:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('1h'),
          run_length='1h',
          interval_label='ending'),
     pd.Timestamp('2020-05-20T10:00Z'),
     pd.Timestamp('2020-05-20T09:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('1h'),
          run_length='1h',
          interval_label='beginning'),
     pd.Timestamp('2020-05-20T10:00Z'),
     pd.Timestamp('2020-05-20T10:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('2h'),
          run_length='4h',
          interval_label='beginning'),
     pd.Timestamp('2020-05-20T05:00Z'),
     pd.Timestamp('2020-05-20T04:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('2h'),
          run_length='4h',
          interval_label='ending'),
     pd.Timestamp('2020-05-20T06:00Z'),
     pd.Timestamp('2020-05-20T04:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('2h'),
          run_length='4h',
          interval_label='ending'),
     pd.Timestamp('2020-05-20T05:00Z'),  # run not finished?
     pd.Timestamp('2020-05-20T00:00Z')),
    (dict(issue_time_of_day=dt.time(hour=0),
          lead_time_to_start=pd.Timedelta('2h'),
          run_length='4h',
          interval_label='ending'),
     pd.Timestamp('2020-05-20T06:00:01Z'),  # run not finished?
     pd.Timestamp('2020-05-20T04:00Z')),
    (dict(issue_time_of_day=dt.time(hour=6),
          lead_time_to_start=pd.Timedelta('24h'),
          run_length='8h',
          interval_label='instantaneous'),
     pd.Timestamp('2020-05-20T04:00Z'),
     pd.Timestamp('2020-05-18T22:00Z')),
])
def test_find_next_issue_time_from_last_forecast(fxargs, last_time, expected,
                                                 site_metadata):
    fx = default_forecast(site_metadata, **fxargs)
    out = utils.find_next_issue_time_from_last_forecast(fx, last_time)
    assert out == expected


@pytest.mark.parametrize('obs_kw,fx_kw,index', [
    pytest.param(dict(), dict(), False,
                 marks=pytest.mark.xfail(strict=True)),
    pytest.param(dict(), dict(), True,
                 marks=pytest.mark.xfail(strict=True)),
    # obs too long
    (dict(interval_length=pd.Timedelta('2h')), dict(), True),
    (dict(interval_length=pd.Timedelta('2h')), dict(), False),
    # short fx runs
    (dict(), dict(run_length=pd.Timedelta('15min')), True),
    (dict(), dict(run_length=pd.Timedelta('15min')), False),
    # day forecast index
    (dict(), dict(run_length=pd.Timedelta('24h')), True),
    pytest.param(dict(), dict(run_length=pd.Timedelta('24h')), False,
                 marks=pytest.mark.xfail(strict=True)),
    # non-instant obs
    (dict(), dict(interval_label='instantaneous'), True),
    (dict(), dict(interval_label='instantaneous'), False),
    # instant, but mismatch length
    (dict(interval_label='instantaneous'),
     dict(interval_label='instantaneous'),
     True),
    (dict(interval_label='instantaneous'),
     dict(interval_label='instantaneous'),
     False),
])
def test_check_persistence_compatibility(obs_kw, fx_kw, index, site_metadata):
    obs_dict = {'interval_label': 'ending',
                'interval_value_type': 'interval_mean',
                'interval_length': pd.Timedelta('30min')}
    fx_dict = {'interval_label': 'ending',
               'interval_value_type': 'interval_mean',
               'interval_length': pd.Timedelta('1h'),
               'run_length': pd.Timedelta('12h')}
    obs_dict.update(obs_kw)
    fx_dict.update(fx_kw)
    obs = default_observation(site_metadata, **obs_dict)
    fx = default_forecast(site_metadata, **fx_dict)
    with pytest.raises(ValueError):
        utils.check_persistence_compatibility(obs, fx, index)
