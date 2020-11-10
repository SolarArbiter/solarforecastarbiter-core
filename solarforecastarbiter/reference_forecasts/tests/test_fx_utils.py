from dataclasses import replace
import datetime as dt


import pandas as pd
import pytest
import pytz


from solarforecastarbiter.reference_forecasts import utils
from solarforecastarbiter.conftest import default_forecast, default_observation


@pytest.fixture
def forecast_hr_begin(site_metadata):
    return default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')


@pytest.mark.parametrize('issuetime,rl,lt,start,expected', [
    ('06:00', '1h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0000Z')] + list(
         pd.date_range(start='20190101T0600Z', end='20190102T0000Z',
                       freq='1h'))),
    ('00:00', '4h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0000Z'), pd.Timestamp('20190101T0400Z'),
      pd.Timestamp('20190101T0800Z'), pd.Timestamp('20190101T1200Z'),
      pd.Timestamp('20190101T1600Z'), pd.Timestamp('20190101T2000Z'),
      pd.Timestamp('20190102T0000Z')]),
    ('16:00', '8h', '3h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0000Z'), pd.Timestamp('20190101T1600Z'),
      pd.Timestamp('20190102T0000Z')]),
    ('00:30', '4h', '120h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0030Z'), pd.Timestamp('20190101T0430Z'),
      pd.Timestamp('20190101T0830Z'), pd.Timestamp('20190101T1230Z'),
      pd.Timestamp('20190101T1630Z'), pd.Timestamp('20190101T2030Z'),
      pd.Timestamp('20190102T0030Z')]),
    ('05:00', '6h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0500Z'), pd.Timestamp('20190101T1100Z'),
      pd.Timestamp('20190101T1700Z'), pd.Timestamp('20190101T2300Z'),
      pd.Timestamp('20190102T0500Z')]),
    ('11:00', '12h', '3h', pd.Timestamp('20190101T2300Z'),
     [pd.Timestamp('20190101T1100Z'), pd.Timestamp('20190101T2300Z'),
      pd.Timestamp('20190102T1100Z')]),
    ('05:00', '12h', '1h', pd.Timestamp('20190101T0000Z'),
     [pd.Timestamp('20190101T0500Z'), pd.Timestamp('20190101T1700Z'),
      pd.Timestamp('20190102T0500Z')]),
    ('05:00', '12h', '1h', pd.Timestamp('20190101T0900-06:00'),
     [pd.Timestamp('20190101T1100-06:00'),
      pd.Timestamp('20190101T2300-06:00'), pd.Timestamp('20190102T1100-06:00')
      ]),
    ('05:00', '12h', '1h', pd.Timestamp('20190101T0900'),
     [pd.Timestamp('20190101T0500'), pd.Timestamp('20190101T1700'),
      pd.Timestamp('20190102T0500')]),
    ('10:00', '12h', '1h', pd.Timestamp('20190101T0900-06:00'),
     [pd.Timestamp('20190101T0400-06:00'), pd.Timestamp('20190101T1600-06:00'),
      pd.Timestamp('20190102T0400-06:00')]),
    ('00:00', '6h', '1h', pd.Timestamp('20190101T1801-06:00'),
     [pd.Timestamp('20190101T0000-06:00'), pd.Timestamp('20190101T0600-06:00'),
      pd.Timestamp('20190101T1200-06:00'), pd.Timestamp('20190101T1800-06:00'),
      pd.Timestamp('20190102T0000-06:00')]),
    ('05:00', '24h', '1h', pd.Timestamp('20190101T0900'),
     [pd.Timestamp('20190101T0500'), pd.Timestamp('20190102T0500')]),
    ('05:00', '2d', '1h', pd.Timestamp('20190101T0900'),
     [pd.Timestamp('20190101T0500'), pd.Timestamp('20190103T0500')]),
    ('05:00', '36h', '1h', pd.Timestamp('20190101T0900'),
     [pd.Timestamp('20190101T0500'), pd.Timestamp('20190102T1700')]),
    ('12:00', '7d', '1h', pd.Timestamp('20190101T0900-07:00'),
     [pd.Timestamp('20190101T0500-07:00'),
      pd.Timestamp('20190108T0500-07:00')]),
    ('07:00', '12h', '1h',
     pd.Timestamp('20200307T0100', tz='America/New_York'),
     [pd.Timestamp('20200307T0200', tz='America/New_York'),
      pd.Timestamp('20200307T1400', tz='America/New_York'),
      # daylight savings
      pd.Timestamp('20200308T0300', tz='America/New_York')]
     )
])
def test_issue_times(single_forecast, issuetime, rl, lt, start,
                     expected):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.datetime.strptime(issuetime, '%H:%M').time(),
        run_length=pd.Timedelta(rl),
        lead_time_to_start=pd.Timedelta(lt))
    out = utils.get_issue_times(fx, start)
    assert out == expected


def test_issue_times_high_freq(single_forecast):
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.time(0),
        run_length=pd.Timedelta('15min'),
        interval_length=pd.Timedelta('5min'),
        lead_time_to_start=pd.Timedelta('5min'))
    out = utils.get_issue_times(fx, pd.Timestamp('20200501T0000-07:00'))
    assert out == list(pd.date_range(
        start='20200501T0000', end='20200502T0000', tz='Etc/GMT+7',
        freq='15min'))


def test_issue_times_high_freq_offset(single_forecast):
    fx = replace(
        single_forecast,
        issue_time_of_day=pytz.timezone('Etc/GMT+7').localize(dt.time(1)),
        run_length=pd.Timedelta('15min'),
        interval_length=pd.Timedelta('5min'),
        lead_time_to_start=pd.Timedelta('5min'))
    out = utils.get_issue_times(fx, pd.Timestamp('20200501T0000-07:00'))
    assert out == [pd.Timestamp('20200501T0000-07:00')] + list(
        pd.date_range(
            start='20200501T0100', end='20200502T0000', tz='Etc/GMT+7',
            freq='15min'))


def test_issue_times_localized(single_forecast):
    fx = replace(
        single_forecast,
        issue_time_of_day=pytz.timezone('Etc/GMT+7').localize(dt.time(12)),
        run_length=pd.Timedelta('12h'),
    )
    out = utils.get_issue_times(fx, pd.Timestamp('20200601T1700-0500'))
    assert out == [pd.Timestamp('20200601T0200-05:00'),
                   pd.Timestamp('20200601T1400-05:00'),
                   pd.Timestamp('20200602T0200-05:00')]


def test_issue_times_localized_dst(single_forecast):
    tzinfo = pd.Timestamp('20200308T0000', tz='America/New_York').tzinfo
    fx = replace(
        single_forecast,
        issue_time_of_day=dt.time(hour=2, tzinfo=tzinfo),
        run_length=pd.Timedelta('12h'),
    )
    out = utils.get_issue_times(fx, pd.Timestamp('20200308T0100-0500'))
    assert out == [pd.Timestamp('20200308T0300-04:00'),
                   pd.Timestamp('20200308T1500-04:00'),
                   pd.Timestamp('20200309T0300-04:00')]


def test_issue_times_fx_gap(forecast_hr_begin):
    # 1-4 utc should not be included in output as 5 is the first
    # issue time for the day.
    out = utils.get_issue_times(forecast_hr_begin,
                                pd.Timestamp('20200505T2000-07:00'))
    assert out == list(pd.date_range(
        start=pd.Timestamp('20200505T0000-07:00'),
        end=pd.Timestamp('20200505T1700-07:00'),
        freq='1h')) + list(pd.date_range(
            start=pd.Timestamp('20200505T2200-07:00'),
            end=pd.Timestamp('20200506T0000-07:00'),
            freq='1h'
        ))


@pytest.mark.parametrize('runtime,expected', [
    (pd.Timestamp('20190501T1100Z'), pd.Timestamp('20190501T1100Z')),
    (pd.Timestamp('20190501T1030Z'), pd.Timestamp('20190501T1100Z')),
    (pd.Timestamp('20190501T0030Z'), pd.Timestamp('20190501T0500Z')),
    (pd.Timestamp('20190501T2359Z'), pd.Timestamp('20190502T0500Z')),
    (pd.Timestamp('20190501T2200Z'), pd.Timestamp('20190501T2300Z')),
    (pd.Timestamp('20190501T0900-07:00'), pd.Timestamp('20190501T1000-07:00'))
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
    issue_time = pd.Timestamp('20190422T2000Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
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
    issue_time = pd.Timestamp('20190422T2000Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
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
    issue_time = pd.Timestamp('20190422T2000Z')
    # obs interval cannot be longer than forecast interval
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time, issue_time)
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
    issue_time = pd.Timestamp('20190422T2000Z')
    # obs interval cannot be longer than 1 hr
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time, issue_time)
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
    issue_time = pd.Timestamp('20190423T0500Z')
    # day ahead doesn't care about obs interval length
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
    assert data_start == pd.Timestamp('20190421T0600Z')
    assert data_end == pd.Timestamp('20190422T0600Z')


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
    issue_time = pd.Timestamp('20190422T2000Z')
    with pytest.raises(ValueError) as excinfo:
        utils.get_data_start_end(observation, forecast, run_time, issue_time)
    assert 'with identical interval length' in str(excinfo.value)


@pytest.mark.parametrize('it,lead,issue', [
    (23, '1h', '20190422T2300Z'),
    (5, '19h', '20190422T0500Z'),
    (5, '19h', '20190422T0000-05:00')
])
def test_get_data_start_end_labels_obs_fx_instant(site_metadata, lead, issue,
                                                  it):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    # interval length of forecast and obs must be equal if interval label is
    # instant
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=it),
        lead_time_to_start=pd.Timedelta(lead),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='instant')
    issue_time = pd.Timestamp(issue)
    run_time = issue_time - pd.Timedelta('75min')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')


def test_get_data_start_end_labels_obs_instant_fx_avg(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    run_time = pd.Timestamp('20190422T1945Z')
    issue_time = pd.Timestamp('20190422T2300Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')


def test_get_data_start_end_labels_obs_instant_fx_avg_ending(site_metadata):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=0),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='ending')
    run_time = pd.Timestamp('20190422T1945Z')
    issue_time = pd.Timestamp('20190422T2300Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
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
    issue_time = pd.Timestamp('20190422T2000Z')
    data_start, data_end = utils.get_data_start_end(observation, forecast,
                                                    run_time, issue_time)
    assert data_start == pd.Timestamp('20190422T193001Z')
    assert data_end == pd.Timestamp('20190422T1945Z')


def test_get_data_start_end_labels_obs_avg_fx_instant(site_metadata):
    run_time = pd.Timestamp('20190422T1945Z')
    issue_time = pd.Timestamp('20190422T2000Z')
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
        utils.get_data_start_end(observation, forecast, run_time, issue_time)
    assert 'made from interval average obs' in str(excinfo.value)


@pytest.mark.parametrize('rl,expected_start,expected_end,rt,lt', [
    ('1d', '20190409T0000Z', '20190410T0000Z', '20190410T2230Z', '1h'),
    ('1d', '20190409T0000Z', '20190410T0000Z', '20190410T0001Z', '1h'),
    ('1d', '20190409T0000Z', '20190410T0000Z', '20190410T0001Z', '25h'),
    ('1d', '20190409T0000Z', '20190410T0000Z', '20190410T0001Z', '49h'),
    ('1d', '20190408T2300Z', '20190409T2300Z', '20190410T2230Z', '24h'),
    # quirky that 23h ahead uses more recent data than 1h ahead, but
    # timestamps are maintained and end is before issue time
    ('1d', '20190409T2200Z', '20190410T2200Z', '20190410T2230Z', '23h'),
    ('2d', '20190408T0000Z', '20190410T0000Z', '20190410T2230Z', '1h'),
    ('36h', '20190409T0000Z', '20190410T1200Z', '20190410T2300Z', '1h'),
    # run time is too early compared to issue time so data_end > run_time
    # and that's ok.
    ('36h', '20190409T0000Z', '20190410T1200Z', '20190410T1100Z', '1h'),
    ('36h', '20190409T0000Z', '20190410T1200Z', '20190410T1100Z', '25h')
])
def test_get_data_start_end_time_dayahead(site_metadata, rl, rt, lt,
                                          expected_start, expected_end):
    observation = default_observation(
        site_metadata,
        interval_length=pd.Timedelta('5min'), interval_label='beginning'
    )

    run_time = pd.Timestamp(rt)
    issue_time = pd.Timestamp('20190410T2300Z')
    forecast = default_forecast(
        site_metadata,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta(lt),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta(rl),
        interval_label='beginning')
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, run_time, issue_time)
    assert data_start == pd.Timestamp(expected_start)
    assert data_end == pd.Timestamp(expected_end)


@pytest.mark.parametrize("variable,expected_start,expected_end,rl", [
    # generate forecast on Wednesday 4/10 that applies to Thursday 4/11 using
    # data from the previous Thursday (4/4)
    ("net_load", "20190404T0000Z", "20190405T0000Z", "1d"),
    ("net_load", "20190404T0000Z", "20190407T0000Z", "3d"),
    # generate forecast on Wednesday 4/10 that applies to Thursday 4/11 using
    # data from the previous day (Tuesday 4/9)
    ("ghi", "20190409T0000Z", "20190410T0000Z", "1d"),
])
def test_get_data_start_end_time_weekahead(site_metadata, variable, rl,
                                           expected_start, expected_end):
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning'
    )

    run_time = pd.Timestamp('20190410T2230Z')
    issue_time = pd.Timestamp('20190410T2300Z')
    # fx from 2019-04-11 00:00
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta(rl),
        interval_label='beginning')
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, run_time, issue_time)
    assert data_start == pd.Timestamp(expected_start)
    assert data_end == pd.Timestamp(expected_end)


@pytest.mark.parametrize("variable,expected_start,expected_end,rl,issue,run", [
    ("net_load", "20190404T0000Z", "20190405T0000Z", "1d",
     '20190410T1600-07:00', '20190410T1550-07:00'),
    ("net_load", "20190410T2150Z", "20190410T2250Z", "12h",
     '20190410T1600-07:00', '20190410T1550-07:00'),
    ("ghi", "20190404T0000Z", "20190405T0000Z", "1d",
     '20190405T1600-07:00', '20190405T1550-07:00'),
    ("ghi", "20190404T0000Z", "20190405T0000Z", "1d",
     '20190405T1600-07:00', '20190405T1750-05:00'),
    ("ghi", "20190405T1450-07:00", "20190405T1550-07:00", "8h",
     '20190405T1600-07:00', '20190405T1550-07:00'),
])
def test_get_data_start_end_time_tz(site_metadata, variable, rl, issue, run,
                                    expected_start, expected_end):
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='ending'
    )
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta(rl),
        interval_label='beginning')
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, pd.Timestamp(run), pd.Timestamp(issue))
    assert data_start == pd.Timestamp(expected_start)
    assert data_end == pd.Timestamp(expected_end)


def test_get_data_start_end_time_weekahead_not_midnight(site_metadata):
    variable = 'net_load'
    observation = default_observation(
        site_metadata, variable=variable,
        interval_length=pd.Timedelta('5min'), interval_label='beginning'
    )

    run_time = pd.Timestamp('20190410T1030Z')
    issue_time = pd.Timestamp('20190410T1200Z')
    # fx from 2019-04-11 12:00
    forecast = default_forecast(
        site_metadata, variable=variable,
        issue_time_of_day=dt.time(hour=12),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, run_time, issue_time)
    assert data_start == pd.Timestamp('20190404T1200Z')
    assert data_end == pd.Timestamp('20190405T1200Z')


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
@pytest.mark.parametrize('issue,exp_start,exp_end', [
    # issue time at 5, 1hr lead/interval
    (pd.Timestamp('20190422T0500'), pd.Timestamp('20190422T0600'),
     pd.Timestamp('20190422T0700')),
    (pd.Timestamp('20190422T0500Z'), pd.Timestamp('20190422T0600Z'),
     pd.Timestamp('20190422T0700Z')),
    (pd.Timestamp('20190422T0000-05:00'), pd.Timestamp('20190422T0100-05:00'),
     pd.Timestamp('20190422T0200-05:00')),
    (pd.Timestamp('20190421T2300-06:00'), pd.Timestamp('20190422T0000-06:00'),
     pd.Timestamp('20190422T0100-06:00')),
    (pd.Timestamp('20190422T0700-07:00'), pd.Timestamp('20190422T0800-07:00'),
     pd.Timestamp('20190422T0900-07:00')),
    # there is a gap in forecasts since issue time is 5 and only hourly run
    pytest.param(pd.Timestamp('20190422T2000-07:00'),  # 3 utc
                 pd.Timestamp('20190422T2100-07:00'),
                 pd.Timestamp('20190422T2200-07:00'),
                 marks=pytest.mark.xfail(strict=True)),
])
def test_get_forecast_start_end_time_tz(
        forecast_hr_begin, adjust_for_interval_label, issue,
        exp_start, exp_end):
    fx_start, fx_end = utils.get_forecast_start_end(
        forecast_hr_begin, issue, adjust_for_interval_label)
    assert fx_start == exp_start
    fx_end_exp = exp_end
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
     pd.Timestamp('2020-05-20T10:00-07:00'),
     pd.Timestamp('2020-05-20T11:00-07:00')),
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
          interval_label='instant'),
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
    # obs interval length > 1h too long
    (dict(interval_length=pd.Timedelta('2h')),
     dict(interval_length=pd.Timedelta('4h')), True),
    (dict(interval_length=pd.Timedelta('2h')),
     dict(interval_length=pd.Timedelta('4h')), False),
    # fx run_length < obs interval length
    (dict(), dict(run_length=pd.Timedelta('15min')), True),
    (dict(), dict(run_length=pd.Timedelta('15min')), False),
    # day forecast index
    (dict(), dict(run_length=pd.Timedelta('24h')), True),
    pytest.param(dict(), dict(run_length=pd.Timedelta('24h')), False,
                 marks=pytest.mark.xfail(strict=True)),
    # non-instant obs
    (dict(), dict(interval_label='instant'), True),
    (dict(), dict(interval_label='instant'), False),
    # instant, but mismatch length
    (dict(interval_label='instant'),
     dict(interval_label='instant'),
     True),
    (dict(interval_label='instant'),
     dict(interval_label='instant'),
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
