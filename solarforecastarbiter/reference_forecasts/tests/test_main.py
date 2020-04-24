from dataclasses import replace
import datetime as dt
from functools import partial
import inspect
from pathlib import Path
import re


import pandas as pd
from pandas.testing import assert_frame_equal
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


def test_run_persistence_weekahead(session, site_metadata, mocker):
    observation = default_observation(
        site_metadata, variable="load",
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190110T1945Z')
    forecast = default_forecast(
        site_metadata, variable="load",
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190111T2300Z')
    mocker.spy(main.persistence, 'persistence_interval')
    out = main.run_persistence(session, observation, forecast, run_time,
                               issue_time)
    assert isinstance(out, pd.Series)
    assert len(out) == 24
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


def test_verify_nwp_forecasts_compatible(ac_power_forecast_metadata):
    fx0 = ac_power_forecast_metadata
    fx1 = replace(fx0, run_length=pd.Timedelta('10h'), interval_label='ending')
    df = pd.DataFrame({'forecast': [fx0, fx1], 'model': ['a', 'b']})
    errs = main._verify_nwp_forecasts_compatible(df)
    assert set(errs) == {'model', 'run_length', 'interval_label'}


@pytest.mark.parametrize('string,expected', [
    ('{"is_reference_forecast": true}', True),
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
