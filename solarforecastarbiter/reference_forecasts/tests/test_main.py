import datetime as dt
from functools import partial

import pandas as pd
# from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter import datamodel
from solarforecastarbiter.reference_forecasts import main, models

# we'll need to do something better once the load_forecast function works
init_time = pd.Timestamp('20190328T1200Z')
start = pd.Timestamp('20190328T1300Z')
end = pd.Timestamp('20190328T1400Z')

index_exp = pd.DatetimeIndex(start=start, end=end, freq='1h')
ghi_exp = pd.Series([0, 10.], index=index_exp)
dni_exp = pd.Series([0, 15.], index=index_exp)
dhi_exp = pd.Series([0, 9.], index=index_exp)
temp_air_exp = pd.Series([10, 11.], index=index_exp)
wind_speed_exp = pd.Series([0, 1.], index=index_exp)
cloud_cover_exp = pd.Series([100., 0.], index=index_exp)
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


# can't figure out how to mock the load_forecast function here, so
# xfail for now. This all needs to change eventually anyways.
@pytest.mark.xfail(raises=NotImplementedError, strict=True)
@pytest.mark.parametrize('model,load_forecast_return_value', [
    pytest.param(
        models.gfs_quarter_deg_3hour_to_hourly_mean,
        load_forecast_return_value_3),
    pytest.param(
        models.gfs_quarter_deg_hourly_to_hourly_mean,
        load_forecast_return_value_3),
    pytest.param(
        models.gfs_quarter_deg_to_hourly_mean,
        load_forecast_return_value_3,
        marks=pytest.mark.xfail(strict=True, raises=NotImplementedError)),
    (models.hrrr_subhourly_to_hourly_mean, load_forecast_return_value_5),
    (models.hrrr_subhourly_to_subhourly_instantaneous,
        load_forecast_return_value_5),
    pytest.param(models.nam_12km_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
    (models.nam_12km_hourly_to_hourly_instantaneous,
     load_forecast_return_value_3),
    pytest.param(models.rap_cloud_cover_to_hourly_mean,
                 load_forecast_return_value_3),
    (models.rap_ghi_to_hourly_mean, load_forecast_return_value_3),
    (models.rap_ghi_to_instantaneous, load_forecast_return_value_3),
])
def test_run(model, load_forecast_return_value, site_powerplant_site_type,
             mocker):
    mocker.patch(
        'solarforecastarbiter.reference_forecasts.forecast.unmix_intervals',
        return_value=cloud_cover_exp)
    site, site_type = site_powerplant_site_type
    out = main.run(site, model, init_time, start, end)
    check_out(out, out_forecast_exp, site_type)


def test_get_data_start_end_labels(site_metadata):
    # common variables
    name = 'Albuquerque Baseline AC Power'
    variable = 'ac_power'
    value_type = 'mean'
    uncertainty = 1

    _observation = partial(
        datamodel.Observation, name=name, variable=variable,
        value_type=value_type, site=site_metadata, uncertainty=uncertainty)

    _forecast = partial(datamodel.Forecast, name=name, value_type=value_type,
                        variable=variable, site=site_metadata)

    observation = _observation(
        interval_length=pd.Timedelta('5min'), interval_label='beginning')
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h'),  # test 1 hr limit on window
        interval_label='beginning')

    # ensure data no later than run time
    run_time = pd.Timestamp('20190422T1945Z')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190422T1845Z')
    assert data_end == pd.Timestamp('20190422T1945Z')

    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('5min'),  # test subhourly limit on window
        interval_label='beginning')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190422T1940Z')
    assert data_end == pd.Timestamp('20190422T1945Z')

    # obs interval cannot be longer than forecast interval
    observation = _observation(
        interval_length=pd.Timedelta('15min'), interval_label='beginning')
    with pytest.raises(ValueError) as excinfo:
        main.get_data_start_end(observation, forecast, run_time)
    assert ("observation.interval_length <= forecast.run_length" in
            str(excinfo.value))

    # obs interval cannot be longer than 1 hr
    observation = _observation(
        interval_length=pd.Timedelta('2h'), interval_label='beginning')
    with pytest.raises(ValueError) as excinfo:
        main.get_data_start_end(observation, forecast, run_time)
    assert 'observation.interval_length <= 1h' in str(excinfo.value)

    # day ahead doesn't care about obs interval length
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1d'),  # day ahead
        interval_label='beginning')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190422T0000Z')

    # instant obs
    observation = _observation(
        interval_length=pd.Timedelta('5min'), interval_label='instant')
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),  # interval_length must be equal
        run_length=pd.Timedelta('1d'),
        interval_label='instant')            # if interval_label also instant
    with pytest.raises(ValueError) as excinfo:
        main.get_data_start_end(observation, forecast, run_time)
    assert 'with identical interval length' in str(excinfo.value)

    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),  # interval_length must be equal
        run_length=pd.Timedelta('1d'),
        interval_label='instant')              # if interval_label also instant
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')

    # instant obs, but interval fx
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='beginning')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190421T0000Z')
    assert data_end == pd.Timestamp('20190421T235959Z')

    # instant obs, but interval fx
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='ending')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190421T000001Z')
    assert data_end == pd.Timestamp('20190422T0000Z')

    # instant obs, but interval fx, intraday
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('15min'),
        interval_label='ending')
    data_start, data_end = main.get_data_start_end(observation, forecast,
                                                   run_time)
    assert data_start == pd.Timestamp('20190422T193001Z')
    assert data_end == pd.Timestamp('20190422T1945Z')

    # instant fx can't be made from interval avg obs
    observation = _observation(
        interval_length=pd.Timedelta('5min'), interval_label='ending')
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1d'),
        interval_label='instant')
    with pytest.raises(ValueError) as excinfo:
        main.get_data_start_end(observation, forecast, run_time)
    assert 'made from interval average obs' in str(excinfo.value)


def test_get_forecast_start_end(site_metadata):
    # common variables
    name = 'Albuquerque Baseline AC Power'
    variable = 'ac_power'
    value_type = 'mean'

    _forecast = partial(datamodel.Forecast, name=name, value_type=value_type,
                        variable=variable, site=site_metadata)

    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')

    # time is same as issue time of day
    issue_time = pd.Timestamp('20190422T0500')
    fx_start, fx_end = main.get_forecast_start_end(forecast, issue_time)
    assert fx_start == pd.Timestamp('20190422T0600')
    assert fx_end == pd.Timestamp('20190422T0700')

    # time is same as issue time of day + n * run_length
    issue_time = pd.Timestamp('20190422T1200')
    fx_start, fx_end = main.get_forecast_start_end(forecast, issue_time)
    assert fx_start == pd.Timestamp('20190422T1300')
    assert fx_end == pd.Timestamp('20190422T1400')

    # time is before issue time of day but otherwise valid
    issue_time = pd.Timestamp('20190422T0400')
    with pytest.raises(ValueError):
        fx_start, fx_end = main.get_forecast_start_end(forecast, issue_time)

    # time is invalid
    issue_time = pd.Timestamp('20190423T0505')
    with pytest.raises(ValueError):
        fx_start, fx_end = main.get_forecast_start_end(forecast, issue_time)

    # instant
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('5min'),
        run_length=pd.Timedelta('1h'),
        interval_label='instant')
    issue_time = pd.Timestamp('20190422T0500')
    fx_start, fx_end = main.get_forecast_start_end(forecast, issue_time)
    assert fx_start == pd.Timestamp('20190422T0600')
    assert fx_end == pd.Timestamp('20190422T065959')


def test_run_persistence_fails(site_metadata, mocker):
    # common variables
    name = 'Albuquerque Baseline AC Power'
    variable = 'ac_power'
    value_type = 'mean'
    uncertainty = 1

    _observation = partial(
        datamodel.Observation, name=name, variable=variable,
        value_type=value_type, site=site_metadata, uncertainty=uncertainty)

    _forecast = partial(datamodel.Forecast, name=name, value_type=value_type,
                        variable=variable, site=site_metadata)

    observation = _observation(
        interval_length=pd.Timedelta('5min'), interval_label='beginning')

    run_time = pd.Timestamp('20190422T1945Z')

    # intraday, index=False
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('1h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    mocker.patch.object(main.persistence, 'persistence_scalar',
                        autospec=True)
    main.run_persistence(observation, forecast, run_time, issue_time)
    assert main.persistence.persistence_scalar.call_count == 1

    # intraday, index=True
    mocker.patch.object(main.persistence, 'persistence_scalar_index',
                        autospec=True)
    main.run_persistence(observation, forecast, run_time, issue_time,
                         index=True)
    assert main.persistence.persistence_scalar_index.call_count == 1

    # day ahead, index = False
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    mocker.patch.object(main.persistence, 'persistence_interval',
                        autospec=True)
    main.run_persistence(observation, forecast, run_time, issue_time)
    assert main.persistence.persistence_interval.call_count == 1

    # index=True not supported for day ahead
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2300Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(observation, forecast, run_time, issue_time,
                             index=True)
    assert 'index=True not supported' in str(excinfo.value)

    forecast = _forecast(
        issue_time_of_day=dt.time(hour=23),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('48h'),  # too long
        interval_label='beginning')

    issue_time = pd.Timestamp('20190423T2300Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(observation, forecast, run_time, issue_time)
    assert 'midnight to midnight' in str(excinfo.value)

    # not midnight to midnight
    forecast = _forecast(
        issue_time_of_day=dt.time(hour=22),
        lead_time_to_start=pd.Timedelta('1h'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='beginning')
    issue_time = pd.Timestamp('20190423T2200Z')
    with pytest.raises(ValueError) as excinfo:
        main.run_persistence(observation, forecast, run_time, issue_time)
    assert 'midnight to midnight' in str(excinfo.value)
