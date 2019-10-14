import time
import datetime
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import preprocessing


THREE_HOURS = pd.date_range(start='2019-03-31T12:00:00',
                            periods=3,
                            freq='60min',
                            tz='MST',
                            name='timestamp')

THREE_HOUR_SERIES = pd.Series(np.arange(1., 4., 1.), index=THREE_HOURS,
                              name='value')

THIRTEEN_10MIN = pd.date_range(start='2019-03-31T12:00:00',
                               periods=13,
                               freq='10min',
                               tz='MST',
                               name='timestamp')

THIRTEEN_10MIN_SERIES = pd.Series((np.arange(0., 13., 1.)/6)+1,
                                  index=THIRTEEN_10MIN)

# Bitwise-flag integers (only test validated and versioned data)
NT_UF = int(0b10011)  # Nighttime, User Flagged and version 0 (19)
CSE_NT = int(0b1000010010)  # Clearsky exceeded, nighttime, and version 0 (530)
CSE = int(0b1000000010)  # Clearsky exceeded and version 0 (514)
OK = int(0b10)  # OK version 0 (2)


@pytest.mark.parametrize('interval_label', ['beginning', 'instant', 'ending'])
@pytest.mark.parametrize('fx_series,obs_series,expected_dt', [
    (THREE_HOUR_SERIES, THREE_HOUR_SERIES, THREE_HOURS),
    (THREE_HOUR_SERIES, THIRTEEN_10MIN_SERIES, THREE_HOURS),
    (THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN)
])
def test_resample_and_align(site_metadata, interval_label,
                            fx_series, obs_series, expected_dt):
    # Create the ForecastObservation to match interval_lengths of data
    observation = datamodel.Observation(
        site=site_metadata, name='dummy obs', variable='ghi',
        interval_value_type='instantaneous', uncertainty=1,
        interval_length=pd.Timedelta(obs_series.index.freq),
        interval_label=interval_label
    )
    forecast = datamodel.Forecast(
        site=site_metadata, name='dummy fx', variable='ghi',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta(fx_series.index.freq),
        interval_label=interval_label,
        issue_time_of_day=datetime.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h')
    )
    fx_obs = datamodel.ForecastObservation(forecast=forecast,
                                           observation=observation)

    # Use local tz
    local_tz = f"Etc/GMT{int(time.timezone/3600):+d}"

    data = {
        fx_obs.forecast: fx_series,
        fx_obs.observation: obs_series
    }
    result = preprocessing.resample_and_align(fx_obs, data, local_tz)

    # Localize datetimeindex
    expected_dt = expected_dt.tz_convert(local_tz)

    pd.testing.assert_index_equal(result.forecast_values.index,
                                  result.observation_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(result.observation_values.index,
                                  expected_dt,
                                  check_categorical=False)


@pytest.mark.parametrize('fx0', [
    pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
    THREE_HOUR_SERIES
])
@pytest.mark.parametrize('fx1', [
    pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
    THREE_HOUR_SERIES
])
@pytest.mark.parametrize('obs', [
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [OK, OK, OK]},
                 index=THREE_HOURS),
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [NT_UF, NT_UF, NT_UF]},
                 index=THREE_HOURS),
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [CSE_NT, CSE, OK]},
                 index=THREE_HOURS),
])
@pytest.mark.parametrize('handle_func', [preprocessing.exclude])
def test_apply_validation(report_objects, fx0, fx1, obs, handle_func):
    report, obs_model, fx0_model, fx1_model = report_objects
    data = {
        obs_model: obs,
        fx0_model: fx0,
        fx1_model: fx1
    }
    result = preprocessing.apply_validation(data,
                                            report.filters[0],
                                            handle_func)

    # Check length and timestamps of observation
    assert len(obs[obs.quality_flag.isin([OK, CSE])]) == \
        len(result[obs_model])
    if not result[obs_model].empty:
        assert obs[obs.quality_flag.isin([OK, CSE])].index.equals(
            result[obs_model].index)
    if not fx0.empty:
        assert result[fx0_model].equals(fx0)
    if not fx1.empty:
        assert result[fx1_model].equals(fx1)


def test_apply_validation_errors(report_objects):
    report, obs_model, fx0_model, fx1_model = report_objects
    obs = pd.DataFrame({'value': [1., 2., 3.],
                        'quality_flag': [OK, OK, OK]},
                        index=THREE_HOURS)
    fx0 = THREE_HOUR_SERIES
    fx1 = THREE_HOUR_SERIES
    data_ok = {
        obs_model: obs,
        fx0_model: fx0,
        fx1_model: fx1
    }
    # Pass a none QualityFlagFilter
    with pytest.raises(TypeError):
        preprocessing.apply_validation(data_ok,
                                       THREE_HOUR_SERIES,
                                       preprocessing.exclude)
    data_bad = {
        obs_model: obs,
        'NotAForecast': fx0,
        fx1_model: fx1
    }
    # Pass a none Forecast
    with pytest.raises(TypeError):
        preprocessing.apply_validation(data_bad,
                                       THREE_HOUR_SERIES,
                                       preprocessing.exclude)


@pytest.mark.parametrize('values,qflags,expectation', [
    (THREE_HOUR_SERIES, None, THREE_HOUR_SERIES),
    (THREE_HOUR_SERIES,
        pd.DataFrame({1: [0, 0, 0]}, index=THREE_HOURS),
        THREE_HOUR_SERIES),
    (THREE_HOUR_SERIES,
        pd.DataFrame({1: [0, 1, 0]}, index=THREE_HOURS),
        THREE_HOUR_SERIES[[True, False, True]]),
    (THREE_HOUR_SERIES,
        pd.DataFrame({1: [1, 1, 0], 2: [0, 1, 0]}, index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, True]]),
    (THREE_HOUR_SERIES,
        pd.DataFrame({1: [1, 1, 1], 2: [1, 1, 1]}, index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, False]]),
    (pd.Series([np.NaN, np.NaN, np.NaN], index=THREE_HOURS),
        None,
        THREE_HOUR_SERIES[[False, False, False]]),
    (pd.Series([1., np.NaN, 3.], index=THREE_HOURS),
        pd.DataFrame({1: [0, 0, 0], 2: [0, 0, 0]}, index=THREE_HOURS),
        THREE_HOUR_SERIES[[True, False, True]]),
    (pd.Series([1., np.NaN, 3.], index=THREE_HOURS),
        pd.DataFrame({1: [1, 1, 0], 2: [1, 0, 0]}, index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, True]])
])
def test_exclude(values, qflags, expectation):
    result = preprocessing.exclude(values, qflags)
    pd.testing.assert_series_equal(result, expectation, check_names=False)
