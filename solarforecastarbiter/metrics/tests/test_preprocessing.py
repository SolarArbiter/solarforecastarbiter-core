import time
import datetime as dt
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

THREE_HOUR_NAN_SERIES = pd.Series([1.0, np.nan, 4.0], index=THREE_HOURS,
                                  name="value")

THREE_HOURS_NAN = THREE_HOURS[[True, False, True]]

THREE_HOURS_EMPTY = pd.DatetimeIndex([], name="timestamp", freq="60min",
                                     tz="MST")

THREE_HOUR_EMPTY_SERIES = pd.Series([], index=THREE_HOURS_EMPTY, name="value")

EMPTY_OBJ_SERIES = pd.Series(
    [],
    dtype=object,
    name="value",
    index=pd.DatetimeIndex([], freq="10min", tz="MST", name="timestamp")
)

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


# TODO: test with different interval label between forecast and obs
# TODO: test with aggregate
@pytest.mark.parametrize('interval_label', ['beginning', 'instant', 'ending'])
@pytest.mark.parametrize('fx_series,obs_series,expected_dt', [
    (THREE_HOUR_SERIES, THREE_HOUR_SERIES, THREE_HOURS),
    (THREE_HOUR_SERIES, THIRTEEN_10MIN_SERIES, THREE_HOURS),
    (THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN),
    (THREE_HOUR_SERIES, THREE_HOUR_NAN_SERIES, THREE_HOURS_NAN),
    (THREE_HOUR_NAN_SERIES, THREE_HOUR_SERIES, THREE_HOURS_NAN),
    (THREE_HOUR_NAN_SERIES, THREE_HOUR_NAN_SERIES, THREE_HOURS_NAN),
    (THREE_HOUR_SERIES, THREE_HOUR_EMPTY_SERIES, THREE_HOURS_EMPTY),
    (THREE_HOUR_EMPTY_SERIES, THREE_HOUR_SERIES, THREE_HOURS_EMPTY),
    (THREE_HOUR_SERIES, EMPTY_OBJ_SERIES, THREE_HOURS_EMPTY),
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
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h')
    )
    fx_obs = datamodel.ForecastObservation(forecast=forecast,
                                           observation=observation)

    # Use local tz
    local_tz = f"Etc/GMT{int(time.timezone/3600):+d}"

    forecast_values, observation_values = preprocessing.resample_and_align(
        fx_obs, fx_series, obs_series, local_tz)

    # Localize datetimeindex
    expected_dt = expected_dt.tz_convert(local_tz)

    pd.testing.assert_index_equal(forecast_values.index,
                                  observation_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(observation_values.index,
                                  expected_dt,
                                  check_categorical=False)


@pytest.mark.parametrize('obs,somecounts', [
    (pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                  columns=['value', 'quality_flag']),
     {'USER FLAGGED': 0}),
    (pd.DataFrame({'value': [1., 2., 3.],
                   'quality_flag': [OK, OK, OK]},
                  index=THREE_HOURS),
     {'USER FLAGGED': 0}),
    (pd.DataFrame({'value': [1., 2., 3.],
                   'quality_flag': [NT_UF, NT_UF, NT_UF]},
                  index=THREE_HOURS),
     {'NIGHTTIME': 3, 'USER FLAGGED': 3, 'STALE VALUES': 0}),
    (pd.DataFrame({'value': [1., 2., 3.],
                   'quality_flag': [CSE_NT, CSE, OK]},
                  index=THREE_HOURS),
     {'NIGHTTIME': 1, 'USER FLAGGED': 0, 'CLEARSKY EXCEEDED': 2}),
])
@pytest.mark.parametrize('handle_func', [preprocessing.exclude])
@pytest.mark.parametrize('filter_', [
    datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
            "NIGHTTIME",
            "LIMITS EXCEEDED",
            "STALE VALUES",
            "INTERPOLATED VALUES",
            "INCONSISTENT IRRADIANCE COMPONENTS",
        )
    ),
    pytest.param(
        datamodel.TimeOfDayFilter((dt.time(12, 0),
                                   dt.time(14, 0))),
        marks=pytest.mark.xfail(strict=True, type=TypeError)
    )
])
def test_apply_validation(obs, handle_func, filter_, somecounts):
    result, counts = preprocessing.apply_validation(obs, filter_,
                                                    handle_func)

    # Check length and timestamps of observation
    assert len(obs[obs.quality_flag.isin([OK, CSE])]) == \
        len(result)
    if not result.empty:
        assert obs[obs.quality_flag.isin([OK, CSE])].index.equals(
            result.index)
    assert set(filter_.quality_flags) == set(counts.keys())
    for k, v in somecounts.items():
        assert counts.get(k, v) == v


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


def test_merge_quality_filters():
    filters = [
        datamodel.QualityFlagFilter(('USER FLAGGED', 'NIGHTTIME',
                                     'CLIPPED VALUES')),
        datamodel.QualityFlagFilter(('SHADED', 'NIGHTTIME',)),
        datamodel.QualityFlagFilter(())
    ]
    out = preprocessing._merge_quality_filters(filters)
    assert set(out.quality_flags) == {'USER FLAGGED', 'NIGHTTIME',
                                      'CLIPPED VALUES', 'SHADED'}


def test_process_forecast_observations(report_objects):
    # TODO: add thorough tests here
    pass


def nooptest_validate_resample_align(mock_data, more_report_objects):
    report, observation, forecast_0, forecast_1 = more_report_objects
    meta = main.create_metadata(report)
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    processed_fxobs_list = main.validate_resample_align(report, meta, data)
    assert len(processed_fxobs_list) == len(report.forecast_observations)
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)
