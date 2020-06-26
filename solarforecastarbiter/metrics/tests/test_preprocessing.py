import time
import json
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import preprocessing


THREE_HOURS = pd.date_range(start='2019-04-01T06:00:00',
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

THREE_HOUR_EMPTY_SERIES = pd.Series([], index=THREE_HOURS_EMPTY, name="value",
                                    dtype='float64')

EMPTY_OBJ_SERIES = pd.Series(
    [],
    dtype=object,
    name="value",
    index=pd.DatetimeIndex([], freq="10min", tz="MST", name="timestamp")
)

THIRTEEN_10MIN = pd.date_range(start='2019-04-01T06:00:00',
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


def create_preprocessing_result(counts):
    """Create preprocessing results in order as
    1. Forecast Undefined Values,
    2. Forecast Values Discarded by Alignment,
    3. Observation Undefined Values,
    4. Observation Values Discarded by Alignment
    """
    return {
        "Forecast " + preprocessing.UNDEFINED_DATA_STRING: counts[0],
        "Forecast " + preprocessing.DISCARD_DATA_STRING: counts[1],
        "Observation " + preprocessing.UNDEFINED_DATA_STRING: counts[2],
        "Observation " + preprocessing.DISCARD_DATA_STRING: counts[3]
    }


@pytest.mark.parametrize('obs_interval_label',
                         ['beginning', 'instant', 'ending'])
@pytest.mark.parametrize('fx_interval_label',
                         ['beginning', 'ending'])
@pytest.mark.parametrize('fx_series,obs_series,expected_dt,expected_res', [
    (THREE_HOUR_SERIES, THREE_HOUR_SERIES, THREE_HOURS, [0]*4),
    (THREE_HOUR_SERIES, THIRTEEN_10MIN_SERIES, THREE_HOURS, [0]*4),
    (THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN_SERIES, THIRTEEN_10MIN, [0]*4),
    (THREE_HOUR_SERIES, THREE_HOUR_NAN_SERIES, THREE_HOURS_NAN, [0, 1, 1, 0]),
    (THREE_HOUR_NAN_SERIES, THREE_HOUR_SERIES, THREE_HOURS_NAN, [1, 0, 0, 1]),
    (THREE_HOUR_NAN_SERIES, THREE_HOUR_NAN_SERIES, THREE_HOURS_NAN, [1, 0, 1, 0]),  # NOQA
    (THREE_HOUR_SERIES, THREE_HOUR_EMPTY_SERIES, THREE_HOURS_EMPTY, [0, 3, 0, 0]),  # NOQA
    (THREE_HOUR_EMPTY_SERIES, THREE_HOUR_SERIES, THREE_HOURS_EMPTY, [0, 0, 0, 3]),  # NOQA
    (THREE_HOUR_SERIES, EMPTY_OBJ_SERIES, THREE_HOURS_EMPTY, [0, 3, 0, 0]),
])
def test_resample_and_align(
        site_metadata, obs_interval_label, fx_interval_label, fx_series,
        obs_series, expected_dt, expected_res):
    # Create the ForecastObservation to match interval_lengths of data
    observation = datamodel.Observation(
        site=site_metadata, name='dummy obs', variable='ghi',
        interval_value_type='instantaneous', uncertainty=1,
        interval_length=pd.Timedelta(obs_series.index.freq),
        interval_label=obs_interval_label
    )
    forecast = datamodel.Forecast(
        site=site_metadata, name='dummy fx', variable='ghi',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta(fx_series.index.freq),
        interval_label=fx_interval_label,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h')
    )
    fx_obs = datamodel.ForecastObservation(forecast=forecast,
                                           observation=observation)

    # Use local tz
    local_tz = f"Etc/GMT{int(time.timezone/3600):+d}"

    fx_vals, obs_vals, ref_vals, res_dict = preprocessing.resample_and_align(
            fx_obs, fx_series, obs_series, None, local_tz)

    # Localize datetimeindex
    expected_dt = expected_dt.tz_convert(local_tz)

    pd.testing.assert_index_equal(fx_vals.index,
                                  obs_vals.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_vals.index,
                                  expected_dt,
                                  check_categorical=False)

    expected_result = create_preprocessing_result(expected_res)
    assert res_dict == expected_result


def test_resample_and_align_fx_aggregate(single_forecast_aggregate):
    fx_series = THREE_HOUR_SERIES
    obs_series = THREE_HOUR_SERIES
    fx_values, obs_values, ref, res_dict = preprocessing.resample_and_align(
        single_forecast_aggregate, fx_series, obs_series, None, 'UTC')

    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  THREE_HOURS,
                                  check_categorical=False)


@pytest.mark.parametrize("label_obs,label_fx,length_obs,length_fx,freq", [
    ("beginning", "ending", "5min", "5min", "5min"),
    ("beginning", "ending", "5min", "1h", "1h"),
    ("ending", "beginning", "5min", "5min", "5min"),
    ("ending", "beginning", "5min", "1h", "1h"),
])
def test_resample_and_align_interval_label(site_metadata, label_obs, label_fx,
                                           length_obs, length_fx, freq):

    observation = datamodel.Observation(
        site=site_metadata, name='dummy obs', variable='ghi',
        interval_value_type='instantaneous', uncertainty=1,
        interval_length=pd.Timedelta(length_obs),
        interval_label=label_obs
    )
    forecast = datamodel.Forecast(
        site=site_metadata, name='dummy fx', variable='ghi',
        interval_value_type='instantaneous',
        interval_length=pd.Timedelta(length_fx),
        interval_label=label_fx,
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta('1h'),
        run_length=pd.Timedelta('12h')
    )
    fx_obs = datamodel.ForecastObservation(forecast=forecast,
                                           observation=observation)

    ts_obs = pd.date_range(
        start='2019-03-31T12:00:00',
        end='2019-05-01T00:00:00',
        freq=length_obs,
        tz='MST',
        name='timestamp'
    )
    ts_fx = pd.date_range(
        start='2019-03-31T12:00:00',
        end='2019-05-01T00:00:00',
        freq=length_fx,
        tz='MST',
        name='timestamp'
    )
    obs_series = pd.Series(index=ts_obs, data=np.random.rand(len(ts_obs)) + 10)
    fx_series = pd.Series(index=ts_fx, data=np.random.rand(len(ts_fx)) + 10)

    # Use local tz
    local_tz = f"Etc/GMT{int(time.timezone/3600):+d}"

    fx_out, obs_out, ref_out, res_dict = preprocessing.resample_and_align(
        fx_obs, fx_series, obs_series, None, local_tz)

    assert obs_out.index.freq == freq


@pytest.mark.parametrize("interval_label", ["beginning", "instant", "ending"])
@pytest.mark.parametrize("tz,local_tz,local_ts", [
    ("UTC", "UTC", ["20190702T0000", "20190702T0100"]),
    ("UTC", "US/Pacific", ["20190701T1700", "20190701T1800"]),
    ("US/Pacific", "UTC", ["20190702T0700", "20190702T0800"]),
    ("US/Central", "US/Pacific", ["20190701T2200", "20190701T2300"]),
    ("US/Pacific", "US/Eastern", ["20190702T0300", "20190702T0400"]),
])
def test_resample_and_align_timezone(site_metadata, interval_label, tz,
                                     local_tz, local_ts):

    expected_dt = pd.DatetimeIndex(local_ts, tz=local_tz)

    # Create the fx/obs pair
    ts = pd.DatetimeIndex(["20190702T0000", "20190702T0100"], tz=tz)
    fx_series = pd.Series([1.0, 4.0], index=ts, name="value")
    obs_series = pd.Series([1.1, 2.7], index=ts, name="value")
    observation = datamodel.Observation(
        site=site_metadata, name='dummy obs', variable='ghi',
        interval_value_type='instantaneous', uncertainty=1,
        interval_length=pd.Timedelta(obs_series.index.freq),
        interval_label=interval_label,
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

    fx_values, obs_values, ref_values, _ = preprocessing.resample_and_align(
        fx_obs, fx_series, obs_series, None, local_tz)

    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  expected_dt,
                                  check_categorical=False)


def test_resample_and_align_with_ref(
        single_forecast_observation_reffx):
    tz = 'UTC'
    fx_obs = single_forecast_observation_reffx
    fx_series = THREE_HOUR_SERIES
    ref_series = THREE_HOUR_SERIES
    obs_series = THREE_HOUR_SERIES
    fx_values, obs_values, ref_values, _ = preprocessing.resample_and_align(
        fx_obs, fx_series, obs_series, ref_series, tz)
    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(fx_values.index,
                                  ref_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  THREE_HOURS,
                                  check_categorical=False)


def test_resample_and_align_ref_less_fx(
        single_forecast_observation_reffx):
    tz = 'UTC'
    fx_obs = single_forecast_observation_reffx
    nine_hour_series = pd.concat([
        THREE_HOUR_SERIES.shift(periods=3, freq='1h'),
        THREE_HOUR_SERIES,
        THREE_HOUR_SERIES.shift(periods=3, freq='-1h')])
    fx_series = THREE_HOUR_SERIES
    ref_series = nine_hour_series
    obs_series = nine_hour_series
    fx_values, obs_values, ref_values, _ = preprocessing.resample_and_align(
        fx_obs, fx_series, obs_series, ref_series, tz)
    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(fx_values.index,
                                  ref_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  THREE_HOURS,
                                  check_categorical=False)


def test_resample_and_align_ref_error_None(
        single_forecast_observation, single_forecast_observation_reffx):
    tz = 'UTC'

    # no ref_fx object, but supplied ref_fx series
    fx_obs = single_forecast_observation
    fx_series = THREE_HOUR_SERIES
    ref_series = THREE_HOUR_SERIES
    obs_series = THREE_HOUR_SERIES
    with pytest.raises(ValueError):
        preprocessing.resample_and_align(
            fx_obs, fx_series, obs_series, ref_series, tz)

    # ref_fx object, but no supplied ref_fx series
    fx_obs = single_forecast_observation_reffx
    fx_series = THREE_HOUR_SERIES
    ref_series = None
    obs_series = THREE_HOUR_SERIES
    with pytest.raises(ValueError):
        preprocessing.resample_and_align(
            fx_obs, fx_series, obs_series, ref_series, tz)


@pytest.mark.parametrize('attr,value', [
    ('interval_label', 'ending'),
    ('interval_length', pd.Timedelta('20min')),
])
def test_resample_and_align_ref_error(
        single_forecast_observation_reffx, attr, value):
    tz = 'UTC'

    changes = {attr: value}
    # ref_fx object parameters are inconsistent with fx object parameters
    ref_fx = single_forecast_observation_reffx.reference_forecast.replace(
        **changes)
    fx_obs = single_forecast_observation_reffx.replace(
        reference_forecast=ref_fx)
    fx_series = THREE_HOUR_SERIES
    ref_series = THREE_HOUR_SERIES
    obs_series = THREE_HOUR_SERIES
    with pytest.raises(ValueError):
        preprocessing.resample_and_align(
            fx_obs, fx_series, obs_series, ref_series, tz)


def test_resample_and_align_ref_error_prob(prob_forecasts, single_observation):
    tz = 'UTC'
    cv = prob_forecasts.constant_values[0].replace(axis='y')
    ref_fx = prob_forecasts.replace(axis='y', constant_values=(cv,))
    fx_obs = datamodel.ForecastObservation(
        prob_forecasts,
        single_observation,
        reference_forecast=ref_fx)
    fx_series = THREE_HOUR_SERIES
    ref_series = THREE_HOUR_SERIES
    obs_series = THREE_HOUR_SERIES
    with pytest.raises(ValueError):
        preprocessing.resample_and_align(
            fx_obs, fx_series, obs_series, ref_series, tz)


def test_resample_and_align_prob(prob_forecasts, single_observation):
    tz = 'UTC'
    fx_obs = datamodel.ForecastObservation(
        prob_forecasts,
        single_observation,
        reference_forecast=prob_forecasts.replace(
            name='reference'))
    fx_data = THREE_HOUR_SERIES.to_frame()
    obs_data = THREE_HOUR_SERIES
    ref_data = THREE_HOUR_SERIES.to_frame()
    fx_values, obs_values, ref_values, _ = preprocessing.resample_and_align(
        fx_obs, fx_data, obs_data, ref_data, tz)
    pd.testing.assert_index_equal(fx_values.index,
                                  ref_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  THREE_HOURS,
                                  check_categorical=False)


def test_resample_and_align_prob_constant_value(
        prob_forecast_constant_value, single_observation):
    tz = 'UTC'
    fx_obs = datamodel.ForecastObservation(
        prob_forecast_constant_value,
        single_observation,
        reference_forecast=prob_forecast_constant_value.replace(
            name='reference'))
    fx_data = THREE_HOUR_SERIES
    obs_data = THREE_HOUR_SERIES
    ref_data = THREE_HOUR_SERIES
    fx_values, obs_values, ref_values, _ = preprocessing.resample_and_align(
        fx_obs, fx_data, obs_data, ref_data, tz)
    pd.testing.assert_index_equal(fx_values.index,
                                  ref_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(fx_values.index,
                                  obs_values.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_values.index,
                                  THREE_HOURS,
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


def test_process_forecast_observations(report_objects, quality_filter,
                                       timeofdayfilter, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    obs_ser = pd.Series(np.arange(8),
                        index=pd.date_range(start='2019-04-01T00:00:00',
                                            periods=8,
                                            freq='15min',
                                            tz='MST',
                                            name='timestamp'))
    obs_df = obs_ser.to_frame('value')
    obs_df['quality_flag'] = OK
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = OK
    data = {
        observation: obs_df,
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter, timeofdayfilter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert logger.warning.called
    assert not logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert all(isinstance(vr, datamodel.ValidationResult)
                   for vr in proc_fxobs.validation_results)
        assert all(isinstance(pr, datamodel.PreprocessingResult)
                   for pr in proc_fxobs.preprocessing_results)
        assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)


def test_process_probabilistic_forecast_observations(
        cdf_and_cv_report_objects, cdf_and_cv_report_data, quality_filter,
        timeofdayfilter, mocker):
    report, *_ = cdf_and_cv_report_objects

    data = cdf_and_cv_report_data

    filters = [quality_filter, timeofdayfilter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert logger.warning.called
    assert not logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert all(isinstance(vr, datamodel.ValidationResult)
                   for vr in proc_fxobs.validation_results)
        assert all(isinstance(pr, datamodel.PreprocessingResult)
                   for pr in proc_fxobs.preprocessing_results)
        if isinstance(proc_fxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            assert isinstance(proc_fxobs.forecast_values, pd.DataFrame)
        else:
            assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)


def test_process_probabilistic_forecast_observations_xy(
        cdf_and_cv_report_objects_xy, cdf_and_cv_report_data_xy,
        quality_filter, timeofdayfilter, mocker):
    report, *_ = cdf_and_cv_report_objects_xy

    data = cdf_and_cv_report_data_xy

    filters = [quality_filter, timeofdayfilter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert logger.warning.called
    assert not logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert all(isinstance(vr, datamodel.ValidationResult)
                   for vr in proc_fxobs.validation_results)
        assert all(isinstance(pr, datamodel.PreprocessingResult)
                   for pr in proc_fxobs.preprocessing_results)
        if isinstance(proc_fxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            assert isinstance(proc_fxobs.forecast_values, pd.DataFrame)
        else:
            assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)


def test_process_forecast_observations_no_data(
        report_objects, quality_filter, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = NT_UF
    data = {
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)


def test_process_forecast_observations_no_cost(report_objects, quality_filter,
                                               timeofdayfilter, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    obs_ser = pd.Series(np.arange(8),
                        index=pd.date_range(start='2019-03-31T12:00:00',
                                            periods=8,
                                            freq='15min',
                                            tz='MST',
                                            name='timestamp'))
    obs_df = obs_ser.to_frame('value')
    obs_df['quality_flag'] = OK
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = OK
    data = {
        observation: obs_df,
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter, timeofdayfilter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    obj_pairs = list(report.report_parameters.object_pairs)
    obj_pairs[-1] = obj_pairs[-1].replace(cost='not in there')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        obj_pairs, filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert logger.warning.called
    cost_warns = 0
    for call in logger.warning.call_args_list:
        if 'Cannot calculate cost metrics for ' in call[0][0]:
            cost_warns += 1
    assert cost_warns == 1
    assert not logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert all(isinstance(vr, datamodel.ValidationResult)
                   for vr in proc_fxobs.validation_results)
        assert all(isinstance(pr, datamodel.PreprocessingResult)
                   for pr in proc_fxobs.preprocessing_results)
        assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)
    assert processed_fxobs_list[-1].cost is None


def test_process_forecast_observations_resample_fail(
        report_objects, quality_filter, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    obs_ser = pd.Series(np.arange(8),
                        index=pd.date_range(start='2019-04-01T00:00:00',
                                            periods=8,
                                            freq='15min',
                                            tz='MST',
                                            name='timestamp'))
    obs_df = obs_ser.to_frame('value')
    obs_df['quality_flag'] = OK
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = OK
    data = {
        observation: obs_df,
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    mocker.patch(
        'solarforecastarbiter.metrics.preprocessing.resample_and_align',
        side_effect=ValueError)
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == 0
    assert logger.error.called


def test_process_forecast_observations_same_name(
        report_objects, quality_filter, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    obs_ser = pd.Series(np.arange(8),
                        index=pd.date_range(start='2019-04-01T00:00:00',
                                            periods=8,
                                            freq='15min',
                                            tz='MST',
                                            name='timestamp'))
    obs_df = obs_ser.to_frame('value')
    obs_df['quality_flag'] = OK
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = OK
    data = {
        observation: obs_df,
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter]
    fxobs = report.report_parameters.object_pairs
    fxobs = list(fxobs) + [fxobs[0], fxobs[0]]
    processed_fxobs_list = preprocessing.process_forecast_observations(
        fxobs,
        filters,
        report.report_parameters.missing_forecast,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        fxobs)
    assert len(set(pfxobs.name for pfxobs in processed_fxobs_list)) == len(
        fxobs)


@pytest.mark.parametrize("method", ['drop', 'forward', '-1', '99.9'])
def test_process_forecast_observations_missing_forecast_types(
        report_objects, quality_filter, mocker, method):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = report_objects  # NOQA
    forecast_ref = report.report_parameters.object_pairs[1].reference_forecast
    obs_ser = pd.Series(np.arange(8),
                        index=pd.date_range(start='2019-04-01T00:00:00',
                                            periods=8,
                                            freq='15min',
                                            tz='MST',
                                            name='timestamp'))
    obs_df = obs_ser.to_frame('value')
    obs_df['quality_flag'] = OK
    agg_df = THREE_HOUR_SERIES.to_frame('value')
    agg_df['quality_flag'] = OK
    data = {
        observation: obs_df,
        forecast_0: THREE_HOUR_SERIES,
        forecast_1: THREE_HOUR_SERIES,
        forecast_ref: THREE_HOUR_SERIES,
        forecast_agg: THREE_HOUR_SERIES,
        aggregate: agg_df
    }
    filters = [quality_filter]
    logger = mocker.patch('solarforecastarbiter.metrics.preprocessing.logger')
    processed_fxobs_list = preprocessing.process_forecast_observations(
        report.report_parameters.object_pairs,
        filters,
        method,
        report.report_parameters.start,
        report.report_parameters.end,
        data, 'MST',
        costs=report.report_parameters.costs)
    assert len(processed_fxobs_list) == len(
        report.report_parameters.object_pairs)
    assert not logger.error.called
    for proc_fxobs in processed_fxobs_list:
        assert isinstance(proc_fxobs, datamodel.ProcessedForecastObservation)
        assert all(isinstance(vr, datamodel.ValidationResult)
                   for vr in proc_fxobs.validation_results)
        assert all(isinstance(pr, datamodel.PreprocessingResult)
                   for pr in proc_fxobs.preprocessing_results)
        assert isinstance(proc_fxobs.forecast_values, pd.Series)
        assert isinstance(proc_fxobs.observation_values, pd.Series)
        pd.testing.assert_index_equal(proc_fxobs.forecast_values.index,
                                      proc_fxobs.observation_values.index)


def test_name_pfxobs_recursion_limit():
    name = 'whoami'
    cn = [name]
    cn += [f'{name}-{i:02d}' for i in range(129)]
    assert preprocessing._name_pfxobs(cn, name) == f'{name}-99'


@pytest.mark.parametrize("obs_index,fx_index,expected_dt", [
    (THREE_HOURS, THREE_HOURS, THREE_HOURS),
    (THIRTEEN_10MIN, THIRTEEN_10MIN, THIRTEEN_10MIN),

    # mismatch interval_length
    pytest.param(THREE_HOURS, THIRTEEN_10MIN, [],
                 marks=pytest.mark.xfail(strict=True, type=ValueError)),
    pytest.param(THIRTEEN_10MIN, THREE_HOURS, [],
                 marks=pytest.mark.xfail(strict=True, type=ValueError)),
])
def test_resample_and_align_event(single_event_forecast_observation,
                                  obs_index, fx_index, expected_dt):

    obs_values = np.random.randint(0, 2, size=len(obs_index), dtype=bool)
    fx_values = np.random.randint(0, 2, size=len(fx_index), dtype=bool)
    obs_series = pd.Series(obs_values, index=obs_index)
    fx_series = pd.Series(fx_values, index=fx_index)

    fxobs = single_event_forecast_observation

    # Use local tz
    local_tz = f"Etc/GMT{int(time.timezone/3600):+d}"

    fx_vals, obs_vals, ref_vals, results = preprocessing.resample_and_align(
        fxobs, fx_series, obs_series, None, local_tz
    )

    assert isinstance(results, dict)

    # Localize datetimeindex
    expected_dt = expected_dt.tz_convert(local_tz)

    pd.testing.assert_index_equal(fx_vals.index,
                                  obs_vals.index,
                                  check_categorical=False)
    pd.testing.assert_index_equal(obs_vals.index,
                                  expected_dt,
                                  check_categorical=False)


@pytest.mark.parametrize("fx_interval_length, obs_interval_length", [
    (60, 60),
    (5, 5),
    pytest.param(30, 10,
                 marks=pytest.mark.xfail(strict=True, type=ValueError)),
    pytest.param(10, 15,
                 marks=pytest.mark.xfail(strict=True, type=ValueError)),
])
def test__resample_event_obs(single_site, single_event_forecast_text,
                             single_event_observation_text,
                             fx_interval_length, obs_interval_length):

    fx_dict = json.loads(single_event_forecast_text)
    fx_dict['site'] = single_site
    fx_dict.update({"interval_length": fx_interval_length})
    fx = datamodel.EventForecast.from_dict(fx_dict)

    obs_dict = json.loads(single_event_observation_text)
    obs_dict['site'] = single_site
    obs_dict.update({"interval_length": obs_interval_length})
    obs = datamodel.Observation.from_dict(obs_dict)

    freq = pd.Timedelta(f"{obs_interval_length}min")
    index = pd.date_range(start="20200301T00Z", end="20200304T00Z", freq=freq)
    obs_series = pd.Series(np.random.randint(0, 2, len(index), dtype=bool),
                           index=index)

    obs_resampled = preprocessing._resample_event_obs(fx, obs, obs_series)
    pd.testing.assert_index_equal(obs_series.index, obs_resampled.index,
                                  check_categorical=False)


@pytest.mark.parametrize("data", [
    [True, True, False],
    [1, 1, 0],
    [1.0, 1.0, 0.0],
    pytest.param([0, 1, 2],
                 marks=pytest.mark.xfail(strict=True, type=TypeError)),
    pytest.param([0.0, 1.0, 2.0],
                 marks=pytest.mark.xfail(strict=True, type=TypeError)),
])
def test__validate_event_dtype(data):
    ser = pd.Series(data)
    print(ser.head())
    ser_conv = preprocessing._validate_event_dtype(ser)

    pd.testing.assert_index_equal(ser.index, ser_conv.index,
                                  check_categorical=False)
    assert ser_conv.dtype == bool


@pytest.mark.parametrize("method", [
    ('drop'), ('forward'), ('1')
])
def test_apply_fill(method):
    n = 10
    dt_range = pd.date_range(start='2020-01-01T00:00',
                             periods=n,
                             freq='30min',
                             name='timestamp')
    data = pd.Series([1]*n, index=dt_range)
    i_rand = (np.append(0., np.round(np.random.rand(n-1))) == 1)
    data[i_rand] = np.nan

    result, count = preprocessing.apply_fill(data, method,
                                             dt_range[0], dt_range[-1])
    assert isinstance(result, pd.Series)
    assert isinstance(result.index, pd.DatetimeIndex)
    if method == 'drop':
        assert result.sum() == n - sum(i_rand)
        assert count == i_rand.sum()
    else:
        assert result.sum() == n
        assert count == i_rand.sum()


@pytest.mark.parametrize("method,exp,exp_count", [
    ('drop',
     pd.Series([1], index=pd.date_range(start='2020-01-01T02:00',
                                        periods=1,
                                        freq='2h',
                                        name='timestamp'),
               dtype=np.float64),
     0),
    ('forward',
     pd.Series([0, 1, 1, 1], index=pd.date_range(start='2020-01-01T00:00',
                                                 periods=4,
                                                 freq='2h',
                                                 name='timestamp'),
               dtype=np.float64),
     3),
    ('1',
     pd.Series([1, 1, 1, 1], index=pd.date_range(start='2020-01-01T00:00',
                                                 periods=4,
                                                 freq='2h',
                                                 name='timestamp'),
               dtype=np.float64),
     3),
])
def test_apply_fill_one_value(method, exp, exp_count):
    # Single value
    start = pd.to_datetime('2020-01-01T00:00')
    end = pd.to_datetime('2020-01-01T6:00')
    data = pd.Series([1], index=pd.date_range(start='2020-01-01T02:00',
                                              periods=1,
                                              freq='2h',
                                              name='timestamp'),
                     dtype=np.float64)
    result, count = preprocessing.apply_fill(data, method,
                                             start=start, end=end)
    pd.testing.assert_series_equal(result, exp)
    assert count == exp_count


@pytest.mark.parametrize("method,exp,exp_count", [
    ('drop',
     pd.Series([], index=pd.DatetimeIndex([], name='timestamp'),
               dtype=np.float64),
     0),
    ('forward',
     pd.Series([0, 0], index=pd.date_range(start='2020-01-01T00:00',
                                           periods=2,
                                           freq='6h',
                                           name='timestamp'),
               dtype=np.float64),
     2),
    ('1',
     pd.Series([1, 1], index=pd.date_range(start='2020-01-01T00:00',
                                           periods=2,
                                           freq='6h',
                                           name='timestamp'),
               dtype=np.float64),
     2),
])
def test_apply_fill_no_values(method, exp, exp_count):
    # Single value
    start = pd.to_datetime('2020-01-01T00:00')
    end = pd.to_datetime('2020-01-01T06:00')
    # No values
    data = pd.Series([], index=pd.DatetimeIndex([], name='timestamp'),
                     dtype=np.float64)
    result, count = preprocessing.apply_fill(data, method,
                                             start=start, end=end)
    pd.testing.assert_series_equal(result, exp)
    assert count == exp_count


def test_apply_fill_unsupported():
    n = 10
    dt_range = pd.date_range(start='2020-01-01T00:00',
                             periods=n,
                             freq='30min',
                             name='timestamp')
    with pytest.raises(ValueError):
        preprocessing.apply_fill(pd.Series(range(n), index=dt_range),
                                 'error',
                                 dt_range[0], dt_range[-1])


@pytest.mark.parametrize("input,exp,n_pre,n_post", [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0, 0),
    ([1, np.nan, 3, np.nan, 5], [1, 3, 5], 0, 0),
    ([np.nan, 2, 3, 4, np.nan], [2, 3, 4], 0, 0),
    ([1, np.nan, np.nan, np.nan, 5], [1, 5], 0, 0),
    ([3, 4, 5], [3, 4, 5], 2, 0),
    ([1, 2], [1, 2], 0, 3),
    ([np.nan]*5, [], 0, 0)
])
def test_apply_fill_drop(input, exp, n_pre, n_post):
    # convert list to series
    n = n_pre + len(input) + n_post
    dt_range = pd.date_range(start='2020-01-01T00:00',
                             periods=n,
                             freq='30min',
                             name='timestamp')
    data = pd.Series(input, index=dt_range[n_pre:n-n_post])
    expected = data.loc[data.isin(exp)]

    # as a Series
    result, count = preprocessing.apply_fill(data, 'drop',
                                             dt_range[0], dt_range[-1])
    pd.testing.assert_series_equal(result, expected,
                                   check_exact=True)
    assert count == len(input) - len(exp)

    # as a DataFrame with 3 columns
    df_data = pd.DataFrame({'1': input,
                            '2': input,
                            '3': input}, index=dt_range[n_pre:n-n_post])
    df_result, count = preprocessing.apply_fill(df_data, 'drop',
                                                dt_range[0], dt_range[-1])
    df_expected = pd.DataFrame({'1': exp,
                                '2': exp,
                                '3': exp}, index=expected.index)
    df_expected = df_expected.astype(data.dtype)
    pd.testing.assert_frame_equal(df_result, df_expected,
                                  dt_range[0], dt_range[-1])
    assert count == (len(input) - len(exp)) * 3


@pytest.mark.parametrize("input,exp,n_pre,n_post", [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0, 0),
    ([1, np.nan, 3, np.nan, 5], [1, 1, 3, 3, 5], 0, 0),
    ([np.nan, 2, 3, 4, np.nan], [0, 2, 3, 4, 4], 0, 0),
    ([1, np.nan, np.nan, np.nan, 5], [1, 1, 1, 1, 5], 0, 0),
    ([3, 4, 5], [0, 0, 3, 4, 5], 2, 0),
    ([1, 2], [1, 2, 2, 2, 2], 0, 3),
    ([np.nan]*5, [0, 0, 0, 0, 0], 0, 0)
])
def test_apply_fill_forward(input, exp, n_pre, n_post):
    # convert list to series
    n = n_pre + len(input) + n_post
    dt_range = pd.date_range(start='2020-01-01T00:00',
                             periods=n,
                             freq='30min',
                             name='timestamp')
    data = pd.Series(input, index=dt_range[n_pre:n-n_post])
    expected = pd.Series(exp, index=dt_range)
    expected = expected.astype(data.dtype)

    # as a Series
    result, count = preprocessing.apply_fill(data, 'forward',
                                             dt_range[0], dt_range[-1])
    pd.testing.assert_series_equal(result, expected,
                                   check_exact=True)
    assert count == data.isna().sum() + n_pre + n_post

    # as a DataFrame with 3 columns
    df_data = pd.DataFrame({'1': input,
                            '2': input,
                            '3': input}, index=dt_range[n_pre:n-n_post])
    df_result, count = preprocessing.apply_fill(df_data, 'forward',
                                                dt_range[0], dt_range[-1])
    df_expected = pd.DataFrame({'1': exp,
                                '2': exp,
                                '3': exp}, index=dt_range)
    df_expected = df_expected.astype(data.dtype)
    pd.testing.assert_frame_equal(df_result, df_expected,
                                  dt_range[0], dt_range[-1])
    assert count == (data.isna().sum() + n_pre + n_post)*3


@pytest.mark.parametrize("input,exp,fvalue,n_pre,n_post", [
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 100, 0, 0),
    ([1, np.nan, 3, np.nan, 5], [1, 100, 3, 100, 5], 100, 0, 0),
    ([np.nan, 2, 3, 4, np.nan], [100, 2, 3, 4, 100], 100, 0, 0),
    ([1, np.nan, np.nan, np.nan, 5], [1, 100, 100, 100, 5], 100, 0, 0),
    ([3, 4, 5], [100, 100, 3, 4, 5], 100, 2, 0),
    ([1, 2], [1, 2, 100, 100, 100], 100, 0, 3),
    ([np.nan]*5, [100]*5, 100, 0, 0)
])
def test_apply_fill_constant(input, exp, fvalue, n_pre, n_post):
    # convert list to series
    n = n_pre + len(input) + n_post
    dt_range = pd.date_range(start='2020-01-01T00:00',
                             periods=n,
                             freq='30min',
                             name='timestamp')
    data = pd.Series(input, index=dt_range[n_pre:n-n_post])
    expected = pd.Series(exp, index=dt_range)
    expected = expected.astype(data.dtype)

    # as a Series
    result, count = preprocessing.apply_fill(data, fvalue,
                                             dt_range[0], dt_range[-1])
    pd.testing.assert_series_equal(result, expected,
                                   check_exact=True)
    assert count == data.isna().sum() + n_pre + n_post

    # as a DataFrame with 3 columns
    df_data = pd.DataFrame({'1': input,
                            '2': input,
                            '3': input}, index=dt_range[n_pre:n-n_post])
    df_result, count = preprocessing.apply_fill(df_data, fvalue,
                                                dt_range[0], dt_range[-1])
    df_expected = pd.DataFrame({'1': exp,
                                '2': exp,
                                '3': exp}, index=dt_range)
    df_expected = df_expected.astype(data.dtype)
    pd.testing.assert_frame_equal(df_result, df_expected,
                                  dt_range[0], dt_range[-1])
    assert count == (data.isna().sum() + n_pre + n_post)*3


@pytest.mark.parametrize("data", [
    (pd.DataFrame(
        {'1': [1, 2, 3, 4, 5],
         '2': [10, np.nan, 30, np.nan, 50],
         '3': [np.nan, 200, 300, 400, np.nan]},
        index=pd.date_range('2020-01-01T00:00', periods=5, freq='1h'),
        dtype=np.float64)),
])
@pytest.mark.parametrize("method,exp,exp_count", [
    ('drop', pd.DataFrame(
        {'1': [3],
         '2': [30],
         '3': [300]},
        index=[pd.Timestamp('2020-01-01T02:00')],
        dtype=np.float64),
        4*3),
    ('forward', pd.DataFrame(
        {'1': [1, 2, 3, 4, 5],
         '2': [10, 10, 30, 30, 50],
         '3': [0, 200, 300, 400, 400]},
        index=pd.date_range('2020-01-01T00:00', periods=5, freq='1h'),
        dtype=np.float64),
        4),
    ('-1', pd.DataFrame(
        {'1': [1, 2, 3, 4, 5],
         '2': [10, -1, 30, -1, 50],
         '3': [-1, 200, 300, 400, -1]},
        index=pd.date_range('2020-01-01T00:00', periods=5, freq='1h'),
        dtype=np.float64),
        4),
])
def test_apply_fill_unstratified_dataframe(data, method, exp, exp_count):
    result, count = preprocessing.apply_fill(data, method,
                                             data.index[0], data.index[-1])
    pd.testing.assert_frame_equal(result, exp)
    assert count == exp_count
