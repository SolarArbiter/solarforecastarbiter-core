import numpy as np
import pandas as pd
import pytest
import itertools
import calendar
import datetime

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (calculator, deterministic)


DETERMINISTIC_METRICS = list(deterministic._MAP.keys())
LIST_OF_CATEGORIES = list(datamodel.ALLOWED_CATEGORIES.keys())


@pytest.fixture()
def create_processed_fxobs(create_datetime_index):
    def _create_processed_fxobs(fxobs, fx_values, obs_values,
                                interval_label=None):

        if not interval_label:
            interval_label = fxobs.forecast.interval_label

        return datamodel.ProcessedForecastObservation(
            fxobs,
            fxobs.forecast.interval_value_type,
            fxobs.forecast.interval_length,
            interval_label,
            forecast_values=pd.Series(
                fx_values, index=create_datetime_index(len(fx_values))),
            observation_values=pd.Series(
                obs_values, index=create_datetime_index(len(obs_values)))
        )

    return _create_processed_fxobs


@pytest.fixture()
def create_datetime_index():
    def _create_datetime_index(n_periods):
        return pd.date_range(start='20190801', periods=n_periods, freq='1h',
                             tz='MST', name='timestamp')

    return _create_datetime_index

# Suppress RuntimeWarnings b/c in some metrics will divide by zero or
# don't handle single values well
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories,metrics', [
    ([], []),
    (LIST_OF_CATEGORIES, DETERMINISTIC_METRICS)
])
def test_calculate_metrics(categories, metrics,
                           create_processed_fxobs,
                           many_forecast_observation):
    proc_fx_obs = []

    for fx_obs in many_forecast_observation:
        proc_fx_obs.append(
            create_processed_fxobs(fx_obs,
                                   np.random.randn(10)+10,
                                   np.random.randn(10)+10)
        )

    ref_fx_obs = create_processed_fxobs(many_forecast_observation[0],
                                        np.random.randn(10)+10,
                                        np.random.randn(10)+10)

    # Error - no metrics
    if len(metrics) == 0:
        with pytest.raises(RuntimeError):
            calculator.calculate_metrics(proc_fx_obs,
                                         categories, metrics,
                                         ref_pair=ref_fx_obs,
                                         normalizer=1.0)
        return

    # All options selected
    all_result = calculator.calculate_metrics(proc_fx_obs,
                                              categories, metrics,
                                              ref_pair=ref_fx_obs,
                                              normalizer=1.0)

    assert isinstance(all_result, list)
    assert len(all_result) == len(proc_fx_obs)

    # One processed pair missing reference forecast but required by metrics
    if any(m for m in metrics if m in deterministic._REQ_REF_FX):
        with pytest.raises(RuntimeError):
            calculator.calculate_metrics([proc_fx_obs[0]],
                                         categories, metrics)
        # drop metrics requiring reference forecast
        list(map(metrics.remove, deterministic._REQ_REF_FX))

    # One processed pair (no reference forecast)
    one_result = calculator.calculate_metrics([proc_fx_obs[0]],
                                              categories, metrics)

    assert isinstance(one_result, list)
    assert len(one_result) == 1


@pytest.mark.parametrize('categories,metrics', [
    ([], []),
    (LIST_OF_CATEGORIES, DETERMINISTIC_METRICS)
])
def test_calculate_metrics_with_probablistic(categories, metrics,
                                             create_processed_fxobs,
                                             single_observation,
                                             prob_forecasts):
    prfxobs = datamodel.ForecastObservation(prob_forecasts, single_observation)
    proc_prfx_obs = create_processed_fxobs(prfxobs,
                                           np.random.randn(10)+10,
                                           np.random.randn(10)+10)

    # Error - ProbabilisticForecast not yet supported
    with pytest.raises(NotImplementedError):
        calculator.calculate_metrics([proc_prfx_obs],
                                     categories, metrics)


# Suppress RuntimeWarnings b/c in some metrics will divide by zero or
# don't handle single values well
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES[1:],
    LIST_OF_CATEGORIES
])
@pytest.mark.parametrize('metrics', [
    [],
    *list(itertools.combinations(DETERMINISTIC_METRICS, 1)),
    DETERMINISTIC_METRICS[0:1],
    DETERMINISTIC_METRICS[0:2],
    DETERMINISTIC_METRICS[1:],
    DETERMINISTIC_METRICS
])
@pytest.mark.parametrize('values,ref_values,normalizer', [
    ((np.random.randn(0), np.random.randn(0)),
     (np.random.randn(0), np.random.randn(0)),
     1.0),
    ((np.random.randn(10)+10, np.random.randn(0)),
     (np.random.randn(10)+10, np.random.randn(0)),
     1.0),
    ((np.random.randn(10)+10, np.random.randn(10)+10),
     (np.random.randn(10)+10, np.random.randn(10)+10),
     1.0),
    ((np.random.randn(10)+10, np.random.randn(10)+10),
     (np.random.randn(10)+10, np.random.randn(10)+10),
     None),
    ((np.random.randn(10)+10, np.random.randn(10)+10),
     None,
     None),
])
def test_calculate_deterministic_metrics(values, categories, metrics,
                                         ref_values, normalizer,
                                         single_forecast_observation,
                                         create_processed_fxobs):
    pair = create_processed_fxobs(single_forecast_observation, *values)

    kws = {}

    if ref_values is not None:
        kws['ref_fx_obs'] = create_processed_fxobs(single_forecast_observation,
                                                   *ref_values)

    if normalizer is not None:
        kws['normalizer'] = normalizer

    # Check if timeseries and metrics provided
    if values[0].size == 0 or values[1].size == 0 or len(metrics) == 0:
        with pytest.raises(RuntimeError):
            calculator.calculate_deterministic_metrics(
                pair, categories, metrics, **kws)

    # Check if reference forecast is required
    elif (ref_values is None and any(m in deterministic._REQ_REF_FX
                                     for m in metrics)):
        # error if no reference forecast given
        with pytest.raises(RuntimeError):
            calculator.calculate_deterministic_metrics(
                pair, categories, metrics, **kws)

        # error if reference forecast given but no reference forecast data
        with pytest.raises(RuntimeError):
            kws['ref_fx_obs'] = create_processed_fxobs(
                single_forecast_observation, np.array([]), np.array([]))
            calculator.calculate_deterministic_metrics(
                pair, categories, metrics, **kws)

    else:
        result = calculator.calculate_deterministic_metrics(
            pair, categories, metrics, **kws)

        # Check results
        assert isinstance(result, dict)
        if (len(metrics) == 0 or len(categories) == 0 or
                len(pair.forecast_values) == 0):
            # Empty results
            assert len(result) == 1
            assert result['name'] == pair.original.forecast.name
        else:
            assert sorted(result.keys()) == sorted(list(categories)+['name'])
            for cat, cat_values in result.items():
                if cat == 'name':
                    assert cat_values == pair.original.forecast.name
                elif cat == 'total':
                    assert sorted(cat_values.keys()) == sorted(metrics)
                    # check metric values
                    for metric, met_value in cat_values.items():
                        assert isinstance(met_value, float)
                else:
                    # check metric values and category values
                    for metric, met_values in cat_values.items():

                        fx_values = pair.forecast_values

                        # has expected groupings
                        if cat == 'month':
                            grps = fx_values.groupby(
                                fx_values.index.month).groups
                            grps = [calendar.month_abbr[g] for g in grps]
                        elif cat == 'hour':
                            grps = fx_values.groupby(
                                fx_values.index.hour).groups
                        elif cat == 'year':
                            grps = fx_values.groupby(
                                fx_values.index.year).groups
                        elif cat == 'date':
                            grps = fx_values.groupby(
                                fx_values.index.date).groups
                        elif cat == 'weekday':
                            grps = fx_values.groupby(
                                fx_values.index.weekday).groups
                            grps = [calendar.day_abbr[g] for g in grps]
                        assert sorted(grps) == sorted(met_values.index)

                        # has valid values
                        assert np.issubdtype(met_values.values.dtype,
                                             np.number)


@pytest.mark.parametrize('metric,fx,obs,ref_fx,norm,expect', [
    ('mae', [], [], None, None, np.NaN),
    ('mae', [1, 1, 1], [0, 1, -1], None, None, 1.0),
    ('mbe', [1, 1, 1], [0, 1, -1], None, None, 1.0),
    ('rmse', [1, 0, 1], [0, -1, 2], None, None, 1.0),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, None, None),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, 2.0, 100/2),
    ('mape', [2, 3, 1], [4, 2, 2], None, None, 50.0),
    ('s', [1, 0, 1], [0, -1, 2], None, None, None),
    ('s', [1, 0, 1], [0, -1, 2], [2, 1, 0], None, 0.5),
    ('r', [3, 2, 1], [1, 2, 3], None, None, -1.0),
    ('r^2', [3, 2, 1], [1, 2, 3], None, None, -3.0),
    ('crmse', [1, 1, 1], [0, 1, 2], None, None, np.sqrt(2/3))
])
def test_apply_deterministic_metric_func(metric, fx, obs, ref_fx, norm, expect,
                                         create_datetime_index):
    fx_series = pd.Series(fx, index=create_datetime_index(len(fx)))
    obs_series = pd.Series(obs, index=create_datetime_index(len(obs)))
    # Check require reference forecast kwarg
    if metric in ['s']:
        if ref_fx is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_deterministic_metric_func(
                    metric, fx_series, obs_series)
        else:
            ref_fx_series = pd.Series(ref_fx,
                                      index=create_datetime_index(len(ref_fx)))
            metric_value = calculator._apply_deterministic_metric_func(
                metric, fx_series, obs_series, ref_fx=ref_fx_series)
            np.testing.assert_approx_equal(metric_value, expect)

    # Check requires normalization kwarg
    elif metric in ['nrmse']:
        if norm is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_deterministic_metric_func(
                    metric, fx_series, obs_series)
        else:
            metric_value = calculator._apply_deterministic_metric_func(
                metric, fx_series, obs_series, normalizer=norm)
            np.testing.assert_approx_equal(metric_value, expect)

    # Does not require kwarg
    else:
        metric_value = calculator._apply_deterministic_metric_func(
            metric, fx_series, obs_series)
        if np.isnan(expect):
            assert np.isnan(metric_value)
        else:
            np.testing.assert_approx_equal(metric_value, expect)


def test_apply_deterministic_bad_metric_func():
    with pytest.raises(KeyError):
        calculator._apply_deterministic_metric_func('BAD METRIC',
                                                    pd.Series(),
                                                    pd.Series())


@pytest.mark.parametrize('index,interval_label,category,result', [
    # category: hour
    (
        pd.DatetimeIndex(
            ['20191210T1300Z', '20191210T1330Z', '20191210T1400Z']
        ),
        'ending',
        'hour',
        pd.Series(data=[-1.0, 0.5], index=[12, 13]),
    ),
    (
        pd.DatetimeIndex(
            ['20191210T1300Z', '20191210T1330Z', '20191210T1400Z']
        ),
        'beginning',
        'hour',
        pd.Series(data=[-0.5, 1.0], index=[13, 14]),
    ),

    # category: month
    (
        pd.DatetimeIndex(
            ['20191130T2330Z', '20191201T0000Z', '20191201T0030Z']
        ),
        'ending',
        'month',
        pd.Series(data=[-0.5, 1.0], index=['Nov', 'Dec']),
    ),
    (
        pd.DatetimeIndex(
            ['20191130T2330Z', '20191201T0000Z', '20191201T0030Z']
        ),
        'beginning',
        'month',
        pd.Series(data=[-1.0, 0.5], index=['Nov', 'Dec']),
    ),

    # category: year
    (
        pd.DatetimeIndex(
            ['20191231T2330Z', '20200101T0000Z', '20200101T0030Z']
        ),
        'ending',
        'year',
        pd.Series(data=[-0.5, 1.0], index=[2019, 2020]),
    ),
    (
        pd.DatetimeIndex(
            ['20191231T2330Z', '20200101T0000Z', '20200101T0030Z']
        ),
        'beginning',
        'year',
        pd.Series(data=[-1.0, 0.5], index=[2019, 2020]),
    ),
])
def test_interval_label(index, interval_label, category, result, create_processed_fxobs):

    # Custom metadata to keep all timestamps in UTC for tests
    site = datamodel.Site(
        name='Albuquerque Baseline',
        latitude=35.05,
        longitude=-106.54,
        elevation=1657.0,
        timezone='UTC',
        provider='Sandia'
    )

    fx_series = pd.Series([0, 1, 2], index=index)
    obs_series = pd.Series([1, 1, 1], index=index)
    fxobs = datamodel.ForecastObservation(
        observation=datamodel.Observation(
            site=site, name='dummy obs', variable='ghi',
            interval_value_type='instantaneous', uncertainty=1,
            interval_length=pd.Timedelta(obs_series.index.freq),
            interval_label=interval_label
        ),
        forecast=datamodel.Forecast(
            site=site, name='dummy fx', variable='ghi',
            interval_value_type='instantaneous',
            interval_length=pd.Timedelta(fx_series.index.freq),
            interval_label=interval_label,
            issue_time_of_day=datetime.time(hour=5),
            lead_time_to_start=pd.Timedelta('1h'),
            run_length=pd.Timedelta('1h')
        )
    )

    proc_fx_obs = datamodel.ProcessedForecastObservation(
        fxobs,
        fxobs.forecast.interval_value_type,
        fxobs.forecast.interval_length,
        fxobs.forecast.interval_label,
        forecast_values=fx_series,
        observation_values=obs_series,
    )

    res = calculator.calculate_deterministic_metrics(proc_fx_obs, [category],
                                                     ['mbe'])
    pd.testing.assert_series_equal(res[category]['mbe'], result)


@pytest.mark.parametrize('label_fx,label_ref', [
    ("beginning", "beginning"),
    ("ending", "ending"),
    pytest.param("beginning", "ending",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    pytest.param("ending", "beginning",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
])
def test_interval_label_match(site_metadata, label_fx, label_ref,
                              create_processed_fxobs,
                              many_forecast_observation):

    categories = LIST_OF_CATEGORIES
    metrics = list(deterministic._MAP.keys())

    proc_fx_obs = []
    for fx_obs in many_forecast_observation:
        proc_fx_obs.append(
            create_processed_fxobs(
                fx_obs,
                np.random.randn(10) + 10,
                np.random.randn(10) + 10,
                interval_label=label_fx,
            )
        )

    proc_ref_obs = create_processed_fxobs(
        many_forecast_observation[0],
        np.random.randn(10) + 10,
        np.random.randn(10) + 10,
        interval_label=label_ref,
    )

    all_results = calculator.calculate_metrics(
        proc_fx_obs,
        categories,
        metrics,
        ref_pair=proc_ref_obs,
    )

    assert isinstance(all_results, list)
    assert len(all_results) == len(proc_fx_obs)
