import numpy as np
import pandas as pd
import pytest
import itertools


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (calculator,
                                          deterministic)


@pytest.fixture()
def create_processed_fxobs(create_datetime_index):
    def _create_processed_fxobs(fxobs, fx_values, obs_values):
        return datamodel.ProcessedForecastObservation(
            fxobs,
            fxobs.forecast.interval_value_type,
            fxobs.forecast.interval_length,
            fxobs.forecast.interval_label,
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
    (calculator.AVAILABLE_CATEGORIES, deterministic.__all__)
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

    # All
    all_result = calculator.calculate_metrics(proc_fx_obs,
                                              categories, metrics,
                                              ref_pair=ref_fx_obs,
                                              normalizer=1.0)

    assert isinstance(all_result, list)
    assert len(all_result) == len(proc_fx_obs)

    # 1 value no options
    if len(metrics) > 0:
        with pytest.raises(AttributeError):
            calculator.calculate_metrics([proc_fx_obs[0]],
                                         categories, metrics)
        # drop metrics requiring reference forecast
        list(map(metrics.remove, deterministic._REQ_REF_FX))
    one_result = calculator.calculate_metrics([proc_fx_obs[0]],
                                              categories, metrics)

    assert isinstance(one_result, list)
    assert len(one_result) == 1


# Suppress RuntimeWarnings b/c in some metrics will divide by zero or
# don't handle single values well
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(calculator.AVAILABLE_CATEGORIES, 1)),
    calculator.AVAILABLE_CATEGORIES[0:1],
    calculator.AVAILABLE_CATEGORIES[0:2],
    calculator.AVAILABLE_CATEGORIES[1:],
    calculator.AVAILABLE_CATEGORIES
])
@pytest.mark.parametrize('metrics', [
    [],
    *list(itertools.combinations(deterministic._MAP.keys(), 1)),
    deterministic.__all__[0:1],
    deterministic.__all__[0:2],
    deterministic.__all__[1:],
    deterministic.__all__
])
@pytest.mark.parametrize('values,ref_values,normalizer', [
    ((np.random.randn(0), np.random.randn(0)),
     (np.random.randn(0), np.random.randn(0)),
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

    # Check if reference forecast is required
    if (ref_values is None and any(m in deterministic._REQ_REF_FX
                                   for m in metrics)):
        with pytest.raises(AttributeError):
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
            assert len(result) == 0
        else:
            assert sorted(result.keys()) == sorted(categories)
            for cat, val_cat in result.items():
                if cat == 'total':
                    assert sorted(val_cat.keys()) == sorted(metrics)
                    # metrics
                    for metric, val_metric in val_cat.items():
                        assert isinstance(val_metric, float)
                else:
                    # category groups
                    for cat_group, metric_group in val_cat.items():
                        assert sorted(metric_group.keys()) == sorted(metrics)

                        # metrics
                        for metric, val_metric in metric_group.items():
                            assert isinstance(val_metric, float)

                        fx_values = pair.forecast_values
                        # has expected groupings
                        if cat == 'month':
                            grps = fx_values.groupby(
                                fx_values.index.month).groups
                        elif cat == 'hour':
                            grps = fx_values.groupby(
                                fx_values.index.hour).groups
                        elif cat == 'year':
                            grps = fx_values.groupby(
                                fx_values.index.year).groups
                        elif cat == 'day':
                            grps = fx_values.groupby(
                                fx_values.index.day).groups
                        elif cat == 'date':
                            grps = fx_values.groupby(
                                fx_values.index.date).groups
                        elif cat == 'weekday':
                            grps = fx_values.groupby(
                                fx_values.index.weekday).groups
                        assert sorted(grps) == sorted(val_cat.keys())


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
