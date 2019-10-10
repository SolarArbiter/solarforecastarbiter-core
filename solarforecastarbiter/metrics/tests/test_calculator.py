import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import calculator, deterministic


_LARGE = 8760


def create_processed_fx_obs(single_forecast_observation,
                            fx_values, obs_values):
    interval_length = pd.Timedelta(fx_values.index.freq)
    assert fx_values.index.equals(obs_values.index)
    # Build ProcessedForecastObservation
    orig = single_forecast_observation
    proc_fx_obs = datamodel.ProcessedForecastObservation(
        original=orig,
        interval_value_type=orig.forecast.interval_value_type,
        interval_length=interval_length,
        interval_label=orig.forecast.interval_label,
        forecast_values=fx_values,
        observation_values=obs_values
    )
    return proc_fx_obs


@pytest.mark.parametrize('fx_values,obs_values,ref_fx_values', [
    (pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
     pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
     None),
    (pd.Series([1.0, 2.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST'), name='value'),
     pd.Series([1.0, 3.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST'), name='value'),
     None),
    (pd.Series([1.0, 2.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST'), name='value'),
     pd.Series([1.0, 3.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST'), name='value'),
     pd.Series([2.0, 3.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST'), name='value')),
    (pd.Series(np.random.normal(1, .5, _LARGE), index=pd.date_range(
        start='20190801', periods=_LARGE, freq='1h', tz='MST'), name='value'),
     pd.Series(np.random.normal(1, .5, _LARGE), index=pd.date_range(
        start='20190801', periods=_LARGE, freq='1h', tz='MST'), name='value'),
     None),
    (pd.Series(np.random.normal(1, .5, _LARGE), index=pd.date_range(
        start='20190801', periods=_LARGE, freq='1h', tz='MST'), name='value'),
     pd.Series(np.random.normal(1, .5, _LARGE), index=pd.date_range(
        start='20190801', periods=_LARGE, freq='1h', tz='MST'), name='value'),
     pd.Series(np.random.normal(1, .5, _LARGE), index=pd.date_range(
        start='20190801', periods=_LARGE, freq='1h', tz='MST'), name='value'))
])
@pytest.mark.parametrize('categories,metrics', [
    (['total'], ['mae']),
    (['total', 'month', 'hour'], ['mae', 'rmse', 'mbe']),
    (calculator.AVAILABLE_CATEGORIES, deterministic._MAP.keys())
])
def test_calculate_deterministic_metrics(single_forecast_observation,
                                         fx_values, obs_values, ref_fx_values,
                                         categories, metrics):
    # Create ProcessedForecastObservation
    proc_fx_obs = create_processed_fx_obs(
        single_forecast_observation, fx_values, obs_values)

    if ref_fx_values is not None:
        proc_ref_fx_obs = create_processed_fx_obs(
            single_forecast_observation, ref_fx_values, obs_values)
    else:
        proc_ref_fx_obs = None

    # Calculations
    needs_ref_fx = any(m in deterministic._REQ_REF_FX for m in metrics)

    # AssertionError if needs ref forecast and none provided
    if ref_fx_values is None and needs_ref_fx:
        with pytest.raises(AssertionError):
            calculator.calculate_deterministic_metrics(
                proc_fx_obs, categories, metrics, proc_ref_fx_obs, 1.0)
    else:
        metric_results = calculator.calculate_deterministic_metrics(
            proc_fx_obs, categories, metrics, proc_ref_fx_obs, 1.0)

        # Checks
        # categories
        assert sorted(metric_results.keys()) == sorted(categories)
        for cat, val_cat in metric_results.items():
            if cat == 'total':
                assert sorted(val_cat.keys()) == sorted(metrics)
            else:
                # category groups
                for cat_group, val_group in val_cat.items():
                    if len(fx_values) != 0:
                        assert sorted(val_group.keys()) == sorted(metrics)

                        # metrics
                        for metric, val_metric in val_group.items():
                            assert np.isfinite(val_metric)

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
