import numpy as np
import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (calculator, preprocessing,
                                          deterministic)


# @pytest.fixture()
# def create_forecast_model():
#     def _create_forecast_model(fx_series, obs_frame):
#         return = datamodel.ProcessedForecastObservation(
#
#         )

@pytest.fixture()
def create_datetime_index():
    def _create_datetime_index(n_periods):
        return pd.date_range(start='20190801', periods=n_periods, freq='1h',
                             tz='MST', name='timestamp')

    return _create_datetime_index


@pytest.mark.parametrize('fx,obs,expect_timestamps', [
    (pd.Series(index=pd.DatetimeIndex([])),
     pd.Series(index=pd.DatetimeIndex([])),
     np.nan),
    (pd.Series(index=pd.DatetimeIndex([])),
     pd.Series([1.0, 1.0], index=pd.date_range(
         start='20190801', periods=2, freq='1h', tz='MST', name='timestamp')),
     np.nan),
    (pd.Series([1.0, 1.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='UTC', name='timestamp')),
     pd.Series(index=pd.DatetimeIndex([])),
     np.nan),
    (pd.Series([1.0, 1.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST')),
     pd.Series([1.0, 1.0], index=pd.date_range(
         start='20190801', periods=2, freq='1h', tz='UTC')),
     pd.Series())
])
def test_calculate_metrics(report_objects, fx, obs, expect_timestamps):
    report, obs_model, fx0_model, fx1_model = report_objects
    metrics = report.metrics
    for fxobs, filter in zip(report.forecast_observations, report.filters):
        pass
        # TODO ProcessedForecastObservation creation
        # out = calculator.calculate_metrics(fxobs,
        #                                    filter,
        #                                    metrics)
        # assert isinstance(out, dict)
        # assert isinstance(out["total"], dict)
        # assert isinstance(out["total"]["mae"], float)
        # for group in ('month', 'day', 'hour'):
        #     assert isinstance(out[group], dict)
        #     if isinstance(expect, pd.Series):
        #         assert isinstance(out[group]["mae"], pd.Series)
        #     else:
        #         assert np.isnan(out[group]["mae"])


# def test_calculate_deterministic_metrics(proc_pair, categories, metrics,
#                                          ref_pair, normalizer):
#     pass


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
