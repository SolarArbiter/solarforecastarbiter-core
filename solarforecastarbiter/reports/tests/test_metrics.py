import pandas as pd
import pytest


from solarforecastarbiter.reports import metrics


@pytest.mark.parametrize('fx,obs', [
    (pd.Series(index=pd.DatetimeIndex([])),
     pd.Series(index=pd.DatetimeIndex([]))),
    (pd.Series(index=pd.DatetimeIndex([])),
     pd.Series([1.0, 1.0], index=pd.date_range(
         start='20190801', periods=2, freq='1h', tz='MST', name='timestamp'))),
    (pd.Series([1.0, 1.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='UTC', name='timestamp')),
     pd.Series(index=pd.DatetimeIndex([]))),
    (pd.Series([1.0, 1.0], index=pd.date_range(
        start='20190801', periods=2, freq='1h', tz='MST')),
     pd.Series([1.0, 1.0], index=pd.date_range(
         start='20190801', periods=2, freq='1h', tz='UTC')))
])
def test_calculate_metrics_runs(report_objects, fx, obs):
    fxobs = report_objects[0].forecast_observations[0]
    metrics.calculate_metrics(fxobs, fx, obs)
