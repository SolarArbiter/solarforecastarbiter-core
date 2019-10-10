import numpy as np
import pandas as pd
import pytest


from solarforecastarbiter.reports import metrics


@pytest.mark.parametrize('fx,obs,expect', [
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
def test_calculate_metrics_runs(report_objects, fx, obs, expect):
    fxobs = report_objects[0].forecast_observations[0]
    out = metrics.calculate_metrics(fxobs, fx, obs)
    assert isinstance(out, dict)
    assert isinstance(out["total"], dict)
    assert isinstance(out["total"]["mae"], float)
    for group in ('month', 'day', 'hour'):
        assert isinstance(out[group], dict)
        if isinstance(expect, pd.Series):
            assert isinstance(out[group]["mae"], pd.Series)
        else:
            assert np.isnan(out[group]["mae"])
