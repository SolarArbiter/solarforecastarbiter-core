import pandas as pd
from pandas.util.testing import assert_series_equal

from solarforecastarbiter.reference_forecasts import forecast


def assert_none_or_series(out, expected):
    assert len(out) == len(expected)
    for o, e in zip(out, expected):
        if e is None:
            assert o is None
        else:
            assert_series_equal(o, e)


def test_resample_args():
    index = pd.DatetimeIndex(start='20190101', freq='15min', periods=5)
    args = [
        None, pd.Series([1, 0, 0, 0, 2], index=index)
    ]
    idx_exp = pd.DatetimeIndex(start='20190101', freq='1h', periods=2)
    expected = [None, pd.Series([0.25, 2.], index=idx_exp)]
    out = forecast.resample_args(*args)
    assert_none_or_series(out, expected)
