import pytest
import numpy as np
import pandas as pd
from solarforecastarbiter.metrics import summary


@pytest.mark.parametrize("ts", [
    [1, 2, 3],
    np.random.rand(10),
    np.random.rand(1000),
])
def test_scalar(ts):
    for metric in summary._DETERMINISTIC_MAP:
        f = summary._DETERMINISTIC_MAP[metric][0]
        assert np.isscalar(f(ts))


@pytest.mark.parametrize('data,expected', [
    (pd.Series([0, 1, 1, 1]), 0.75),
    (pd.Series([0, 1, 1, 1, np.nan]), 0.75),
    ([0, 1, 1, 1, np.nan], 0.75),
    # dataframe is reduced to a single number!
    (pd.DataFrame({
        'observation': [0, 1, 1, 1, np.nan],
        'forecast': [0, 0, 0, 1, np.nan]
     }),
     0.5),
])
def test_yes_fraction(data, expected):
    assert summary.yes_fraction(data) == expected


@pytest.mark.parametrize('data,expected', [
    (pd.Series([0, 1, 1, 1]), 0.25),
    (pd.Series([0, 1, 1, 1, np.nan]), 0.25),
    ([0, 1, 1, 1, np.nan], 0.25),
    # dataframe is reduced to a single number!
    (pd.DataFrame({
        'observation': [0, 1, 1, 1, np.nan],
        'forecast': [0, 0, 0, 1, np.nan]
     }),
     0.5),
])
def test_no_fraction(data, expected):
    out = summary.no_fraction(data)
    assert out == expected
