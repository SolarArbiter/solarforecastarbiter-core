from solarforecastarbiter.reports import figures

import pandas as pd
import numpy as np

import pandas.testing as pdt

import pytest


@pytest.fixture
def metrics():
    # important that these are not ordered alphabetically (GH 204)
    metrics = [
        {
            'name': 'Forecast 1 BBBB',
            'Total': {'mae': 74.},
            'Month of the year': {'mae': pd.Series(74., index=[8])},
            'Day of the month': {'mae': pd.Series(74., index=[21])},
            'Hour of the day': {'mae': pd.Series([74., 75.], index=[12, 13])}
        },
        {
            'name': 'Forecast 1 AAAA',
            'Total': {'mae': 74.},
            'Month of the year': {'mae': pd.Series(74., index=[8])},
            'Day of the month': {'mae': pd.Series(74., index=[21])},
            'Hour of the day': {'mae': pd.Series([74., 75.], index=[12, 13])}
        }
    ]
    return metrics


@pytest.fixture
def metrics_month():
    index = pd.MultiIndex(
        levels=[['mae'], ['Forecast 1 BBBB', 'Forecast 1 AAAA'], [8]],
        codes=[[0, 0], [0, 1], [0, 0]],
        names=['metric', 'forecast', 'Month of the year'])
    return pd.Series([74., 74.], index=index)


def test_construct_metrics_cds(metrics):
    cds = figures.construct_metrics_cds(metrics, 'Total')
    assert cds.data['forecast'][0] == 'Forecast 1 BBBB'
    cds = figures.construct_metrics_cds(metrics, 'Total',
                                        rename=figures.abbreviate)
    assert cds.data['forecast'][0] == 'For. 1 BBBB'


def test_construct_metrics_series(metrics, metrics_month):
    # same idea applies to month, day, hour, etc groupings
    # could test more, but wait for refactoring
    out = figures.construct_metrics_series(metrics, 'Month of the year')
    pdt.assert_series_equal(out, metrics_month)


def test_construct_metrics_cds2(metrics, metrics_month):
    out = figures.construct_metrics_cds2(metrics_month, 'mae')
    expected = {
        'Month of the year': np.array([8]),
        'Forecast 1 BBBB': np.array([74]),
        'Forecast 1 AAAA': np.array([74])
    }
    assert out.data == expected


@pytest.mark.parametrize('arg,expected', [
    ('a', 'a'),
    ('abcd', 'abc.'),
    ('ABCDEFGHIJKLMNOP', 'ABCDEFGHIJKLMNOP'),
    ('University of Arizona OASIS Day Ahead GFS ghi',
     'Uni. of Ari. OASIS Day Ahe. GFS ghi')
])
def test_abbreviate(arg, expected):
    out = figures.abbreviate(arg)
    assert out == expected
