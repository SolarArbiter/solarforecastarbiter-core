from solarforecastarbiter.reports import figures

import pandas as pd

import pytest


def test_construct_metrics_cds():
    metrics = [
        {
            'name': 'University of Arizona OASIS Day Ahead GFS ghi',
            'total': {'mae': 74.},
            'month': {'mae': pd.Series(74., index=[8])},
            'day': {'mae': pd.Series(74., index=[21])},
            'hour': {'mae': pd.Series([74., 75.], index=[12, 13])}
        },
        {
            'name': 'University of Arizona OASIS Intraday NAM',
            'total': {'mae': 74.},
            'month': {'mae': pd.Series(74., index=[8])},
            'day': {'mae': pd.Series(74., index=[21])},
            'hour': {'mae': pd.Series([74., 75.], index=[12, 13])}
        }
    ]
    cds = figures.construct_metrics_cds(metrics, 'total')
    assert cds.data['forecast'][0] == \
        'University of Arizona OASIS Day Ahead GFS ghi'
    cds = figures.construct_metrics_cds(metrics, 'total',
                                        rename=figures.abbreviate)
    assert cds.data['forecast'][0] == 'Uni. of Ari. OASIS Day Ahe. GFS ghi'


@pytest.mark.parametrize('arg,expected', [
    ('University of Arizona OASIS Day Ahead GFS ghi',
     'Uni. of Ari. OASIS Day Ahe. GFS ghi')
])
def test_abbreviate(arg, expected):
    out = figures.abbreviate(arg)
    assert out == expected
