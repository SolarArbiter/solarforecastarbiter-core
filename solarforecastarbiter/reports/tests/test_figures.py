from solarforecastarbiter.reports import figures


import pytest


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
