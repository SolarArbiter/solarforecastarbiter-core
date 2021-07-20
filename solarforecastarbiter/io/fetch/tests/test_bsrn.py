"""
Modified from pvlib python pvlib/tests/test_bsrn.py.
See LICENSES/PVLIB-PYTHON_LICENSE
"""

import inspect
import gzip
import os
from pathlib import Path
import re

import pandas as pd
import pytest

from solarforecastarbiter.io.fetch import bsrn
from pandas.testing import assert_index_equal, assert_frame_equal

DATA_DIR = Path(os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))) / 'data'


@pytest.mark.parametrize('testfile,open_func,mode,expected_index', [
    ('bsrn-pay0616.dat.gz', gzip.open, 'rt',
     pd.date_range(start='20160601', periods=43200, freq='1min', tz='UTC')),
    ('bsrn-lr0100-pay0616.dat', open, 'r',
     pd.date_range(start='20160601', periods=43200, freq='1min', tz='UTC')),
])
def test_read_bsrn(testfile, open_func, mode, expected_index):
    with open_func(DATA_DIR / testfile, mode) as buf:
        data = bsrn.parse_bsrn(buf)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


@pytest.mark.parametrize('year', [2020, '2020'])
@pytest.mark.parametrize('month', [1, '01', '1'])
def test_read_bsrn_month_from_nasa_larc(year, month, requests_mock):
    # all 2020-01 int/str variants should produce this url
    expected_url = 'https://cove.larc.nasa.gov/BSRN/LRC49/2020/lrc0120.dat'
    with open(DATA_DIR / 'bsrn-lr0100-pay0616.dat') as f:
        content = f.read()
    matcher = re.compile('https://cove.larc.nasa.gov/BSRN/LRC49/.*')
    r = requests_mock.register_uri('GET', matcher, content=content.encode())
    out = bsrn.read_bsrn_month_from_nasa_larc(year, month)
    assert isinstance(out, pd.DataFrame)
    assert r.last_request.url == expected_url


@pytest.mark.parametrize('start,end,ncalls', [
    ('20200101', '20210101', 13),
    ('20200101', '20200103', 1)
])
def test_read_bsrn_from_nasa_larc(start, end, ncalls, mocker):
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    m = mocker.patch(
        'solarforecastarbiter.io.fetch.bsrn.read_bsrn_month_from_nasa_larc')
    m.return_value = pd.DataFrame()
    out = bsrn.read_bsrn_from_nasa_larc(start, end)
    assert m.call_count == ncalls
    assert isinstance(out, pd.DataFrame)


def test_read_bsrn_from_nasa_larc_now_limiter(mocker):
    mocked_now = mocker.patch('pandas.Timestamp.now')
    mocked_now.return_value = pd.Timestamp('2021-02-16 15:15:15', tz='UTC')
    start = pd.Timestamp('20210101', tz='UTC')
    end = pd.Timestamp('20210228', tz='UTC')
    expected = pd.DataFrame(1., columns=['ghi'], index=pd.date_range(
        start='20210101', end='20210131T2359', freq='1min', tz='UTC'))
    m = mocker.patch(
        'solarforecastarbiter.io.fetch.bsrn.read_bsrn_month_from_nasa_larc')
    m.return_value = expected
    out = bsrn.read_bsrn_from_nasa_larc(start, end)
    m.assert_called_once_with(2021, 1)
    assert_frame_equal(out, expected)
