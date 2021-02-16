"""
Copied from pvlib python pvlib/tests/test_bsrn.py.
See LICENSES/PVLIB-PYTHON_LICENSE
"""

import inspect
import gzip
import os
from pathlib import Path

import pandas as pd
import pytest

from solarforecastarbiter.io.fetch import bsrn
from pandas.testing import assert_index_equal

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
