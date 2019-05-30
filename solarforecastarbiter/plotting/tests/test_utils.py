import pandas as pd
from pandas.testing import assert_index_equal
import pytest


from solarforecastarbiter.plotting import utils


@pytest.mark.parametrize('var,exp', [
    ('ghi', 'GHI (W/m^2)'),
    ('dc_power', 'DC Power (MW)')
])
def test_format_variable_name(var, exp):
    out = utils.format_variable_name(var)
    assert out == exp


@pytest.mark.parametrize('dobj,removal', [
    (pd.Series, slice(5, 10)),
    (pd.DataFrame, slice(12, 15))
])
def test_align_index(dobj, removal):
    index = pd.date_range(start='now', freq='5min',
                          periods=20, name='timestamp')
    data = dobj(index=index)
    data = data.drop(index[removal])
    out = utils.align_index(data, pd.Timedelta('5min'))
    assert_index_equal(out.index, index)


def test_align_index_new_length():
    index = pd.date_range(start='now', freq='5min',
                          periods=20, name='timestamp')
    data = pd.Series(index=index)
    out = utils.align_index(data, pd.Timedelta('1min'))
    nindex = pd.date_range(start=index[0], end=index[-1], freq='1min',
                           name='timestamp')
    assert_index_equal(out.index, nindex)


def test_align_index_limit():
    index = pd.date_range(start='now', freq='5min',
                          periods=20, name='timestamp')
    data = pd.Series(index=index)
    out = utils.align_index(data, pd.Timedelta('5min'),
                            limit=pd.Timedelta('60min'))
    nindex = pd.date_range(start=index[-13], end=index[-1], freq='5min',
                           name='timestamp')
    assert_index_equal(out.index, nindex)


@pytest.mark.parametrize('label,method', [
    ('instant', 'line'),
    ('beginning', 'step'),
    ('ending', 'step'),
    pytest.param('other', '', marks=pytest.mark.xfail(raises=ValueError))
])
def test_line_or_step(label, method):
    out = utils.line_or_step(label)
    assert out[0] == method
    assert isinstance(out[1], dict)
    assert isinstance(out[2], dict)
