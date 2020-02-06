import itertools

import pandas as pd
from pandas.testing import assert_series_equal

import pytest

from solarforecastarbiter.reference_forecasts import forecast


def assert_none_or_series(out, expected):
    assert len(out) == len(expected)
    for o, e in zip(out, expected):
        if e is None:
            assert o is None
        else:
            assert_series_equal(o, e)


def test_resample():
    index = pd.date_range(start='20190101', freq='15min', periods=5)
    arg = pd.Series([1, 0, 0, 0, 2], index=index)
    idx_exp = pd.date_range(start='20190101', freq='1h', periods=2)
    expected = pd.Series([0.25, 2.], index=idx_exp)
    out = forecast.resample(arg)
    assert_series_equal(out, expected)
    assert forecast.resample(None) is None


@pytest.fixture
def rfs_series():
    return pd.Series([1, 2],
                     index=pd.DatetimeIndex(['20190101 01', '20190101 02']))


@pytest.mark.parametrize(
    'start,end,start_slice,end_slice,fill_method,exp_val,exp_idx', [
        (None, None, None, None, 'interpolate', [1, 1.5, 2],
         ['20190101 01', '20190101 0130', '20190101 02']),
        ('20190101', '20190101 0230', None, None, 'interpolate',
         [1, 1, 1, 1.5, 2, 2],
         ['20190101', '20190101 0030', '20190101 01', '20190101 0130',
          '20190101 02', '20190101 0230']),
        ('20190101', '20190101 02', '20190101 0030', '20190101 0130', 'bfill',
         [1., 1, 2], ['20190101 0030', '20190101 01', '20190101 0130'])
    ]
)
def test_reindex_fill_slice(rfs_series, start, end, start_slice, end_slice,
                            fill_method, exp_val, exp_idx):
    exp = pd.Series(exp_val, index=pd.DatetimeIndex(exp_idx))
    out = forecast.reindex_fill_slice(
        rfs_series, freq='30min', start=start, end=end,
        start_slice=start_slice, end_slice=end_slice, fill_method=fill_method)
    assert_series_equal(out, exp)


def test_reindex_fill_slice_some_nan():
    rfs_series = pd.Series([1, 2, None, 4], index=pd.DatetimeIndex([
        '20190101 01', '20190101 02', '20190101 03', '20190101 04',
    ]))
    start, end, start_slice, end_slice, fill_method = \
        None, None, None, None, 'interpolate'
    exp_val = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    exp_idx = [
        '20190101 01', '20190101 0130', '20190101 02', '20190101 0230',
        '20190101 03', '20190101 0330', '20190101 04']
    exp = pd.Series(exp_val, index=pd.DatetimeIndex(exp_idx))
    out = forecast.reindex_fill_slice(
        rfs_series, freq='30min', start=start, end=end,
        start_slice=start_slice, end_slice=end_slice, fill_method=fill_method)
    assert_series_equal(out, exp)


def test_reindex_fill_slice_all_nan():
    arg = pd.Series([None]*3, index=pd.DatetimeIndex(
        ['20190101 01', '20190101 02', '20190101 03']))
    out = forecast.reindex_fill_slice(arg, freq='30min')
    exp = pd.Series([None]*5, index=pd.DatetimeIndex(
        ['20190101 01', '20190101 0130', '20190101 02', '20190101 0230',
         '20190101 03']))
    assert_series_equal(out, exp)


def test_reindex_fill_slice_empty():
    out = forecast.reindex_fill_slice(pd.Series(dtype=float), freq='30min')
    assert_series_equal(out, pd.Series(dtype=float))


def test_reindex_fill_slice_none():
    out = forecast.reindex_fill_slice(None, freq='30min')
    assert out is None


def test_cloud_cover_to_ghi_linear():
    cloud_cover = pd.Series([0, 50, 100.])
    ghi_clear = pd.Series([1000, 1000, 1000.])
    out = forecast.cloud_cover_to_ghi_linear(cloud_cover, ghi_clear)
    expected = pd.Series([1000, 675, 350.])
    assert_series_equal(out, expected)
    out = forecast.cloud_cover_to_ghi_linear(cloud_cover, ghi_clear, offset=20)
    expected = pd.Series([1000, 600, 200.])
    assert_series_equal(out, expected)


@pytest.mark.xfail(raises=AssertionError, strict=True)
def test_cloud_cover_to_irradiance_ghi_clear():
    index = pd.date_range(start='20190101', periods=3, freq='1h')
    cloud_cover = pd.Series([0, 50, 100.], index=index)
    ghi_clear = pd.Series([10, 10, 1000.], index=index)
    zenith = pd.Series([90.0, 89.9, 45], index=index)
    out = forecast.cloud_cover_to_irradiance_ghi_clear(
        cloud_cover, ghi_clear, zenith
    )
    # https://github.com/pvlib/pvlib-python/issues/681
    ghi_exp = pd.Series([10., 6.75, 350.])
    dni_exp = pd.Series([0., 0., 4.74198165e+01])
    dhi_exp = pd.Series([10., 6.75, 316.46912616])
    assert_series_equal(out[0], ghi_exp)
    assert_series_equal(out[1], dni_exp)
    assert_series_equal(out[2], dhi_exp)


@pytest.mark.xfail(raises=AssertionError, strict=True)
def test_cloud_cover_to_irradiance():
    index = pd.date_range(start='20190101', periods=3, freq='1h')
    cloud_cover = pd.Series([0, 50, 100.], index=index)
    latitude = 32.2
    longitude = -110.9
    elevation = 700
    zenith = pd.Series([90.0, 89.9, 45], index=index)
    apparent_zenith = pd.Series([89.9, 89.85, 45], index=index)
    out = forecast.cloud_cover_to_irradiance(
        latitude, longitude, elevation, cloud_cover, apparent_zenith, zenith
    )
    # https://github.com/pvlib/pvlib-python/issues/681
    ghi_exp = pd.Series([10., 6.75, 350.], index=index)
    dni_exp = pd.Series([0., 0., 4.74198165e+01], index=index)
    dhi_exp = pd.Series([10., 6.75, 316.46912616], index=index)
    assert_series_equal(out[0], ghi_exp)
    assert_series_equal(out[1], dni_exp)
    assert_series_equal(out[2], dhi_exp)


@pytest.mark.parametrize('mixed,expected', [
    ([1, 1/2, 1/3, 1/4, 1/5, 1/6], [1., 0, 0, 0, 0, 0]),
    ([0, 0, 0, 0, 0, 1/6], [0, 0, 0, 0, 0, 1.]),
    ([0, 0, 0, 0, 0, 1/6, 1, 1/2, 1/3, 1/4, 1/5, 1/6],
     [0, 0, 0, 0, 0, 1., 1., 0, 0, 0, 0, 0]),
    ([65.0, 66.0, 44.0, 32.0, 30.0, 26.0],  # GH 144
     [65.0, 67.0, 0.0, 0.0, 22.0, 6.0]),  # 4th element is -4 if no clipping
    ([1, 1/2], [1., 0]),
    ([0, 1/2], [0, 1.]),
    ([0, 1/2, 1, 1/2], [0, 1., 1., 0])
])
def test_unmix_intervals(mixed, expected):
    npts = len(mixed)
    if npts in [2, 4]:
        index = pd.date_range(start='20190101 03Z', freq='3h', periods=npts)
    else:
        index = pd.date_range(start='20190101 01Z', freq='1h', periods=npts)
    mixed_s = pd.Series(mixed, index=index)
    out = forecast.unmix_intervals(mixed_s)
    expected_s = pd.Series(expected, index=index)
    assert_series_equal(out, expected_s)


# allowed times are 1, 7, 13, 19Z
_1h_allowed_0700 = [0, 6, 12, 18]
_1h_not_allowed_0700 = [x for x in range(0, 24) if x not in _1h_allowed_0700]
# allowed times are 3, 9, 15, 21Z
_3h_allowed_0700 = [2, 8, 14, 20]
_3h_not_allowed_0700 = [x for x in range(0, 24) if x not in _3h_allowed_0700]


@pytest.mark.parametrize('hr,freq', (
    list(itertools.product(_1h_allowed_0700, ['1h'])) +
    list(itertools.product(_3h_allowed_0700, ['3h']))
))
def test_unmix_intervals_tz(hr, freq):
    periods = 6 if freq == '1h' else 2
    # should work.
    index = pd.date_range(
        start=f'20190101 {hr:02}-0700', freq=freq, periods=periods)
    # we only care to test tz functionality, not values
    mixed_s = pd.Series([0] * len(index), index=index)
    out = forecast.unmix_intervals(mixed_s)
    expected_s = pd.Series([0] * len(index), index=index)
    assert_series_equal(out, expected_s)


@pytest.mark.parametrize('hr,freq', (
    list(itertools.product(_1h_not_allowed_0700, ['1h'])) +
    list(itertools.product(_3h_not_allowed_0700, ['3h']))
))
def test_unmix_intervals_tz_fail(hr, freq):
    periods = 6 if freq == '1h' else 2
    # should not work. 13-0700 is 20Z
    index = pd.date_range(
        start=f'20190101 {hr:02}-0700', freq=freq, periods=periods)
    with pytest.raises(ValueError):
        mixed_s = pd.Series([0] * len(index), index=index)
        forecast.unmix_intervals(mixed_s)


def test_unmix_intervals_invalidfreq():
    index = pd.date_range(start='20190101 01Z', freq='2h', periods=3)
    mixed_s = pd.Series([1, 1/3, 1/6], index=index)
    with pytest.raises(ValueError):
        forecast.unmix_intervals(mixed_s)


def test_unmix_intervals_two_freq():
    index = pd.DatetimeIndex(['20190101 01', '20190101 02', '20190101 04'])
    mixed_s = pd.Series([1, 1/3, 1/6], index=index)
    with pytest.raises(ValueError):
        forecast.unmix_intervals(mixed_s)
