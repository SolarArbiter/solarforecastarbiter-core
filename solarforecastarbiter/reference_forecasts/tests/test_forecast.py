import pandas as pd
from pandas.util.testing import assert_series_equal

import pytest

from solarforecastarbiter.reference_forecasts import forecast


def assert_none_or_series(out, expected):
    assert len(out) == len(expected)
    for o, e in zip(out, expected):
        if e is None:
            assert o is None
        else:
            assert_series_equal(o, e)


def test_resample_args():
    index = pd.date_range(start='20190101', freq='15min', periods=5)
    args = [
        None, pd.Series([1, 0, 0, 0, 2], index=index)
    ]
    idx_exp = pd.date_range(start='20190101', freq='1h', periods=2)
    expected = [None, pd.Series([0.25, 2.], index=idx_exp)]
    out = forecast.resample_args(*args)
    assert_none_or_series(out, expected)


def test_resample():
    index = pd.date_range(start='20190101', freq='15min', periods=5)
    arg = pd.Series([1, 0, 0, 0, 2], index=index)
    idx_exp = pd.date_range(start='20190101', freq='1h', periods=2)
    expected = pd.Series([0.25, 2.], index=idx_exp)
    out = forecast.resample(arg)
    assert_series_equal(out, expected)
    assert forecast.resample(None) is None


def test_interpolate():
    index = pd.date_range(start='20190101', freq='15min', periods=2)
    arg = pd.Series([0, 1.5], index=index)
    out = forecast.interpolate(arg)
    assert_series_equal(out, arg)

    idx_exp = pd.date_range(start='20190101', freq='5min', periods=4)
    expected = pd.Series([0., 0.5, 1., 1.5], index=idx_exp)
    out = forecast.interpolate(arg, freq='5min')
    assert_series_equal(out, expected)

    assert forecast.interpolate(None) is None


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
    ([1, 1/2], [1., 0]),
    ([0, 1/2], [0, 1.]),
    ([0, 1/2, 1, 1/2], [0, 1., 1., 0])
])
def test_unmix_intervals(mixed, expected):
    if len(mixed) in [2, 4]:
        freq = '3h'
    else:
        freq = '1h'
    index = pd.date_range(start='20190101 01', freq=freq, periods=len(mixed))
    mixed_s = pd.Series(mixed, index=index)
    out = forecast.unmix_intervals(mixed_s)
    expected_s = pd.Series(expected, index=index)
    assert_series_equal(out, expected_s)


def test_unmix_intervals_invalidfreq():
    index = pd.date_range(start='20190101 01', freq='2h', periods=3)
    mixed_s = pd.Series([1, 1/3, 1/6], index=index)
    with pytest.raises(ValueError):
        forecast.unmix_intervals(mixed_s)


def test_unmix_intervals_two_freq():
    index = pd.DatetimeIndex(['20190101 01', '20190101 02', '20190101 04'])
    mixed_s = pd.Series([1, 1/3, 1/6], index=index)
    with pytest.raises(ValueError):
        forecast.unmix_intervals(mixed_s)
