"""
Created on Wed Feb 27 15:12:14 2019

@author: cwhanse
"""

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from datetime import datetime
import pytz
import pytest
from solarforecastarbiter.validation import validator
import pvlib
from pvlib.location import Location


@pytest.fixture
def irradiance_QCRad():
    output = pd.DataFrame(
        columns=['ghi', 'dhi', 'dni', 'solar_zenith', 'dni_extra',
                 'ghi_limit_flag', 'dhi_limit_flag', 'dni_limit_flag',
                 'consistent_components', 'diffuse_ratio_limit'],
        data=np.array([[-100, 100, 100, 30, 1370, 0, 1, 1, 0, 0],
                       [100, -100, 100, 30, 1370, 1, 0, 1, 0, 0],
                       [100, 100, -100, 30, 1370, 1, 1, 0, 0, 1],
                       [1000, 100, 900, 0, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 15, 1370, 1, 1, 1, 1, 1],
                       [1000, 200, 800, 60, 1370, 0, 1, 1, 0, 1],
                       [1000, 300, 850, 80, 1370, 0, 0, 1, 0, 1],
                       [1000, 500, 800, 90, 1370, 0, 0, 1, 0, 1],
                       [500, 100, 1100, 0, 1370, 1, 1, 1, 0, 1],
                       [1000, 300, 1200, 0, 1370, 1, 1, 1, 0, 1],
                       [500, 600, 100, 60, 1370, 1, 1, 1, 0, 0],
                       [500, 600, 400, 80, 1370, 0, 0, 1, 0, 0],
                       [500, 500, 300, 80, 1370, 0, 0, 1, 1, 1],
                       [0, 0, 0, 93, 1370, 1, 1, 1, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
              'bool', 'bool', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_ghi_limits_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out = validator.check_ghi_limits_QCRad(expected['ghi'],
                                               expected['solar_zenith'],
                                               expected['dni_extra'])
    assert_series_equal(ghi_out, ghi_out_expected)


def test_check_dhi_limits_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    dhi_out_expected = expected['dhi_limit_flag']
    dhi_out = validator.check_dhi_limits_QCRad(expected['dhi'],
                                               expected['solar_zenith'],
                                               expected['dni_extra'])
    assert_series_equal(dhi_out, dhi_out_expected)


def test_check_dni_limits_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    dni_out_expected = expected['dni_limit_flag']
    dni_out = validator.check_dni_limits_QCRad(expected['dni'],
                                               expected['solar_zenith'],
                                               expected['dni_extra'])
    assert_series_equal(dni_out, dni_out_expected)


def test_check_irradiance_limits_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    ghi_out_expected = expected['ghi_limit_flag']
    ghi_out, dhi_out, dni_out = validator.check_irradiance_limits_QCRad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'])
    assert_series_equal(ghi_out, ghi_out_expected)
    assert dhi_out is None
    assert dni_out is None

    dhi_out_expected = expected['dhi_limit_flag']
    ghi_out, dhi_out, dni_out = validator.check_irradiance_limits_QCRad(
        expected['solar_zenith'], expected['dni_extra'], ghi=expected['ghi'],
        dhi=expected['dhi'])
    assert_series_equal(dhi_out, dhi_out_expected)

    dni_out_expected = expected['dni_limit_flag']
    ghi_out, dhi_out, dni_out = validator.check_irradiance_limits_QCRad(
        expected['solar_zenith'], expected['dni_extra'],
        dni=expected['dni'])
    assert_series_equal(dni_out, dni_out_expected)


def test_check_irradiance_consistency_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    cons_comp, diffuse = validator.check_irradiance_consistency_QCRad(
        expected['ghi'], expected['solar_zenith'], expected['dni_extra'],
        expected['dhi'], expected['dni'])
    assert_series_equal(cons_comp, expected['consistent_components'])
    assert_series_equal(diffuse, expected['diffuse_ratio_limit'])


@pytest.fixture
def weather():
    output = pd.DataFrame(columns=['air_temperature', 'wind_speed',
                                   'relative_humidity',
                                   'extreme_temp_flag', 'extreme_wind_flag',
                                   'extreme_rh_flag'],
                          data=np.array([[-40, -5, -5, 0, 0, 0],
                                         [10, 10, 50, 1, 1, 1],
                                         [140, 55, 105, 0, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_temperature_limits(weather):
    expected = weather
    result_expected = expected['extreme_temp_flag']
    result = validator.check_temperature_limits(expected['air_temperature'])
    assert_series_equal(result, result_expected)


def test_check_wind_limits(weather):
    expected = weather
    result_expected = expected['extreme_wind_flag']
    result = validator.check_wind_limits(expected['wind_speed'])
    assert_series_equal(result, result_expected)


def test_check_rh_limits(weather):
    expected = weather
    data = expected['relative_humidity']
    result_expected = expected['extreme_rh_flag']
    result = validator.check_rh_limits(data)
    result.name = 'extreme_rh_flag'
    assert_series_equal(result, result_expected)


def test_check_ac_power_limits():
    index = pd.date_range(
        start='20200401 0700', freq='2h', periods=6, tz='UTC')
    power = pd.Series([0, -0.1, 0.1, 1, 1.1, -0.1], index=index)
    day_night = pd.Series([0, 0, 0, 1, 1, 1], index=index, dtype='bool')
    capacity = 1.
    expected = pd.Series([1, 0, 0, 1, 0, 0], index=index).astype(bool)
    out = validator.check_ac_power_limits(power, day_night, capacity)
    assert_series_equal(out, expected)


def test_check_dc_power_limits():
    index = pd.date_range(
        start='20200401 0700', freq='2h', periods=6, tz='UTC')
    power = pd.Series([0, -0.1, 0.1, 1, 1.3, -0.1], index=index)
    day_night = pd.Series([0, 0, 0, 1, 1, 1], index=index, dtype='bool')
    capacity = 1.
    expected = pd.Series([1, 0, 0, 1, 0, 0], index=index).astype(bool)
    out = validator.check_dc_power_limits(power, day_night, capacity)
    assert_series_equal(out, expected)


def test_check_limits():
    # testing with input type Series
    expected = pd.Series(data=[True, False])
    data = pd.Series(data=[3, 2])
    result = validator._check_limits(val=data, lb=2.5)
    assert_series_equal(expected, result)
    result = validator._check_limits(val=data, lb=3, lb_ge=True)
    assert_series_equal(expected, result)

    data = pd.Series(data=[3, 4])
    result = validator._check_limits(val=data, ub=3.5)
    assert_series_equal(expected, result)
    result = validator._check_limits(val=data, ub=3, ub_le=True)
    assert_series_equal(expected, result)

    result = validator._check_limits(val=data, lb=3, ub=4, lb_ge=True,
                                     ub_le=True)
    assert all(result)
    result = validator._check_limits(val=data, lb=3, ub=4)
    assert not any(result)

    with pytest.raises(ValueError):
        validator._check_limits(val=data)


@pytest.fixture
def location():
    return Location(latitude=35.05, longitude=-106.5, altitude=1619,
                    name="Albuquerque", tz="MST")


@pytest.fixture
def times():
    MST = pytz.timezone('MST')
    return pd.date_range(start=datetime(2018, 6, 15, 12, 0, 0, tzinfo=MST),
                         end=datetime(2018, 6, 15, 13, 0, 0, tzinfo=MST),
                         freq='10T')


def test_check_ghi_clearsky(mocker, location, times):
    clearsky = location.get_clearsky(times)
    # modify to create test conditions
    ghi = clearsky['ghi'].copy()
    ghi.iloc[0] *= 0.5
    ghi.iloc[-1] *= 2.0
    clear_times = np.tile(True, len(times))
    clear_times[-1] = False
    expected = pd.Series(index=times, data=clear_times)
    result = validator.check_ghi_clearsky(ghi, clearsky['ghi'])
    assert_series_equal(result, expected)


def test_check_poa_clearsky(mocker, times):
    dt = pd.date_range(start=datetime(2019, 6, 15, 12, 0, 0),
                       freq='15T', periods=5)
    poa_global = pd.Series(index=dt, data=[800, 1000, 1200, -200, np.nan])
    poa_clearsky = pd.Series(index=dt, data=1000)
    result = validator.check_poa_clearsky(poa_global, poa_clearsky)
    expected = pd.Series(index=dt, data=[True, True, False, True, False])
    assert_series_equal(result, expected)
    result = validator.check_poa_clearsky(poa_global, poa_clearsky, kt_max=1.2)
    expected = pd.Series(index=dt, data=[True, True, True, True, False])
    assert_series_equal(result, expected)


def test_check_day_night():
    MST = pytz.timezone('MST')
    times = [datetime(2018, 6, 15, 12, 0, 0, tzinfo=MST),
             datetime(2018, 6, 15, 22, 0, 0, tzinfo=MST)]
    expected = pd.Series(data=[True, False], index=times)
    solar_zenith = pd.Series(data=[11.8, 114.3], index=times)
    result = validator.check_day_night(solar_zenith)
    assert_series_equal(result, expected)


@pytest.mark.parametrize('closed,solar_zenith,expected', (
    ('left', [89]*11 + [80]*13,
     pd.Series([False, True], index=pd.DatetimeIndex(
            ['20200917 0000', '20200917 0100']))),
    ('right', [89]*11 + [80]*13,
     pd.Series([False, True], index=pd.DatetimeIndex(
        ['20200917 0100', '20200917 0200']))),
    ('left', [89]*10 + [80]*14,
     pd.Series([True, True], index=pd.DatetimeIndex(
            ['20200917 0000', '20200917 0100']))),
    ('right', [89]*10 + [80]*14,
     pd.Series([True, True], index=pd.DatetimeIndex(
        ['20200917 0100', '20200917 0200'])))
))
def test_check_day_night_interval(closed, solar_zenith, expected):
    interval_length = pd.Timedelta('1h')
    index = pd.date_range(
        start='20200917 0000', end='20200917 0200', closed=closed,
        freq='5min')
    solar_zenith = pd.Series(solar_zenith, index=index)
    result = validator.check_day_night_interval(
        solar_zenith, closed, interval_length
    )
    assert_series_equal(result, expected)


def test_check_day_night_interval_no_infer():
    interval_length = pd.Timedelta('1h')
    closed = 'left'
    index = pd.DatetimeIndex(
        ['20200917 0100', '20200917 0200', '20200917 0400'])
    solar_zenith = pd.Series([0]*3, index=index)
    with pytest.raises(ValueError, match='contains gaps'):
        validator.check_day_night_interval(
            solar_zenith, closed, interval_length
        )


def test_check_day_night_interval_irregular():
    interval_length = pd.Timedelta('1h')
    solar_zenith_interval_length = pd.Timedelta('5min')
    closed = 'left'
    index1 = pd.date_range(
        start='20200917 0000', end='20200917 0100', closed=closed,
        freq=solar_zenith_interval_length)
    index2 = pd.date_range(
        start='20200917 0200', end='20200917 0300', closed=closed,
        freq=solar_zenith_interval_length)
    index = index1.union(index2)
    solar_zenith = pd.Series([89]*11 + [80]*13, index=index)
    result = validator.check_day_night_interval(
        solar_zenith, closed, interval_length,
        solar_zenith_interval_length=solar_zenith_interval_length
    )
    expected = pd.Series([False, False, True], index=pd.DatetimeIndex(
        ['20200917 0000', '20200917 0100', '20200917 0200']))
    assert_series_equal(result, expected)


def test_check_timestamp_spacing(times):
    assert_series_equal(
        validator.check_timestamp_spacing(times, times.freq),
        pd.Series(True, index=times))

    assert_series_equal(
        validator.check_timestamp_spacing(times[[0]], times.freq),
        pd.Series(True, index=[times[0]]))

    assert_series_equal(
        validator.check_timestamp_spacing(times[[0, 2, 3]], times.freq),
        pd.Series([True, False, True], index=times[[0, 2, 3]]))

    assert_series_equal(
        validator.check_timestamp_spacing(times, '30min'),
        pd.Series([True] + [False] * (len(times) - 1), index=times))


def test_detect_stale_values():
    data = [1.0, 1.001, 1.001, 1.001, 1.001, 1.001001, 1.001, 1.001, 1.2, 1.3]
    x = pd.Series(data=data)
    res1 = validator.detect_stale_values(x, window=3)
    res2 = validator.detect_stale_values(x, rtol=1e-8, window=2)
    res3 = validator.detect_stale_values(x, window=7)
    res4 = validator.detect_stale_values(x, window=8)
    res5 = validator.detect_stale_values(x, rtol=1e-8, window=4)
    res6 = validator.detect_stale_values(x[1:], window=3)
    res7 = validator.detect_stale_values(x[1:8], window=3)
    assert_series_equal(res1, pd.Series([False, False, False, True, True, True,
                                         True, True, False, False]))
    assert_series_equal(res2, pd.Series([False, False, True, True, True, False,
                                         False, True, False, False]))
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False]))
    assert not all(res4)
    assert_series_equal(res5, pd.Series([False, False, False, False, True,
                                         False, False, False, False, False]))
    assert_series_equal(res6, pd.Series(index=x[1:].index,
                                        data=[False, False, True, True, True,
                                              True, True, False, False]))
    assert_series_equal(res7, pd.Series(index=x[1:8].index,
                                        data=[False, False, True, True, True,
                                              True, True]))
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    y = pd.Series(data=data)
    res = validator.detect_stale_values(y, window=3)
    assert_series_equal(res, pd.Series([False, False, True, True, False, False,
                                        False]))
    res = validator.detect_stale_values(y, window=3, atol=1e-3)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    res = validator.detect_stale_values(y, window=3, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, False,
                                        False]))
    res = validator.detect_stale_values(y, window=3, atol=2e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    with pytest.raises(ValueError):
        validator.detect_stale_values(x, window=1)


def test_detect_interpolation():
    data = [1.0, 1.001, 1.002001, 1.003, 1.004, 1.001001, 1.001001, 1.001001,
            1.2, 1.3, 1.5, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    x = pd.Series(data=data)
    res1 = validator.detect_interpolation(x, window=3)
    assert_series_equal(res1, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res2 = validator.detect_interpolation(x, window=3, rtol=1e-2)
    assert_series_equal(res2, pd.Series([False, False, True, True, True,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res3 = validator.detect_interpolation(x, window=5)
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, False, False, False,
                                         False, False, False, False, False,
                                         True, False]))
    res4 = validator.detect_interpolation(x, window=3, atol=1e-2)
    assert_series_equal(res4, pd.Series([False, False, True, True, True,
                                         True, True, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    y = pd.Series(data=data)
    res = validator.detect_interpolation(y, window=3, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        False]))
    res = validator.detect_stale_values(y, window=3, atol=1e-4)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    with pytest.raises(ValueError):
        validator.detect_interpolation(x, window=2)


@pytest.fixture
def ghi_clearsky():
    MST = pytz.timezone('Etc/GMT+7')
    dt = pd.date_range(start=datetime(2019, 4, 3, 5, 0, 0, tzinfo=MST),
                       periods=60, freq='15T')
    loc = pvlib.location.Location(latitude=35, longitude=-110, tz=MST)
    cs = loc.get_clearsky(dt)
    return cs['ghi']


@pytest.fixture
def ghi_clipped(ghi_clearsky):
    ghi_clipped = ghi_clearsky.copy()
    ghi_clipped = np.minimum(ghi_clearsky, 800)
    ghi_clipped.iloc[12:17] = np.minimum(ghi_clearsky, 300)
    ghi_clipped.iloc[18:20] = np.minimum(ghi_clearsky, 300)
    ghi_clipped.iloc[26:28] *= 0.5
    ghi_clipped.iloc[36:] = np.minimum(ghi_clearsky, 400)
    return ghi_clipped


def test_detect_clipping(ghi_clipped):
    placeholder = pd.Series(index=ghi_clipped.index, data=False)
    expected = placeholder.copy()
    # for window=4 and fraction_in_window=0.75
    expected.iloc[3:6] = True
    expected.iloc[14:17] = True
    expected.iloc[18:20] = True
    expected.iloc[25] = True
    expected.iloc[30:36] = True
    expected.iloc[38:46] = True
    expected.iloc[56:60] = True
    flags = validator.detect_clipping(ghi_clipped, window=4,
                                      fraction_in_window=0.75, rtol=5e-3,
                                      levels=4)
    assert_series_equal(flags, expected)


def test_detect_clipping_some_nans(ghi_clipped):
    placeholder = pd.Series(index=ghi_clipped.index, data=False)
    expected = placeholder.copy()
    # for window=4 and fraction_in_window=0.75
    expected.iloc[3:6] = True
    expected.iloc[14:17] = True
    expected.iloc[18:20] = True
    expected.iloc[25] = True
    expected.iloc[30:36] = True
    expected.iloc[38:46] = True
    expected.iloc[56:60] = True
    inp = ghi_clipped.copy()
    inp.iloc[48] = np.nan
    flags = validator.detect_clipping(inp, window=4,
                                      fraction_in_window=0.75, rtol=5e-3,
                                      levels=4)
    assert_series_equal(flags, expected)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_detect_clearsky_ghi(ghi_clearsky):
    flags = validator.detect_clearsky_ghi(ghi_clearsky, ghi_clearsky)
    # first 7 and last 6 values are judged not clear due to night (ghi=0)
    # and rapid change in ghi with sunrise and sunset
    assert all(flags[7:-6])
    assert not flags[0:7].any() and not flags[-6:].any()
    ghi_cloud = ghi_clearsky.copy()
    ghi_cloud[12:15] *= 0.5
    flags = validator.detect_clearsky_ghi(ghi_cloud, ghi_clearsky)
    assert all(flags[7:12]) and all(flags[15:-6])


def test_detect_clearsky_ghi_warn_interval_length(ghi_clearsky):
    with pytest.warns(RuntimeWarning):
        flags = validator.detect_clearsky_ghi(ghi_clearsky[::4],
                                              ghi_clearsky[::4])
    assert (flags == 0).all()


def test_detect_clearsky_ghi_warn_regular_interval(ghi_clearsky):
    with pytest.warns(RuntimeWarning):
        ser = ghi_clearsky[:-2].append(ghi_clearsky[-1:])
        flags = validator.detect_clearsky_ghi(ser, ser)
    assert (flags == 0).all()


def test_detect_clearsky_ghi_one_val(ghi_clearsky):
    ser = ghi_clearsky[:1]
    assert len(ser) == 1
    with pytest.warns(RuntimeWarning):
        flags = validator.detect_clearsky_ghi(ser, ser)
    assert (flags == 0).all()


@pytest.mark.parametrize('inp,expected', [
    (pd.Timedelta('15min'), 6),
    (pd.Timedelta('1h'), 3)
])
def test_stale_interpolated_window(inp, expected):
    assert validator.stale_interpolated_window(inp) == expected
