# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:12:14 2019

@author: cwhanse
"""

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
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
                 'ghi_physical_limit_flag', 'dhi_physical_limit_flag',
                 'dni_physical_limit_flag', 'consistent_components',
                 'diffuse_ratio_limit'],
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
                       [500, 500, 300, 80, 1370, 0, 0, 1, 1, 1]]))
    dtypes = ['float64', 'float64', 'float64', 'float64', 'float64',
              'bool', 'bool', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_irradiance_limits_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    ghi_in = expected[['ghi', 'solar_zenith', 'dni_extra']]
    ghi_out_expected = expected[['ghi_physical_limit_flag']]
    ghi_out = validator.check_irradiance_limits_QCRad(ghi_in)
    assert_frame_equal(ghi_out, ghi_out_expected)

    dhi_in = expected[['ghi', 'solar_zenith', 'dni_extra', 'dhi']]
    dhi_out_expected = expected[['ghi_physical_limit_flag',
                                 'dhi_physical_limit_flag']]
    dhi_out = validator.check_irradiance_limits_QCRad(dhi_in, test_dhi=True)
    assert_frame_equal(dhi_out, dhi_out_expected)

    dni_in = expected[['ghi', 'solar_zenith', 'dni_extra', 'dni']]
    dni_out_expected = expected[['ghi_physical_limit_flag',
                                 'dni_physical_limit_flag']]
    dni_out = validator.check_irradiance_limits_QCRad(dni_in, test_dni=True)
    assert_frame_equal(dni_out, dni_out_expected)


def test_check_irradiance_limits_QCRad_fail(irradiance_QCRad):
    expected = irradiance_QCRad
    with pytest.raises(KeyError):
        validator.check_irradiance_limits_QCRad(expected['ghi'])
    with pytest.raises(KeyError):
        validator.check_irradiance_limits_QCRad(
            expected[['dni_extra', 'solar_zenith']])
    with pytest.raises(KeyError):
        validator.check_irradiance_limits_QCRad(
            expected[['ghi', 'dni_extra', 'solar_zenith']], test_dni=True)
    with pytest.raises(KeyError):
        validator.check_irradiance_limits_QCRad(
            expected[['ghi', 'dni_extra', 'solar_zenith']], test_dhi=True)


def test_check_irradiance_consistency_QCRad(irradiance_QCRad):
    expected = irradiance_QCRad
    components = expected[['ghi', 'solar_zenith', 'dni_extra', 'dni', 'dhi']]
    result_expected = expected[['consistent_components',
                                'diffuse_ratio_limit']]
    result = validator.check_irradiance_consistency_QCRad(components)
    assert_frame_equal(result, result_expected)


def test_check_irradiance_consistency_QCRad_fail(irradiance_QCRad):
    expected = irradiance_QCRad
    with pytest.raises(KeyError):
        validator.check_irradiance_consistency_QCRad(expected['ghi'])


@pytest.fixture
def weather():
    output = pd.DataFrame(columns=['temp_air', 'wind_speed',
                                   'relative_humidity',
                                   'extreme_temp_flag', 'extreme_wind_flag',
                                   'extreme_rh_flag'],
                          data=np.array([[-20, -5, -5, 0, 0, 0],
                                         [10, 10, 50, 1, 1, 1],
                                         [140, 75, 105, 0, 0, 0]]))
    dtypes = ['float64', 'float64', 'float64', 'bool', 'bool', 'bool']
    for (col, typ) in zip(output.columns, dtypes):
        output[col] = output[col].astype(typ)
    return output


def test_check_temperature_limits(weather):
    expected = weather
    data = expected[['temp_air', 'wind_speed']]
    result_expected = expected[['extreme_temp_flag']]
    result = validator.check_temperature_limits(data)
    assert_frame_equal(result, result_expected)


def test_check_temperature_limits_fail(weather):
    expected = weather
    with pytest.raises(KeyError):
        validator.check_temperature_limits(expected[['wind_speed']])


def test_check_wind_limits(weather):
    expected = weather
    data = expected[['temp_air', 'wind_speed']]
    result_expected = expected[['extreme_wind_flag']]
    result = validator.check_wind_limits(data)
    assert_frame_equal(result, result_expected)


def test_check_wind_limits_fail(weather):
    expected = weather
    with pytest.raises(KeyError):
        validator.check_wind_limits(expected[['temp_air']])


def test_check_rh_limits(weather):
    expected = weather
    data = expected[['relative_humidity']]
    result_expected = expected[['extreme_rh_flag']]
    result = validator.check_wind_limits(data)
    assert_series_equal(result, result_expected)


def test_check_rh_limits_fail(weather):
    expected = weather
    with pytest.raises(KeyError):
        validator.check_wind_limits(expected[['temp_air']])


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


def test_get_solarposition(mocker, location, times):
    m = mocker.spy(pvlib.solarposition, 'get_solarposition')
    validator.get_solarposition(location, times)
    validator.get_solarposition(location, times, pressure=100000)
    validator.get_solarposition(location, times, method='ephemeris')
    assert m.call_count == 3


def test_get_clearsky(mocker, location, times):
    m = mocker.spy(pvlib.clearsky, 'ineichen')
    validator.get_clearsky(location, times)
    assert m.call_count == 1
    m = mocker.spy(pvlib.clearsky, 'haurwitz')
    validator.get_clearsky(location, times, model='haurwitz')
    assert m.call_count == 1


def test_check_ghi_clearsky(mocker, location, times):
    clearsky = location.get_clearsky(times)
    # modify to create test conditions
    irrad = clearsky.copy()
    irrad.iloc[0] *= 0.5
    irrad.iloc[-1] *= 2.0
    clear_times = np.tile(True, len(times))
    clear_times[-1] = False
    expected = pd.DataFrame(index=times, data=clear_times,
                            columns=['ghi_clearsky'])
    result = validator.check_ghi_clearsky(irrad, clearsky=clearsky)
    assert_frame_equal(result, expected)
    with pytest.raises(ValueError):
        validator.check_ghi_clearsky(irrad)
    result = validator.check_ghi_clearsky(irrad, location=location)
    assert_frame_equal(result, expected)


def test_check_irradiance_day_night(location):
    MST = pytz.timezone('MST')
    times = [datetime(2018, 6, 15, 12, 0, 0, tzinfo=MST),
             datetime(2018, 6, 15, 22, 0, 0, tzinfo=MST)]
    expected = pd.DataFrame(index=times, columns=['daytime'],
                            data=[True, False])
    solar_position = pd.DataFrame(index=times, columns=['zenith'],
                                  data=[11.8, 114.3])
    result = validator.check_irradiance_day_night(
        times, solar_position=solar_position)
    assert_frame_equal(result, expected)
    result = validator.check_irradiance_day_night(times, location=location)
    assert_frame_equal(result, expected)
    with pytest.raises(ValueError):
        validator.check_irradiance_day_night(times)


def test_check_timestamp_spacing(times):
    assert validator.check_timestamp_spacing(times)
    assert validator.check_timestamp_spacing(pd.DatetimeIndex([times[0]]))
    assert validator.check_timestamp_spacing(times[[0, 2]])
    assert not validator.check_timestamp_spacing(times[[0, 2, 3]])
    MST = pytz.timezone('MST')
    times2 = pd.DatetimeIndex([datetime(2018, 6, 15, 12, 0, 0, tzinfo=MST),
                               datetime(2018, 6, 15, 12, 0, 57, tzinfo=MST),
                               datetime(2018, 6, 15, 12, 2, 13, tzinfo=MST),
                               datetime(2018, 6, 15, 12, 2, 46, tzinfo=MST)])
    assert validator.check_timestamp_spacing(times2, freq='1T')
    assert not validator.check_timestamp_spacing(times2, freq='5S')


def test_detect_stale_values():
    data = [1.0, 1.001, 1.001, 1.001, 1.001, 1.001001, 1.001, 1.001, 1.2, 1.3]
    x = pd.Series(data=data)
    res1 = validator.detect_stale_values(x)
    res2 = validator.detect_stale_values(x, rtol=1e-8, window=2)
    res3 = validator.detect_stale_values(x, window=7)
    res4 = validator.detect_stale_values(x, window=8)
    res5 = validator.detect_stale_values(x, rtol=1e-8, window=4)
    res6 = validator.detect_stale_values(x[1:])
    res7 = validator.detect_stale_values(x[1:8])
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
    res = validator.detect_stale_values(y)
    assert_series_equal(res, pd.Series([False, False, True, True, False, False,
                                        False]))
    res = validator.detect_stale_values(y, atol=1e-3)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    res = validator.detect_stale_values(y, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, False,
                                        False]))
    res = validator.detect_stale_values(y, atol=2e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    with pytest.raises(ValueError):
        validator.detect_stale_values(x, window=1)


def test_detect_interpolation():
    data = [1.0, 1.001, 1.002001, 1.003, 1.004, 1.001001, 1.001001, 1.001001,
            1.2, 1.3, 1.5, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    x = pd.Series(data=data)
    res1 = validator.detect_interpolation(x)
    assert_series_equal(res1, pd.Series([False, False, False, False, False,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res2 = validator.detect_interpolation(x, rtol=1e-2)
    assert_series_equal(res2, pd.Series([False, False, True, True, True,
                                         False, False, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    res3 = validator.detect_interpolation(x, window=5)
    assert_series_equal(res3, pd.Series([False, False, False, False, False,
                                         False, False, False, False, False,
                                         False, False, False, False, False,
                                         True, False]))
    res4 = validator.detect_interpolation(x, atol=1e-2)
    assert_series_equal(res4, pd.Series([False, False, True, True, True,
                                         True, True, True, False, False,
                                         False, False, False, True, True, True,
                                         False]))
    data = [0.0, 0.0, 0.0, -0.0, 0.00001, 0.000010001, -0.00000001]
    y = pd.Series(data=data)
    res = validator.detect_interpolation(y, atol=1e-5)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        False]))
    res = validator.detect_stale_values(y, atol=1e-4)
    assert_series_equal(res, pd.Series([False, False, True, True, True, True,
                                        True]))
    with pytest.raises(ValueError):
        validator.detect_interpolation(x, window=2)
