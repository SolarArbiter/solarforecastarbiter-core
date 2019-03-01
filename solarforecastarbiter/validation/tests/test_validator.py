# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:12:14 2019

@author: cwhanse
"""

import pandas as pd
import numpy as np
import pytest
from solarforecastarbiter.validation import validator
from pandas.util.testing import assert_frame_equal, assert_series_equal


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
                                   'extreme_temp_flag', 'extreme_wind_flag'],
                          data=np.array([[-20, -5, 0, 0],
                                         [10, 10, 1, 1],
                                         [140, 75, 0, 0]]))
    dtypes = ['float64', 'float64', 'bool', 'bool']
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


def test_check_limits(val, lb=None, ub=None, lb_ge=False, ub_le=False):
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
    result = validator._check_limits(val=data, lb=3, ub_ge=True)
    assert_series_equal(expected, result)

    result = validator._check_limits(val=data, lb=3, ub=4, lb_ge=True,
                                     ub_ge=True)
    pytest.assertTrue(all(result))
    result = validator._check_limits(val=data, lb=3, ub=4)
    pytest.assertFalse(any(result))
