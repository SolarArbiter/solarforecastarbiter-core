import datetime
from functools import partial

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from pvlib.location import Location

from solarforecastarbiter import pvmodel


# from pvlib
@pytest.fixture
def golden_mst():
    return Location(39.742476, -105.1786, 'MST', 1830.14)


# from pvlib
@pytest.fixture
def expected_solpos():
    return pd.DataFrame({'elevation': 39.872046,
                         'apparent_zenith': 50.111622,
                         'azimuth': 194.340241,
                         'apparent_elevation': 39.888378},
                        index=['2003-10-17T12:30:30Z'])


# modified from pvlib
def test_calculate_solar_position(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    solar_position = pvmodel.calculate_solar_position(golden_mst.latitude,
                                                      golden_mst.longitude,
                                                      golden_mst.altitude,
                                                      times)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos,
                       solar_position[expected_solpos.columns],
                       check_less_precise=3)


# modified from pvlib
def test_complete_irradiance_components():
    index = pd.DatetimeIndex(
        ['20190101', '20190101', '20190101', '20190629'], tz='UTC')
    ghi = pd.Series([0, 50, 1000, 1000], index=index)
    zenith = pd.Series([120, 85, 10, 10], index=index)
    expected_dni = pd.Series([
        -0.0, 96.71926718388598, 794.2056511357252, 842.3392765276444
    ], index=index)
    expected_dhi = pd.Series([
        0.0, 41.57036043057922, 217.86011727542888, 170.45774980888152
    ], index=index)

    dni, dhi = pvmodel.complete_irradiance_components(ghi, zenith)

    assert_series_equal(expected_dni, dni, check_names=False)
    assert_series_equal(expected_dhi, dhi, check_names=False)


def test_clearsky():
    assert False


@pytest.fixture
def times():
    return pd.date_range(start='20190101', periods=2, freq='12H',
                         tz='Etc/GMT+7')


@pytest.fixture
def apparent_zenith(times):
    return pd.Series([163.25196176,  62.70113907], index=times)


@pytest.fixture
def azimuth(times):
    return pd.Series([356.76293291, 178.88940279], index=times)


@pytest.fixture
def ghi(times):
    return pd.Series([0., 458.632494], index=times)


@pytest.fixture
def dni(times):
    return pd.Series([0., 1000.], index=times)


@pytest.fixture
def dhi(times):
    return pd.Series([0., 100.], index=times)


def test_aoi_func_factory(modeling_parameters, apparent_zenith, azimuth):
    aoi_func = pvmodel.aoi_func_factory(modeling_parameters)
    assert isinstance(aoi_func, partial)
    out = aoi_func(apparent_zenith, azimuth)
    index = apparent_zenith.index
    expected_fixed = (
        30,
        180,
        pd.Series([166.69070004,  32.70998989], index=index)
    )
    expected_tracking = (
        pd.Series([None, 2.15070219], index=index),
        pd.Series([None, 90.], index=index),
        pd.Series([None, 62.68029182], index=index)
    )
    if isinstance(out[0], (float, int)):
        assert out[0] == expected_fixed[0]
        assert out[1] == expected_fixed[1]
        assert_series_equal(out[2], expected_fixed[2], check_names=False)
    else:
        for o, e in zip(out, expected_tracking):
            assert_series_equal(o, e, check_names=False)


def test_aoi_func_factory_fail():
    with pytest.raises(TypeError):
        pvmodel.aoi_func_factory(None)


@pytest.fixture
def aoi_func(modeling_parameters):
    return pvmodel.aoi_func_factory(modeling_parameters)


def test_calculate_poa_effective(aoi_func, apparent_zenith, azimuth,
                                 ghi, dni, dhi):
    out = pvmodel.calculate_poa_effective(
        aoi_func, apparent_zenith, azimuth, ghi, dni, dhi)
    index = apparent_zenith.index
    expected_fixed = pd.Series([0., 1161.768016], index=index)
    expected_tracking = pd.Series([None, 1030.2303106183322], index=index)
    try:
        assert_series_equal(expected_fixed, out)
    except AssertionError:
        assert_series_equal(expected_tracking, out)


def test_calculate_power():
    assert False


@pytest.fixture
def temp_air(times):
    return pd.Series([0., 40.], index=times)


@pytest.fixture
def wind_speed(times):
    return pd.Series([0., 10.], index=times)


def test_irradiance_to_power(modeling_parameters, apparent_zenith, azimuth,
                             ghi, dni, dhi, temp_air, wind_speed):
    out = pvmodel.irradiance_to_power(
        modeling_parameters, apparent_zenith, azimuth, ghi, dni, dhi,
        temp_air=temp_air, wind_speed=wind_speed)
    index = apparent_zenith.index
    expected_fixed = pd.Series([None, 0.0029317797254337485], index=index)
    expected_tracking = pd.Series([None, 0.003], index=index)
    try:
        assert_series_equal(expected_fixed, out)
    except AssertionError:
        assert_series_equal(expected_tracking, out)
