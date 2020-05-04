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


@pytest.fixture
def poa_effective(times):
    return pd.Series([0., 1000.], index=times)


@pytest.fixture
def temp_air(times):
    return pd.Series([0., 40.], index=times)


@pytest.fixture
def wind_speed(times):
    return pd.Series([0., 10.], index=times)


def test_clearsky(golden_mst, apparent_zenith):
    latitude, longitude = golden_mst.latitude, golden_mst.longitude
    elevation = golden_mst.altitude
    out = pvmodel.calculate_clearsky(
        latitude, longitude, elevation, apparent_zenith)
    expected = pd.DataFrame(
        [[  0.        ,   0.        ,   0.        ],  # noqa
         [499.27823691, 963.70500214,  57.2923921 ]], # noqa
        columns=['ghi', 'dni', 'dhi'],
        index=apparent_zenith.index)
    assert_frame_equal(expected, out)


def fixed_or_tracking(system_type, expected_fixed, expected_tracking, out):
    if system_type == 'fixed':
        assert_series_equal(expected_fixed, out)
    elif system_type == 'tracking':
        assert_series_equal(expected_tracking, out)
    else:
        raise ValueError('system_type must be fixed or tracking')


def test_aoi_func_factory(modeling_parameters_system_type, apparent_zenith,
                          azimuth):
    modeling_parameters, system_type = modeling_parameters_system_type
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
    if system_type == 'fixed':
        assert out[0] == expected_fixed[0]
        assert out[1] == expected_fixed[1]
        assert_series_equal(out[2], expected_fixed[2], check_names=False)
    elif system_type == 'tracking':
        for o, e in zip(out, expected_tracking):
            assert_series_equal(o, e, check_names=False)
    else:
        raise Exception


def test_aoi_func_factory_fail():
    with pytest.raises(TypeError):
        pvmodel.aoi_func_factory(None)


@pytest.fixture
def aoi_func_system_type(modeling_parameters_system_type):
    modeling_parameters, system_type = modeling_parameters_system_type
    return pvmodel.aoi_func_factory(modeling_parameters), system_type


def test_calculate_poa_effective(aoi_func_system_type, apparent_zenith,
                                 azimuth, ghi, dni, dhi):
    aoi_func, system_type = aoi_func_system_type
    out = pvmodel.calculate_poa_effective(
        aoi_func, apparent_zenith, azimuth, ghi, dni, dhi)
    index = apparent_zenith.index
    expected_fixed = pd.Series([0., 1161.768016], index=index)
    expected_tracking = pd.Series([0., 1030.2303106183322], index=index)
    fixed_or_tracking(system_type, expected_fixed, expected_tracking, out)


@pytest.mark.parametrize(
    'dc_capacity,temperature_coefficient,dc_loss_factor,'
    'ac_capacity,ac_loss_factor,expected', (
        (10, -0.5, 0, 10, 0, (0., 7.58301313763377)),
        (5, 0.5, 0, 10, 0, (0., 5.)),  # pvwatts_ac max out is pdc0
        (10, 0.000, 20, 10, 20, (0., 6.1789760000000005)),
        (10, 0.000, -20, 10, -20, (0., 12.)),
        (15, -0.5, 0, 10, 0, (0., 10.)),
    ),
    ids=[
        '10:10 DC:AC, neg temp co, no loss',
        '5:10 DC:AC, pos temp co, no loss',
        '10:10 DC:AC, no temp co, losses',
        '10:10 DC:AC, no temp co, enhancing losses',
        '15:10 DC:AC, neg temp co, no loss',
    ])
def test_calculate_power(dc_capacity, temperature_coefficient, dc_loss_factor,
                         ac_capacity, ac_loss_factor, poa_effective, temp_air,
                         wind_speed, expected):
    out = pvmodel.calculate_power(
        dc_capacity, temperature_coefficient, dc_loss_factor,
        ac_capacity, ac_loss_factor, poa_effective, temp_air,
        wind_speed)
    assert_series_equal(pd.Series(expected, index=poa_effective.index), out)


def test_irradiance_to_power(modeling_parameters_system_type, apparent_zenith,
                             azimuth, ghi, dni, dhi, temp_air, wind_speed):
    modeling_parameters, system_type = modeling_parameters_system_type
    out = pvmodel.irradiance_to_power(
        modeling_parameters, apparent_zenith, azimuth, ghi, dni, dhi,
        temp_air=temp_air, wind_speed=wind_speed)
    index = apparent_zenith.index
    expected_fixed = pd.Series([0., 0.003], index=index)
    expected_tracking = pd.Series([0., 0.00293178], index=index)
    fixed_or_tracking(system_type, expected_fixed, expected_tracking, out)
