# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:08:20 2019

@author: cwhanse
"""

from pvlib.tools import cosd
from pvlib.irradiance import clearsky_index
import numpy as np
import pandas as pd

QCRAD_LIMITS = {'ghi_ub': {'mult': 1.5, 'exp': 1.2, 'min': 100},
                'dhi_ub': {'mult': 0.95, 'exp': 1.2, 'min': 50},
                'dni_ub': {'mult': 1.0, 'exp': 0.0, 'min': 0},
                'ghi_lb': -4, 'dhi_lb': -4, 'dni_lb': -4}

QCRAD_CONSISTENCY = {
    'ghi_ratio':
        {'low_zenith':
            {'zenith_bounds': [0.0, 75], 'ghi_bounds': [50, np.Inf],
             'ratio_bounds': [0.92, 1.08]},
         'high_zenith':
            {'zenith_bounds': [75, 93], 'ghi_bounds': [50, np.Inf],
             'ratio_bounds': [0.85, 1.15]}},
    'dhi_ratio':
        {'low_zenith':
            {'zenith_bounds': [0.0, 75], 'ghi_bounds': [50, np.Inf],
             'ratio_bounds': [0.0, 1.05]},
         'high_zenith':
            {'zenith_bounds': [75, 93], 'ghi_bounds': [50, np.Inf],
             'ratio_bounds': [0.0, 1.10]}}}


def _check_limits(val, lb=None, ub=None, lb_ge=False, ub_le=False):
    if lb_ge:
        lb_op = np.greater_equal
    else:
        lb_op = np.greater
    if ub_le:
        ub_op = np.less_equal
    else:
        ub_op = np.less

    if (lb is not None) & (ub is not None):
        return lb_op(val, lb) & ub_op(val, ub)
    elif lb is not None:
        return lb_op(val, lb)
    elif ub is not None:
        return ub_op(val, ub)
    else:
        raise ValueError('must provide either upper or lower bound')


def _QCRad_ub(dni_extra, sza, lim):
    return lim['mult'] * dni_extra * cosd(sza)**lim['exp'] + lim['min']


def check_irradiance_limits_QCRad(irrad, test_dhi=False, test_dni=False,
                                  limits=None):
    """
    Tests for physical limits on GHI using the QCRad criteria.

    Also tests for physical limits on DHI and DNI if test_dhi and test_dni,
    respectively, are set and these data are present.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters:
    -----------
    irrad : DataFrame
        ghi : float
            Global horizontal irradiance in W/m^2
        solar_zenith : float
            Solar zenith angle in degrees
        dni_extra : float
            Extraterrestrial normal irradiance in W/m^2
        dhi : float, optional
            Diffuse horizontal irradiance in W/m^2
        dni : float, optional
            Direct normal irradiance in W/m^2
    test_dhi : boolean, default False
    test_dni : boolean, default False
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns:
    --------
    flags : DataFrame
        physical_limit_flag : boolean
            True if value passes physically-possible test
        climate_limit_flag : boolean
            True if value passes climatological test
    """

    if not limits:
        limits = QCRAD_LIMITS

    flags = pd.DataFrame(index=irrad.index, data=None,
                         columns=['ghi_physical_limit_flag'])

    ghi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                       limits['ghi_ub'])

    flags['ghi_physical_limit_flag'] = _check_limits(irrad['ghi'],
                                                     limits['ghi_lb'],
                                                     ghi_ub)

    if test_dhi:
        dhi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           limits['dhi_ub'])
        flags['dhi_physical_limit_flag'] = _check_limits(irrad['dhi'],
                                                         limits['dhi_lb'],
                                                         dhi_ub)

    if test_dni:
        dni_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           limits['dni_ub'])
        flags['dni_physical_limit_flag'] = _check_limits(irrad['dni'],
                                                         limits['dni_lb'],
                                                         dni_ub)

    return flags


def _get_bounds(bounds):
    return (bounds['ghi_bounds'][0], bounds['ghi_bounds'][1],
            bounds['zenith_bounds'][0], bounds['zenith_bounds'][1],
            bounds['ratio_bounds'][0], bounds['ratio_bounds'][1])


def _check_irrad_ratio(ratio, ghi, sza, bounds):
    # unpack bounds dict
    ghi_lb, ghi_ub, sza_lb, sza_ub, ratio_lb, ratio_ub = _get_bounds(bounds)
    # for zenith set lb_ge to handle edge cases, e.g., zenith=0
    return ((_check_limits(sza, lb=sza_lb, ub=sza_ub, lb_ge=True)) &
            (_check_limits(ghi, lb=ghi_lb, ub=ghi_ub)) &
            (_check_limits(ratio, lb=ratio_lb, ub=ratio_ub)))


def check_irradiance_consistency_QCRad(irrad, param=None):
    """
    Checks consistency of GHI, DHI and DNI.

    Parameters:
    -----------
    irrad : DataFrame
        ghi : float
            Global horizontal irradiance in W/m^2
        solar_zenith : float
            Solar zenith angle in degrees
        dni_extra : float
            Extraterrestrial normal irradiance in W/m^2
        dhi : float
            Diffuse horizontal irradiance in W/m^2
        dni : float
            Direct normal irradiance in W/m^2

    param : dict
        keys are 'ghi_ratio' and 'dhi_ratio'. For each key, value is a dict
        with keys 'high_zenith' and 'low_zenith'; for each of these keys,
        value is a dict with keys 'zenith_bounds', 'ghi_bounds', and
        'ratio_bounds' and value is an ordered pair [lower, upper]
        of float.

    Returns:
    --------
    flags : DataFrame
        consistent_components : boolean
            True if ghi, dhi and dni components are consistent.
        diffuse_ratio_limit : boolean
            True if diffuse to ghi ratio passes limit test.
    """

    if not param:
        param = QCRAD_CONSISTENCY

    # sum of components
    component_sum = irrad['dni'] * cosd(irrad['solar_zenith']) + \
        irrad['dhi']
    ghi_ratio = irrad['ghi'] / component_sum
    dhi_ratio = irrad['dhi'] / irrad['ghi']

    flags = pd.DataFrame(index=irrad.index, data=None,
                         columns=['consistent_components',
                                  'diffuse_ratio_limit'])

    bounds = param['ghi_ratio']
    flags['consistent_components'] = (
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=irrad['solar_zenith'],
                           bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=irrad['solar_zenith'],
                           bounds=bounds['low_zenith']))

    bounds = param['dhi_ratio']
    flags['diffuse_ratio_limit'] = (
        _check_irrad_ratio(ratio=dhi_ratio, ghi=irrad['ghi'],
                           sza=irrad['solar_zenith'],
                           bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=dhi_ratio, ghi=irrad['ghi'],
                           sza=irrad['solar_zenith'],
                           bounds=bounds['low_zenith']))
    return flags


def check_temperature_limits(weather, temp_limits=(-10., 50.)):
    """ Checks for extreme temperatures.

    Parameters:
    -----------
    weather : DataFrame
        temp_air : float
            Air temperature in Celsius
    temp_limits : tuple, default (-10, 50)
        (lower bound, upper bound) for temperature.

    Returns:
    --------
    flags : DataFrame
        True if temp_air > lower bound and temp_air < upper bound.
    """
    temp_air = weather['temp_air']

    flags = pd.DataFrame(index=weather.index, data=None,
                         columns=['extreme_temp_flag'])
    flags['extreme_temp_flag'] = _check_limits(temp_air, lb=temp_limits[0],
                                               ub=temp_limits[1])
    return flags


def check_wind_limits(weather, wind_limits=(0., 60.)):
    """ Checks for extreme wind speeds.

    Parameters:
    -----------
    weather : DataFrame
        wind_speed : float
            Wind speed m/s
    wind_limits : tuple, default (0, 60)
        (lower bound, upper bound) for wind speed.

    Returns:
    --------
    flags : DataFrame
        True if wind_speed > lower bound and wind_speed < upper bound.
    """
    wind_speed = weather['wind_speed']

    flags = pd.DataFrame(index=weather.index, data=None,
                         columns=['extreme_wind_flag'])
    flags['extreme_wind_flag'] = _check_limits(wind_speed, lb=wind_limits[0],
                                               ub=wind_limits[1])
    return flags


def get_solarposition(location, times, **kwargs):
    """ Calculates solar position.

    Wraps pvlib.location.Location.get_solarposition.

    Parameters:
    -----------
    location : pvlib.Location
    times : DatetimeIndex
    optional kwargs include
        pressure : float, Pa
        temperature : float, degrees C
        method : str, default 'nrel_numpy'
            Other values are 'nrel_c', 'nrel_numba', 'pyephem', and 'ephemeris'
    Other kwargs are passed to the underlying solar position function
    specified by method kwarg.

    Returns
    -------
    solar_position : DataFrame
        Columns depend on the ``method`` kwarg, but always include
        ``zenith`` and ``azimuth``.
    """
    return location.get_solarposition(times, **kwargs)


def get_clearsky(location, times, **kwargs):
    """ Calculates clear-sky GHI, DNI, DHI.

    Wraps pvlib.location.Location.get_clearsky.

    Parameters:
    -----------
    location : pvlib.Location
    times : DatetimeIndex
    optional kwargs include
        solar_position : None or DataFrame, default None
        dni_extra: None or numeric, default None
        model: str, default 'ineichen'
            Other values are 'haurwitz', 'simplified_solis'.
    Other kwargs are passed to the underlying solar position function
    specified by model kwarg.

    Returns
    -------
    clearsky : DataFrame
        Column names are: ``ghi, dni, dhi``.
    """
    return location.get_clearsky(times, **kwargs)


def check_ghi_clearsky(irrad, clearsky=None, location=None, kt_max=1.1):
    """
    Flags GHI values greater than clearsky values.

    If clearsky is not provided, a Location is required and clear-sky
    irradiance is calculated by pvlib.location.Location.get_clearsky.

    Parameters:
    -----------
    irrad : DataFrame
        ghi : float
            Global horizontal irradiance in W/m^2
    clearsky : DataFrame, default None
        ghi : float
            Global horizontal irradiance in W/m^2
    location : Location, default None
        instance of pvlib.location.Location
    kt_max : float
        maximum clearness index that defines when ghi exceeds clear-sky value.

    Returns:
    --------
    flags : DataFrame
        ghi_clearsky : boolean
            True if ghi is less than or equal to clear sky value.
    """
    times = irrad.index

    if clearsky is None and location is None:
        raise ValueError("Either clearsky or location is required")
    elif clearsky is None and location is not None:
        clearsky = get_clearsky(location, times)

    flags = pd.DataFrame(index=times, data=None, columns=['ghi_clearsky'])
    kt = clearsky_index(irrad['ghi'], clearsky['ghi'],
                        max_clearsky_index=np.Inf)
    flags['ghi_clearsky'] = _check_limits(kt, ub=kt_max, ub_le=True)
    return flags
