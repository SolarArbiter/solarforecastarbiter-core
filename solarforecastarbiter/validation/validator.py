# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:08:20 2019

@author: cwhanse
"""

from pvlib.tools import cosd
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


def _apply_limit(val, lb=None, ub=None, lb_ge=False, ub_le=False):
    if (lb is not None) & (ub is not None):
        if lb_ge and ub_le:
            return (val >= lb) & (val <= ub)
        elif lb_ge and not ub_le:
            return (val >= lb) & (val < ub)
        elif not lb_ge and ub_le:
            return (val > lb) & (val <= ub)
        else:
            return (val > lb) & (val < ub)
    elif lb is not None:
        if lb_ge:
            return (val >= lb)
        else:
            return (val > lb)
    elif ub is not None:
        if ub_le:
            return (val <= ub)
        else:
            return (val < ub)
    else:
        pass


def _QCRad_ub(dni_extra, sza, coeff):
    return coeff['mult'] * dni_extra * cosd(sza)**coeff['exp'] + coeff['min']


def check_irradiance_limits_QCRad(irrad, test_dhi=False, test_dni=False,
                                  coeff=QCRAD_LIMITS):
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
    coeff : dict, default QCRAD_LIMITS
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

    flags = pd.DataFrame(index=irrad.index, data=None,
                         columns=['ghi_physical_limit_flag'])

    try:
        ghi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           coeff['ghi_ub'])
    except KeyError:
        raise KeyError('Requires solar_zenith and dni_extra')

    try:
        flags['ghi_physical_limit_flag'] = _apply_limit(irrad['ghi'],
                                                        coeff['ghi_lb'],
                                                        ghi_ub)
    except KeyError:
        raise KeyError('ghi not found')

    if test_dhi:
        try:
            dhi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                               coeff['dhi_ub'])
            flags['dhi_physical_limit_flag'] = _apply_limit(irrad['dhi'],
                                                            coeff['dhi_lb'],
                                                            dhi_ub)
        except KeyError:
            raise KeyError('dhi not found')

    if test_dni:
        try:
            dni_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                               coeff['dni_ub'])
            flags['dni_physical_limit_flag'] = _apply_limit(irrad['dni'],
                                                            coeff['dni_lb'],
                                                            dni_ub)
        except KeyError:
            raise KeyError('dni not found')

    return flags


def _get_bounds(bounds):
    return (bounds['ghi_bounds'][0], bounds['ghi_bounds'][1],
            bounds['zenith_bounds'][0], bounds['zenith_bounds'][1],
            bounds['ratio_bounds'][0], bounds['ratio_bounds'][1])


def _check_irrad_ratio(ratio, ghi, sza, bounds):
    # unpack bounds dict
    ghi_lb, ghi_ub, sza_lb, sza_ub, ratio_lb, ratio_ub = _get_bounds(bounds)
    # for zenith set lb_ge to handle edge cases, e.g., zenith=0
    return ((_apply_limit(sza, lb=sza_lb, ub=sza_ub, lb_ge=True)) &
            (_apply_limit(ghi, lb=ghi_lb, ub=ghi_ub)) &
            (_apply_limit(ratio, lb=ratio_lb, ub=ratio_ub)))


def check_irradiance_consistency_QCRad(irrad, param=QCRAD_CONSISTENCY):
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

    # sum of components
    try:
        component_sum = irrad['dni'] * cosd(irrad['solar_zenith']) + \
            irrad['dhi']
        ghi_ratio = irrad['ghi'] / component_sum
        dhi_ratio = irrad['dhi'] / irrad['ghi']
    except KeyError:
        raise KeyError('Requires ghi, dhi, dni and solar_zenith')

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


def check_temperature_limits(weather, temp_limits=[-10., 50.]):
    """ Checks for extreme temperatures.

    Parameters:
    -----------
    weather : DataFrame
        temp_air : float
            Air temperature in Celsius
    temp_limits : list, default [-10, 50]
        (lower bound, upper bound) for temperature.

    Returns:
    --------
    flags : DataFrame
        True if temp_air > lower bound and temp_air < upper bound.
    """
    try:
        temp_air = weather['temp_air']
    except KeyError:
        raise KeyError('temp_air not found')

    flags = pd.DataFrame(index=weather.index, data=None,
                         columns=['extreme_temp_flag'])
    flags['extreme_temp_flag'] = _apply_limit(temp_air, lb=temp_limits[0],
                                              ub=temp_limits[1])
    return flags


def check_wind_limits(weather, wind_limits=[0., 60.]):
    """ Checks for extreme wind speeds.

    Parameters:
    -----------
    weather : DataFrame
        wind_speed : float
            Wind speed m/s
    wind_limits : list, default [0, 60]
        (lower bound, upper bound) for wind speed.

    Returns:
    --------
    flags : DataFrame
        True if wind_speed > lower bound and wind_speed < upper bound.
    """
    try:
        wind_speed = weather['wind_speed']
    except KeyError:
        raise KeyError('wind_speed not found')

    flags = pd.DataFrame(index=weather.index, data=None,
                         columns=['extreme_wind_flag'])
    flags['extreme_wind_flag'] = _apply_limit(wind_speed, lb=wind_limits[0],
                                              ub=wind_limits[1])
    return flags
