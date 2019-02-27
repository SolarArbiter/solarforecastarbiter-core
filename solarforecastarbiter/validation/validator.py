# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:08:20 2019

@author: cwhanse
"""

from pvlib.tools import cosd
import numpy as np

QCRAD_LIMITS = {'ghi_ub': {'mult': 1.5, 'exp': 1.2, 'min': 100},
                'dhi_ub': {'mult': 0.95, 'exp': 1.2, 'min': 50},
                'dni_ub': {'mult': 1.0, 'exp': 0.0, 'min': 0},
                'ghi_lb': -4, 'dhi_lb': -4, 'dni_lb': -4}

QCRAD_CONSISTENCY = {'ghi_ratio': {'high_zenith':
                                      {'zenith_bounds': [0.0, 75],
                                       'ghi_bounds': [50, np.Inf],
                                       'ratio_bounds': [0.85, 1.15]},
                                   'low_zenith':
                                       {'zenith_bounds': [75, 93],
                                        'ghi_bounds': [50, np.Inf],
                                        'ratio_bounds': [0.92, 1.08]}},
                    'dhi_ratio': {'high_zenith':
                                      {'zenith_bounds': [0.0, 75],
                                       'ghi_bounds': [50, np.Inf],
                                       'ratio_bounds': [0.0, 1.10]},
                                   'low_zenith':
                                       {'zenith_bounds': [75, 93],
                                        'ghi_bounds': [50, np.Inf],
                                        'ratio_bounds': [0.0, 1.05]}}}


def _apply_limit(val, lb=None, ub=None):
    if (lb is not None) & (ub is not None):
        return (val > lb) & (val < ub)
    elif lb is not None:
        return (val > lb)
    elif ub is not None:
        return (val < ub)
    else:
        pass


def _QCRad_ub(dni_extra, sza, coeff):
    return coeff['mult'] * dni_extra * cosd(sza)**coeff['exp'] + coeff['min']


def check_irradiance_limits_QCRad(irrad, coeff=QCRAD_LIMITS):
    """
    Applies limits using the QCRad criteria.

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
    coeff : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns:
    --------
    irrad : DataFrame
        With two columns added for each irradiance value:
        physical_limit_flag : boolean
            True if value passes physically-possible test
        climate_limit_flag : boolean
            True if value passes climatological test
    """

    try:
        ghi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           coeff['ghi_ub'])
    except KeyError:
        raise KeyError('Requires solar_zenith and dni_extra')

    try:
        irrad['ghi_physical_limit_flag'] = _apply_limit(irrad['ghi'],
             coeff['ghi_lb'], ghi_ub)
    except KeyError:
        raise KeyError('ghi not found')

    try:
        dhi_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           coeff['dhi_ub'])
        irrad['dhi_physical_limit_flag'] = _apply_limit(irrad['dhi'],
             coeff['dhi_lb'], dhi_ub)
    except KeyError:
        pass

    try:
        dni_ub = _QCRad_ub(irrad['dni_extra'], irrad['solar_zenith'],
                           coeff['dni_ub'])
        irrad['dni_physical_limit_flag'] = _apply_limit(irrad['dni'],
             coeff['dni_lb'], dni_ub)
    except KeyError:
        pass

    return irrad


def _get_bounds(bounds):
    return (bounds['ghi_bounds'][0], bounds['ghi_bounds'][1],
            bounds['zenith_bounds'][0], bounds['zenith_bounds'][1],
            bounds['ratio_bounds'][0], bounds['ratio_bounds'][1])


def _check_irrad_ratio(ratio, ghi, sza, bounds):
    # unpack bounds dict
    ghi_lb, ghi_ub, sza_lb, sza_ub, ratio_lb, ratio_ub = _get_bounds(bounds)
    return ((_apply_limit(sza, lb=sza_lb, ub=sza_ub)) &
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
        irrad : DataFrame
        With two columns added for each irradiance value:
        consistent_components : boolean
            True if ghi, dhi and dni components are consistent.
        diffuse_ratio_limit : boolean
            True if diffuse to ghi ratio passes limit test.
    """

    # sum of components
    try:
        component_sum = irrad['dni'] * cosd(irrad['solar_zenith']) + irrad['dhi']
        ghi_ratio = irrad['ghi'] / component_sum
        dhi_ratio = irrad['dhi'] / irrad['ghi']
    except KeyError:
        raise KeyError('Requires ghi, dhi, dni and solar_zenith')

    bounds = param['ghi_ratio']
    irrad['consistent_components'] = (
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=irrad['solar_zenith'],
                           bounds=bounds['high_zenith']) &
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=irrad['solar_zenith'],
                           bounds=bounds['low_zenith']))

    bounds = param['dhi_ratio']
    irrad['diffuse_ratio_limit'] = (
        _check_irrad_ratio(ratio=dhi_ratio, ghi=irrad['ghi'],
                           sza=irrad['solar_zenith'],
                           bounds=bounds['high_zenith']) &
        _check_irrad_ratio(ratio=dhi_ratio, ghi=irrad['ghi'],
                           sza=irrad['solar_zenith'],
                           bounds=bounds['low_zenith']))
    return irrad

