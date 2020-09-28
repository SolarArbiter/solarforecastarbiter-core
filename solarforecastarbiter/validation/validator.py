"""
Created on Fri Feb 15 14:08:20 2019

@author: cwhanse
"""
import warnings


import numpy as np
import pandas as pd
from pvlib.tools import cosd
from pvlib.irradiance import clearsky_index
from pvlib.clearsky import detect_clearsky as _detect_clearsky


from solarforecastarbiter.validation.quality_mapping import mask_flags


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

DAY_NIGHT_MAX_ZENITH = 87.


def _check_limits(val, lb=None, ub=None, lb_ge=False, ub_le=False):
    """ Returns True where lb < (or <=) val < (or <=) ub
    """
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
    cosd_sza = cosd(sza)
    cosd_sza[cosd_sza < 0] = 0
    return lim['mult'] * dni_extra * cosd_sza**lim['exp'] + lim['min']


@mask_flags('LIMITS EXCEEDED')
def check_ghi_limits_QCRad(ghi, solar_zenith, dni_extra, limits=None):
    """
    Tests for physical limits on GHI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    ghi_limit_flag : Series
        True if value passes physically-possible test
    """
    if not limits:
        limits = QCRAD_LIMITS
    ghi_ub = _QCRad_ub(dni_extra, solar_zenith, limits['ghi_ub'])

    ghi_limit_flag = _check_limits(ghi, limits['ghi_lb'], ghi_ub)
    ghi_limit_flag.name = 'ghi_limit_flag'

    return ghi_limit_flag


@mask_flags('LIMITS EXCEEDED')
def check_dhi_limits_QCRad(dhi, solar_zenith, dni_extra, limits=None):
    """
    Tests for physical limits on DHI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    dhi : Series
        Diffuse horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    dhi_limit_flag : Series
        True if value passes physically-possible test
    """
    if not limits:
        limits = QCRAD_LIMITS

    dhi_ub = _QCRad_ub(dni_extra, solar_zenith, limits['dhi_ub'])

    dhi_limit_flag = _check_limits(dhi, limits['dhi_lb'], dhi_ub)
    dhi_limit_flag.name = 'dhi_limit_flag'

    return dhi_limit_flag


@mask_flags('LIMITS EXCEEDED')
def check_dni_limits_QCRad(dni, solar_zenith, dni_extra, limits=None):
    """
    Tests for physical limits on DNI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    dni : Series
        Direct normal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    dni_limit_flag : Series
        True if value passes physically-possible test
    """
    if not limits:
        limits = QCRAD_LIMITS

    dni_ub = _QCRad_ub(dni_extra, solar_zenith, limits['dni_ub'])

    dni_limit_flag = _check_limits(dni, limits['dni_lb'], dni_ub)
    dni_limit_flag.name = 'dni_limit_flag'

    return dni_limit_flag


@mask_flags('LIMITS EXCEEDED')
def check_irradiance_limits_QCRad(solar_zenith, dni_extra, ghi=None, dhi=None,
                                  dni=None, limits=None):
    """
    Tests for physical limits on GHI, DHI or DNI using the QCRad criteria.

    Test passes if a value > lower bound and value < upper bound. Lower bounds
    are constant for all tests. Upper bounds are calculated as

    .. math::
        ub = min + mult * dni_extra * cos( solar_zenith)^exp

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    ghi : Series or None, default None
        Global horizontal irradiance in W/m^2
    dhi : Series or None, default None
        Diffuse horizontal irradiance in W/m^2
    dni : Series or None, default None
        Direct normal irradiance in W/m^2
    limits : dict, default QCRAD_LIMITS
        for keys 'ghi_ub', 'dhi_ub', 'dni_ub', value is a dict with
        keys {'mult', 'exp', 'min'}. For keys 'ghi_lb', 'dhi_lb', 'dni_lb',
        value is a float.

    Returns
    -------
    ghi_limit_flag : Series or None, default None
        True if value passes physically-possible test
    dhi_limit_flag : Series or None, default None
    dhi_limit_flag : Series or None, default None

    References
    ----------
    [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.
    """
    if not limits:
        limits = QCRAD_LIMITS

    if ghi is not None:
        ghi_limit_flag = check_ghi_limits_QCRad(ghi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        ghi_limit_flag = None

    if dhi is not None:
        dhi_limit_flag = check_dhi_limits_QCRad(dhi, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dhi_limit_flag = None

    if dni is not None:
        dni_limit_flag = check_dni_limits_QCRad(dni, solar_zenith, dni_extra,
                                                limits=limits)
    else:
        dni_limit_flag = None

    return ghi_limit_flag, dhi_limit_flag, dni_limit_flag


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


@mask_flags('INCONSISTENT IRRADIANCE COMPONENTS')
def check_irradiance_consistency_QCRad(ghi, solar_zenith, dni_extra, dhi, dni,
                                       param=None):
    """
    Checks consistency of GHI, DHI and DNI. Not valid for night time.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2
    solar_zenith : Series
        Solar zenith angle in degrees
    dni_extra : Series
        Extraterrestrial normal irradiance in W/m^2
    dhi : Series
        Diffuse horizontal irradiance in W/m^2
    dni : Series
        Direct normal irradiance in W/m^2
    param : dict
        keys are 'ghi_ratio' and 'dhi_ratio'. For each key, value is a dict
        with keys 'high_zenith' and 'low_zenith'; for each of these keys,
        value is a dict with keys 'zenith_bounds', 'ghi_bounds', and
        'ratio_bounds' and value is an ordered pair [lower, upper]
        of float.

    Returns
    -------
    consistent_components : Series
        True if ghi, dhi and dni components are consistent.
    diffuse_ratio_limit : Series
        True if diffuse to ghi ratio passes limit test.

    References
    ----------
    [1] C. N. Long and Y. Shi, An Automated Quality Assessment and Control
        Algorithm for Surface Radiation Measurements, The Open Atmospheric
        Science Journal 2, pp. 23-37, 2008.
    """

    if not param:
        param = QCRAD_CONSISTENCY

    # sum of components
    component_sum = dni * cosd(solar_zenith) + dhi
    ghi_ratio = ghi / component_sum
    dhi_ratio = dhi / ghi

    bounds = param['ghi_ratio']
    consistent_components = (
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=solar_zenith, bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=ghi_ratio, ghi=component_sum,
                           sza=solar_zenith, bounds=bounds['low_zenith']))
    consistent_components.name = 'consistent_components'

    bounds = param['dhi_ratio']
    diffuse_ratio_limit = (
        _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                           bounds=bounds['high_zenith']) |
        _check_irrad_ratio(ratio=dhi_ratio, ghi=ghi, sza=solar_zenith,
                           bounds=bounds['low_zenith']))
    diffuse_ratio_limit.name = 'diffuse_ratio_limit'

    return consistent_components, diffuse_ratio_limit


@mask_flags('LIMITS EXCEEDED')
def check_temperature_limits(temp_air, temp_limits=(-35., 50.)):
    """ Checks for extreme temperatures.

    Parameters
    ----------
    temp_air : Series
        Air temperature in Celsius
    temp_limits : tuple, default (-35, 50)
        (lower bound, upper bound) for temperature.

    Returns
    -------
    extreme_temp_flag : Series
        True if temp_air > lower bound and temp_air < upper bound.
    """
    extreme_temp_flag = _check_limits(temp_air, lb=temp_limits[0],
                                      ub=temp_limits[1])
    extreme_temp_flag.name = 'extreme_temp_flag'
    return extreme_temp_flag


@mask_flags('LIMITS EXCEEDED')
def check_wind_limits(wind_speed, wind_limits=(0., 50.)):
    """ Checks for extreme wind speeds.

    Parameters
    ----------
    wind_speed : Series
        Wind speed in m/s
    wind_limits : tuple, default (0, 50)
        (lower bound, upper bound) for wind speed.

    Returns
    -------
    extreme_wind_flag : Series
        True if wind_speed > lower bound and wind_speed < upper bound.
    """
    extreme_wind_flag = _check_limits(wind_speed, lb=wind_limits[0],
                                      ub=wind_limits[1],
                                      lb_ge=True)
    extreme_wind_flag.name = 'extreme_wind_flag'
    return extreme_wind_flag


@mask_flags('LIMITS EXCEEDED')
def check_rh_limits(rh, rh_limits=(0, 100)):
    """ Checks for extreme relative humidity.

    Parameters
    ----------
    rh : Series
        Relative humidity in %
    rh_limits : tuple, default (0, 100)
        (lower bound, upper bound) for relative humidity

    Returns
    -------
    flags : Series
        True if rh >= lower bound and rh <= upper bound.
    """
    flags = _check_limits(rh, lb=rh_limits[0], ub=rh_limits[1], lb_ge=True,
                          ub_le=True)
    return flags


@mask_flags('LIMITS EXCEEDED')
def check_ac_power_limits(power, day_night, capacity,
                          capacity_limit_low=-0.05,
                          capacity_limit_high_day=1.05,
                          capacity_limit_high_night=0.05):
    """Check for extreme AC power.

    Parameters
    ----------
    power : Series
        DC or AC power.
    day_night : Series
        True for day time points, False for night time points.
    capacity : float
        AC capacity.
    capacity_limit_low : float
        Lower bound in fraction of capacity.
    capacity_limit_high_day : float
        Upper bound in fraction of capacity for day time.
    capacity_limit_high_night : float
        Upper bound in fraction of capacity for night time.

    Returns
    -------
    flags : Series
        True for values that are within the limits:
          * power > capacity * capacity_limit_low, AND
            * power < capacity * capacity_limit_high_day AND day_night, OR
            * power < capacity * capacity_limit_high_night AND NOT day_night
    """
    flags = _check_power_limits(
        power, day_night, capacity, capacity_limit_low,
        capacity_limit_high_day, capacity_limit_high_night
        )

    return flags


@mask_flags('LIMITS EXCEEDED')
def check_dc_power_limits(power, day_night, capacity,
                          capacity_limit_low=-0.05,
                          capacity_limit_high_day=1.20,
                          capacity_limit_high_night=0.05):
    """Check for extreme AC power.

    Parameters
    ----------
    power : Series
        DC or AC power.
    day_night : Series
        True for day time points, False for night time points.
    capacity : float
        DC capacity.
    capacity_limit_low : float
        Lower bound in fraction of capacity.
    capacity_limit_high_day : float
        Upper bound in fraction of capacity for day time.
    capacity_limit_high_night : float
        Upper bound in fraction of capacity for night time.

    Returns
    -------
    flags : Series
        True for values that are within the limits:
          * power > capacity * capacity_limit_low, AND
            * power < capacity * capacity_limit_high_day AND day_night, OR
            * power < capacity * capacity_limit_high_night AND NOT day_night
    """
    flags = _check_power_limits(
        power, day_night, capacity, capacity_limit_low,
        capacity_limit_high_day, capacity_limit_high_night
        )

    return flags


def _check_power_limits(
        power, day_night, capacity, capacity_limit_low,
        capacity_limit_high_day, capacity_limit_high_night
        ):

    # convert fractions to absolute values
    capacity_low = capacity * capacity_limit_low
    capacity_high_day = capacity * capacity_limit_high_day
    capacity_high_night = capacity * capacity_limit_high_night

    flag_low = _check_limits(power, lb=capacity_low)
    flag_high_day = _check_limits(power, ub=capacity_high_day) & day_night
    flag_high_night = _check_limits(power, ub=capacity_high_night) & ~day_night

    # composite constructed such that True values within limits for day
    # or night. False values exceed any limit.
    flags = flag_low & (flag_high_day | flag_high_night)
    return flags


@mask_flags('CLEARSKY EXCEEDED')
def check_ghi_clearsky(ghi, ghi_clearsky, kt_max=1.1):
    """
    Flags GHI values greater than clearsky values.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2
    ghi_clearsky : Series
         Global horizontal irradiance in W/m^2 under clear sky conditions
    kt_max : float
        maximum clearness index that defines when ghi exceeds clear-sky value.

    Returns
    -------
    flags : Series
        True if ghi is less than or equal to clear sky value.
    """
    kt = clearsky_index(ghi, ghi_clearsky, max_clearsky_index=np.Inf)
    flags = _check_limits(kt, ub=kt_max, ub_le=True)
    return flags


@mask_flags('CLEARSKY EXCEEDED')
def check_poa_clearsky(poa_global, poa_clearsky, kt_max=1.1):
    """
    Flags plane of array irradiance values greater than clearsky values.

    Parameters
    ----------
    poa_global : Series
        Plane of array irradiance in W/m^2
    poa_clearsky : Series
        Plane of array irradiance under clear sky conditions, in W/m^2
    kt_max : float
        maximum allowed ratio of poa_global to poa_clearsky

    Returns
    -------
    flags : Series
        True if poa_global is less than or equal to clear sky value.
    """
    kt = clearsky_index(poa_global, poa_clearsky,
                        max_clearsky_index=np.Inf)
    flags = _check_limits(kt, ub=kt_max, ub_le=True)
    return flags


@mask_flags('NIGHTTIME')
def check_day_night(solar_zenith, max_zenith=DAY_NIGHT_MAX_ZENITH):
    """Check for day/night periods based on solar zenith.

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees.
    max_zenith : float
        Maximum zenith angle for a daylight time.

    Returns
    -------
    daytime : Series
        True when solar zenith is less than max_zenith.
    """
    # True = daytime. False = nighttime.
    daytime = _check_limits(solar_zenith, ub=max_zenith)
    return daytime


@mask_flags('NIGHTTIME')
def check_day_night_interval(
        solar_zenith, closed, interval_length,
        solar_zenith_interval_length=None, fraction_of_interval=0.1,
        max_zenith=DAY_NIGHT_MAX_ZENITH):
    """Check for day/night periods based on solar zenith.

    Interval average data may be analyzed by supplying higher resolution
    solar zenith data and parameters that describe the desired intervals.

    Parameters
    ----------
    solar_zenith : Series
        Solar zenith angle in degrees.
    closed : {'left', 'right'}
        None for instantaneous data, 'left' for interval beginning labeled
        data, 'right' for interval ending labeled data.
    interval_length : Timedelta
        Interval length to resample day/night periods to.
    solar_zenith_interval_length : None or Timedelta
        If None, attempt to infer.
        Required if solar_zenith contains gaps.
    fraction_of_interval : float
        The fraction of the points in an interval that must be daytime
        in order to mark the interval as daytime.
    max_zenith : float
        Maximum zenith angle for a daylight time.

    Returns
    -------
    daytime : Series
        True when sufficient points within an interval are less than
        max_zenith. Index conforms to solar_zenith resampled to
        interval_length.

    Raises
    ------
    ValueError
        If solar_zenith contains gaps and solar_zenith_interval_length is not
        provided.
    """
    # True = daytime. False = nighttime.
    daytime = _check_limits(solar_zenith, ub=max_zenith)
    # number of daytime minutes within the interval
    daytime_sum = daytime.resample(
        interval_length, closed=closed, label=closed
    ).sum()
    # if not provided, try to interval length for normalization.
    # this will raise if the index has gaps.
    if solar_zenith_interval_length is None:
        solar_zenith_interval_length = pd.infer_freq(
            solar_zenith.index, warn=False)
        if solar_zenith_interval_length is None:
            raise ValueError(
                'solar_zenith.index contains gaps so the freq could not be '
                'inferred and solar_zenith_interval_length was not provided. '
                'Fill the gaps or pass solar_zenith_interval_length.')
    # If points corresponding to fraction_of_interval is daytime,
    # then the interval is daytime
    count_threshold = int(
        interval_length * fraction_of_interval / solar_zenith_interval_length
    )
    daytime = daytime_sum > count_threshold
    return daytime


@mask_flags('UNEVEN FREQUENCY')
def check_timestamp_spacing(times, freq):
    """ Checks if spacing between times conforms to freq.

    Parameters
    ----------
    times : DatetimeIndex
    freq : string or Timedelta
        Expected frequency of times

    Returns
    -------
    flags : Series
        True when the difference between one time and the time before
        conforms to freq
    """
    if not isinstance(freq, pd.Timedelta):
        freq = pd.Timedelta(freq)
    # first value is NaT, rest are timedeltas
    delta = times.to_series().diff()
    flags = delta == freq
    flags.iloc[0] = True
    return flags


def _all_close_to_first(x, rtol=1e-5, atol=1e-8):
    """ Returns True if all values in x are close to x[0].

    Parameters
    ----------
    x : array
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Returns
    -------
    Boolean
    """
    return np.allclose(a=x, b=x[0], rtol=rtol, atol=atol)


@mask_flags('STALE VALUES', invert=False)
def detect_stale_values(x, window=6, rtol=1e-5, atol=1e-8):
    """ Detects stale data.

    For a window of length N, the last value (index N-1) is considered stale
    if all values in the window are close to the first value (index 0).

    Parameters
    ----------
    x : Series
        data to be processed
    window : int, default 6
        number of consecutive values which, if unchanged, indicates stale data
    rtol : float, default 1e-5
        relative tolerance for detecting a change in data values
    atol : float, default 1e-8
        absolute tolerance for detecting a change in data values

    Parameters rtol and atol have the same meaning as in numpy.allclose

    Returns
    -------
    flags : Series
        True if the value is part of a stale sequence of data

    Raises
    ------
        ValueError if window < 2
    """
    if window < 2:
        raise ValueError(f'window set to {window}, must be at least 2')

    flags = x.rolling(window=window).apply(
        _all_close_to_first, raw=True, kwargs={'rtol': rtol, 'atol': atol}
        ).fillna(False).astype(bool)
    return flags


def stale_interpolated_window(interval_length):
    """Returns the recommended window size for detect stale and
    interpolation functions"""
    if interval_length < pd.Timedelta('1h'):
        return 6
    else:
        return 3


@mask_flags('INTERPOLATED VALUES', invert=False)
def detect_interpolation(x, window=6, rtol=1e-5, atol=1e-8):
    """ Detects sequences of data which appear linear.

    Sequences are linear if the first difference appears to be constant.
    For a window of length N, the last value (index N-1) is flagged
    if all values in the window appear to be a line segment.

    Parameters
    ----------
    x : series
        data to be processed
    window : int, default 6
        number of sequential values that, if the first difference is constant,
        are classified as a linear sequence
    rtol : float, default 1e-5
        tolerance relative to max(abs(x.diff()) for detecting a change
    atol : float, default 1e-8
        absolute tolerance for detecting a change in first difference

    Returns
    -------
    flags : Series
        True if the value is part of a linear sequence

    Raises
    ------
        ValueError if window < 3
    """
    if window < 3:
        raise ValueError(f'window set to {window}, must be at least 3')

    # reduce window by 1 because we're passing the first difference
    flags = detect_stale_values(x.diff(periods=1), window=window-1, rtol=rtol,
                                atol=atol)
    return flags


def detect_levels(x, count=3, num_bins=100):
    """ Detects plateau levels in data.

    Parameters
    ----------
    x : Series
        data to be processed
    count : int
        number of plateaus to return
    num_bins : int
        number of bins to use in histogram that finds plateau levels

    Returns
    -------
    levels : list of tuples
        (left, right) values of the interval in x with a detected plateau, in
        decreasing order of count of x values in the interval. List length is
        given by the kwarg count
    """
    nonan = x[~np.isnan(x)]
    hist, bin_edges = np.histogram(nonan, bins=num_bins, density=True)
    level_index = np.argsort(hist * -1)  # decreasing order
    levels = [(bin_edges[i], bin_edges[i + 1]) for i in level_index[:count]]
    return levels, bin_edges


def _label_clipping(x, window, frac):
    """ Returns Series with True at the end of each window with
    sum(x(window)) >= window * frac.
    """
    tmp = x.rolling(window).sum()
    y = (tmp >= window * frac) & x.astype(bool)
    return y


@mask_flags('CLIPPED VALUES', invert=False)
def detect_clipping(ac_power, window=4, fraction_in_window=0.75, rtol=5e-3,
                    levels=2):
    """ Detects clipping in a series of AC power.

    Possible clipped power levels are found by detect_levels. Within each
    sliding window, clipping is indicated when at least fraction_in_window
    of points are close to a clipped power level.

    Parameters
    ----------
    ac_power : Series
        data to be processed

    window : int
        number of data points defining the length of a rolling window

    fraction_in_window : float
        fraction of points which indicate clipping if AC power at each point
        is close to the plateau level

    rtol : float
        a point is close to a clipped level M if
        abs(ac_power - M) < rtol * max(ac_power)

    levels : int
        number of clipped power levels to consider.

    Returns
    -------
    flags : Series
        True when clipping is indicated.
    """
    num_bins = np.ceil(1.0 / rtol).astype(int)
    flags = pd.Series(index=ac_power.index, data=False)
    power_plateaus, bins = detect_levels(ac_power, count=levels,
                                         num_bins=num_bins)
    for lower, upper in power_plateaus:
        temp = pd.Series(index=ac_power.index, data=0.0)
        temp.loc[(ac_power >= lower) & (ac_power <= upper)] = 1.0
        flags = flags | _label_clipping(temp, window=window,
                                        frac=fraction_in_window)
    return flags


@mask_flags('CLEARSKY', invert=False)
def detect_clearsky_ghi(ghi, ghi_clearsky):
    """ Identifies times when GHI is consistent with clear sky conditions.

    Uses the function pvlib.clearsky.detect_clearsky. Assumes ghi data with
    regular (constant) time intervals which must be 15 minutes or less.

    Parameters
    ----------
    ghi : Series
        Global horizontal irradiance in W/m^2

    ghi_clearsky : Series
         Global horizontal irradiance in W/m^2 under clear sky conditions

    Returns
    -------
    flags : Series
        True when clear sky conditions are indicated.

    Notes
    -----
    Clear-sky conditions are inferred when each of six criteria are met; see
    `pvlib.clearsky.detect_clearsky` for references and details. Threshold
    values for each criterion were originally developed for ten minute windows
    containing one-minute data [1]. As indicated in [2], the algorithm also
    works for longer windows and data at different intervals, if threshold
    criteria are roughly scaled to the window length. Here, the threshold
    values are based on [1] with the scaling indicated in [2].

    Warns
    -----
    RuntimeWarning
        If `pvlib.clearsky.detect_clearsky` cannot be applied to the input.

    References
    ----------
    [1] Reno, M.J. and C.W. Hansen, "Identification of periods of clear
    sky irradiance in time series of GHI measurements" Renewable Energy,
    v90, p. 520-531, 2016.

    [2] B. H. Ellis, M. Deceglie and A. Jain, "Automatic Detection of
    Clear-Sky Periods From Irradiance Data," in IEEE Journal of Photovoltaics,
    vol. 9, no. 4, pp. 998-1005, July 2019. doi: 10.1109/JPHOTOV.2019.2914444
    """
    if len(ghi) < 2:
        warnings.warn(
            'At least two datapoints are required for detect_clearsky_ghi',
            RuntimeWarning
        )
        return pd.Series(0, index=ghi.index)
    # determine window length in minutes, 10 x interval for intervals <= 15m
    delta = ghi.index.to_series().diff()
    delta_minutes = delta[1] / np.timedelta64(1, '60s')
    deltas_same = (delta[1:] == delta[1]).all()
    if delta_minutes <= 15 and deltas_same:
        window_length = np.minimum(10*delta_minutes, 60.0)
        scale_factor = window_length / 10
        flags = _detect_clearsky(ghi, ghi_clearsky, ghi.index, window_length,
                                 lower_line_length=-5*scale_factor,
                                 upper_line_length=10*scale_factor,
                                 slope_dev=8*scale_factor)
        return flags
    else:
        warnings.warn(
            'detect_clearsky requires regular time intervals of 15m or less',
            RuntimeWarning
        )
        return pd.Series(0, index=ghi.index)
