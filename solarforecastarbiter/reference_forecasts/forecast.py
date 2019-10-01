"""
Functions for forecasting.
"""
import datetime

import pandas as pd
import numpy as np

from solarforecastarbiter import pvmodel

# some functions originally implemented in pvlib-python.
# find pvlib-python license in licenses directory.


def cloud_cover_to_ghi_linear(cloud_cover, ghi_clear, offset=35):
    """
    Convert cloud cover to GHI using a linear relationship.

    0% cloud cover returns ghi_clear.

    100% cloud cover returns offset*ghi_clear.

    Parameters
    ----------
    cloud_cover: numeric
        Cloud cover in %.
    ghi_clear: numeric
        GHI under clear sky conditions.
    offset: numeric, default 35
        Determines the minimum GHI.

    Returns
    -------
    ghi: numeric
        Cloudy sky GHI.

    References
    ----------
    Larson et. al. "Day-ahead forecasting of solar power output from
    photovoltaic plants in the American Southwest" Renewable Energy
    91, 11-20 (2016).
    """

    offset = offset / 100.
    cloud_cover = cloud_cover / 100.
    ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
    return ghi


def cloud_cover_to_irradiance_ghi_clear(cloud_cover, ghi_clear, zenith):
    """
    Estimates irradiance from cloud cover in the following steps:

    1. Estimate cloudy sky GHI using a function of cloud_cover and
       ghi_clear: :py:func:`cloud_cover_to_ghi_linear`
    2. Estimate cloudy sky DNI and DHI using the Erbs model.

    Parameters
    ----------
    site : datamodel.Site
    cloud_cover : Series
        Cloud cover in %.
    zenith : Series
        Solar zenith

    Returns
    -------
    ghi : pd.Series, dni : pd.Series, dhi : pd.Series
    """
    ghi = cloud_cover_to_ghi_linear(cloud_cover, ghi_clear)
    dni, dhi = pvmodel.complete_irradiance_components(ghi, zenith)
    return ghi, dni, dhi


def cloud_cover_to_irradiance(latitude, longitude, elevation, cloud_cover,
                              apparent_zenith, zenith):
    """
    Estimates irradiance from cloud cover in the following steps:

    1. Determine clear sky GHI using Ineichen model and climatological
       turbidity.
    2. Estimate cloudy sky GHI using a function of cloud_cover and
       ghi_clear: :py:func:`cloud_cover_to_ghi_linear`
    3. Estimate cloudy sky DNI and DHI using the Erbs model.

    Don't use this function if you already have clear sky GHI. Instead,
    use :py:func:`cloud_cover_to_irradiance_ghi_clear`

    Parameters
    ----------
    latitude : float
    longitude : float
    elevation : float
    cloud_cover : Series
        Cloud cover in %.
    apparent_zenith : Series
        Solar apparent zenith
    zenith : Series
        Solar zenith

    Returns
    -------
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series

    See also
    --------
    cloud_cover_to_irradiance_ghi_clear
    cloud_cover_to_ghi_linear
    """
    cs = pvmodel.calculate_clearsky(latitude, longitude, elevation,
                                    apparent_zenith)
    ghi, dni, dhi = cloud_cover_to_irradiance_ghi_clear(
        cloud_cover, cs['ghi'], zenith)
    return ghi, dni, dhi


def resample(arg, freq='1h', label=None):
    """Resamples an argument, allowing for None. Use with map.
    Useful for applying resample to unknown model output (e.g. AC power
    is None if no plant metadata is provided).

    Parameters
    ----------
    arg : pd.Series or None
    freq : str
    label : str
        Sets pandas resample's label and closed kwargs.

    Returns
    -------
    pd.Series or None
    """
    if arg is None:
        return None
    else:
        return arg.resample(freq, label=label, closed=label).mean()


def reindex_fill_slice(arg, freq='5min', label=None, start=None, end=None,
                       start_slice=None, end_slice=None,
                       fill_method='interpolate'):
    """Reindex data to shorter intervals (create NaNs) from `start` to
    `end`, fill NaNs in gaps using `fill_method`, fill NaNs before first
    valid time using bfill, fill NaNs after last valid time using ffill,
    then slice output from `start_slice` to `end_slice`.
    """
    if arg is None or arg.empty:
        return arg
    if start is None:
        start_reindex = arg.index[0]
    else:
        start_reindex = min(pd.Timestamp(start), arg.index[0])
    if end is None:
        end_reindex = arg.index[-1]
    else:
        end_reindex = max(pd.Timestamp(end), arg.index[-1])
    index = pd.date_range(start=start_reindex, end=end_reindex, freq=freq,
                          closed=label)
    reindexed = arg.reindex(index)
    filled = getattr(reindexed, fill_method)()
    sliced = filled.loc[start_slice:end_slice].bfill().ffill()
    return sliced


def unmix_intervals(mixed, lower=0, upper=100):
    """Convert mixed interval averages into pure interval averages.

    For example, the GFS 3 hour output contains the following data:

    * forecast hour 3: average cloud cover from 0 - 3 hours
    * forecast hour 6: average cloud cover from 0 - 6 hours
    * forecast hour 9: average cloud cover from 6 - 9 hours
    * forecast hour 12: average cloud cover from 6 - 12 hours

    and so on. This function returns:

    * forecast hour 3: average cloud cover from 0 - 3 hours
    * forecast hour 6: average cloud cover from 3 - 6 hours
    * forecast hour 9: average cloud cover from 6 - 9 hours
    * forecast hour 12: average cloud cover from 9 - 12 hours

    Parameters
    ----------
    data : pd.Series
        The first time must be the first output of a cycle.
    lower : None or float
        Lower bound. Useful for handling numerical precision issues in
        input data.
    upper : None or float
        Upper bound of output. Useful for handling numerical precision
        issues in input data.

    Returns
    -------
    pd.Series
        Data is the unmixed interval average with ending label.
    """
    intervals = (mixed.index[1:] - mixed.index[:-1]).unique()
    if len(intervals) > 1:
        raise ValueError('multiple interval lengths detected. slice forecasts '
                         'into sections with unique interval lengths first.')
    interval = intervals[0]
    start = mixed.index[0]
    mixed_vals = np.array(mixed)
    if interval == pd.Timedelta('1h'):
        _check_start_time(start, interval)
        # mixed_1...mixed_6 are the values in the raw forecast file at
        # each hour of the mixed interval period. Let f1...f6 be unmixed,
        # true hourly average values. The relationship is:
        # mixed_1 = f1
        # mixed_2 = (f1 + f2) / 2
        # mixed_3 = (f1 + f2 + f3) / 3
        # mixed_4 = (f1 + f2 + f3 + f4) / 4
        # mixed_5 = (f1 + f2 + f3 + f4 + f5) / 5
        # mixed_6 = (f1 + f2 + f3 + f4 + f5 + f6) / 6
        # some algebra will show that the f1...f6 can be obtained as
        # coded below.
        # the cycle repeats itself after 6 hours so
        # mixed_7 = f7
        # mixed_8 = (f8 + f9) / 2 ...
        # To efficiently compute the values for all forecast times,
        # we use slices for every 6th element, calculate the forecasts
        # at every 6th point, then interleave them by constructing a 2D
        # array and reshaping it to a 1D array.
        mixed_1 = mixed_vals[0::6]
        mixed_2 = mixed_vals[1::6]
        mixed_3 = mixed_vals[2::6]
        mixed_4 = mixed_vals[3::6]
        mixed_5 = mixed_vals[4::6]
        mixed_6 = mixed_vals[5::6]
        f1 = mixed_1
        f2 = 2 * mixed_2 - mixed_1
        f3 = 3 * mixed_3 - 2 * mixed_2
        f4 = 4 * mixed_4 - 3 * mixed_3
        f5 = 5 * mixed_5 - 4 * mixed_4
        f6 = 6 * mixed_6 - 5 * mixed_5
        f = np.array([f1, f2, f3, f4, f5, f6])
    elif interval == pd.Timedelta('3h'):
        _check_start_time(start, interval)
        # similar to above, but
        # mixed_3 = f_0_3
        # mixed_6 = (f_0_3 + f_3_6) / 2
        f3 = mixed_vals[0::2]
        f6 = 2 * mixed_vals[1::2] - f3
        f = np.array([f3, f6])
    else:
        raise ValueError('mixed period must be 6 hours and data interval must '
                         'be 3 hours or 1 hour')
    unmixed = pd.Series(f.reshape(-1, order='F'), index=mixed.index)
    unmixed = unmixed.clip(lower=0, upper=100)
    return unmixed


def _check_start_time(start, interval):
    """
    start : pd.Timestamp
        Must be timezone aware.
    interval : pd.Timedelta
        Time between data points.
    """
    # for this to work, the first value must belong to the
    # first time of a mixed interval period. Assuming GFS data...
    start_time = start.tz_convert('UTC').time()
    allowed_times_interval = {
        pd.Timedelta('1h'): (1, 7, 13, 19),
        pd.Timedelta('3h'): (3, 9, 15, 21)
    }
    allowed_times = allowed_times_interval[interval]
    if any(start_time == datetime.time(t) for t in allowed_times):
        return
    else:
        raise ValueError(f'for {interval} mixed intervals, start time must '
                         f'be one of {allowed_times}Z hours')


def sort_gefs_frame(frame):
    """Sort a DataFrame from a GEFS forecast. Column 0 is the smallest
    value at each time. Column 20 is the largest value at each time.
    """
    if frame is None:
        return frame
    else:
        return pd.DataFrame(np.sort(frame), index=frame.index)
