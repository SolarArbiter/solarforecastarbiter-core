"""
Functions for forecasting.
"""

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
        Estimated GHI.

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


def cloud_cover_to_irradiance_clearsky_scaling(cloud_cover, ghi_clear,
                                               solar_zenith):
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

    Returns
    -------
    ghi : pd.Series, dni : pd.Series, dhi : pd.Series
    """
    ghi = cloud_cover_to_ghi_linear(cloud_cover, ghi_clear)
    dni, dhi = pvmodel.complete_irradiance_components(ghi, solar_zenith)
    return ghi, dni, dhi


def cloud_cover_to_irradiance_clearsky_scaling_solpos(
        latitude, longitude, elevation, cloud_cover, apparent_zenith, zenith):
    """
    Estimates irradiance from cloud cover in the following steps:

    1. Calculate solar position for the site.
    2. Determine clear sky GHI using Ineichen model and climatological
       turbidity.
    3. Estimate cloudy sky GHI using a function of cloud_cover and
       ghi_clear: :py:func:`cloud_cover_to_ghi_linear`
    4. Estimate cloudy sky DNI and DHI using the Erbs model.

    Don't use this function if you already have solar position.

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
    cloud_cover_to_irradiance_clearsky_scaling
    cloud_cover_to_ghi_linear
    """
    cs = pvmodel.calculate_clearsky(latitude, longitude, elevation,
                                    apparent_zenith)
    ghi, dni, dhi = cloud_cover_to_irradiance_clearsky_scaling(
        cloud_cover, cs['ghi'], zenith)
    return ghi, dni, dhi


def resample_args(*args, freq='1h'):
    # this one uses map for fun
    def f(arg):
        if arg is None:
            return None
        else:
            return arg.resample(freq).mean()
    return list(map(f, args))


def resample(arg, freq='1h'):
    if arg is None:
        return None
    else:
        return arg.resample(freq).mean()


def interpolate_args(*args, freq='15min'):
    # could add how kwarg to resample_args and lookup method with
    # getattr but this seems much more clear
    # this one uses a list comprehension for different fun
    resampled_args = [
        arg if arg is None else arg.resample(freq).interpolate()
        for arg in args]
    return resampled_args


def interpolate(arg, freq='15min'):
    # could add how kwarg to resample and lookup method with
    # getattr but this seems much more clear
    if arg is None:
        return None
    else:
        return arg.resample(freq).interpolate()


def slice_args(*args, start, end):
    resampled_args = [arg if arg is None else arg.iloc[start:end]
                      for arg in args]
    return resampled_args


def unmix_intervals(cloud_cover):
    raise NotImplementedError
