"""
Make benchmark irradiance and power forecasts.
"""
from inspect import signature

from solarforecastarbiter import datamodel, pvmodel


def run(site, model):
    """
    Calculate benchmark irradiance and power forecasts for a site.

    The meaning of the timestamps (instantaneous or interval average)
    is determined by the model processing function.

    Parameters
    ----------
    site : datamodel.Site

    Returns
    -------
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    temp_air : None or pd.Series
    wind_speed : None or pd.Series
    ac_power : None or pd.Series

    Examples
    --------
    The following code would return hourly average forecasts derived
    from the subhourly HRRR model.

    >>> from solarforecastarbiter import datamodel
    >>> from solarforecastarbiter.reference_forecasts import models
    >>> modeling_parameters = datamodel.FixedTiltModelingParameters(
    ...     ac_capacity=10, dc_capacity=15,
    ...     temperature_coefficient=-0.004, dc_loss_factor=0,
    ...     ac_loss_factor=0)
    >>> power_plant = datamodel.SolarPowerPlant(
    ...     name='Test plant', latitude=32.2, longitude=-110.9,
    ...     elevation=715, timezone='America/Phoenix',
    ...     modeling_parameters = modeling_parameters)
    >>> ghi, dni, dhi, temp_air, wind_speed, ac_power = run(
    ...     power_plant, hrrr_subhourly_to_hourly_mean)
    """

    *solpos_forecast, resampler = model(
        site.latitude, site.longitude, site.elevation)

    if isinstance(site, datamodel.SolarPowerPlant):
        solpos_forecast = maybe_calc_solar_position(site, *solpos_forecast)
        ac_power = run_power(site.modeling_parameters, *solpos_forecast)
    else:
        ac_power = None

    # resample data after power calculation
    ret = list(map(resampler, (*solpos_forecast[2:], ac_power)))
    return ret


def maybe_calc_solar_position(site, apparent_zenith, azimuth, *args):
    if apparent_zenith is None or azimuth is None:
        solar_position = pvmodel.calculate_solar_position(
            site.latitude, site.longitude, site.elevation, args[0].index)
        (apparent_zenith, azimuth) = \
            solar_position['apparent_zenith'], solar_position['azimuth']
    return (apparent_zenith, azimuth) + (*args)


def run_irradiance(site, model, solar_position_func):
    """
    Calculate benchmark irradiance forecast for a Site. Also returns
    solar position and weather data that can be used for power modeling.

    Assumes all returned values are instantaneous. So if you want to
    make a hourly average power forecast then you probably want to ask
    for a subhourly irradiance forecast.

    Parameters
    ----------
    site : datamodel.Site
    model : function
        See solarforecastarbiter.reference_forecasts.models for examples
    solar_position_func : function
        Function that takes one DatetimeIndex argument and returns a
        DataFrame of solar position variables.

    Returns
    -------
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    temp_air : None or pd.Series
    wind_speed : None or pd.Series
    """
    args = site.latitude, site.longitude
    # figure out what metadata and functions the model needs
    # alternatively always pass the full site and solar_position_func
    # to model functions even if they don't need all metadata or the function
    sig = signature(model)
    kwargs = {}
    if 'elevation' in sig.parameters:
        kwargs['elevation'] = site.elevation
    if 'solar_position_func' in sig.parameters:
        kwargs['solar_position_func'] = solar_position_func
    ghi, dni, dhi, temp_air, wind_speed, resample_func = model(*args, **kwargs)
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def run_power(modeling_parameters, ghi, dni, dhi, temp_air, wind_speed,
              solar_position_func):
    solar_position = solar_position_func(ghi.index)
    apparent_zenith = solar_position['apparent_zenith']
    azimuth = solar_position['azimuth']
    ac_power = pvmodel.irradiance_to_power(
        modeling_parameters, apparent_zenith, azimuth, ghi, dni, dhi,
        temp_air=temp_air, wind_speed=wind_speed)
    return ac_power
