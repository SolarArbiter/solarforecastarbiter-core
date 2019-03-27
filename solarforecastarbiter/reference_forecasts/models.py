"""
Default processing steps for some weather models.

All functions assume that weather is a DataFrame.
"""

from functools import partial

import pandas as pd

from solarforecastarbiter import pvmodel
from solarforecastarbiter.io import load_forecast
from solarforecastarbiter.reference_forecasts import forecast


def hrrr_subhourly_to_subhourly_instantaneous(latitude, longitude):
    """
    Subhourly (15 min) instantantaneous HRRR forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, 'hrrr_subhourly')
    resample_func = partial(forecast.resample_args, freq='15min')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def hrrr_subhourly_to_hourly_mean(latitude, longitude):
    """
    Hourly mean HRRR forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, 'hrrr_subhourly')
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def rap_to_instantaneous(latitude, longitude):
    """
    Hourly instantantaneous RAP forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(latitude, longitude,
                                                        'rap')
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def rap_irrad_to_hourly_mean(latitude, longitude):
    """
    Take hourly RAP instant irradiance and convert it to hourly average
    data.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, 'rap')
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def rap_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                   solar_position_func):
    """
    Take hourly RAP instant cloud cover and convert it to hourly average
    data.
    """
    variables = load_forecast(
        latitude, longitude, 'rap',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    cloud_cover, temp_air, wind_speed = forecast.resample_args(*variables,
                                                               freq='15min')
    solar_position = solar_position_func(cloud_cover.index)
    cs = pvmodel.calculate_clearsky(latitude, longitude, elevation,
                                    solar_position['apparent_zenith'])
    ghi, dni, dhi = forecast.cloud_cover_to_irradiance_clearsky_scaling(
        cloud_cover, cs['ghi'], solar_position['zenith']
    )
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def rap_cloud_cover_to_hourly_mean_alternative(latitude, longitude, elevation,
                                               solar_position_func):
    """
    Take hourly RAP instant cloud cover and convert it to hourly average
    data.
    """
    variables = load_forecast(
        latitude, longitude, 'rap',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    cloud_cover, temp_air, wind_speed = forecast.interpolate_args(*variables,
                                                                  freq='15min')
    ghi, dni, dhi = \
        forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos_func(
            latitude, longitude, elevation, cloud_cover, solar_position_func)
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def gfs_3hour_to_hourly_mean(latitude, longitude, elevation,
                             solar_position_func):
    """
    Take 3 hr GFS and convert it to hourly average data.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(latitude, longitude,
                                                      'gfs_3h')
    cloud_cover = forecast.unmix_intervals(cloud_cover)
    cloud_cover, temp_air, wind_speed = forecast.interpolate_args(
        cloud_cover, temp_air, wind_speed, freq='15min')
    ghi, dni, dhi = \
        forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos_func(
            latitude, longitude, elevation, cloud_cover, solar_position_func)
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def gfs_hourly_to_hourly_mean(latitude, longitude, elevation,
                              solar_position_func):
    """
    Take 1 hr GFS and convert it to hourly average data.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(latitude, longitude,
                                                      'gfs_1h')
    # hourly data only available through 96h?
    start = cloud_cover.index[0]
    end = start + pd.Timedelta('96h')  # not sure about this time limit
    ghi, temp_air, wind_speed = forecast.slice_args(
        cloud_cover, temp_air, wind_speed, start, end)
    cloud_cover, temp_air, wind_speed = forecast.interpolate_args(
        cloud_cover, temp_air, wind_speed, freq='15min')
    ghi, dni, dhi = \
        forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos_func(
            latitude, longitude, elevation, cloud_cover, solar_position_func)
    resample_func = partial(forecast.resample_args, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def nam_to_instantaneous(latitude, longitude, solar_position_func):
    """
    Hourly instantantaneous forecast. Max forecast horizon 48 hours.
    """
    ghi, temp_air, wind_speed = load_forecast(latitude, longitude, 'nam')
    # hourly data only available through 48h?
    start = ghi.index[0]
    end = start + pd.Timedelta('48h')  # not sure about this time limit
    ghi, temp_air, wind_speed = forecast.slice_args(ghi, temp_air, wind_speed,
                                                    start, end)
    solar_position = solar_position_func(ghi.index)
    dni, dhi = pvmodel.complete_irradiance_components(
        ghi, solar_position['zenith'])
    resample_func = partial(forecast.resample_args, freq='15min')
    return ghi, dni, dhi, temp_air, wind_speed, resample_func


def nam_1_3_hour_to_hourly_mean(weather):
    """

    """
    weather = forecast.hourly_cloud_cover_to_subhourly(weather)
    return weather
