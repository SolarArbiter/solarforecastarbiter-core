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
    resampler = partial(forecast.resample, freq='15min')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def hrrr_subhourly_to_hourly_mean(latitude, longitude):
    """
    Hourly mean HRRR forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, 'hrrr_subhourly')
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def rap_to_instantaneous(latitude, longitude):
    """
    Hourly instantantaneous RAP forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(latitude, longitude,
                                                        'rap')
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def rap_irrad_to_hourly_mean(latitude, longitude):
    """
    Take hourly RAP instantantaneous irradiance and convert it to hourly
    average data.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, 'rap')
    resample = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resample


def rap_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                   solar_position_func):
    """
    Take hourly RAP instantantaneous cloud cover and convert it to
    hourly average data.
    """
    variables = load_forecast(
        latitude, longitude, 'rap',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    resampler = partial(forecast.resample, freq='15')
    cloud_cover, temp_air, wind_speed = list(map(resampler, variables))
    solar_position = solar_position_func(cloud_cover.index)
    cs = pvmodel.calculate_clearsky(latitude, longitude, elevation,
                                    solar_position['apparent_zenith'])
    ghi, dni, dhi = forecast.cloud_cover_to_irradiance_clearsky_scaling(
        cloud_cover, cs['ghi'], solar_position['zenith']
    )
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def rap_cloud_cover_to_hourly_mean_alternative(latitude, longitude, elevation,
                                               solar_position_func):
    """
    Take hourly RAP instantantaneous cloud cover and convert it to
    hourly average data.
    """
    variables = load_forecast(
        latitude, longitude, 'rap',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    interpolator = partial(forecast.interpolate_args, freq='15min')
    cloud_cover, temp_air, wind_speed = list(map(interpolator, variables))
    ghi, dni, dhi = \
        forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos_func(
            latitude, longitude, elevation, cloud_cover, solar_position_func)
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def gfs_3hour_to_hourly_mean(latitude, longitude, elevation,
                             solar_position_func):
    """
    Take 3 hr GFS and convert it to hourly average data.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(latitude, longitude,
                                                      'gfs_3h')
    cloud_cover = forecast.unmix_intervals(cloud_cover)
    interpolator = partial(forecast.interpolate_args, freq='15min')
    cloud_cover, temp_air, wind_speed = list(
        map(interpolator, (cloud_cover, temp_air, wind_speed)))
    ghi, dni, dhi = \
        forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos_func(
            latitude, longitude, elevation, cloud_cover, solar_position_func)
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


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
    resampler = partial(forecast.resample, freq='1h')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


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
    resampler = partial(forecast.resample, freq='15min')
    return ghi, dni, dhi, temp_air, wind_speed, resampler


def nam_1_3_hour_to_hourly_mean(weather):
    """

    """
    weather = forecast.hourly_cloud_cover_to_subhourly(weather)
    return weather
