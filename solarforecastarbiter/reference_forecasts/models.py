"""
Default processing steps for NOAA weather models.

All functions in this module have the same signature.

The functions accept:

  * latitude : float
  * longitude : float
  * elevation : float
  * init_time : pd.Timestamp
      Full datetime of a model initialization
  * start : pd.Timestamp
  * end : pd.Timestamp

The functions return a tuple of:

  * ghi : pd.Series
  * dni : pd.Series
  * dhi : pd.Series
  * temp_air : pd.Series
  * wind_speed : pd.Series
  * resampler : function
      A function that resamples data to the appropriate frequency.
      This function is applied to all variables after power is
      calculated.
  * solar_position_calculator : function
      A function that returns solar position at the ghi forecast times
      (after interpolation, before resampling). The function will return
      results immediately if solar position is already known or will
      call a solar position calculation algorithm and then return.

Most of the functions return forecast data interpolated to 5 minute
frequency. Interpolation to 5 minutes reduces the errors associated with
solar position.
"""

from functools import partial

from solarforecastarbiter import pvmodel
from solarforecastarbiter.io.nwp import load_forecast
from solarforecastarbiter.reference_forecasts import forecast


def hrrr_subhourly_to_subhourly_instantaneous(latitude, longitude, elevation,
                                              init_time, start, end):
    """
    Subhourly (15 min) instantantaneous HRRR forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'hrrr_subhourly')
    resampler = partial(forecast.resample, freq='15min')
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def hrrr_subhourly_to_hourly_mean(latitude, longitude, elevation,
                                  init_time, start, end):
    """
    Hourly mean HRRR forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'hrrr_subhourly')
    interpolator = partial(forecast.interpolate, freq='5min')
    ghi, dni, dhi, temp_air, wind_speed = list(
        map(interpolator, (ghi, dni, dhi, temp_air, wind_speed)))
    resampler = partial(forecast.resample, freq='1h')
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def rap_to_instantaneous(latitude, longitude, elevation,
                         init_time, start, end):
    """
    Hourly instantantaneous RAP forecast.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'rap')
    resampler = partial(forecast.resample, freq='1h')
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def rap_irrad_to_hourly_mean(latitude, longitude, elevation,
                             init_time, start, end):
    """
    Take hourly RAP instantantaneous irradiance and convert it to hourly
    average forecasts.
    """
    ghi, dni, dhi, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'rap')
    interpolator = partial(forecast.interpolate, freq='5min')
    ghi, dni, dhi, temp_air, wind_speed = list(
        map(interpolator, (ghi, dni, dhi, temp_air, wind_speed)))
    resampler = partial(forecast.resample, freq='1h')
    solar_pos_calculator = partial(
        pvmodel.calculate_solar_position, latitude, longitude, elevation,
        ghi.index)
    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def _resample_using_cloud_cover(latitude, longitude, elevation,
                                cloud_cover, temp_air, wind_speed):
    # 5 minutes is probably better than 15
    interpolator = partial(forecast.interpolate, freq='5min')
    cloud_cover, temp_air, wind_speed = list(
        map(interpolator, (cloud_cover, temp_air, wind_speed)))
    solar_position = pvmodel.calculate_solar_position(
        latitude, longitude, elevation, cloud_cover.index)
    ghi, dni, dhi = forecast.cloud_cover_to_irradiance_clearsky_scaling_solpos(
        latitude, longitude, elevation, cloud_cover,
        solar_position['apparent_zenith'], solar_position['azimuth'])
    resampler = partial(forecast.resample, freq='1h')

    def solar_pos_calculator(): return solar_position

    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def rap_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                   init_time, start, end):
    """
    Take hourly RAP instantantaneous cloud cover and convert it to
    hourly average forecasts.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'rap',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, temp_air, wind_speed)


def gfs_3hour_to_hourly_mean(latitude, longitude, elevation,
                             init_time, start, end):
    """
    Take 3 hr GFS and convert it to hourly average data.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'gfs_3h')
    cloud_cover = forecast.unmix_intervals(cloud_cover)
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, temp_air, wind_speed)


def gfs_hourly_to_hourly_mean(latitude, longitude, elevation,
                              init_time, start, end):
    """
    Take 1 hr GFS and convert it to hourly average data.
    Max forecast horizon 96? hours.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'gfs_1h')
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, temp_air, wind_speed)


def nam_to_hourly_instantaneous(latitude, longitude, elevation,
                                init_time, start, end):
    """
    Hourly instantantaneous forecast. Max forecast horizon 48? hours.
    """
    ghi, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'nam')
    solar_position = pvmodel.calculate_solar_position(
        latitude, longitude, elevation, ghi.index)
    dni, dhi = pvmodel.complete_irradiance_components(
        ghi, solar_position['zenith'])
    resampler = partial(forecast.resample, freq='15min')
    def solar_pos_calculator(): return solar_position

    return ghi, dni, dhi, temp_air, wind_speed, resampler, solar_pos_calculator


def nam_cloud_cover_to_hourly_mean(latitude, longitude, elevation,
                                   init_time, start, end):
    """
    Hourly average forecast. Max forecast horizon 72 hours.
    """
    cloud_cover, temp_air, wind_speed = load_forecast(
        latitude, longitude, init_time, start, end, 'nam',
        variables=('cloud_cover', 'temp_air', 'wind_speed'))
    return _resample_using_cloud_cover(latitude, longitude, elevation,
                                       cloud_cover, temp_air, wind_speed)
