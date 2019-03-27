"""
Default processing steps for some weather models.

All functions assume that weather is a DataFrame.
"""

from solarforecastarbiter.reference_forecasts import forecast


def hrrr_subhourly(weather):
    """
    HRRR subhourly probably already has all of the data that we want.
    """
    return weather


def hrrr_hourly(weather):
    weather = forecast.hourly_ghi_to_subhourly(weather)
    return weather


rap = hrrr_hourly


def gfs(weather):
    weather = forecast.unmix_intervals(weather)
    weather = forecast.interpolate_to(weather, freq='1h')
    weather = forecast.hourly_cloud_cover_to_subhourly(weather)
    return weather


def nam(weather):
    weather = forecast.hourly_cloud_cover_to_subhourly(weather)
    return weather
