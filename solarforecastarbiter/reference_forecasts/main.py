"""
Make benchmark irradiance and power forecasts.
"""

# from solarforecastarbiter.io import load_forecast
from solarforecastarbiter import pvmodel


def run(site, model):
    """
    Calculate benchmark irradiance forecast for a Site. Also returns
    solar position and weather data that can be used for power modeling.

    Assumes all returned values are instantaneous. So if you want to
    make a hourly average power forecast then you probably want to ask
    for a subhourly irradiance forecast.

    Parameters
    ----------
    site : datamodel.Site

    Returns
    -------
    Tuple of:

    apparent_zenith : None or pd.Series
    azimuth : None or pd.Series
    ghi : pd.Series
    dni : pd.Series
    dhi : pd.Series
    temp_air : None or pd.Series
    wind_speed : None or pd.Series

    See also
    --------
    pvmodel.irradiance_to_power
    """

    # get point forecast data from raw or slightly processed model files
    # assuming forecast is a DataFrame of data that is actually in the
    # model.
    forecast = load_forecast(site.latitude, site.longitude, model)

    # interpolate forecast to finer time steps as required.
    # could also be def interpolator(forecast): return forecast for some models
    # interpolator is determined by some TBD combination of
    # 1. the forecast model that was loaded
    # 2. arg/kwarg string
    # 3. arg/kwarg function
    fx_interp = interpolator(forecast)

    # solar position is always needed for power forecasts.
    # it's not needed for some irradiance forecasts.
    # but we only want to calculate it once, so the best control flow
    # is not yet clear to me. perhaps None
    solar_position = pvmodel.calculate_solar_position(
        site.latitude, site.longitude, site.elevation, fx_interp.index)

    # same story as for interp, but also needs solar position
    fx_processed = processor(fx_interp, solar_position)

    keys = ('ghi', 'dni', 'dhi', 'temp_air', 'wind_speed')
    tuples = solar_position['apparent_zenith'], solar_position['azimuth']
    tuples = tuples + [fx_processed.get(key) for key in keys]
    return tuples
