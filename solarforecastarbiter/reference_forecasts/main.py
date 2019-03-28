"""
Make benchmark irradiance and power forecasts.
"""

from solarforecastarbiter import datamodel, pvmodel


def run(site, model, init_time, start, end):
    """
    Calculate benchmark irradiance and power forecasts for a site.

    The meaning of the timestamps (instantaneous or interval average)
    is determined by the model processing function.

    It's currently the user's job to determine time parameters that
    correspond to a particular Forecast Evaluation Time Series.

    Parameters
    ----------
    site : datamodel.Site
    model : function
        Model loading and processing function.
        See :py:mod:`solarforecastarbiter.reference_forecasts.models`
        for options.
    init_time : pd.Timestamp
        Model initialization time.
    start : pd.Timestamp
        Start of the forecast.
    end : pd.Timestamp
        End of the forecast.

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
    >>> init_time = pd.Timestamp('20190328T1200Z')
    >>> start = pd.Timestamp('20190328T1300Z')  # typical available time
    >>> end = pd.Timestamp('20190329T1300Z')  # 24 hour forecast
    >>> modeling_parameters = datamodel.FixedTiltModelingParameters(
    ...     ac_capacity=10, dc_capacity=15,
    ...     temperature_coefficient=-0.004, dc_loss_factor=0,
    ...     ac_loss_factor=0)
    >>> power_plant = datamodel.SolarPowerPlant(
    ...     name='Test plant', latitude=32.2, longitude=-110.9,
    ...     elevation=715, timezone='America/Phoenix',
    ...     modeling_parameters = modeling_parameters)
    >>> ghi, dni, dhi, temp_air, wind_speed, ac_power = run(
    ...     power_plant, hrrr_subhourly_to_hourly_mean, init_time,
    ...     start, end)
    """

    *solpos_forecast, resampler, solar_position_calculator = model(
        site.latitude, site.longitude, site.elevation)

    if isinstance(site, datamodel.SolarPowerPlant):
        solar_position = solar_position_calculator()
        ac_power = pvmodel.irradiance_to_power(
            site.modeling_parameters, solar_position['apparent_zenith'],
            solar_position['azimuth'], *solpos_forecast)
    else:
        ac_power = None

    # resample data after power calculation
    resampled = list(map(resampler, (*solpos_forecast, ac_power)))
    return resampled
