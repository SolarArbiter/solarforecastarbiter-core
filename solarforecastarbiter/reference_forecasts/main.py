"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
import pandas as pd

from solarforecastarbiter import datamodel, pvmodel


# maybe rename run_nwp
# maybe rework in terms of forecast run *issue time* as described in
# https://solarforecastarbiter.org/usecases/#forecastrun
# and datamodel.Forecast (as demonstrated in run_persistence below)
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
    ...     power_plant, models.hrrr_subhourly_to_hourly_mean,
    ...     init_time, start, end)
    """

    *forecast, resampler, solar_position_calculator = model(
        site.latitude, site.longitude, site.elevation,
        init_time, start, end)

    if isinstance(site, datamodel.SolarPowerPlant):
        solar_position = solar_position_calculator()
        ac_power = pvmodel.irradiance_to_power(
            site.modeling_parameters, solar_position['apparent_zenith'],
            solar_position['azimuth'], *forecast)
    else:
        ac_power = None

    # resample data after power calculation
    resampled = list(map(resampler, (*forecast, ac_power)))
    return resampled


def run_persistence(observation, forecast, issue_time, model):
    """
    Run a persistence forecast for an *observation* using a *model*.

    Model can one of:

      1. irradiance or power persistence
      2. irradiance persistence using clear sky index
      3. power persistence using AC power index

    Persistence window is fixed according to:

      * Intraday forecasts: Maximum of observation interval length,
        forecast interval length, forecast lead time to start. No
        longer than 1 hour.
      * Day ahead forecasts: Equal to forecast interval length.

    Parameters
    ----------
    observation : datamodel.Observation
    forecast : datamodel.Forecast
    issue_time : pd.Timestamp
        Issue time of the forecast run.
    model : function
        Model loading and processing function. See
        :py:mod:`solarforecastarbiter.reference_forecasts.persistence`
        for options.


    Alternative and incomplete Parameters
    ----------
    observation : datamodel.Observation
    model : function
        Model loading and processing function. See
        :py:mod:`solarforecastarbiter.reference_forecasts.persistence`
        for options.
    issue_time : pd.Timestamp
        Time at which the forecast begins
    start : pd.Timestamp
        Forecast start
    end : pd.Timestamp
        Forecast end

    Returns
    -------
    forecast : pd.Series
    """
    check_issue_time(forecast, issue_time)

    # intra day persistence and "day ahead" persistence require
    # fairly different parameters.
    # is this a sufficiently robust way to distinguish?
    intraday = forecast.run_length < pd.Timedelta('1d')
    if intraday:
        # time window over which observation data will be used to create
        # persistence forecast.
        # assumes observation.interval_length <= 1h
        window = min(
            pd.Timedelta('1h'),
            max(observation.interval_length, forecast.interval_length,
                forecast.lead_time_to_start)
        )
        data_end = issue_time
        data_start = data_end - window
    else:
        # day ahead persistence: tomorrow's forecast is equal to yesterday's
        # observations. So, forecast always uses obs > 24 hr old at each valid
        # time. Smarter TBD approach might be to use today's observations up
        # until issue_time, and use yesterday's observations for issue_time
        # until end of day. So, forecast *never* uses obs > 24 hr old at each
        # valid time.
        data_end = issue_time.floor('1d')
        data_start = data_end - pd.Timedelta('1d')
        window = forecast.interval_length

    # account for observation interval label.
    # you could argue this should be elsewhere or at least a utility
    # function in io, but putting it here for now
    if observation.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    elif observation.interval_label == 'ending':
        data_start += pd.Timedelta('1s')

    # specify forecast start and end times. question: should we also
    # specify forecast interval label here? see notes in persistence.py
    # and in resampling below
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length

    # finally, we call the desired persistence model function
    fx = model(observation, window, data_start, data_end, forecast_start,
               forecast_end, forecast.interval_length)

    # make interval label consistent with input forecast
    # not confident that this code does what it's supposed to do all the time
    # deal with that once design is better understood
    if forecast.interval_label == 'ending':
        fx = fx.resample(fx.index.freq, label='right').mean()

    return fx


# put in validator?
def check_issue_time(forecast, issue_time):
    """
    Check that the issue time is compatible with the forecast.

    Raises
    ------
    ValueError if forecast and issue_time are incompatible
    """
    start = issue_time.floor() + forecast.issue_time_of_day
    index = pd.DatetimeIndex(start=start, end=start+pd.Timedelta('1d'),
                             freq=forecast.run_length)
    if issue_time not in index:
        ValueError(('Incompatible forecast.issue_time_of_day %s,'
                    'forecast.run_length %s, and issue_time %s') %
                   forecast.issue_time_of_day, forecast.run_length, issue_time)
