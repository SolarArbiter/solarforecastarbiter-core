"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
import pandas as pd

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter import persistence


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
        NWP model loading and processing function.
        See :py:mod:`solarforecastarbiter.reference_forecasts.models`
        for options.
    init_time : pd.Timestamp
        NWP model initialization time.
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
    Run a persistence *forecast* for an *observation* using a *model*.

    Model can one of:

      1. persistence: irradiance or power persistence
      2. index persistence: irradiance persistence using clear sky index
         or power persistence using AC power index

    The persistence *window* is the time over which the persistence
    quantity (irradiance, power, clear sky index, or power index) is
    averaged over. The persistence window is automatically determined
    based on the type of persistence forecast desired:

      * Intraday persistence forecasts: Maximum of observation interval
        length, forecast interval length, and forecast lead time to
        start. No longer than 1 hour.
      * Day ahead forecasts: Equal to forecast interval length.

    Users that would like more flexibility may use the lower-level
    functions in
    :py:mod:`solarforecastarbiter.reference_forecasts.persistence`.

    Parameters
    ----------
    observation : datamodel.Observation
        The metadata of the observation to be used to create the
        forecast.
    forecast : datamodel.Forecast
        The metadata of the desired forecast.
    issue_time : pd.Timestamp
        Issue time of the forecast run.
    model : str or function
        Forecast model. If string, must be *persistence* or
        *index_persistence*. If function, must have the same signature
        as the functions in
        :py:mod:`solarforecastarbiter.reference_forecasts.persistence`.

    Returns
    -------
    forecast : pd.Series
        Forecast conforms to the metadata specified by the *forecast*
        argument.
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
        # time. Smarter approach might be to use today's observations up
        # until issue_time, and use yesterday's observations for issue_time
        # until end of day. So, forecast *never* uses obs > 24 hr old at each
        # valid time. Arguably too much for a reference forecast.
        data_end = issue_time.floor('1d')
        data_start = data_end - pd.Timedelta('1d')
        window = forecast.interval_length

    # to ensure that each observation contributes to the correct forecast,
    # data_end, data_start need to be nudged if observations are instantaneous
    if datamodel.CLOSED_MAPPING[observation.interval_label] is None:
        if forecast.interval_label == 'ending':
            data_start += pd.Timedelta('1s')
        else:
            data_end -= pd.Timedelta('1s')

    # specify forecast start and end times
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length

    if isinstance(model, str):
        # Extract the function with this name in the persistence module.
        #
        # Instead of using vars, we could be more explicit and define our own
        # dict here or in persistence.py:
        # models = {'persistence': persistence.persistence,
        #           'index_persistence': persistence.index_persistence}
        # model = models[model]
        # Using vars gives us one fewer thing to update if we want to change
        # anything and is less of a hassle if we want to use the same pattern
        # for the NWP processing models.
        try:
            model = vars(persistence)[model.replace(' ', '_')]
        except KeyError:
            raise ValueError(
                'Invalid model option. See doc string for options.')

    # finally, we call the desired persistence model function
    fx = model(observation, window, data_start, data_end, forecast_start,
               forecast_end, forecast.interval_length)

    # make interval label consistent with input forecast
    # not confident that this code does what it's supposed to do all the time
    # deal with that once design is better understood
    closed_fx = datamodel.CLOSED_MAPPING[forecast.interval_label]
    fx = fx.resample(fx.index.freq, closed=closed_fx).mean()

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
