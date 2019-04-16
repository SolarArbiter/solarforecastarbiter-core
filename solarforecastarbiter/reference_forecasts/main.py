"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
import pandas as pd

from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.reference_forecasts import persistence


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


def run_persistence(observation, forecast, run_time, issue_time, index=False):
    """
    Run a persistence *forecast* for an *observation*.

    For intraday forecasts, the *index* argument controls if the
    forecast is constructed using persistence of the measured values
    (*index = False*) or persistence using clear sky index or AC power
    index.

    For day ahead forecasts, only persistence of measured values
    (*index = False*) is supported.

    Forecasts may be run operationally or retrospectively. For
    operational forecasts, *run_time* is typically set to now. For
    retrospective forecasts, *run_time* is the time by which the
    forecast should be run so that it could have been be delivered for
    the *issue_time*. Forecasts will only use data with timestamps
    before *run_time*.

    The persistence *window* is the time over which the persistence
    quantity (irradiance, power, clear sky index, or power index) is
    averaged. The persistence window is automatically determined
    from the *forecast* attributes:

      * Intraday persistence forecasts:
           *window = run_time - forecast.run_length.
           No longer than 1 hour.
      * Day ahead forecasts:
          *window = forecast.interval_length*.

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
    run_time : pd.Timestamp
        Run time of the forecast.
    issue_time : pd.Timestamp
        Issue time of the forecast run.
    index : bool, default False
        If False, use persistence of observed value. If True, use
        persistence of clear sky or AC power index.

    Returns
    -------
    forecast : pd.Series
        Forecast conforms to the metadata specified by the *forecast*
        argument.

    Raises
    ------
    ValueError
        If forecast and issue_time are incompatible.
    ValueError
        If persistence window < observation.interval_length.
    ValueError
        If forecast.run_length = 1 day and forecast period is not
        midnight to midnight.
    ValueError
        If forecast.run_length = 1 day and index=True.
    ValueError
        If instantaneous forecast and instantaneous observation interval
        lengths do not match.
    ValueError
        If average observations are used to make instantaneous forecast.
    """
    forecast_start, forecast_end = get_forecast_start_end(forecast, issue_time)
    intraday = _is_intraday(forecast)

    data_start, data_end = get_data_start_end(
        observation, forecast, run_time, forecast_start, forecast_end)

    if intraday and index:
        fx = persistence.persistence_scalar_index(
            observation, data_start, data_end, forecast_start, forecast_end,
            forecast.interval_length, forecast.interval_label)
    elif intraday and not index:
        fx = persistence.persistence_scalar(
            observation, data_start, data_end, forecast_start, forecast_end,
            forecast.interval_length, forecast.interval_label)
    elif not intraday and not index:
        fx = persistence.persistence_interval(
            observation, data_start, data_end, forecast_start,
            forecast.interval_length, forecast.interval_label)
    else:
        raise ValueError(
            'index=True not supported for forecasts with run_length >= 1day')

    return fx


def get_forecast_start_end(forecast, issue_time):
    """
    Get absolute forecast start from *forecast* object parameters and
    absolute *issue_time*.

    Parameters
    ----------
    forecast : datamodel.Forecast
    issue_time : pd.Timestamp

    Returns
    -------
    forecast_start : pd.Timestamp
        Start time of forecast issued at issue_time
    forecast_end : pd.Timestamp
        End time of forecast issued at issue_time

    Raises
    ------
    ValueError if forecast and issue_time are incompatible
    """
    first_issue_time = pd.Timestamp.combine(issue_time.floor('1D'),
                                            forecast.issue_time_of_day)
    issue_times = pd.date_range(start=first_issue_time,
                                end=first_issue_time+pd.Timedelta('1d'),
                                freq=forecast.run_length)
    if issue_time not in issue_times:
        ValueError(('Incompatible forecast.issue_time_of_day %s,'
                    'forecast.run_length %s, and issue_time %s') %
                   forecast.issue_time_of_day, forecast.run_length, issue_time)
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length
    if 'instant' in forecast.interval_label:
        forecast_end -= pd.Timedelta('1s')
    return forecast_start, forecast_end


def _is_intraday(forecast):
    """Is the forecast intraday?"""
    # intra day persistence and "day ahead" persistence require
    # fairly different parameters.
    # is this a sufficiently robust way to distinguish?
    return forecast.run_length < pd.Timedelta('1d')


def get_data_start_end(observation, forecast, run_time, forecast_start,
                       forecast_end):
    """
    Determine the data start and data end times for a persistence
    forecast.

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp
    """
    if _is_intraday(forecast):
        # time window over which observation data will be used to create
        # persistence forecast.
        if (observation.interval_length > forecast.run_length or
                observation.interval_length > pd.Timedelta('1h')):
            raise ValueError(
                'Intraday persistence requires observation.interval_length '
                '<= forecast.run_length and observation.interval_length <= 1h')
        # no longer than 1 hour
        window = min(forecast.run_length, pd.Timedelta('1hr'))
        data_end = run_time
        data_start = data_end - window
    else:
        # day ahead persistence: tomorrow's forecast is equal to yesterday's
        # observations. So, forecast always uses obs > 24 hr old at each valid
        # time. Smarter approach might be to use today's observations up
        # until issue_time, and use yesterday's observations for issue_time
        # until end of day. So, forecast *never* uses obs > 24 hr old at each
        # valid time. Arguably too much for a reference forecast.
        data_end = run_time.floor('1d')
        data_start = data_end - pd.Timedelta('1d')
        if (forecast_start.round('1d') != forecast_start or
                forecast_end - forecast_start > pd.Timedelta('1d')):
            raise ValueError(
                'Day ahead persistence requires midnight to midnight periods')

    # to ensure that each observation data point contributes to the correct
    # forecast, the data_end and data_start values may need to be nudged
    if 'instant' in observation.interval_label:
        # instantaneous observations require care.
        # persistence models return forecasts with same closure as obs
        if 'instant' in forecast.interval_label:
            if forecast.interval_length != observation.interval_length:
                raise ValueError('Instantaneous forecast requires '
                                 'instantaneous observation '
                                 'with identical interval length.')
            else:
                data_end -= pd.Timedelta('1s')
        elif forecast.interval_label == 'beginning':
            data_start += pd.Timedelta('1s')
        else:
            data_end -= pd.Timedelta('1s')
    else:
        if 'instant' in forecast.interval_label:
            raise ValueError('Instantaneous forecast cannot be made from '
                             'interval average observations')
    return data_start, data_end
