"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
from functools import partial
import json
import logging


import pandas as pd


from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io.fetch import nwp as fetch_nwp
from solarforecastarbiter.reference_forecasts import persistence, models


logger = logging.getLogger(__name__)


# maybe rework in terms of forecast run *issue time* as described in
# https://solarforecastarbiter.org/usecases/#forecastrun
# and datamodel.Forecast (as demonstrated in run_persistence below)
def run_nwp(site, model, init_time, start, end):
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
    return datamodel.NWPOutput(*resampled)


def run_persistence(session, observation, forecast, run_time, issue_time,
                    index=False):
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
           *window = run_time - forecast.run_length*.
           No longer than 1 hour.
      * Day ahead forecasts:
          *window = forecast.interval_length*.

    Users that would like more flexibility may use the lower-level
    functions in
    :py:mod:`solarforecastarbiter.reference_forecasts.persistence`.

    Parameters
    ----------
    session : api.Session
        The session object to use to request data from the
        SolarForecastArbiter API.
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
    if not intraday:
        # raise ValueError if not intraday and not midnight to midnight
        _check_midnight_to_midnight(forecast_start, forecast_end)

    data_start, data_end = get_data_start_end(
        observation, forecast, run_time)

    def load_data(observation, data_start, data_end):
        df = session.get_observation_values(observation.observation_id,
                                            data_start, data_end,
                                            observation.interval_label)
        df = df.tz_convert(observation.site.timezone)
        return df['value']

    if intraday and index:
        fx = persistence.persistence_scalar_index(
            observation, data_start, data_end, forecast_start, forecast_end,
            forecast.interval_length, forecast.interval_label, load_data)
    elif intraday and not index:
        fx = persistence.persistence_scalar(
            observation, data_start, data_end, forecast_start, forecast_end,
            forecast.interval_length, forecast.interval_label, load_data)
    elif not intraday and not index:
        fx = persistence.persistence_interval(
            observation, data_start, data_end, forecast_start,
            forecast.interval_length, forecast.interval_label, load_data)
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
        raise ValueError(
            ('Incompatible forecast.issue_time_of_day %s, '
             'forecast.run_length %s, and issue_time %s') % (
             forecast.issue_time_of_day, forecast.run_length, issue_time))
    forecast_start = issue_time + forecast.lead_time_to_start
    forecast_end = forecast_start + forecast.run_length
    if (
            'instant' in forecast.interval_label
            or forecast.interval_label == 'beginning'
    ):
        forecast_end -= pd.Timedelta('1s')
    else:
        forecast_start += pd.Timedelta('1s')
    return forecast_start, forecast_end


def _is_intraday(forecast):
    """Is the forecast intraday?"""
    # intra day persistence and "day ahead" persistence require
    # fairly different parameters.
    # is this a sufficiently robust way to distinguish?
    return forecast.run_length < pd.Timedelta('1d')


def _check_midnight_to_midnight(forecast_start, forecast_end):
    if (forecast_start.round('1d') != forecast_start or
            forecast_end - forecast_start > pd.Timedelta('1d')):
        raise ValueError(
            'Day ahead persistence requires midnight to midnight periods')


def _intraday_start_end(observation, forecast, run_time):
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
    return data_start, data_end


def _dayahead_start_end(run_time):
    # day ahead persistence: tomorrow's forecast is equal to yesterday's
    # observations. So, forecast always uses obs > 24 hr old at each valid
    # time. Smarter approach might be to use today's observations up
    # until issue_time, and use yesterday's observations for issue_time
    # until end of day. So, forecast *never* uses obs > 24 hr old at each
    # valid time. Arguably too much for a reference forecast.
    data_end = run_time.floor('1d')
    data_start = data_end - pd.Timedelta('1d')
    return data_start, data_end


def _adjust_for_instant_obs(data_start, data_end, observation, forecast):
    # instantaneous observations require care.
    # persistence models return forecasts with same closure as obs
    if 'instant' in forecast.interval_label:
        if forecast.interval_length != observation.interval_length:
            raise ValueError('Instantaneous forecast requires instantaneous '
                             'observation with identical interval length.')
        else:
            data_end -= pd.Timedelta('1s')
    elif forecast.interval_label == 'beginning':
        data_end -= pd.Timedelta('1s')
    else:
        data_start += pd.Timedelta('1s')
    return data_start, data_end


def get_data_start_end(observation, forecast, run_time):
    """
    Determine the data start and data end times for a persistence
    forecast.

    Returns
    -------
    data_start : pd.Timestamp
    data_end : pd.Timestamp
    """
    if _is_intraday(forecast):
        data_start, data_end = _intraday_start_end(observation, forecast,
                                                   run_time)
    else:
        data_start, data_end = _dayahead_start_end(run_time)

    # to ensure that each observation data point contributes to the correct
    # forecast, the data_end and data_start values may need to be nudged
    if 'instant' in observation.interval_label:
        data_start, data_end = _adjust_for_instant_obs(data_start, data_end,
                                                       observation, forecast)
    else:
        if 'instant' in forecast.interval_label:
            raise ValueError('Instantaneous forecast cannot be made from '
                             'interval average observations')
    return data_start, data_end


def get_init_time(run_time, fetch_metadata):
    """Determine the most recent init time for which all forecast data is
    available."""
    run_finish = (pd.Timedelta(fetch_metadata['delay_to_first_forecast']) +
                  pd.Timedelta(fetch_metadata['avg_max_run_length']))
    freq = fetch_metadata['update_freq']
    init_time = (run_time - run_finish).floor(freq=freq)
    return init_time


def process_forecast_groups(forecasts, session, run_time, issue_time,
                            load_forecast):
    forecast_df = pd.DataFrame(
        [(fx.forecast_id, fx,
          json.loads(fx.extra_parameters).get('piggyback_on',
                                              fx.forecast_id))
         for fx in forecasts],
        columns=['forecast_id', 'forecast', 'piggyback_on']
        ).set_index('forecast_id')
    for run_for, group in forecast_df.groupby('piggyback_on'):
        key_fx = group.loc[run_for].forecast
        extra_params = json.loads(key_fx.extra_parameters)

        try:
            fetch_metadata = getattr(
                fetch_nwp, extra_params['fetch_metadata'])
        except AttributeError:
            if len(group) == 1:
                logger.info(
                    'No fetch_metadata defined for %s, not creating '
                    'reference forecast', key_fx.name)
            else:

                logger.warning(
                    'No fetch_metadata for the key forecast, %s, '
                    'not generating forecasts for group',
                    key_fx.name)
            continue
        init_time = get_init_time(run_time, fetch_metadata)
        forecast_start, forecast_end = get_forecast_start_end(
            key_fx, issue_time)
        model = getattr(models, extra_params['model'])
        # for testing
        model = partial(model, load_forecast=load_forecast)
        nwp_result = run_nwp(key_fx.site, model, init_time,
                             forecast_start, forecast_end)
        for fx_id, fx in group['forecast'].iteritems():
            fx_vals = getattr(nwp_result, fx.variable)
            if fx_vals is None:
                continue
            session.post_forecast_values(fx_id, fx_vals)
