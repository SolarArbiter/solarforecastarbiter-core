"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
import json
import logging


import pandas as pd


from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io import api
from solarforecastarbiter.io.fetch import nwp as fetch_nwp
from solarforecastarbiter.reference_forecasts import persistence, models, utils


logger = logging.getLogger(__name__)


def run_nwp(forecast, model, run_time, issue_time):
    """
    Calculate benchmark irradiance and power forecasts for a Forecast.

    Forecasts may be run operationally or retrospectively. For
    operational forecasts, *run_time* is typically set to now. For
    retrospective forecasts, *run_time* is the time by which the
    forecast should be run so that it could have been be delivered for
    the *issue_time*. Forecasts will only use data with timestamps
    before *run_time*.

    Parameters
    ----------
    forecast : datamodel.Forecast
        The metadata of the desired forecast.
    model : function
        NWP model loading and processing function.
        See :py:mod:`solarforecastarbiter.reference_forecasts.models`
        for options.
    run_time : pd.Timestamp
        Run time of the forecast.
    issue_time : pd.Timestamp
        Issue time of the forecast run.

    Returns
    -------
    datamodel.NWPOutput

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
    fetch_metadata = fetch_nwp.model_map[models.get_nwp_model(model)]
    init_time = utils.get_init_time(run_time, fetch_metadata)
    start, end = utils.get_forecast_start_end(forecast, issue_time)
    site = forecast.site

    *forecasts, resampler, solar_position_calculator = model(
        site.latitude, site.longitude, site.elevation,
        init_time, start, end)

    if isinstance(site, datamodel.SolarPowerPlant):
        solar_position = solar_position_calculator()
        ac_power = pvmodel.irradiance_to_power(
            site.modeling_parameters, solar_position['apparent_zenith'],
            solar_position['azimuth'], *forecasts)
    else:
        ac_power = None

    # resample data after power calculation
    resampled = list(map(resampler, (*forecasts, ac_power)))
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
    forecast_start, forecast_end = utils.get_forecast_start_end(
        forecast, issue_time)
    intraday = utils._is_intraday(forecast)
    if not intraday:
        # raise ValueError if not intraday and not midnight to midnight
        utils._check_midnight_to_midnight(forecast_start, forecast_end)

    data_start, data_end = utils.get_data_start_end(
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


def _verify_nwp_forecasts_compatible(fx_group):
    """Verify that all the forecasts grouped by piggyback_on are compatible
    """
    errors = []
    if not len(fx_group.model.unique()) == 1:
        errors.append('model')
    for var in ('issue_time_of_day', 'lead_time_to_start', 'interval_length',
                'run_length', 'interval_label', 'interval_value_type',
                'site'):
        if len({getattr(fx, var) for fx in fx_group.forecast}) > 1:
            errors.append(var)
    return errors


def find_reference_nwp_forecasts(forecasts, run_time=None):
    """
    Sort through all *forecasts* to find those that should be generated
    by the Arbiter from NWP models. The forecast must have a *model* key
    in *extra_parameters* (formated as a JSON string). If *piggyback_on*
    is also defined in *extra_parameters*, it should be the forecast_id
    of another forecast that has the same parameters, including site,
    except the variable.

    Parameters
    ----------
    forecasts : list of datamodel.Forecasts
        The forecasts that should be filtered to find references.
    run_time : pandas.Timestamp or None, default None
        The run_time of that forecast generation is taking place. If not
        None, the next issue time for each forecast is added to the output.

    Returns
    -------
    pandas.DataFrame
        Of NWP reference forecasts With index of forecast_id and columns
        (forecast, piggyback_on, model, next_issue_time).
    """
    df_vals = []
    for fx in forecasts:
        if fx.extra_parameters == '':
            continue
        try:
            extra_parameters = json.loads(fx.extra_parameters)
        except json.JSONDecodeError:
            logger.warning(
                'Failed to decode extra_parameters for %s: %s as JSON',
                fx.name, fx.forecast_id)
            continue

        try:
            model = extra_parameters['model']
        except KeyError:
            if 'piggyback_on' in extra_parameters:
                logger.error(
                    'Forecast, %s: %s, has piggyback_on in extra_parameters'
                    ' but no model. Cannot make forecast.',
                    fx.name, fx.forecast_id)
            else:
                logger.debug(
                    'Not model found for %s:%s, no forecast will be made',
                    fx.name, fx.forecast_id)
            continue

        if run_time is not None:
            next_issue_time = utils.get_next_issue_time(fx, run_time)
        else:
            next_issue_time = None
        piggyback_on = extra_parameters.get('piggyback_on', fx.forecast_id)
        df_vals.append((fx.forecast_id, fx, piggyback_on, model,
                        next_issue_time))

    forecast_df = pd.DataFrame(
        df_vals, columns=['forecast_id', 'forecast', 'piggyback_on', 'model',
                          'next_issue_time']
        ).set_index('forecast_id')
    return forecast_df


def process_nwp_forecast_groups(session, run_time, forecast_df):
    """
    Groups NWP forecasts based on piggyback_on, calculates the forecast as
    appropriate for *run_time*, and uploads the values to the API.

    Parameters
    ----------
    session : io.api.APISession
        API session for uploading forecast values
    run_time : pandas.Timestamp
        Run time of the forecast. Also used along with the forecast metadata
        to determine the issue_time of the forecast.
    forecast_df : pandas.DataFrame
        Dataframe of the forecast objects as procduced by
        :py:func:`solarforecastarbiter.reference_forecasts.main.find_reference_nwp_forecasts``.
    """  # NOQA
    for run_for, group in forecast_df.groupby('piggyback_on'):
        errors = _verify_nwp_forecasts_compatible(group)
        if errors:
            logger.error(
                'Not all forecasts compatible in group with %s. '
                'The following parameters may differ: %s', run_for, errors)
            continue
        try:
            key_fx = group.loc[run_for].forecast
        except KeyError:
            logger.error('Forecast, %s,  that others are piggybacking on not '
                         'found', run_for)
            continue
        model = getattr(models, group.loc[run_for].model)
        issue_time = group.loc[run_for].next_issue_time
        if issue_time is None:
            issue_time = utils.get_next_issue_time(key_fx, run_time)

        nwp_result = run_nwp(key_fx, model, run_time, issue_time)
        for fx_id, fx in group['forecast'].iteritems():
            fx_vals = getattr(nwp_result, fx.variable)
            if fx_vals is None:
                logger.warning('No forecast produced for %s in group with %s',
                               fx_id, run_for)
                continue
            session.post_forecast_values(fx_id, fx_vals)


def make_latest_nwp_forecasts(token, run_time, issue_buffer, base_url=None):
    """
    Make all reference NWP forecasts for *run_time* that are within
    *issue_buffer* of the next issue time for the forecast.

    Parameters
    ----------
    token : str
        Access token for the API
    run_time : pandas.Timestamp
        Run time of the forecast generation
    issue_buffer : pandas.Timedelta
        Maximum time between *run_time* and the next initialization time of
        each forecast that will be updated
    base_url : str or None, default None
        Alternate base_url of the API
    """
    session = api.APISession(token, base_url=base_url)
    forecasts = session.list_forecasts()
    forecast_df = find_reference_nwp_forecasts(forecasts, run_time)
    execute_for = forecast_df[
        forecast_df.next_issue_time <= run_time + issue_buffer]
    if execute_for.empty:
        logger.info('No forecasts to be made at %s', run_time)
        return
    return process_nwp_forecast_groups(session, run_time, execute_for)
