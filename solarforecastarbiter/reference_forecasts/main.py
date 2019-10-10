"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
from collections import namedtuple
import itertools
import json
import logging
import re


import pandas as pd


from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.io import api
from solarforecastarbiter.io.fetch import nwp as fetch_nwp
from solarforecastarbiter.reference_forecasts import persistence, models, utils


logger = logging.getLogger(__name__)


def run_nwp(forecast, model, run_time, issue_time):
    """
    Calculate benchmark irradiance and power forecasts for a Forecast or
    ProbabilisticForecast.

    Forecasts may be run operationally or retrospectively. For
    operational forecasts, *run_time* is typically set to now. For
    retrospective forecasts, *run_time* is the time by which the
    forecast should be run so that it could have been be delivered for
    the *issue_time*. Forecasts will only use data with timestamps
    before *run_time*.

    Parameters
    ----------
    forecast : datamodel.Forecast or datamodel.ProbabilisticForecast
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
    ghi : pd.Series or pd.DataFrame
    dni : pd.Series or pd.DataFrame
    dhi : pd.Series or pd.DataFrame
    air_temperature : pd.Series or pd.DataFrame
    wind_speed : pd.Series or pd.DataFrame
    ac_power : pd.Series or pd.DataFrame

    Series are returned for deterministic forecasts, DataFrames are
    returned for probabilisic forecasts.

    Examples
    --------
    The following code would return hourly average forecasts derived
    from the subhourly HRRR model.

    .. testsetup::

       import datetime
       from solarforecastarbiter import datamodel
       from solarforecastarbiter.reference_forecasts import models
       from solarforecastarbiter.reference_forecasts.main import *

    >>> run_time = pd.Timestamp('20190515T0200Z')
    >>> issue_time = pd.Timestamp('20190515T0000Z')
    >>> modeling_parameters = datamodel.FixedTiltModelingParameters(
    ...     surface_tilt=30, surface_azimuth=180,
    ...     ac_capacity=10, dc_capacity=15,
    ...     temperature_coefficient=-0.004, dc_loss_factor=0,
    ...     ac_loss_factor=0)
    >>> power_plant = datamodel.SolarPowerPlant(
    ...     name='Test plant', latitude=32.2, longitude=-110.9,
    ...     elevation=715, timezone='America/Phoenix',
    ...     modeling_parameters=modeling_parameters)
    >>> forecast = datamodel.Forecast(
    ...     name='Test plant fx',
    ...     site=power_plant,
    ...     variable='ac_power',
    ...     interval_label='ending',
    ...     interval_value_type='mean',
    ...     interval_length='1h',
    ...     issue_time_of_day=datetime.time(hour=0),
    ...     run_length=pd.Timedelta('24h'),
    ...     lead_time_to_start=pd.Timedelta('0h'))
    >>> ghi, dni, dhi, temp_air, wind_speed, ac_power = run_nwp(
    ...     forecast, models.hrrr_subhourly_to_hourly_mean,
    ...     run_time, issue_time)
    """
    fetch_metadata = fetch_nwp.model_map[models.get_nwp_model(model)]
    # absolute date and time for model run most recently available
    # as of run_time
    init_time = utils.get_init_time(run_time, fetch_metadata)
    # absolute start and end times. interval_label still controls
    # inclusive/exclusive
    start, end = utils.get_forecast_start_end(forecast, issue_time)
    site = forecast.site
    logger.info(
        'Calculating forecast for model %s starting at %s from %s to %s',
        model, init_time, start, end)
    # model will account for interval_label
    *forecasts, resampler, solar_position_calculator = model(
        site.latitude, site.longitude, site.elevation,
        init_time, start, end, forecast.interval_label)

    if isinstance(site, datamodel.SolarPowerPlant):
        solar_position = solar_position_calculator()
        if isinstance(forecasts[0], pd.DataFrame):
            # must iterate over columns because pvmodel.irradiance_to_power
            # calls operations that do not properly broadcast Series along
            # a DataFrame time index. pvlib.irradiance.haydavies operation
            # (AI = dni_ens / dni_extra) is the known culprit, though there
            # may be more.
            ac_power = {}
            for col in forecasts[0].columns:
                member_fx = [fx.get(col) for fx in forecasts]
                member_ac_power = pvmodel.irradiance_to_power(
                    site.modeling_parameters,
                    solar_position['apparent_zenith'],
                    solar_position['azimuth'],
                    *member_fx)
                ac_power[col] = member_ac_power
            ac_power = pd.DataFrame(ac_power)
        else:
            ac_power = pvmodel.irradiance_to_power(
                site.modeling_parameters, solar_position['apparent_zenith'],
                solar_position['azimuth'], *forecasts)
    else:
        ac_power = None

    # resample data after power calculation
    resampled = list(map(resampler, (*forecasts, ac_power)))
    nwpoutput = namedtuple(
        'NWPOutput', ['ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed',
                      'ac_power'])
    return nwpoutput(*resampled)


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
           *window = forecast.run_length*.
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
        forecast, issue_time, False)
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


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def _verify_nwp_forecasts_compatible(fx_group):
    """Verify that all the forecasts grouped by piggyback_on are compatible
    """
    errors = []
    if not len(fx_group.model.unique()) == 1:
        errors.append('model')
    for var in ('issue_time_of_day', 'lead_time_to_start', 'interval_length',
                'run_length', 'interval_label', 'interval_value_type',
                'site'):
        if not all_equal(getattr(fx, var) for fx in fx_group.forecast):
            errors.append(var)
    return errors


def _is_reference_forecast(extra_params_string):
    match = re.search('is_reference_forecast(["\\s\\:]*)true',
                      extra_params_string, re.I)
    return match is not None


def find_reference_nwp_forecasts(forecasts, run_time=None):
    """
    Sort through all *forecasts* to find those that should be generated
    by the Arbiter from NWP models. The forecast must have a *model* key
    in *extra_parameters* (formatted as a JSON string). If *piggyback_on*
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
        NWP reference forecasts with index of forecast_id and columns
        (forecast, piggyback_on, model, next_issue_time).
    """
    df_vals = []
    for fx in forecasts:
        # more explicit than filter()
        if not _is_reference_forecast(fx.extra_parameters):
            logger.debug('Forecast %s is not labeled as a reference forecast',
                         fx.forecast_id)
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
            logger.error(
                'Forecast, %s: %s, has no model. Cannot make forecast.',
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
        :py:func:`solarforecastarbiter.reference_forecasts.main.find_reference_nwp_forecasts`.
    """  # NOQA
    for run_for, group in forecast_df.groupby('piggyback_on'):
        logger.info('Computing forecasts for group %s', run_for)
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
        try:
            nwp_result = run_nwp(key_fx, model, run_time, issue_time)
        except FileNotFoundError as e:
            logger.error('Could not process group of %s, %s', run_for, str(e))
            continue
        for fx_id, fx in group['forecast'].iteritems():
            fx_vals = getattr(nwp_result, fx.variable)
            if fx_vals is None:
                logger.warning('No forecast produced for %s in group with %s',
                               fx_id, run_for)
                continue
            logger.info('Posting values %s for %s:%s issued at %s',
                        len(fx_vals), fx.name, fx_id, issue_time)
            session.post_forecast_values(fx_id, fx_vals)


def make_latest_nwp_forecasts(token, run_time, issue_buffer, base_url=None):
    """
    Make all reference NWP forecasts for *run_time* that are within
    *issue_buffer* of the next issue time for the forecast. For example,
    this function may run in a cronjob every five minutes with *run_time*
    set to now. By setting *issue_buffer* to '5min', only forecasts that
    should be issued in the next five minutes will be generated on each
    run.

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
    process_nwp_forecast_groups(session, run_time, execute_for)
