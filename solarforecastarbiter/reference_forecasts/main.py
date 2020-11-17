"""
Make benchmark irradiance and power forecasts.

The functions in this module use the
:py:mod:`solarforecastarbiter.datamodel` objects.
"""
from collections import namedtuple, defaultdict
import itertools
import json
import logging
import re


import pandas as pd


from solarforecastarbiter import datamodel, pvmodel
from solarforecastarbiter.utils import generate_continuous_chunks
from solarforecastarbiter.io import api
from solarforecastarbiter.io.fetch import nwp as fetch_nwp
from solarforecastarbiter.io.utils import adjust_timeseries_for_interval_label
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
    ...     temperature_coefficient=-0.4, dc_loss_factor=0,
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
    ...     interval_value_type='interval_mean',
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


def _default_load_data(session):
    def load_data(observation, data_start, data_end):
        df = session.get_observation_values(observation.observation_id,
                                            data_start, data_end,
                                            observation.interval_label)
        df = df.tz_convert(observation.site.timezone)
        return df['value']
    return load_data


def run_persistence(session, observation, forecast, run_time, issue_time,
                    index=False, load_data=None):
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

    - Intraday persistence forecasts:

      + ``window = forecast.run_length``. No longer than 1 hour.

    - Day ahead forecasts (all but net load) and week ahead forecasts (net
      load only):

      + ``window = forecast.interval_length``.

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
    load_data : function
        Function to load the observation data 'value' series given
        (observation, data_start, data_end) arguments. Typically,
        calls `session.get_observation_values` and selects the 'value'
        column. May also have data preloaded to then slice from
        data_start to data_end.

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
        If data is required from after run_time.
    ValueError
        If persistence window < observation.interval_length.
    ValueError
        If forecast.run_length => 1 day and index=True.
    ValueError
        If instantaneous forecast and instantaneous observation interval
        lengths do not match.
    ValueError
        If average observations are used to make instantaneous forecast.

    Notes
    -----
    For non-intraday net load forecasts, this function will use a weekahead
    persistence due to the fact that net load exhibits stronger correlation
    week-to-week than day-to-day. For example, the net load on a Monday tends
    to look more similar to the previous Monday that it does to the previous
    day (Sunday).
    """
    utils.check_persistence_compatibility(observation, forecast, index)
    forecast_start, forecast_end = utils.get_forecast_start_end(
        forecast, issue_time, False)
    intraday = utils._is_intraday(forecast)

    if load_data is None:
        load_data = _default_load_data(session)
    data_start, data_end = utils.get_data_start_end(
        observation, forecast, run_time, issue_time)
    if data_end > run_time:
        raise ValueError(
            'Persistence forecast requires data from after run_time')

    if isinstance(forecast, datamodel.ProbabilisticForecast):
        cvs = [f.constant_value for f in forecast.constant_values]
        fx = persistence.persistence_probabilistic(
            observation, data_start, data_end, forecast_start, forecast_end,
            forecast.interval_length, forecast.interval_label, load_data,
            forecast.axis, cvs)
    elif intraday and index:
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
    else:  # pragma: no cover
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
    match = re.search(r'is_reference_forecast(["\s\:]*)true',
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


def _post_forecast_values(session, fx, fx_vals, model_str):
    if isinstance(fx, datamodel.ProbabilisticForecast):
        if not model_str.startswith('gefs'):
            raise ValueError(
                'Can only process probabilisic forecast from GEFS')

        if not isinstance(fx_vals, pd.DataFrame) or len(fx_vals.columns) != 21:
            raise TypeError(
                'Could not post probabilistic forecast values: '
                'forecast values in unknown format')
        # adjust columns to be constant values
        cv_df = fx_vals.rename(columns={i: i * 5.0 for i in range(22)})
        for cv_fx in fx.constant_values:
            # will raise a KeyError if no match
            cv_vals = cv_df[cv_fx.constant_value]
            logger.debug('Posting %s values to %s', len(cv_vals),
                         cv_fx.forecast_id)
            session.post_probabilistic_forecast_constant_value_values(
                cv_fx.forecast_id, cv_vals
            )
    else:
        session.post_forecast_values(fx.forecast_id, fx_vals)


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
        _process_single_group(session, run_for, group, run_time)


def _process_single_group(session, run_for, group, run_time):
    logger.info('Computing forecasts for group %s at %s', run_for, run_time)
    errors = _verify_nwp_forecasts_compatible(group)
    if errors:
        logger.error(
            'Not all forecasts compatible in group with %s. '
            'The following parameters may differ: %s', run_for, errors)
        return
    try:
        key_fx = group.loc[run_for].forecast
    except KeyError:
        logger.error('Forecast, %s, that others are piggybacking on not '
                     'found', run_for)
        return
    model_str = group.loc[run_for].model
    model = getattr(models, model_str)
    issue_time = group.loc[run_for].next_issue_time
    if issue_time is None:
        issue_time = utils.get_next_issue_time(key_fx, run_time)
    try:
        nwp_result = run_nwp(key_fx, model, run_time, issue_time)
    except FileNotFoundError as e:
        logger.error('Could not process group of %s, %s', run_for, str(e))
        return
    for fx_id, fx in group['forecast'].iteritems():
        fx_vals = getattr(nwp_result, fx.variable)
        if fx_vals is None:
            logger.warning('No forecast produced for %s in group with %s',
                           fx_id, run_for)
            continue
        logger.info('Posting values %s for %s:%s issued at %s',
                    len(fx_vals), fx.name, fx_id, issue_time)
        _post_forecast_values(session, fx, fx_vals, model_str)


def make_latest_nwp_forecasts(token, run_time, issue_buffer, base_url=None):
    """
    Make all reference NWP forecasts for *run_time* that are within
    *issue_buffer* of the next issue time for the forecast. For example,
    this function may run in a cronjob every five minutes with *run_time*
    set to now. By setting *issue_buffer* to '5min', only forecasts that
    should be issued in the next five minutes will be generated on each
    run. Only forecasts that belong to the same provider/organization
    of the token user will be updated.

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
    session, forecast_df = _get_nwp_forecast_df(token, run_time, base_url)
    execute_for = forecast_df[
        forecast_df.next_issue_time <= run_time + issue_buffer]
    if execute_for.empty:
        logger.info('No forecasts to be made at %s', run_time)
        return
    process_nwp_forecast_groups(session, run_time, execute_for)


def _get_nwp_forecast_df(token, run_time, base_url):
    session = api.APISession(token, base_url=base_url)
    user_info = session.get_user_info()
    forecasts = session.list_forecasts()
    forecasts += session.list_probabilistic_forecasts()
    forecasts = [fx for fx in forecasts
                 if fx.provider == user_info['organization']]
    forecast_df = find_reference_nwp_forecasts(forecasts, run_time)
    return session, forecast_df


def _nwp_issue_time_generator(fx, gap_start, gap_end):
    # max_run_time is the forecast issue time that will generate forecast
    # that end before gap_end
    max_run_time = utils.find_next_issue_time_from_last_forecast(
        fx, gap_end - pd.Timedelta('1ns'))
    # next_issue_time is the forecast issue time that will generate forecast
    # values at gap_start
    next_issue_time = utils.find_next_issue_time_from_last_forecast(
            fx, gap_start)
    while next_issue_time < max_run_time:
        yield next_issue_time
        next_issue_time = utils.get_next_issue_time(
            fx, next_issue_time + pd.Timedelta('1ns'))


def _find_group_gaps(session, forecasts, start, end):
    times = set()
    for forecast in forecasts:
        gaps = session.get_value_gaps(forecast, start, end)
        for gap in gaps:
            times |= set(_nwp_issue_time_generator(
                forecast, gap[0], gap[1]))
    return sorted(times)


def fill_nwp_forecast_gaps(token, start, end, base_url=None):
    """
    Make all reference NWP forecasts that are missing from *start* to *end*.
    Only forecasts that belong to the same provider/organization
    of the token user will be updated.

    Parameters
    ----------
    token : str
        Access token for the API
    start : pandas.Timestamp
        Start of the period to check and fill forecast gaps
    end : pandas.Timestamp
        End of the period to check and fill forecast gaps
    base_url : str or None, default None
        Alternate base_url of the API
    """
    session, forecast_df = _get_nwp_forecast_df(token, None, base_url)
    # go through each group separately
    for run_for, group in forecast_df.groupby('piggyback_on'):
        issue_times = _find_group_gaps(session, group.forecast.to_list(),
                                       start, end)
        group = group.copy()
        for issue_time in issue_times:
            group.loc[:, 'next_issue_time'] = issue_time
            _process_single_group(session, run_for, group, issue_time)


def _is_reference_persistence_forecast(extra_params_string):
    match = re.search(r'is_reference_persistence_forecast(["\s\:]*)true',
                      extra_params_string, re.I)
    return match is not None


def _ref_persistence_check(fx, observation_dict, user_info, session):
    if not _is_reference_persistence_forecast(fx.extra_parameters):
        logger.debug(
            'Forecast %s is not labeled as a reference '
            'persistence forecast',  fx.forecast_id)
        return

    if not fx.provider == user_info['organization']:
        logger.debug(
            "Forecast %s is not in user's organization",
            fx.forecast_id)
        return

    try:
        extra_parameters = json.loads(fx.extra_parameters)
    except json.JSONDecodeError:
        logger.warning(
            'Failed to decode extra_parameters for %s: %s as JSON',
            fx.name, fx.forecast_id)
        return

    try:
        observation_id = extra_parameters['observation_id']
    except KeyError:
        logger.error(
            'Forecast, %s: %s, has no observation_id to base forecasts'
            ' off of. Cannot make persistence forecast.',
            fx.name, fx.forecast_id)
        return
    if observation_id not in observation_dict:
        logger.error(
            'Observation %s not in set of given observations.'
            ' Cannot generate persistence forecast for %s: %s.',
            observation_id, fx.name, fx.forecast_id)
        return
    observation = observation_dict[observation_id]

    index = extra_parameters.get('index_persistence', False)
    obs_mint, obs_maxt = session.get_observation_time_range(observation_id)
    if pd.isna(obs_maxt):  # no observations to use anyway
        logger.info(
            'No observation values to use for %s: %s from observation %s',
            fx.name, fx.forecast_id, observation_id)
        return
    return observation, index, obs_mint, obs_maxt


def generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time):
    """Sort through all *forecasts* to find those that should be generated
    by the Arbiter from persisting Observation values. The forecast
    must have ``'is_reference_persistence_forecast': true`` and an
    observation_id in Forecast.extra_parameters (formatted as a JSON
    string). A boolean value for "index_persistence" in
    Forecast.extra_parameters controls whether the persistence
    forecast should be made adjusting for clear-sky/AC power index or
    not.

    Parameters
    ----------
    session : solarforecastarbiter.io.api.APISession
    forecasts : list of datamodel.Forecasts
        The forecasts that should be filtered to find references.
    observations : list of datamodel.Observations
        Observations that will are available to use to fetch values
        and make persistence forecasts.
    max_run_time : pandas.Timestamp
        The maximum run time/issue time for any forecasts. Usually now.

    Returns
    -------
    generator of (Forecast, Observation, index, data_start, issue_times)
    """
    user_info = session.get_user_info()
    observation_dict = {obs.observation_id: obs for obs in observations}
    out = namedtuple(
        'PersistenceParameters',
        ['forecast', 'observation', 'index', 'data_start',
         'issue_times'])

    for fx in forecasts:
        obs_ind_mint_maxt = _ref_persistence_check(
            fx, observation_dict, user_info, session)
        if obs_ind_mint_maxt is None:
            continue
        observation, index, obs_mint, obs_maxt = obs_ind_mint_maxt
        # probably split this out to generate issues times for only gaps vs
        # latest
        if isinstance(fx, datamodel.ProbabilisticForecast):
            fx_mint, fx_maxt = \
                session.get_probabilistic_forecast_constant_value_time_range(
                    fx.constant_values[0].forecast_id)
        else:
            fx_mint, fx_maxt = session.get_forecast_time_range(fx.forecast_id)
        # find the next issue time for the forecast based on the last value
        # in the forecast series
        if pd.isna(fx_maxt):
            # if there is no forecast yet, go back a bit from the last
            # observation. Don't use the start of observations, since it
            # could really stress the workers if we have a few years of
            # data before deciding to make a persistence fx
            next_issue_time = utils.get_next_issue_time(
                fx, obs_maxt - fx.run_length)
        else:
            next_issue_time = utils.find_next_issue_time_from_last_forecast(
                fx, fx_maxt)

        data_start, _ = utils.get_data_start_end(
            observation, fx, next_issue_time, next_issue_time)
        issue_times = tuple(_issue_time_generator(
            observation, fx, obs_mint, obs_maxt,
            next_issue_time, max_run_time))

        if len(issue_times) == 0:
            continue

        yield out(fx, observation, index, data_start, issue_times)


def _issue_time_generator(observation, fx, obs_mint, obs_maxt, next_issue_time,
                          max_run_time):
    # now find all the run times that can be made based on the
    # last observation timestamp
    while next_issue_time <= max_run_time:
        data_start, data_end = utils.get_data_start_end(
            observation, fx, next_issue_time, next_issue_time)
        if data_end > obs_maxt:
            break

        if data_start > obs_mint:
            yield next_issue_time
        next_issue_time = utils.get_next_issue_time(
            fx, next_issue_time + pd.Timedelta('1ns'))


def _preload_load_data(session, obs, data_start, data_end):
    """Fetch all the data required at once and slice as appropriate.
    Much more efficient when generating many persistence forecasts from
    the same observation.
    """
    obs_data = session.get_observation_values(
        obs.observation_id, data_start, data_end
    ).tz_convert(obs.site.timezone)['value']

    def load_data(observation, data_start, data_end):
        data = obs_data.loc[data_start:data_end]
        return adjust_timeseries_for_interval_label(
            data, observation.interval_label, data_start, data_end)
    return load_data


def make_latest_persistence_forecasts(token, max_run_time, base_url=None):
    """Make all reference persistence forecasts that need to be made up to
    *max_run_time*.

    Parameters
    ----------
    token : str
        Access token for the API
    max_run_time : pandas.Timestamp
        Last possible run time of the forecast generation
    base_url : str or None, default None
        Alternate base_url of the API
    """
    session = api.APISession(token, base_url=base_url)
    forecasts = session.list_forecasts()
    observations = session.list_observations()
    params = generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time)
    for fx, obs, index, data_start, issue_times in params:
        _pers_loop(session, fx, obs, index, data_start, max_run_time,
                   issue_times)


def _pers_loop(session, fx, obs, index, data_start, data_end, issue_times):
    load_data = _preload_load_data(session, obs, data_start, data_end)
    out = defaultdict(list)
    logger.info('Making persistence forecast for %s:%s from %s to %s',
                fx.name, fx.forecast_id, issue_times[0], issue_times[-1])
    for issue_time in issue_times:
        run_time = issue_time
        try:
            fx_out = run_persistence(
                session, obs, fx, run_time, issue_time,
                index=index, load_data=load_data)
        except ValueError as e:
            logger.error('Unable to generate persistence forecast: %s', e)
        else:
            if hasattr(fx, 'constant_values'):
                cv_ids = [f.forecast_id for f in fx.constant_values]
                for id_, fx_ser in zip(cv_ids, fx_out):
                    out[id_].append(fx_ser)
            else:
                out[fx.forecast_id].append(fx_out)
    for id_, serlist in out.items():
        if len(serlist) > 0:
            ser = pd.concat(serlist)
            for cser in generate_continuous_chunks(ser, fx.interval_length):
                if type(fx) == datamodel.Forecast:
                    session.post_forecast_values(id_, cser)
                else:
                    session.post_probabilistic_forecast_constant_value_values(
                        id_, cser)


def make_latest_probabilistic_persistence_forecasts(
        token, max_run_time, base_url=None):
    """Make all reference probabilistic persistence forecasts that need to
    be made up to *max_run_time*.

    Parameters
    ----------
    token : str
        Access token for the API
    max_run_time : pandas.Timestamp
        Last possible run time of the forecast generation
    base_url : str or None, default None
        Alternate base_url of the API
    """
    session = api.APISession(token, base_url=base_url)
    forecasts = session.list_probabilistic_forecasts()
    observations = session.list_observations()
    params = generate_reference_persistence_forecast_parameters(
        session, forecasts, observations, max_run_time)
    for fx, obs, index, data_start, issue_times in params:
        _pers_loop(session, fx, obs, index, data_start, max_run_time,
                   issue_times)


def generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end):
    """Sort through all *forecasts* to find those with gaps in the data
    that should be generated by the Arbiter from persisting
    Observation values. The forecast must have
    ``'is_reference_persistence_forecast': true`` and an
    observation_id in Forecast.extra_parameters (formatted as a JSON
    string). A boolean value for "index_persistence" in
    Forecast.extra_parameters controls whether the persistence
    forecast should be made adjusting for clear-sky/AC power index or
    not.

    Parameters
    ----------
    session : solarforecastarbiter.io.api.APISession
    forecasts : list of datamodel.Forecasts
        The forecasts that should be filtered to find references.
    observations : list of datamodel.Observations
        Observations that will are available to use to fetch values
        and make persistence forecasts.
    start : pandas.Timestamp
        The start of the period to search for missing forecast values.
    end : pandas.Timestamp
        The end of the period to search for missing forecast values.

    Returns
    -------
    generator of (Forecast, Observation, index, data_start, data_end, issue_times)

    """  # NOQA: E501
    user_info = session.get_user_info()
    observation_dict = {obs.observation_id: obs for obs in observations}
    out = namedtuple(
        'PersistenceGapParameters',
        ['forecast', 'observation', 'index', 'data_start', 'data_end',
         'issue_times'])
    for fx in forecasts:
        obs_ind_mint_maxt = _ref_persistence_check(
            fx, observation_dict, user_info, session)
        if obs_ind_mint_maxt is None:
            continue
        observation, index, obs_mint, obs_maxt = obs_ind_mint_maxt

        times = set()
        gaps = session.get_value_gaps(fx, start, end)
        for gap in gaps:
            times |= set(_issue_time_generator(
                observation, fx, obs_mint, obs_maxt, gap[0],
                gap[1] - pd.Timedelta('1ns')))
        issue_times = tuple(sorted(times))
        if len(issue_times) == 0:
            continue

        # get_data_start_end only looks for start/end of a single
        # forecast run, so need to do for first and last issue times
        # to get full range of data possibly needed
        data_start, _ = utils.get_data_start_end(
            observation, fx, issue_times[0], issue_times[0])
        _, data_end = utils.get_data_start_end(
            observation, fx, issue_times[-1], issue_times[-1])
        yield out(fx, observation, index, data_start, data_end, issue_times)


def _fill_persistence_gaps(token, start, end, base_url, forecast_fnc):
    session = api.APISession(token, base_url=base_url)
    forecasts = getattr(session, forecast_fnc)()
    observations = session.list_observations()
    params = generate_reference_persistence_forecast_gaps_parameters(
        session, forecasts, observations, start, end)
    for fx, obs, index, data_start, data_end, issue_times in params:
        _pers_loop(session, fx, obs, index, data_start, data_end,
                   issue_times)


def fill_persistence_forecasts_gaps(token, start, end, base_url=None):
    """Make all reference persistence forecasts that need to be made
    between start and end.

    Parameters
    ----------
    token : str
        Access token for the API
    start : pandas.Timestamp
        The start of the period to search for missing forecast values.
    end : pandas.Timestamp
        The end of the period to search for missing forecast values.
    base_url : str or None, default None
        Alternate base_url of the API

    """
    _fill_persistence_gaps(token, start, end, base_url, 'list_forecasts')


def fill_probabilistic_persistence_forecasts_gaps(
        token, start, end, base_url=None):
    """Make all reference probabilistic persistence forecasts that need to
    be made between start and end.

    Parameters
    ----------
    token : str
        Access token for the API
    start : pandas.Timestamp
        The start of the period to search for missing forecast values.
    end : pandas.Timestamp
        The end of the period to search for missing forecast values.
    base_url : str or None, default None
        Alternate base_url of the API

    """
    _fill_persistence_gaps(token, start, end, base_url,
                           'list_probabilistic_forecasts')
