import datetime as dt
from functools import lru_cache, partial
import json
import logging


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.utils import merge_ranges
from solarforecastarbiter.datamodel import Observation, ProbabilisticForecast
from solarforecastarbiter.io.reference_observations.default_forecasts import (
    CURRENT_NWP_VARIABLES, is_in_nwp_domain)
from solarforecastarbiter.reference_forecasts.utils import (
    check_persistence_compatibility)


logger = logging.getLogger('reference_data')


def decode_extra_parameters(metadata):
    """Returns a dictionary parsed from the json string stored
    in extra_parameters

    Parameters
    ----------
    metadata
        A SolarForecastArbiter.datamodel class with an extra_parameters
        attribute

    Returns
    -------
    dict
        The extra parameters as a python dictionary

    Raises
    ------
    ValueError
        If parameters cannot be decoded or are None. Or if missing the
        required keys: network, network_api_id, network_api_abbreviation
        and observation_interval_length.
    """
    try:
        params = json.loads(metadata.extra_parameters)
    except (json.decoder.JSONDecodeError, TypeError):
        raise ValueError(f'Could not read extra parameters of {metadata.name}')
    required_keys = ['network', 'network_api_id', 'network_api_abbreviation',
                     'observation_interval_length']
    if not all([key in params for key in required_keys]):
        raise ValueError(f'{metadata.name} is missing required extra '
                         'parameters.')
    return params


def check_network(networks, metadata):
    """Decodes extra_parameters and checks if an object is in
    the network.

    Parameters
    ----------
    networks: list of str
        A list of networks to check against.
    metadata
        An instantiated dataclass from the datamodel.

    Returns
    -------
    bool
        True if the site belongs to the network, or one of the
        networks.
    """
    if type(networks) == str:
        networks = [networks]
    try:
        extra_params = decode_extra_parameters(metadata)
    except ValueError:
        return False
    try:
        in_network = extra_params['network'] in networks
    except KeyError:
        return False
    else:
        return in_network


@lru_cache(maxsize=4)
def existing_observations(api):
    return {obs.name: obs for obs in api.list_observations()}


@lru_cache(maxsize=4)
def existing_forecasts(api):
    out = {fx.name: fx for fx in api.list_forecasts()}
    out.update({fx.name: fx for fx in api.list_probabilistic_forecasts()})
    return out


@lru_cache(maxsize=4)
def existing_sites(api):
    return {site.name: site for site in api.list_sites()}


def filter_by_networks(object_list, networks):
    """Returns a copy of object_list with all objects that are not in the
    network removed.

    Parameters
    ----------
    object_list: list
        List of datamodel objects.
    networks: string or list
        Network or list of networks to check for.

    Returns
    -------
    filtered
        List of filtered datamodel objects.
    """
    filtered = [obj for obj in object_list if check_network(networks, obj)]
    return filtered


def create_observation(api, site, variable, extra_params=None, **kwargs):
    """ Creates a new Observation for the variable and site. Kwargs can be
    provided to overwrite the default arguments to the Observation constructor.
    Kwarg options are documented in 'Other Parameters' below but users should
    reference the SolarForecastArbiter API for valid Observation field values.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.site
        A site object.
    variable : string
        Variable measured in the observation.
    extra_params : dict, optional
        If provided, this dict will be serialized as the 'extra_parameters'
        field of the observation, otherwise the site's field is copied over.
        Must contain the keys 'network_api_length', 'network_api_id', and
        'observation_interval_length'.

    Other Parameters
    ----------------
    name: string
        Defaults to `<site.name> <variable>`
    interval_label: string
        Defaults to 'ending'
    interval_value_type: string
        Defaults to 'interval_mean'
    uncertainty: float or None
        Defaults to None.

    Returns
    -------
    created
        The datamodel object of the newly created observation.

    Raises
    ------
    KeyError
        When the extra_parameters, either loaded from the site or provided
        by the user is missing 'network_api_abbreviation'
        or 'observation_interval_length'

    """
    # Copy network api data from the site, and get the observation's
    # interval length
    if extra_params:
        extra_parameters = extra_params
    else:
        try:
            extra_parameters = decode_extra_parameters(site)
        except ValueError:
            logger.warning(f'Cannot create observations for site {site.name}'
                           'missing required extra parameters')
            return
    site_name = site_name_no_network(site)
    observation_name = f'{site_name} {variable}'
    # Some site names are too long and exceed the API's limits,
    # in those cases. Use the abbreviated version.
    if len(observation_name) > 64:
        site_abbreviation = extra_parameters["network_api_abbreviation"]
        observation_name = f'{site_abbreviation} {variable}'

    observation = Observation.from_dict({
        'name': kwargs.get('name', observation_name),
        'interval_label': kwargs.get('interval_label', 'ending'),
        'interval_length': extra_parameters['observation_interval_length'],
        'interval_value_type': kwargs.get('interval_value_type',
                                          'interval_mean'),
        'site': site,
        'uncertainty': kwargs.get('uncertainty'),
        'variable': variable,
        'extra_parameters': json.dumps(extra_parameters)
    })

    return check_and_post_observation(api, observation)


def check_and_post_observation(api, observation):
    existing = existing_observations(api)
    if observation.name in existing:
        logger.info('Observation, %s, already exists', observation.name)
        return existing[observation.name]

    try:
        created = api.create_observation(observation)
    except HTTPError as e:
        logger.error(f'Failed to create {observation.variable} observation '
                     f'at Site {observation.site.name}.')
        logger.debug(f'HTTP Error: {e.response.text}')
    else:
        logger.info(f"Observation {created.name} created successfully.")
        return created


def _utcnow():
    return pd.Timestamp.now(tz='UTC')


def get_last_site_timestamp(api, observations, end):
    """Get the last value timestamp from the API as the minimum of the
    last timestamp for each observation at that site. The result of
    this function is often used to make new data queries, so a limit
    of end - 7 days is set to avoid excessive queries to external
    sources.

    Parameters
    ---------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site_observations : list of solarforecastarbiter.datamodel.Observation
        A list of reference Observations for a site to search.
    end : pd.Timestamp
        Typically, set to now.

    Returns
    -------
    pandas.Timestamp

    """
    # update cli.py as appropriate if behaviour is changed
    out = end
    updated = False
    for obs in observations:
        maxt = api.get_observation_time_range(obs.observation_id)[1]
        # <= so that even if maxt == end updated -> true
        # effectively ignores all NaT values unless all observations
        # for the site are NaT, then use weekago
        if pd.notna(maxt) and maxt <= out:
            out = maxt
            updated = True

    weekago = end - pd.Timedelta('7d')
    if not updated or out < weekago:
        out = weekago
    return out


def update_site_observations(api, fetch_func, site, observations,
                             start, end, *, gaps_only=False):
    """Updates data for all reference observations at a given site
    for the period between start and end.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    fetch_func : function
        A function that requests data and returns a DataFrame for a given site.
        The function should accept the parameters (api, site, start, end) as
        they appear in this function.
    site : solarforecastarbiter.datamodel.Site
        The Site with observations to update.
    observations : list of solarforecastarbiter.datamodel.Observation
        A full list of reference Observations to search.
    start : pandas.Timestamp or None
        Start time to get data for. If None, try finding the last
        value in the API and use that time (with a limit of 7 days from start
        to end). If None and no values in the API, use end - 7 day.
    end : pandas.Timestamp or None
        End time to get data for. If None, use now.
    gaps_only : bool, default False
        If True, only update periods between start and end where there
        are data gaps.
    """
    site_observations = [obs for obs in observations if obs.site == site]
    if end is None:
        end = _utcnow()
    if start is None:
        start = get_last_site_timestamp(api, site_observations, end)
    if gaps_only:
        gaps = _find_data_gaps(api, site_observations, start, end)
        for gstart, gend in gaps:
            _post_data(api, fetch_func, site, site_observations, gstart, gend)
    else:
        _post_data(api, fetch_func, site, site_observations, start, end)


def _find_data_gaps(api, site_observations, start, end):
    ranges = []
    for obs in site_observations:
        ranges += api.get_observation_value_gaps(
            obs.observation_id, start, end)
    return merge_ranges(ranges)


def _post_data(api, fetch_func, site, site_observations, start, end):
    logger.debug('Fetching data for %s from %s to %s', site.name, start, end)
    obs_df = fetch_func(api, site, start, end)
    # must be sorted for proper inexact start:end slicing
    data_in_range = obs_df.sort_index()[start:end]
    if data_in_range.empty:
        return
    for obs in site_observations:
        post_observation_data(api, obs, data_in_range, start, end)


def _prepare_data_to_post(data, variable, observation, start, end,
                          resample_how=None):
    """Manipulate the data including reindexing to observation.interval_label
    to prepare for posting"""
    data = data[[variable]]
    data = data.rename(columns={variable: 'value'})
    # ensure data is sorted before slicing and for optimal order in the
    # database
    data = data.sort_index()

    if resample_how:
        resampler = data.resample(observation.interval_length)
        data = getattr(resampler, resample_how)()

    # remove all future values, some files have forward filled nightly data
    data = data[start:min(end, _utcnow())]

    if data.empty:
        return data
    # reindex the data to put nans where required
    # we don't extend the new index to start, end, since reference
    # data has some lag time from the end it was requested from
    # and it isn't necessary to keep the nans between uploads in db
    new_index = pd.date_range(start=data.index[0], end=data.index[-1],
                              freq=observation.interval_length)
    data = data.reindex(new_index)
    # set quality flags
    data['quality_flag'] = data['value'].isna().astype(int)
    return data


def post_observation_data(api, observation, data, start, end):
    """Posts data to an observation between start and end.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    observation : solarforecastarbiter.datamodel.Observation
        Data model object corresponding to the Observation to update.
    data : pandas.DataFrame
        Dataframe of values to post containing a column labeled with
        the Observation's variable.
    start : datetime-like
        The beginning of the period to update.
    end : datetime-like
        The end of the period to update.

    Raises
    ------
    IndexError
        If the data provided has an empty index.
    """
    logger.info(
        f'Updating {observation.name} from '
        f'{start} to {end}.')
    try:
        extra_parameters = decode_extra_parameters(observation)
    except ValueError:
        return
    # check for a non-standard variable label in extra_parameters
    variable = extra_parameters.get('network_data_label',
                                    observation.variable)
    # check if the raw observation needs to be resampled before posting
    resample_how = extra_parameters.get('resample_how', None)
    try:
        var_df = _prepare_data_to_post(data, variable, observation,
                                       start, end, resample_how=resample_how)
    except KeyError:
        logger.error(f'{variable} could not be found in the data file '
                     f'from {data.index[0]} to {data.index[-1]}'
                     f'for Observation {observation.name}')
        return
    except AttributeError:
        logger.error(f'{variable} could not be resampled using method '
                     f'{resample_how} for Observation {observation.name}')
        return

    # skip post id data is empty, if there are nans, should still post
    if var_df.empty:
        logger.warning(
            f'{observation.name} data empty from '
            f'{data.index[0]} to {data.index[-1]}.')
        return
    try:
        logger.debug(f'Posting data to {observation.name} between '
                     f'{var_df.index[0]} and {var_df.index[-1]}.')
        api.post_observation_values(observation.observation_id, var_df)
    except HTTPError as e:
        logger.error(f'Posting data to {observation.name} failed.')
        logger.debug(f'HTTP Error: {e.response.text}.')


def clean_name(string):
    """Removes all disallowed characters from a string and converts
    underscores to spaces.
    """
    return string.translate(string.maketrans('_', ' ', ':(){}/\\[]@-.'))


def site_name_no_network(site):
    """Removes the prefixed network from a site name for prepending
    to an observation.
    """
    extra_params = decode_extra_parameters(site)
    network = extra_params['network']
    # only select the site name after the network name and a space.
    if site.name.startswith(network):
        return site.name[len(network) + 1:]
    else:
        return site.name


def _make_fx_name(site_name, template_name, variable):
    fx_name = f'{site_name} {template_name} {variable}'
    # Some site names are too long and exceed the API's limits,
    # in those cases. Use the abbreviated version.
    if len(fx_name) > 63:
        for old, new in (
            ('Persistence', 'Pers'),
            ('persistence', 'pers'),
            ('Fifteen-minute', '15 min'),
            ('Five-minute', '5 min'),
        ):
            template_name = template_name.replace(old, new)
        suffix = f'{template_name} {variable}'
        # ensure name can containe at least first word of site
        # name and the suffix
        if (len(site_name.split(' ')[0]) + len(suffix)) > 62:
            raise ValueError('Template/site name too long together')
        while len(fx_name) > 63:
            # drop the last word
            site_name = ' '.join(site_name.split(' ')[:-1])
            fx_name = f"{site_name} {suffix}"
        logger.warning("Forecast name truncated to %s", fx_name)
    return fx_name


def create_one_forecast(api, site, template_forecast, variable,
                        creation_validation=lambda x: True, **extra_params):
    """Creates a new Forecast or ProbabilisticForecast for the variable
    and site based on the template forecast.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.site
        A site object.
    template_forecast : solarforecastarbiter.datamodel.Forecast
        A Forecast or ProbabilisticForecast object that will only have name,
        site, variable, and issue_time_of_day replaced. New keys may be added
        to extra parameters.
    variable : string
        Variable measured in the forecast.
    creation_validation : function
        Function that expects a Forecast or ProbabilisticForecast object
        and raises a ValueError if the forecast is invalid just before it
        is created.
    **extra_params : dict
        Other key, value pairs to add to the extra_parameters of the Forecast
        object.

    Returns
    -------
    created
        The datamodel object of the newly created forecast.
    """
    extra_parameters = json.loads(template_forecast.extra_parameters)
    extra_parameters.update(extra_params)
    site_name = site_name_no_network(site)
    fx_name = _make_fx_name(site_name, template_forecast.name, variable)
    # adjust issue_time_of_day to localtime for standard time, not DST
    issue_datetime = pd.Timestamp.combine(
        dt.date(2019, 2, 1), template_forecast.issue_time_of_day,
        ).tz_localize(site.timezone).tz_convert('UTC')
    # make sure this is the first possible issue for the UTC day
    orig_date = issue_datetime.floor('1d')
    while issue_datetime - template_forecast.run_length >= orig_date:
        issue_datetime -= template_forecast.run_length
    issue_time_of_day = issue_datetime.time()

    forecast = template_forecast.replace(
        name=fx_name, extra_parameters=json.dumps(extra_parameters),
        site=site, variable=variable, issue_time_of_day=issue_time_of_day,
    )
    existing = existing_forecasts(api)
    if (
            forecast.name in existing and
            existing[forecast.name].site == forecast.site
    ):
        logger.info('Forecast, %s, already exists', forecast.name)
        return existing[forecast.name]

    if isinstance(forecast, ProbabilisticForecast):
        create_func = api.create_probabilistic_forecast
    else:
        create_func = api.create_forecast

    try:
        creation_validation(forecast)
    except ValueError as exc:
        logger.error('Validation failed on creation of %s forecast '
                     'at Site %s with message %s', variable, site.name, exc)
        return
    try:
        created = create_func(forecast)
    except HTTPError as e:
        logger.error(f'Failed to create {variable} forecast at Site '
                     f'{site.name}.')
        logger.debug(f'HTTP Error: {e.response.text}')
    else:
        logger.info(f"Forecast {created.name} created successfully.")
        return created


def create_nwp_forecasts(api, site, variables, templates):
    """Create Forecast objects for each of variables, if NWP forecasts
    can be made for that variable. Each Forecast in templates will be
    updated with the appropriate parameters for each variable. Forecasts
    will also be grouped together via 'piggyback_on'.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.Site
        A site object.
    variables : list-like
        List of variables to make a new forecast for each of the template
        forecasts
    templates : list of datamodel.Forecasts or datamodel.ProbabilisticForecast
        Forecasts that will be used as templates for many fields. See
        :py:func:`solarforecastarbiter.io.reference_data.common.create_one_forecast`
        for the fields that are required vs overwritten.

    Raises
    ------
    ValueError
        If the site is outside the domain of the current NWP forecasts.
    """  # NOQA
    if not is_in_nwp_domain(site):
        raise ValueError(
            f'Site {site.name} is outside the domain of the current NWP '
            'forecasts')
    vars_ = set(variables)
    diff = vars_ - CURRENT_NWP_VARIABLES
    if diff:
        logger.warning('NWP forecasts for %s cannot currently be made',
                       diff)
    vars_ = vars_ & CURRENT_NWP_VARIABLES
    if len(vars_) == 0:
        return []

    if 'ac_power' in vars_:
        primary = 'ac_power'
        vars_.remove('ac_power')
    elif 'ghi' in vars_:
        primary = 'ghi'
        vars_.remove('ghi')
    else:
        # pick random var
        primary = vars_.pop()

    created = []
    for template_fx in templates:
        logger.info('Creating forecasts based on %s at site %s',
                    template_fx.name, site.name)
        primary_fx = create_one_forecast(api, site, template_fx, primary)
        created.append(primary_fx)
        piggyback_on = primary_fx.forecast_id
        for var in vars_:
            created.append(
                create_one_forecast(api, site, template_fx, var,
                                    piggyback_on=piggyback_on))
    return created


def create_persistence_forecasts(api, site, variables, templates):
    """Create persistence Forecast objects for each Observation at the
    ``site`` with variable in ``variables``. Each Forecast in templates
     will be updated with the appropriate parameters for each variable.
    By default, *index* persistence forecasts are made for variables
    with valid index persistence functions namely (ghi, dni, dhi, ac_power).

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.Site
        A site object with Observations whose data will be persisted.
    variables : list-like
        List of variables to make a new forecast for each of the template
        forecasts
    templates : list of datamodel.Forecasts or datamodel.ProbabilisticForecast
        Forecasts that will be used as templates for many fields. See
        :py:func:`solarforecastarbiter.io.reference_data.common.create_one_forecast`
        for the fields that are required vs overwritten.

    """  # NOQA
    created = []
    for obs in api.list_observations():
        if obs.site != site or obs.variable not in variables:
            continue
        for template_fx in templates:
            logger.info('Creating forecast based on %s and observation %s',
                        template_fx.name, obs.name)
            use_index = (
                template_fx.run_length < pd.Timedelta('1d') and
                obs.variable in ('ghi', 'dni', 'dhi', 'ac_power')
            )
            validation_func = partial(check_persistence_compatibility, obs,
                                      index=use_index)
            # net_load might go here, although other changes might be required
            fx_id = create_one_forecast(api, site, template_fx, obs.variable,
                                        creation_validation=validation_func,
                                        observation_id=obs.observation_id,
                                        index_persistence=use_index)
            created.append(fx_id)
    return created


def create_forecasts(api, site, variables, templates):
    """Create Forecast objects (NWP based and persistence) for each of
    variables. Each Forecast in templates will be updated with the
    appropriate parameters for each variable.

    Templates with the 'is_reference_persistence_forecast' key in
    'extra_parameters' are assumed to be persistence forecasts, and others
    are assumed to be NWP forecasts.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.Site
        A site object.
    variables : list-like
        List of variables to make a new forecast for each of the template
        forecasts
    templates : list of datamodel.Forecasts or datamodel.ProbabilisticForecast
        Forecasts that will be used as templates for many fields. See
        :py:func:`solarforecastarbiter.io.reference_data.common.create_one_forecast`
        for the fields that are required vs overwritten.

    Returns
    -------
    list
        A list of all Forecast/ProbabilisticForecast objects created

    See Also
    --------
    solarforecastarbiter.io.reference_data.common.create_nwp_forecasts
    solarforecastarbiter.io.reference_data.common.create_persistence_forecasts

    """  # NOQA: E501
    persistence_templates = []
    nwp_templates = []
    for template in templates:
        if 'is_reference_persistence_forecast' in template.extra_parameters:
            persistence_templates.append(template)
        else:
            nwp_templates.append(template)
    nwp_created = create_nwp_forecasts(api, site, variables, nwp_templates)
    persist_created = create_persistence_forecasts(
        api, site, variables, persistence_templates)
    return nwp_created + persist_created


def apply_json_site_parameters(json_sitefile, site):
    """Updates site metadata with modeling parameters found in a json file.

    Parameters
    ----------
    json_sitefile: str
        Absolute path of a json file with a 'sites' key containing a list of
        sites in the Solar Forecast Arbiter JSON format.
    site: dict

    Returns
    -------
    dict
        Copy of inputs plus a new key 'modeling_parameters'.
    """
    with open(json_sitefile) as fp:
        sites_metadata = json.load(fp)['sites']
    site_api_id = str(site['extra_parameters']['network_api_id'])
    for site_metadata in sites_metadata:
        site_extra_params = json.loads(site_metadata['extra_parameters'])
        if str(site_extra_params['network_api_id']) == site_api_id:
            site_out = site.copy()
            site_out['modeling_parameters'] = site_metadata[
                'modeling_parameters']
            site_out['extra_parameters'].update(site_extra_params)
            return site_out
    return site
