import json
import logging


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.datamodel import Observation


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
    """
    try:
        params = json.loads(metadata.extra_parameters)
    except (json.decoder.JSONDecodeError, TypeError):
        raise ValueError(f'Could not read extra parameters of {metadata.name}')
    required_keys = ['network_api_id', 'network_api_abbreviation',
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
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.site
        A site object.
    variable : string
        Variable measured in the observation.
    extra_params : dict, optional
        If provided, this dict will be serialized as the 'extra_parameters'
        field of the observation, otherwise the site's field is copied over.
        Must contain the key 'observation_interval_length'.

    Other Parameters
    ----------------
    name: string
        Defaults to `<site.name> <variable>`
    interval_label: string
        Defaults to 'ending'
    interval_value_type: string
        Defaults to 'interval_mean'
    uncertainty: float
        Defaults to 0.

    Returns
    -------
    created
        The datamodel object of the newly created observation.

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
        'uncertainty': kwargs.get('uncertainty', 0),
        'variable': variable,
        'extra_parameters': json.dumps(extra_parameters)
    })
    try:
        created = api.create_observation(observation)
    except HTTPError as e:
        logger.error(f'Failed to create {variable} observation at Site '
                     f'{site.name}.')
        logger.debug(f'HTTP Error: {e.response.text}')
    else:
        logger.info(f"{created.name} created successfully.")
        return created


def update_site_observations(api, fetch_func, site, observations,
                             start, end):
    """Updates data for all reference observations at a given site
    for the period between start and end.

    Prameters
    ---------
    api : solarforecastarbiter.io.APISession
        An active Reference user session.

    fetch_func : function
        A function that requests data and returns a DataFrame for a given site.
        The function should accept the parameters (api, site, start end) as
        they appear in this function.
    site : solarforecastarbiter.datamodel.Site
        The Site with observations to update.
    observations : list of solarforecastarbiter.datamodel.Observation
        A full list of reference Observations to search.
    """
    obs_df = fetch_func(api, site, start, end)
    data_in_range = obs_df[start:end]
    if data_in_range.empty:
        logger.warning(f'Data for site {site.name} contained no entries '
                       f'from {start} to {end}.')
        return
    site_observations = [obs for obs in observations if obs.site == site]
    for obs in site_observations:
        post_observation_data(api, obs, data_in_range, start, end)


def post_observation_data(api, observation, data, start, end):
    """
    Parameters
    ----------
    api : solarforecastarbiter.io.APISession
        An active Reference user session.
    observation : solarforecastarbiter.datamodel.Observation
        Data model object corresponding to the Observation to update.
    data : pandas.DataFrame
        Dataframe of values to post containing a column labelled with
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
    try:
        var_df = data[[variable]]
    except KeyError:
        logger.error(f'{variable} could not be found in the data file '
                     f'from {data.index[0]} to {data.index[-1]}'
                     f'for Observation {observation.name}')
        return
    var_df = var_df.rename(columns={variable: 'value'})
    var_df['quality_flag'] = 0
    # remove all future values, some files have forward filled nightly data
    var_df = var_df[start:min(end, pd.Timestamp.now(tz='UTC'))]
    # Drop NaNs and skip post if empty.
    var_df = var_df.dropna()
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
    return string.translate(string.maketrans('_', ' ', '(){}/\\[]@-.'))


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
