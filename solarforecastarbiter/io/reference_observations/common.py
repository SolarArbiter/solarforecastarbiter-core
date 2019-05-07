import json
import logging


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
    return json.loads(metadata.extra_parameters)


def check_network(networks, metadata):
    """Decodes extra_parameters and checks if an object is in
    the network.

    Parameters
    ----------
    network: string or list
        The name of the network to check for or a list of
        networks to check against.
    metadata
        An instantiated dataclass from teh datamodel.

    Returns
    -------
    bool
        True if the site belongs to the surfrad network.
    """
    extra_params = decode_extra_parameters(metadata)
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
        List of datamodel objects to filer
    networks: list
        List of networks to check for.

    Returns
    -------
    filtered
        List of filtered objects objects.
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
        extra_parameters = decode_extra_parameters(site)
    observation = Observation.from_dict({
        'name': kwargs.get('name', f'{site.name} {variable}'),
        'interval_label': kwargs.get('interval_label','ending'),
        'interval_length': extra_parameters['observation_interval_length'],
        'interval_value_type': kwargs.get('interval_value_type','interval_mean'),
        'site': site,
        'uncertainty': kwargs.get('uncertainty', 0),
        'variable': variable,
        'extra_parameters': json.dumps(extra_parameters)
    })
    created = api.create_observation(observation)
    logger.info(f"{created.name} created successfully.")
    return created
