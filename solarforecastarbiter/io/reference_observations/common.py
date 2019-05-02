import json


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
