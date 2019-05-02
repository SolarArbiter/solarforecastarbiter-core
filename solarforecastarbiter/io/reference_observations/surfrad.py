"""Functions for Creating and Updating NOAA SURFRAD related objects
within the SolarForecastArbiter
"""
import json
import logging
from urllib.error import URLError


import pandas as pd


from pvlib import iotools
from solarforecastarbiter.io.utils import observation_df_to_json_payload as obs_to_payload # NOQA
from solarforecastarbiter.datamodel import Site, Observation


# Format strings for the location of a surfrad data file. Expects the following
# variables:
#   abrv: The SURFRAD site's abbreviated name, i.e. Bondville, IL is 'bon'
#   year: Year as a 4 digit number.
#   year_2d: Year as a 0 padded 2 digit  number
#   jday: day of the year as a 3 digit number.
SURFRAD_FTP_DIR = "ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad"
REALTIME_URL = SURFRAD_FTP_DIR + "/realtime/{abrv}/{abrv}{year_2d}{jday}.dat"
ARCHIVE_URL = SURFRAD_FTP_DIR + "/{abrv}/{year}/{abrv}{year_2d}{jday}.dat"

# A list of variables that are available in all SURFRAD files to
# parse and create observations for.
surfrad_variables = ['ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed']
rename_mapping = {'temp_air': 'air_temperature'}

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


def fetch(api, site, start, end, realtime=False):
    """Retrieve observation data for a surfrad site between start and end.

    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : datamodel.Site
        Site object with the appropriate metadata.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    realtime : bool
        Whether or not to look for realtime data. Note that this data is
        raw, unverified data from the instruments.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    if realtime:
        url_format = REALTIME_URL
    else:
        url_format = ARCHIVE_URL
    extra_params = decode_extra_parameters(site)
    abbreviation = extra_params['network_api_abbreviation']
    date_range = pd.date_range(start, end)
    single_day_dfs = []
    for day in date_range():
        filename = url_format.format(abrv=abbreviation,
                                     year=day.year,
                                     year_2d=day.strftime('%y'),
                                     jday=day.strftime('%j'))
        logger.info('Requesting SURFRAD site {}.'.format(site.name))
        try:
            surfrad_day = iotools.read_surfrad(filename)[0]
        except URLError:
            logger.warning(f'Could not SURFRAD data for site {site.name}')
            logger.debug(f'Failed SURFRAD URL: {site.name}')
            continue
        else:
            single_day_dfs.append(surfrad_day)
    all_period_data = pd.concat(single_day_dfs)
    all_period_data = all_period_data.rename(rename_mapping)
    return all_period_data


def create_site(api, site):
    """
    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : dict
        Dictionary describing the site to post. This will be instantiated as
        a datamodel.Site object and the value of 'extra_parameters' will be
        serialized to json.

    Returns
    -------
    uuid : string
        UUID of the created Site.
    """
    site.update({'extra_parameters': json.dumps(site['extra_parameters'])})
    site_to_create = Site.from_dict(site)
    return api.create_site(site_to_create)


def create_observation(api, site, variable):
    """ Creates a new Observation for the variable and site.

    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.site
        A site object.

    Returns
    -------
    uuid : string
        The uuid of the newly created Observation.

    Raises
    ------
    ValueError
        If the post failed.
    """
    extra_parameters = decode_extra_parameters(site)
    observation = Observation.from_dict({
        'name': f"{site.name} {variable}",
        'interval_label': 'ending',
        'interval_length': extra_parameters['observation_interval_length'],
        'interval_value_type': 'interval_mean',
        'site': site,
        'uncertainty': 0,
        'variable': variable,
        'extra_parameters': site.extra_parameters
    })
    created = api.create_observation(observation)
    logger.info(f"{created.name} created successfully.")


def initialize_site_observations(api, site):
    """Creates an observaiton at the site for each variable in surfrad_variables.

    Parameters
    ----------
    site : datamodel.Site
        The site object for which to create Observations.
    """
    for variable in surfrad_variables:
        try:
            create_observation(api, site, variable)
        except ValueError as e:
            logger.error(e)


def initialize_metadata_objects(api, sites):
    """Create all Sites and associated Observations for Surfrad
    stations.

    Parameters
    ----------
    api: solarforecastarbiter.io.api.APISession
        An active Reference user session.
    """
    for site_dict in sites:
        site = create_site(api, site_dict)
        initialize_site_observations(api, site)


def update_observation_data(api, start, end):
    """Post new observation data to a list of Surfrad Observations
    from start to end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    sites = api.list_sites()
    observations = api.list_observations()
    for site in sites:
        obs_df = fetch(api, site, start, end)
        site_observations = observations
        for observation in site_observations:
            logger.info(
                'Updating {observation.name} from '
                '{start.strftime("%Y%m%dT%H%MZ")} '
                'to {end.strftime("%Y%m%dT%H%MZ")}.')
            api.post_observation_values(observation.observation_id,
                                        obs_df[[observation.variable]])
